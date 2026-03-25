"""
Memory Profiling Utilities for Goldcrest Architecture

This module provides memory profiling and isolation capabilities to measure
and compare memory usage across different execution phases in the plugin system.
"""
import psutil
import torch
import gc
import json
import time
import threading
import subprocess
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS = True


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time"""
    rss_mb: float = 0.0  # Resident Set Size in MB
    vms_mb: float = 0.0  # Virtual Memory Size in MB
    gpu_allocated_mb: float = 0.0  # GPU memory allocated in MB
    gpu_cached_mb: float = 0.0  # GPU memory cached in MB
    timestamp: float = 0.0
    phase_name: Optional[str] = None

    # Aliases for backward compatibility
    @property
    def gpu_allocated(self) -> float:
        """Alias for gpu_allocated_mb."""
        return self.gpu_allocated_mb

    @property
    def gpu_reserved(self) -> float:
        """Alias for gpu_cached_mb (reserved/cached memory)."""
        return self.gpu_cached_mb

    @property
    def cpu_used(self) -> float:
        """Alias for rss_mb (CPU RSS memory)."""
        return self.rss_mb

    @staticmethod
    def create(
        gpu_allocated: float = 0.0,
        gpu_reserved: float = 0.0,
        cpu_used: float = 0.0,
        timestamp: Optional[float] = None,
        **kwargs,
    ) -> "MemorySnapshot":
        """
        Factory method with simplified/alternative parameter names.

        Args:
            gpu_allocated: GPU allocated memory in MB.
            gpu_reserved: GPU reserved/cached memory in MB.
            cpu_used: CPU RSS memory in MB.
            timestamp: Timestamp (defaults to current time).
            **kwargs: Additional fields (rss_mb, vms_mb, etc.).

        Returns:
            MemorySnapshot instance.
        """
        return MemorySnapshot(
            rss_mb=kwargs.get("rss_mb", cpu_used),
            vms_mb=kwargs.get("vms_mb", 0.0),
            gpu_allocated_mb=kwargs.get("gpu_allocated_mb", gpu_allocated),
            gpu_cached_mb=kwargs.get("gpu_cached_mb", gpu_reserved),
            timestamp=timestamp if timestamp is not None else time.time(),
            phase_name=kwargs.get("phase_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemorySnapshot":
        """Create MemorySnapshot from serialized dictionary."""
        return MemorySnapshot(
            rss_mb=float(data.get("rss_mb", 0.0) or 0.0),
            vms_mb=float(data.get("vms_mb", 0.0) or 0.0),
            gpu_allocated_mb=float(data.get("gpu_allocated_mb", 0.0) or 0.0),
            gpu_cached_mb=float(data.get("gpu_cached_mb", 0.0) or 0.0),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
            phase_name=data.get("phase_name"),
        )


@dataclass
class MemoryProfile:
    """Memory profile for a specific execution phase"""
    phase_name: str = ""
    pre_execution: Optional[MemorySnapshot] = None
    post_execution: Optional[MemorySnapshot] = None
    peak_memory: Optional[MemorySnapshot] = None

    def __post_init__(self):
        """Initialize default snapshots if not provided."""
        if self.pre_execution is None:
            self.pre_execution = MemorySnapshot(timestamp=time.time())
        if self.post_execution is None:
            self.post_execution = MemorySnapshot(timestamp=time.time())
    
    @property
    def memory_delta(self) -> Dict[str, float]:
        """Calculate memory delta between pre and post execution"""
        return {
            "rss_delta_mb": self.post_execution.rss_mb - self.pre_execution.rss_mb,
            "vms_delta_mb": self.post_execution.vms_mb - self.pre_execution.vms_mb,
            "gpu_allocated_delta_mb": self.post_execution.gpu_allocated_mb - self.pre_execution.gpu_allocated_mb,
            "gpu_cached_delta_mb": self.post_execution.gpu_cached_mb - self.pre_execution.gpu_cached_mb
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "phase_name": self.phase_name,
            "pre_execution": self.pre_execution.to_dict(),
            "post_execution": self.post_execution.to_dict(),
            "peak_memory": self.peak_memory.to_dict() if self.peak_memory else None,
            "memory_delta": self.memory_delta
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryProfile":
        """Create MemoryProfile from serialized dictionary."""
        pre = MemorySnapshot.from_dict(data.get("pre_execution", {}) or {})
        post = MemorySnapshot.from_dict(data.get("post_execution", {}) or {})
        peak_data = data.get("peak_memory")
        peak = MemorySnapshot.from_dict(peak_data) if isinstance(peak_data, dict) else None
        phase_name = data.get("phase_name") or pre.phase_name or post.phase_name or ""
        return MemoryProfile(
            phase_name=phase_name,
            pre_execution=pre,
            post_execution=post,
            peak_memory=peak,
        )


class MemoryProfiler:
    """
    Memory profiler for isolating and monitoring memory usage during plugin execution

    Provides context managers and utilities for measuring memory consumption
    across different phases of the pipeline execution.
    """

    def __init__(self, isolate: bool = False):
        """
        Initialize the memory profiler.

        Args:
            isolate: If True, enables subprocess isolation mode for profiling.
                     This is useful for accurate memory measurement without
                     contamination from the parent process.
        """
        self.profiles: Dict[str, MemoryProfile] = {}
        self.baseline_memory = self._get_memory_snapshot("baseline")
        self._monitoring_active = False
        self.isolate = isolate
    
    def _get_memory_snapshot(self, phase_name: Optional[str] = None, sync_gpu: bool = False) -> MemorySnapshot:
        """
        Get current memory usage snapshot
        
        Args:
            phase_name: Optional name for this snapshot
            
        Returns:
            MemorySnapshot with current memory usage
        """
        try:
            if sync_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # GPU memory (if CUDA available)
            gpu_allocated = 0.0
            gpu_cached = 0.0
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024

            
            return MemorySnapshot(
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                gpu_allocated_mb=gpu_allocated,
                gpu_cached_mb=gpu_cached,
                timestamp=time.time(),
                phase_name=phase_name
            )
            
        except Exception as e:
            # Fallback to minimal memory info if psutil fails
            return MemorySnapshot(
                rss_mb=0.0,
                vms_mb=0.0,
                gpu_allocated_mb=0.0,
                gpu_cached_mb=0.0,
                timestamp=time.time(),
                phase_name=phase_name
            )
    
    @contextmanager
    def profile(
        self,
        phase_name: str,
        cleanup_before: bool = True,
        sync_gpu: bool = True,
    ):
        """
        Simplified context manager to profile memory usage (alias for profile_execution).

        Args:
            phase_name: Name of the execution phase
            cleanup_before: Whether to run garbage collection before profiling
            sync_gpu: Synchronize GPU before snapshots for more accurate stats

        Yields:
            MemoryProfiler instance for additional operations during execution
        """
        with self.profile_execution(
            phase_name,
            cleanup_before=cleanup_before,
            sync_gpu=sync_gpu,
        ) as profiler:
            yield profiler

    @contextmanager
    def profile_execution(
        self,
        phase_name: str,
        cleanup_before: bool = True,
        sync_gpu: bool = True,
        sample_interval: float = 0.02,
        track_cuda_peak: bool = True,
    ):
        """
        Context manager to profile memory usage during execution phase

        Args:
            phase_name: Name of the execution phase
            cleanup_before: Whether to run garbage collection before profiling
            sync_gpu: Synchronize GPU before snapshots for more accurate stats
            sample_interval: Sampling interval in seconds for CPU RSS/VMS peaks
            track_cuda_peak: Whether to record CUDA peak allocated/reserved memory

        Yields:
            MemoryProfiler instance for additional operations during execution
        """
        if cleanup_before:
            self._cleanup_memory(sync_gpu=sync_gpu)
        
        # Capture pre-execution memory
        pre_memory = self._get_memory_snapshot(f"{phase_name}_pre", sync_gpu=sync_gpu)
        
        # Start lightweight background sampler to capture peak memory during phase
        self._monitoring_active = True
        stop_event = threading.Event()
        peak_holder = {"snapshot": pre_memory}
        if torch.cuda.is_available() and track_cuda_peak:
            if sync_gpu:
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def _monitor_peak():
            local_peak = pre_memory
            # For GPU, track peak via torch if available
            # Sample RSS/VMS periodically
            while not stop_event.is_set():
                snap = self._get_memory_snapshot(f"{phase_name}_sample", sync_gpu=False)
                # Update peak by RSS first; tie-break with VMS
                if (snap.rss_mb > local_peak.rss_mb) or (
                    abs(snap.rss_mb - local_peak.rss_mb) < 1e-6 and snap.vms_mb > local_peak.vms_mb
                ):
                    local_peak = snap
                time.sleep(max(0.001, float(sample_interval)))
            # If CUDA, try to reflect true GPU peaks
            if torch.cuda.is_available() and track_cuda_peak:
                # Create a final snapshot with CUDA peak numbers
                peak_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_res = torch.cuda.max_memory_reserved() / 1024 / 1024
                local_peak = MemorySnapshot(
                    rss_mb=local_peak.rss_mb,
                    vms_mb=local_peak.vms_mb,
                    gpu_allocated_mb=max(local_peak.gpu_allocated_mb, peak_alloc),
                    gpu_cached_mb=max(local_peak.gpu_cached_mb, peak_res),
                    timestamp=time.time(),
                    phase_name=f"{phase_name}_peak",
                )
            peak_holder["snapshot"] = local_peak

        monitor_thread = threading.Thread(target=_monitor_peak, daemon=True)
        monitor_thread.start()

        try:
            yield self
        finally:
            # Stop monitoring and wait briefly for thread to finish
            self._monitoring_active = False
            stop_event.set()
            monitor_thread.join(timeout=0.2)
            # Capture post-execution memory
            post_memory = self._get_memory_snapshot(f"{phase_name}_post", sync_gpu=sync_gpu)
            
            # Create and store profile
            profile = MemoryProfile(
                phase_name=phase_name,
                pre_execution=pre_memory,
                post_execution=post_memory,
                peak_memory=peak_holder.get("snapshot")
            )
            
            self.profiles[phase_name] = profile
    
    def _cleanup_memory(self, sync_gpu: bool = True) -> None:
        """Perform memory cleanup before profiling"""
        # Clear GPU cache if available
        if torch.cuda.is_available():
            try:
                if sync_gpu:
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if sync_gpu:
                    torch.cuda.synchronize()  # Wait for operations to complete
            except:
                pass  # Ignore CUDA errors
        
        # Run garbage collection
        gc.collect()
        
        # Small delay to let memory settle
        time.sleep(0.1)
    
    def get_profile(self, phase_name: str) -> Optional[MemoryProfile]:
        """
        Get memory profile for a specific phase
        
        Args:
            phase_name: Name of the execution phase
            
        Returns:
            MemoryProfile or None if not found
        """
        return self.profiles.get(phase_name)
    
    def compare_phases(self, phase1: str, phase2: str) -> Dict[str, float]:
        """
        Compare memory usage between two execution phases
        
        Args:
            phase1: First phase name
            phase2: Second phase name
            
        Returns:
            Dictionary with memory usage comparison
            
        Raises:
            ValueError: If either phase is not found
        """
        if phase1 not in self.profiles:
            raise ValueError(f"Phase '{phase1}' not found in profiles")
        if phase2 not in self.profiles:
            raise ValueError(f"Phase '{phase2}' not found in profiles")
        
        profile1 = self.profiles[phase1]
        profile2 = self.profiles[phase2]
        
        delta1 = profile1.memory_delta
        delta2 = profile2.memory_delta
        
        return {
            "rss_difference_mb": delta2["rss_delta_mb"] - delta1["rss_delta_mb"],
            "vms_difference_mb": delta2["vms_delta_mb"] - delta1["vms_delta_mb"],
            "gpu_allocated_difference_mb": delta2["gpu_allocated_delta_mb"] - delta1["gpu_allocated_delta_mb"],
            "gpu_cached_difference_mb": delta2["gpu_cached_delta_mb"] - delta1["gpu_cached_delta_mb"],
            "phase1_total_delta_mb": delta1["rss_delta_mb"],
            "phase2_total_delta_mb": delta2["rss_delta_mb"]
        }
    
    def get_memory_efficiency_score(self, phase_name: str) -> float:
        """
        Calculate memory efficiency score for a phase (lower is better)
        
        Args:
            phase_name: Name of the execution phase
            
        Returns:
            Efficiency score (MB of memory used per second)
        """
        if phase_name not in self.profiles:
            return float('inf')
        
        profile = self.profiles[phase_name]
        delta = profile.memory_delta
        time_delta = profile.post_execution.timestamp - profile.pre_execution.timestamp
        
        if time_delta <= 0:
            return float('inf')
        
        # Memory used per second (lower is better)
        return delta["rss_delta_mb"] / time_delta
    
    def export_report(self, output_path: str) -> None:
        """
        Export comprehensive memory usage report
        
        Args:
            output_path: Path to save the JSON report
        """
        report = {
            "baseline_memory": self.baseline_memory.to_dict(),
            "execution_profiles": {
                phase_name: profile.to_dict()
                for phase_name, profile in self.profiles.items()
            },
            "summary": self._generate_summary(),
            "timestamp": time.time()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    def import_report(self, report: Dict[str, Any], replace: bool = True) -> None:
        """Load serialized report data into this profiler."""
        if replace:
            self.profiles.clear()
        baseline = report.get("baseline_memory")
        if isinstance(baseline, dict):
            self.baseline_memory = MemorySnapshot.from_dict(baseline)
        for phase_name, profile_data in (report.get("execution_profiles") or {}).items():
            if not isinstance(profile_data, dict):
                continue
            profile = MemoryProfile.from_dict(profile_data)
            if not profile.phase_name:
                profile.phase_name = phase_name
            self.profiles[phase_name] = profile

    def load_report(self, report_path: Union[str, Path], replace: bool = True) -> Dict[str, Any]:
        """Load a JSON report from disk and import its profiles."""
        path = Path(report_path)
        report = json.loads(path.read_text())
        self.import_report(report, replace=replace)
        return report

    def profile_execution_subprocess(
        self,
        command: List[str],
        report_path: Union[str, Path],
        phase_name: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        replace: bool = True,
    ) -> MemoryProfile:
        """Run a subprocess that emits a MemoryProfiler report, then load it."""
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            timeout=timeout,
        )
        self.load_report(report_path, replace=replace)
        if phase_name is None:
            if len(self.profiles) == 1:
                return next(iter(self.profiles.values()))
            if self.profiles:
                first_key = next(iter(self.profiles.keys()))
                return self.profiles[first_key]
            raise ValueError("No profiles found in subprocess report")
        if phase_name not in self.profiles:
            raise ValueError(f"Phase '{phase_name}' not found in subprocess report")
        return self.profiles[phase_name]
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all phases"""
        if not self.profiles:
            return {}
        
        # Calculate totals
        total_memory_delta = sum(
            profile.memory_delta["rss_delta_mb"]
            for profile in self.profiles.values()
        )
        
        # Find peak memory usage
        peak_rss = max(
            (profile.peak_memory.rss_mb if profile.peak_memory else profile.post_execution.rss_mb)
            for profile in self.profiles.values()
        )
        
        # Calculate average efficiency
        efficiency_scores = [
            self.get_memory_efficiency_score(phase_name)
            for phase_name in self.profiles.keys()
        ]
        valid_scores = [score for score in efficiency_scores if score != float('inf')]
        avg_efficiency = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return {
            "total_phases": len(self.profiles),
            "total_memory_delta_mb": total_memory_delta,
            "peak_memory_usage_mb": peak_rss,
            "baseline_memory_mb": self.baseline_memory.rss_mb,
            "average_efficiency_score": avg_efficiency,
            "most_efficient_phase": min(
                self.profiles.keys(),
                key=self.get_memory_efficiency_score,
                default=None
            ),
            "least_efficient_phase": max(
                self.profiles.keys(),
                key=self.get_memory_efficiency_score,
                default=None
            )
        }
    
    def print_summary(self) -> None:
        """Print memory usage summary to console"""
        summary = self._generate_summary()
        
        print("="*60)
        print("MEMORY PROFILER SUMMARY")
        print("="*60)
        print(f"Baseline Memory Usage: {summary.get('baseline_memory_mb', 0):.1f} MB")
        print(f"Peak Memory Usage: {summary.get('peak_memory_usage_mb', 0):.1f} MB")
        print(f"Total Memory Delta: {summary.get('total_memory_delta_mb', 0):.1f} MB")
        print(f"Total Phases Profiled: {summary.get('total_phases', 0)}")
        print()
        
        if self.profiles:
            print("Phase Details:")
            for phase_name, profile in self.profiles.items():
                delta = profile.memory_delta
                print(f"  {phase_name}:")
                print(f"    Memory Delta: {delta['rss_delta_mb']:.1f} MB")
                print(f"    Efficiency Score: {self.get_memory_efficiency_score(phase_name):.2f} MB/s")
        
        print("="*60)
    
    def clear_profiles(self) -> None:
        """Clear all stored memory profiles"""
        self.profiles.clear()


# Convenience function for quick profiling
def profile_memory_usage(phase_name: str, cleanup_before: bool = True, sync_gpu: bool = True):
    """
    Decorator/context manager factory for quick memory profiling
    
    Args:
        phase_name: Name of the phase to profile
        cleanup_before: Whether to cleanup memory before profiling
        sync_gpu: Synchronize GPU before snapshots for more accurate stats
        
    Returns:
        Context manager for memory profiling
    """
    profiler = MemoryProfiler()
    return profiler.profile_execution(phase_name, cleanup_before, sync_gpu=sync_gpu)
