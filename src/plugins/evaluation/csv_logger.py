"""
CSV Logging Integration for Evaluation Plugins

Enhanced CSV structure implementation with result comparison and analysis tools.
Migrated and enhanced from existing CSV logging functionality.
"""
import csv
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict


@dataclass
class ExperimentSession:
    """Tracks a complete experiment session"""
    experiment_id: str
    start_time: datetime
    config_file: str
    user: Optional[str] = None
    end_time: Optional[datetime] = None
    status: str = "running"
    error_message: Optional[str] = None


@dataclass
class ModelRecord:
    """Model loading and information record"""
    experiment_id: str
    timestamp: datetime
    model_name: str
    model_type: str  # "baseline" or "compressed"
    parameter_count: int
    size_mb: float
    loading_time: float
    device: str
    precision: str
    architecture_info: Dict[str, Any]


@dataclass
class EvaluationRecord:
    """Evaluation results record"""
    experiment_id: str
    timestamp: datetime
    model_type: str  # "baseline" or "compressed"
    evaluation_type: str  # "lm_eval" or "profile"
    plugin_name: str
    task_name: str
    metric_name: str
    metric_value: Union[float, int, str]
    evaluation_params: Dict[str, Any]
    execution_time: float


@dataclass
class CompressionRecord:
    """Compression results record"""
    experiment_id: str
    timestamp: datetime
    compression_method: str
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_time: float
    parameters: Dict[str, Any]
    memory_usage_mb: float


class CSVLogger:
    """
    Enhanced CSV logging system for evaluation plugins
    
    Provides structured CSV logging with experiment tracking,
    result comparison, and analysis capabilities.
    """
    
    def __init__(self, output_dir: str = "./logs/csv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current experiment session
        self.current_session: Optional[ExperimentSession] = None
        
        # CSV file paths
        self.experiments_file = self.output_dir / "experiments.csv"
        self.models_file = self.output_dir / "models.csv"
        self.evaluations_file = self.output_dir / "evaluations.csv"
        self.compressions_file = self.output_dir / "compressions.csv"
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
    
    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with appropriate headers"""
        # Experiments file
        if not self.experiments_file.exists():
            with open(self.experiments_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'experiment_id', 'start_time', 'end_time', 'duration_minutes',
                    'config_file', 'user', 'status', 'error_message'
                ])
                writer.writeheader()
        
        # Models file
        if not self.models_file.exists():
            with open(self.models_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'experiment_id', 'timestamp', 'model_name', 'model_type',
                    'parameter_count', 'size_mb', 'loading_time', 'device',
                    'precision', 'architecture_info'
                ])
                writer.writeheader()
        
        # Evaluations file  
        if not self.evaluations_file.exists():
            with open(self.evaluations_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'experiment_id', 'timestamp', 'model_type', 'evaluation_type',
                    'plugin_name', 'task_name', 'metric_name', 'metric_value',
                    'evaluation_params', 'execution_time'
                ])
                writer.writeheader()
        
        # Compressions file
        if not self.compressions_file.exists():
            with open(self.compressions_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'experiment_id', 'timestamp', 'compression_method',
                    'original_size_mb', 'compressed_size_mb', 'compression_ratio',
                    'compression_time', 'parameters', 'memory_usage_mb'
                ])
                writer.writeheader()
    
    def start_experiment(self, experiment_name: str, config_file: str, 
                        user: Optional[str] = None) -> str:
        """
        Start new experiment logging session
        
        Args:
            experiment_name: Name/identifier for the experiment
            config_file: Path to configuration file used
            user: Optional user identifier
            
        Returns:
            Generated experiment ID
        """
        # Generate unique experiment ID
        timestamp = datetime.now()
        experiment_id = f"{experiment_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create session
        self.current_session = ExperimentSession(
            experiment_id=experiment_id,
            start_time=timestamp,
            config_file=config_file,
            user=user,
            status="running"
        )
        
        # Log to experiments CSV
        self._write_experiment_record(self.current_session)
        
        return experiment_id
    
    def end_experiment(self, status: str = "completed", 
                      error_message: Optional[str] = None) -> None:
        """
        End current experiment session
        
        Args:
            status: Final experiment status (completed, failed, cancelled)
            error_message: Optional error message if status is failed
        """
        if not self.current_session:
            return
        
        self.current_session.end_time = datetime.now()
        self.current_session.status = status
        self.current_session.error_message = error_message
        
        # Update experiments CSV
        self._update_experiment_record(self.current_session)
        
        self.current_session = None
    
    def log_model_info(self, model_name: str, model_type: str,
                      model_info: Dict[str, Any], loading_time: float,
                      device: str, precision: str) -> None:
        """
        Log model loading and information
        
        Args:
            model_name: Model identifier  
            model_type: "baseline" or "compressed"
            model_info: Dictionary with model analysis results
            loading_time: Time taken to load model (seconds)
            device: Computation device
            precision: Model precision (float32, float16, etc.)
        """
        if not self.current_session:
            raise RuntimeError("No active experiment session")
        
        record = ModelRecord(
            experiment_id=self.current_session.experiment_id,
            timestamp=datetime.now(),
            model_name=model_name,
            model_type=model_type,
            parameter_count=model_info.get('num_parameters', 0),
            size_mb=model_info.get('size_mb', 0.0),
            loading_time=loading_time,
            device=device,
            precision=precision,
            architecture_info=model_info
        )
        
        self._write_model_record(record)
    
    def log_evaluation_results(self, model_type: str, evaluation_type: str,
                             plugin_name: str, task_results: Dict[str, Any],
                             evaluation_params: Dict[str, Any],
                             execution_time: float) -> None:
        """
        Log evaluation results
        
        Args:
            model_type: "baseline" or "compressed"
            evaluation_type: "lm_eval" or "profile"  
            plugin_name: Name of evaluation plugin
            task_results: Dictionary of task results
            evaluation_params: Evaluation parameters used
            execution_time: Total evaluation time (seconds)
        """
        if not self.current_session:
            raise RuntimeError("No active experiment session")
        
        timestamp = datetime.now()
        
        # Flatten task results into individual metric records
        for task_name, task_data in task_results.items():
            if isinstance(task_data, dict):
                for metric_name, metric_value in task_data.items():
                    record = EvaluationRecord(
                        experiment_id=self.current_session.experiment_id,
                        timestamp=timestamp,
                        model_type=model_type,
                        evaluation_type=evaluation_type,
                        plugin_name=plugin_name,
                        task_name=task_name,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        evaluation_params=evaluation_params,
                        execution_time=execution_time
                    )
                    self._write_evaluation_record(record)
            else:
                # Single value result
                record = EvaluationRecord(
                    experiment_id=self.current_session.experiment_id,
                    timestamp=timestamp,
                    model_type=model_type,
                    evaluation_type=evaluation_type,
                    plugin_name=plugin_name,
                    task_name=task_name,
                    metric_name="result",
                    metric_value=task_data,
                    evaluation_params=evaluation_params,
                    execution_time=execution_time
                )
                self._write_evaluation_record(record)
    
    def log_compression_results(self, compression_method: str,
                              compression_results: Dict[str, Any],
                              parameters: Dict[str, Any]) -> None:
        """
        Log compression results
        
        Args:
            compression_method: Compression method used
            compression_results: Dictionary with compression results
            parameters: Compression parameters
        """
        if not self.current_session:
            raise RuntimeError("No active experiment session")
        
        record = CompressionRecord(
            experiment_id=self.current_session.experiment_id,
            timestamp=datetime.now(),
            compression_method=compression_method,
            original_size_mb=compression_results.get('original_size_mb', 0.0),
            compressed_size_mb=compression_results.get('compressed_size_mb', 0.0),
            compression_ratio=compression_results.get('compression_ratio', 1.0),
            compression_time=compression_results.get('compression_time', 0.0),
            parameters=parameters,
            memory_usage_mb=compression_results.get('memory_usage_mb', 0.0)
        )
        
        self._write_compression_record(record)
    
    def _write_experiment_record(self, session: ExperimentSession) -> None:
        """Write experiment record to CSV"""
        with open(self.experiments_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'start_time', 'end_time', 'duration_minutes',
                'config_file', 'user', 'status', 'error_message'
            ])
            
            duration_minutes = 0.0
            if session.end_time:
                duration = session.end_time - session.start_time
                duration_minutes = duration.total_seconds() / 60.0
            
            writer.writerow({
                'experiment_id': session.experiment_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else '',
                'duration_minutes': duration_minutes,
                'config_file': session.config_file,
                'user': session.user or '',
                'status': session.status,
                'error_message': session.error_message or ''
            })
    
    def _update_experiment_record(self, session: ExperimentSession) -> None:
        """Update experiment record with final status"""
        # Read all records
        rows = []
        with open(self.experiments_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Update matching record
        duration_minutes = 0.0
        if session.end_time:
            duration = session.end_time - session.start_time
            duration_minutes = duration.total_seconds() / 60.0
        
        for row in rows:
            if row['experiment_id'] == session.experiment_id:
                row['end_time'] = session.end_time.isoformat() if session.end_time else ''
                row['duration_minutes'] = str(duration_minutes)
                row['status'] = session.status
                row['error_message'] = session.error_message or ''
                break
        
        # Write back all records
        with open(self.experiments_file, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def _write_model_record(self, record: ModelRecord) -> None:
        """Write model record to CSV"""
        with open(self.models_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'timestamp', 'model_name', 'model_type',
                'parameter_count', 'size_mb', 'loading_time', 'device',
                'precision', 'architecture_info'
            ])
            
            writer.writerow({
                'experiment_id': record.experiment_id,
                'timestamp': record.timestamp.isoformat(),
                'model_name': record.model_name,
                'model_type': record.model_type,
                'parameter_count': record.parameter_count,
                'size_mb': record.size_mb,
                'loading_time': record.loading_time,
                'device': record.device,
                'precision': record.precision,
                'architecture_info': json.dumps(record.architecture_info)
            })
    
    def _write_evaluation_record(self, record: EvaluationRecord) -> None:
        """Write evaluation record to CSV"""
        with open(self.evaluations_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'timestamp', 'model_type', 'evaluation_type',
                'plugin_name', 'task_name', 'metric_name', 'metric_value',
                'evaluation_params', 'execution_time'
            ])
            
            writer.writerow({
                'experiment_id': record.experiment_id,
                'timestamp': record.timestamp.isoformat(),
                'model_type': record.model_type,
                'evaluation_type': record.evaluation_type,
                'plugin_name': record.plugin_name,
                'task_name': record.task_name,
                'metric_name': record.metric_name,
                'metric_value': record.metric_value,
                'evaluation_params': json.dumps(record.evaluation_params),
                'execution_time': record.execution_time
            })
    
    def _write_compression_record(self, record: CompressionRecord) -> None:
        """Write compression record to CSV"""
        with open(self.compressions_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'timestamp', 'compression_method',
                'original_size_mb', 'compressed_size_mb', 'compression_ratio',
                'compression_time', 'parameters', 'memory_usage_mb'
            ])
            
            writer.writerow({
                'experiment_id': record.experiment_id,
                'timestamp': record.timestamp.isoformat(),
                'compression_method': record.compression_method,
                'original_size_mb': record.original_size_mb,
                'compressed_size_mb': record.compressed_size_mb,
                'compression_ratio': record.compression_ratio,
                'compression_time': record.compression_time,
                'parameters': json.dumps(record.parameters),
                'memory_usage_mb': record.memory_usage_mb
            })


class ResultComparator:
    """
    Result comparison and analysis tools for evaluation results
    """
    
    def __init__(self, csv_logger: CSVLogger):
        self.csv_logger = csv_logger
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare results across multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'experiments': experiment_ids,
            'model_comparison': {},
            'evaluation_comparison': {},
            'compression_comparison': {}
        }
        
        # Load evaluation results for each experiment
        for exp_id in experiment_ids:
            eval_results = self._load_evaluation_results(exp_id)
            model_results = self._load_model_results(exp_id)
            compression_results = self._load_compression_results(exp_id)
            
            comparison_results['model_comparison'][exp_id] = model_results
            comparison_results['evaluation_comparison'][exp_id] = eval_results
            comparison_results['compression_comparison'][exp_id] = compression_results
        
        return comparison_results
    
    def analyze_compression_efficiency(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze compression efficiency for an experiment
        
        Args:
            experiment_id: Experiment to analyze
            
        Returns:
            Dictionary with efficiency analysis
        """
        eval_results = self._load_evaluation_results(experiment_id)
        compression_results = self._load_compression_results(experiment_id)
        
        analysis = {
            'experiment_id': experiment_id,
            'compression_ratio': 0.0,
            'performance_retention': {},
            'efficiency_score': 0.0
        }
        
        # Get compression ratio
        if compression_results:
            analysis['compression_ratio'] = compression_results[0].get('compression_ratio', 1.0)
        
        # Calculate performance retention (baseline vs compressed)
        baseline_results = {}
        compressed_results = {}
        
        for result in eval_results:
            if result['model_type'] == 'baseline':
                key = f"{result['task_name']}_{result['metric_name']}"
                baseline_results[key] = float(result['metric_value'])
            elif result['model_type'] == 'compressed':
                key = f"{result['task_name']}_{result['metric_name']}"
                compressed_results[key] = float(result['metric_value'])
        
        for key in baseline_results:
            if key in compressed_results:
                if baseline_results[key] != 0:
                    retention = compressed_results[key] / baseline_results[key]
                    analysis['performance_retention'][key] = retention
        
        # Calculate efficiency score (compression ratio * average retention)
        if analysis['performance_retention']:
            avg_retention = sum(analysis['performance_retention'].values()) / len(analysis['performance_retention'])
            analysis['efficiency_score'] = analysis['compression_ratio'] * avg_retention
        
        return analysis
    
    def _load_evaluation_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load evaluation results for an experiment"""
        results = []
        
        if self.csv_logger.evaluations_file.exists():
            with open(self.csv_logger.evaluations_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['experiment_id'] == experiment_id:
                        results.append(row)
        
        return results
    
    def _load_model_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load model results for an experiment"""
        results = []
        
        if self.csv_logger.models_file.exists():
            with open(self.csv_logger.models_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['experiment_id'] == experiment_id:
                        results.append(row)
        
        return results
    
    def _load_compression_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load compression results for an experiment"""
        results = []
        
        if self.csv_logger.compressions_file.exists():
            with open(self.csv_logger.compressions_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['experiment_id'] == experiment_id:
                        results.append(row)
        
        return results