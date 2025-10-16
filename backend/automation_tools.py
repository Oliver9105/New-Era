"""
Feature 3: Automation Tools
Advanced automation for data collection, analysis, and prediction
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class AutomationTask:
    name: str
    function: Callable
    schedule_type: str  # 'interval', 'cron', 'once'
    interval: int = None  # seconds for interval type
    cron_expression: str = None
    enabled: bool = True
    last_run: datetime = None
    next_run: datetime = None
    run_count: int = 0
    error_count: int = 0

class AutomationManager:
    def __init__(self):
        self.active = False
        self.tasks = {}
        self.automation_thread = None
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.automation_history = []
        self.performance_metrics = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_execution_time': 0,
            'last_execution': None
        }

    def initialize(self):
        """Initialize the automation manager"""
        self.active = True
        self._setup_default_tasks()
        self._start_automation_loop()
        logger.info("Automation Manager initialized")

    def is_active(self):
        return self.active

    def _setup_default_tasks(self):
        """Setup default automation tasks"""
        # Data collection task
        self.add_task(
            name="data_collection",
            function=self._automated_data_collection,
            schedule_type="interval",
            interval=60  # Every minute
        )

        # Pattern analysis task
        self.add_task(
            name="pattern_analysis",
            function=self._automated_pattern_analysis,
            schedule_type="interval",
            interval=300  # Every 5 minutes
        )

        # Model retraining task
        self.add_task(
            name="model_retraining",
            function=self._automated_model_retraining,
            schedule_type="interval",
            interval=3600  # Every hour
        )

        # Health check task
        self.add_task(
            name="health_check",
            function=self._automated_health_check,
            schedule_type="interval",
            interval=180  # Every 3 minutes
        )

        # Data cleanup task
        self.add_task(
            name="data_cleanup",
            function=self._automated_data_cleanup,
            schedule_type="interval",
            interval=86400  # Daily
        )

    def add_task(self, name: str, function: Callable, schedule_type: str, **kwargs) -> bool:
        """Add a new automation task"""
        try:
            task = AutomationTask(
                name=name,
                function=function,
                schedule_type=schedule_type,
                **kwargs
            )
            
            # Calculate next run time
            if schedule_type == "interval" and kwargs.get('interval'):
                task.next_run = datetime.utcnow() + timedelta(seconds=kwargs['interval'])
            elif schedule_type == "once":
                task.next_run = datetime.utcnow() + timedelta(seconds=1)
            
            self.tasks[name] = task
            logger.info(f"Added automation task: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add task {name}: {e}")
            return False

    def remove_task(self, name: str) -> bool:
        """Remove an automation task"""
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Removed automation task: {name}")
            return True
        return False

    def enable_task(self, name: str) -> bool:
        """Enable a task"""
        if name in self.tasks:
            self.tasks[name].enabled = True
            return True
        return False

    def disable_task(self, name: str) -> bool:
        """Disable a task"""
        if name in self.tasks:
            self.tasks[name].enabled = False
            return True
        return False

    def _start_automation_loop(self):
        """Start the main automation loop"""
        def automation_loop():
            while not self.stop_event.is_set():
                try:
                    self._process_scheduled_tasks()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Error in automation loop: {e}")
                    time.sleep(5)

        self.automation_thread = threading.Thread(target=automation_loop, daemon=True)
        self.automation_thread.start()

    def _process_scheduled_tasks(self):
        """Process tasks that are scheduled to run"""
        current_time = datetime.utcnow()
        
        for task_name, task in self.tasks.items():
            if not task.enabled:
                continue
                
            if task.next_run and current_time >= task.next_run:
                self._execute_task(task)

    def _execute_task(self, task: AutomationTask):
        """Execute a single automation task"""
        try:
            start_time = time.time()
            logger.info(f"Executing automation task: {task.name}")
            
            # Execute the task function
            result = task.function()
            
            execution_time = time.time() - start_time
            
            # Update task statistics
            task.last_run = datetime.utcnow()
            task.run_count += 1
            
            # Schedule next run
            if task.schedule_type == "interval" and task.interval:
                task.next_run = datetime.utcnow() + timedelta(seconds=task.interval)
            elif task.schedule_type == "once":
                task.enabled = False  # Disable one-time tasks
            
            # Update performance metrics
            self.performance_metrics['total_runs'] += 1
            self.performance_metrics['successful_runs'] += 1
            self.performance_metrics['last_execution'] = datetime.utcnow().isoformat()
            
            # Calculate average execution time
            if self.performance_metrics['total_runs'] == 1:
                self.performance_metrics['average_execution_time'] = execution_time
            else:
                self.performance_metrics['average_execution_time'] = (
                    (self.performance_metrics['average_execution_time'] * (self.performance_metrics['total_runs'] - 1) + execution_time) 
                    / self.performance_metrics['total_runs']
                )
            
            # Store execution history
            self.automation_history.append({
                'task_name': task.name,
                'timestamp': task.last_run.isoformat(),
                'execution_time': execution_time,
                'status': 'success',
                'result': result if isinstance(result, (dict, list, str, int, float)) else 'completed'
            })
            
            logger.info(f"Task {task.name} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            task.error_count += 1
            self.performance_metrics['failed_runs'] += 1
            
            self.automation_history.append({
                'task_name': task.name,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e)
            })
            
            logger.error(f"Task {task.name} failed: {e}")

    def start_automation(self, schedule: str = 'continuous') -> Dict:
        """Start automation with specified schedule"""
        try:
            if schedule == 'continuous':
                # Enable all tasks
                for task in self.tasks.values():
                    task.enabled = True
            elif schedule == 'minimal':
                # Enable only essential tasks
                essential_tasks = ['data_collection', 'health_check']
                for task_name, task in self.tasks.items():
                    task.enabled = task_name in essential_tasks
            elif schedule == 'analysis_only':
                # Enable only analysis tasks
                analysis_tasks = ['pattern_analysis', 'model_retraining']
                for task_name, task in self.tasks.items():
                    task.enabled = task_name in analysis_tasks
            
            return {
                'status': 'started',
                'schedule': schedule,
                'active_tasks': len([t for t in self.tasks.values() if t.enabled]),
                'total_tasks': len(self.tasks)
            }
        except Exception as e:
            return {'error': str(e)}

    def stop_automation(self) -> Dict:
        """Stop all automation"""
        try:
            for task in self.tasks.values():
                task.enabled = False
            
            return {
                'status': 'stopped',
                'total_tasks': len(self.tasks)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_status(self) -> Dict:
        """Get automation status"""
        active_tasks = [name for name, task in self.tasks.items() if task.enabled]
        
        task_details = {}
        for name, task in self.tasks.items():
            task_details[name] = {
                'enabled': task.enabled,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'run_count': task.run_count,
                'error_count': task.error_count,
                'schedule_type': task.schedule_type,
                'interval': task.interval
            }

        return {
            'automation_active': self.active,
            'total_tasks': len(self.tasks),
            'active_tasks': len(active_tasks),
            'task_details': task_details,
            'performance_metrics': self.performance_metrics,
            'recent_history': self.automation_history[-10:] if self.automation_history else []
        }

    # Default automation task functions
    def _automated_data_collection(self) -> Dict:
        """Automated data collection task"""
        try:
            # Simulate data collection
            from .web_data_access import WebDataCollector
            collector = WebDataCollector()
            
            if not collector.active:
                collector.initialize()
            
            result = collector.collect_data('auto')
            
            return {
                'action': 'data_collection',
                'timestamp': datetime.utcnow().isoformat(),
                'sources_collected': result.get('successful_sources', 0),
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Automated data collection failed: {e}")
            return {'action': 'data_collection', 'status': 'failed', 'error': str(e)}

    def _automated_pattern_analysis(self) -> Dict:
        """Automated pattern analysis task"""
        try:
            # Simulate pattern analysis
            from .app_analysis import GameAnalyzer
            analyzer = GameAnalyzer()
            
            if not analyzer.active:
                analyzer.initialize()
            
            patterns = analyzer.get_patterns()
            trends = analyzer.analyze_trends(1)  # Last hour
            
            return {
                'action': 'pattern_analysis',
                'timestamp': datetime.utcnow().isoformat(),
                'patterns_found': len(patterns.get('patterns', [])),
                'trends_analyzed': len(trends.get('trends', [])),
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Automated pattern analysis failed: {e}")
            return {'action': 'pattern_analysis', 'status': 'failed', 'error': str(e)}

    def _automated_model_retraining(self) -> Dict:
        """Automated model retraining task"""
        try:
            # Simulate model retraining
            from .data_modeling import PredictionEngine
            engine = PredictionEngine()
            
            if not engine.active:
                engine.initialize()
            
            result = engine.train_models('ensemble')
            
            return {
                'action': 'model_retraining',
                'timestamp': datetime.utcnow().isoformat(),
                'models_trained': result.get('models_trained', 0),
                'training_accuracy': result.get('accuracy', 0),
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Automated model retraining failed: {e}")
            return {'action': 'model_retraining', 'status': 'failed', 'error': str(e)}

    def _automated_health_check(self) -> Dict:
        """Automated health check task"""
        try:
            # Check system health
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_status': 'healthy',
                'components_checked': 0,
                'issues_found': 0
            }

            # Check API endpoints
            try:
                from .api_explorer import APIExplorer
                api_explorer = APIExplorer()
                if api_explorer.active:
                    health = api_explorer.monitor_api_health()
                    health_status['api_health'] = health
                    health_status['components_checked'] += 1
            except Exception:
                health_status['issues_found'] += 1

            # Check data collection
            try:
                from .web_data_access import WebDataCollector
                collector = WebDataCollector()
                if collector.active:
                    quality = collector.monitor_data_quality()
                    health_status['data_quality'] = quality
                    health_status['components_checked'] += 1
            except Exception:
                health_status['issues_found'] += 1

            return {
                'action': 'health_check',
                'status': 'completed',
                'health_status': health_status
            }
        except Exception as e:
            logger.error(f"Automated health check failed: {e}")
            return {'action': 'health_check', 'status': 'failed', 'error': str(e)}

    def _automated_data_cleanup(self) -> Dict:
        """Automated data cleanup task"""
        try:
            # Cleanup old data
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            # Remove old automation history
            original_count = len(self.automation_history)
            self.automation_history = [
                entry for entry in self.automation_history
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            cleaned_count = original_count - len(self.automation_history)

            return {
                'action': 'data_cleanup',
                'timestamp': datetime.utcnow().isoformat(),
                'records_cleaned': cleaned_count,
                'cutoff_date': cutoff_date.isoformat(),
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Automated data cleanup failed: {e}")
            return {'action': 'data_cleanup', 'status': 'failed', 'error': str(e)}

    def batch_task_execution(self, task_names: List[str]) -> Dict:
        """Execute multiple tasks in batch"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {
                executor.submit(self._execute_task, self.tasks[task_name]): task_name
                for task_name in task_names if task_name in self.tasks
            }
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = {'status': 'completed', 'result': result}
                except Exception as e:
                    results[task_name] = {'status': 'failed', 'error': str(e)}

        return {
            'batch_execution': True,
            'timestamp': datetime.utcnow().isoformat(),
            'tasks_executed': len(task_names),
            'results': results
        }

    def create_custom_task(self, task_config: Dict) -> Dict:
        """Create a custom automation task"""
        try:
            def custom_function():
                # Execute custom code
                code = task_config.get('code', '')
                if code:
                    exec(code)
                return {'custom_task': 'executed'}

            success = self.add_task(
                name=task_config['name'],
                function=custom_function,
                schedule_type=task_config.get('schedule_type', 'interval'),
                interval=task_config.get('interval', 300)
            )

            return {
                'task_created': success,
                'task_name': task_config['name'],
                'schedule_type': task_config.get('schedule_type', 'interval')
            }
        except Exception as e:
            return {'error': str(e)}

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
