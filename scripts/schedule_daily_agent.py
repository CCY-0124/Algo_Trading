"""
schedule_daily_agent.py

Windows task scheduler helper for daily agent execution.

Features:
- Create Windows scheduled task
- Configure daily execution
- Test task creation
- Remove task if needed
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TaskScheduler:
    """
    Windows Task Scheduler helper for LLM backtest agent.
    """
    
    def __init__(self, task_name: str = "LLMBacktestAgentDaily"):
        """
        Initialize task scheduler.
        
        :param task_name: Name of the scheduled task
        """
        self.task_name = task_name
        self.project_root = Path(__file__).parent.parent
        self.script_path = self.project_root / "scripts" / "llm_backtest_agent.py"
        self.python_exe = sys.executable
    
    def create_task(self, run_time: str = "09:00", config_path: str = None):
        """
        Create Windows scheduled task.
        
        :param run_time: Time to run daily (HH:MM format, e.g., "09:00")
        :param config_path: Path to config file (optional)
        """
        logging.info(f"Creating scheduled task: {self.task_name}")
        
        # Build command
        cmd_parts = [f'"{self.python_exe}"', f'"{self.script_path}"']
        if config_path:
            cmd_parts.append(f'--config "{config_path}"')
        
        command = ' '.join(cmd_parts)
        
        # Create task using schtasks
        # Format: schtasks /create /tn "TaskName" /tr "Command" /sc daily /st "Time" /f
        schtasks_cmd = [
            'schtasks',
            '/create',
            '/tn', self.task_name,
            '/tr', command,
            '/sc', 'daily',
            '/st', run_time,
            '/f'  # Force (overwrite if exists)
        ]
        
        try:
            result = subprocess.run(
                schtasks_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info(f"✅ Scheduled task created successfully!")
            logging.info(f"  Task name: {self.task_name}")
            logging.info(f"  Run time: {run_time} daily")
            logging.info(f"  Command: {command}")
            logging.info(f"\nOutput: {result.stdout}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Failed to create scheduled task")
            logging.error(f"Error: {e.stderr}")
            logging.error(f"\nYou can manually create the task using:")
            logging.error(f'schtasks /create /tn "{self.task_name}" /tr "{command}" /sc daily /st {run_time}')
            return False
    
    def delete_task(self):
        """
        Delete the scheduled task.
        """
        logging.info(f"Deleting scheduled task: {self.task_name}")
        
        try:
            result = subprocess.run(
                ['schtasks', '/delete', '/tn', self.task_name, '/f'],
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info(f"✅ Task deleted successfully")
            logging.info(f"Output: {result.stdout}")
            return True
        
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Failed to delete task: {e.stderr}")
            return False
    
    def query_task(self):
        """
        Query task information.
        """
        logging.info(f"Querying task: {self.task_name}")
        
        try:
            result = subprocess.run(
                ['schtasks', '/query', '/tn', self.task_name, '/fo', 'list', '/v'],
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info(f"Task information:\n{result.stdout}")
            return True
        
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Task not found or query failed: {e.stderr}")
            return False
    
    def run_task_now(self):
        """
        Run the task immediately (for testing).
        """
        logging.info(f"Running task now: {self.task_name}")
        
        try:
            result = subprocess.run(
                ['schtasks', '/run', '/tn', self.task_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info(f"✅ Task started")
            logging.info(f"Output: {result.stdout}")
            return True
        
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Failed to run task: {e.stderr}")
            return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Windows Task Scheduler Helper')
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create scheduled task'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete scheduled task'
    )
    parser.add_argument(
        '--query',
        action='store_true',
        help='Query task information'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run task now'
    )
    parser.add_argument(
        '--time',
        type=str,
        default='09:00',
        help='Daily run time (HH:MM format, default: 09:00)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default='LLMBacktestAgentDaily',
        help='Task name (default: LLMBacktestAgentDaily)'
    )
    
    args = parser.parse_args()
    
    scheduler = TaskScheduler(task_name=args.task_name)
    
    if args.create:
        scheduler.create_task(run_time=args.time, config_path=args.config)
    elif args.delete:
        scheduler.delete_task()
    elif args.query:
        scheduler.query_task()
    elif args.run:
        scheduler.run_task_now()
    else:
        # Default: create task
        print("No action specified. Creating scheduled task by default...")
        print(f"Use --help for more options")
        scheduler.create_task(run_time=args.time, config_path=args.config)


if __name__ == "__main__":
    main()
