"""
ETL Scheduler for V7P3R Chess Engine Analytics

This module provides scheduling and orchestration for the ETL process,
supporting both local execution and Google Cloud Scheduler integration.
It can be used to configure periodic ETL jobs with monitoring and alerting.
"""

import os
import yaml
import json
import time
import logging
import argparse
import datetime
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging/etl_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("etl_scheduler")

class ETLScheduler:
    """
    Scheduler for V7P3R Chess Engine ETL jobs.
    
    This class provides both local scheduling and Google Cloud Scheduler
    integration for ETL jobs, with monitoring and alerting capabilities.
    """
    
    def __init__(self, config_path: str = "config/etl_config.yaml"):
        """
        Initialize the ETL scheduler.
        
        Args:
            config_path: Path to the ETL configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.etl_script_path = Path("engine_utilities/etl_processor.py")
        self.last_run_file = Path("logging/etl_last_run.json")
        
        # Initialize GCP components if enabled
        self.use_gcp = self.config.get('schedule', {}).get('use_gcp', False)
        self.project_id = self.config.get('cloud', {}).get('project_id')
        self.location_id = self.config.get('cloud', {}).get('location_id', 'us-central1')
        
        logger.info(f"ETL Scheduler initialized with {'GCP' if self.use_gcp else 'local'} mode")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    def _save_last_run(self, status: str, metrics: Optional[Dict[str, Any]] = None):
        """Save information about the last ETL run."""
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "metrics": metrics or {}
        }
        
        os.makedirs(os.path.dirname(self.last_run_file), exist_ok=True)
        with open(self.last_run_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_last_run(self) -> Optional[Dict[str, Any]]:
        """Get information about the last ETL run."""
        if not os.path.exists(self.last_run_file):
            return None
        
        try:
            with open(self.last_run_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading last run file: {e}")
            return None
    
    def run_etl_job(self, limit: Optional[int] = None, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run an ETL job locally.
        
        Args:
            limit: Maximum number of records to process
            start_date: Start date for data extraction (YYYY-MM-DD)
            end_date: End date for data extraction (YYYY-MM-DD)
            
        Returns:
            Dictionary with job status and metrics
        """
        logger.info(f"Running ETL job with limit={limit}, start_date={start_date}, end_date={end_date}")
        
        # Build command
        cmd = ["python", "-m", "engine_utilities.etl_processor"]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])
        cmd.extend(["--config", self.config_path])
        
        # Run ETL process
        try:
            start_time = time.time()
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            duration = time.time() - start_time
            
            # Parse output to extract metrics
            metrics = {
                "duration_seconds": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            # Try to parse more detailed metrics from stdout
            try:
                # Look for JSON output in the result
                json_start = result.stdout.find('{')
                json_end = result.stdout.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result.stdout[json_start:json_end]
                    detailed_metrics = json.loads(json_str)
                    metrics.update(detailed_metrics)
            except Exception as e:
                logger.warning(f"Failed to parse detailed metrics: {e}")
            
            # Save last run information
            self._save_last_run("success", metrics)
            
            logger.info(f"ETL job completed successfully in {duration:.2f} seconds")
            return {"status": "success", "metrics": metrics}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ETL job failed with return code {e.returncode}: {e.stderr}")
            metrics = {
                "duration_seconds": time.time() - start_time,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "return_code": e.returncode,
                "error": str(e)
            }
            self._save_last_run("failed", metrics)
            return {"status": "failed", "metrics": metrics}
        
        except Exception as e:
            logger.error(f"Error running ETL job: {e}")
            metrics = {
                "duration_seconds": time.time() - start_time,
                "error": str(e)
            }
            self._save_last_run("error", metrics)
            return {"status": "error", "metrics": metrics}
    
    def setup_gcp_scheduler(self, cron_schedule: Optional[str] = None):
        """
        Set up a Google Cloud Scheduler job for the ETL process.
        This requires Google Cloud Scheduler API to be enabled in your project.
        
        Args:
            cron_schedule: Cron expression for job schedule (defaults to config)
            
        Returns:
            Success status and scheduler job name if successful
        """
        if not self.use_gcp:
            logger.warning("GCP scheduling is not enabled in the configuration")
            return {"status": "skipped", "reason": "GCP scheduling not enabled"}
        try:
            # Import GCP libraries here to avoid dependencies when not using GCP
            try:
                from google.cloud import scheduler_v1
                from google.api_core import exceptions
            except ImportError:
                logger.error("Google Cloud Scheduler libraries not installed")
                return {
                    "status": "error",
                    "message": "Google Cloud Scheduler libraries not installed. Install with 'pip install google-cloud-scheduler'"
                }
            
            if not self.project_id:
                raise ValueError("GCP project ID not configured")
            
            # Use provided schedule or fallback to config
            schedule = cron_schedule or self.config.get('schedule', {}).get('cron', '0 2 * * *')
            
            # Create a Cloud Scheduler client
            client = scheduler_v1.CloudSchedulerClient()
            
            # Construct the fully qualified location path
            parent = f"projects/{self.project_id}/locations/{self.location_id}"
            
            # Define the job name
            job_name = f"{parent}/jobs/v7p3r-etl-job"
            
            # Create a job to hit an HTTP endpoint
            http_target = scheduler_v1.HttpTarget(
                uri=self.config.get('schedule', {}).get('http_endpoint', "https://your-app-url/etl/trigger"),
                http_method=scheduler_v1.HttpMethod.POST,
                body=json.dumps({
                    "config_path": self.config_path,
                    "auth_token": self.config.get('schedule', {}).get('auth_token', "")
                }).encode(),
                headers={"Content-Type": "application/json"}
            )
            
            job = scheduler_v1.Job(
                name=job_name,
                description="V7P3R Chess Engine ETL Process",
                schedule=schedule,
                time_zone=self.config.get('schedule', {}).get('timezone', "UTC"),
                http_target=http_target
            )
            
            # Check if job already exists
            try:
                existing_job = client.get_job(name=job_name)
                logger.info(f"Updating existing job: {job_name}")
                # Convert dictionary to proper Job object
                request = scheduler_v1.UpdateJobRequest(job=job)
                client.update_job(request=request)
                return {"status": "updated", "job": job_name}
            except exceptions.NotFound:
                # Job doesn't exist, create it
                logger.info(f"Creating new job: {job_name}")
                client.create_job(parent=parent, job=job)
                return {"status": "created", "job": job_name}
            
        except ImportError as e:
            logger.error(f"Missing GCP libraries: {e}. Run 'pip install google-cloud-scheduler'")
            return {"status": "error", "reason": f"Missing GCP libraries: {e}"}
        
        except Exception as e:
            logger.error(f"Error setting up GCP scheduler: {e}")
            return {"status": "error", "reason": str(e)}
    
    def setup_local_scheduler(self):
        """
        Set up a local scheduler using cron (Linux/macOS) or Task Scheduler (Windows).
        
        Returns:
            Success status and scheduler details
        """
        import platform
        system = platform.system()
        
        cron_schedule = self.config.get('schedule', {}).get('cron', '0 2 * * *')
        
        if system == "Windows":
            return self._setup_windows_scheduler(cron_schedule)
        else:  # Linux or macOS
            return self._setup_unix_scheduler(cron_schedule)
    
    def _setup_windows_scheduler(self, cron_schedule: str) -> Dict[str, Any]:
        """Set up Task Scheduler on Windows."""
        try:
            # Convert cron expression to Windows schedule format
            parts = cron_schedule.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid cron expression: {cron_schedule}")
            
            minute, hour, day, month, weekday = parts
            
            # Get current directory as working directory
            current_dir = os.path.abspath(os.curdir)
            
            # Build the command
            script_path = os.path.join(current_dir, "engine_utilities", "etl_scheduler.py")
            task_cmd = f'"{sys.executable}" "{script_path}" --run-job'
            
            # Create a unique task name
            task_name = "V7P3RChessETL"
            
            # Build the schtasks command
            cmd = [
                "schtasks", "/create", "/tn", task_name, "/tr", task_cmd,
                "/sc", "daily", "/st", f"{hour.replace('*', '0')}:{minute.replace('*', '0')}",
                "/f"  # Force creation/overwrite
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Windows Task Scheduler job created: {task_name}")
            return {"status": "success", "platform": "windows", "task": task_name}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating Windows task: {e.stderr}")
            return {"status": "error", "reason": e.stderr}
        
        except Exception as e:
            logger.error(f"Error setting up Windows scheduler: {e}")
            return {"status": "error", "reason": str(e)}
    
    def _setup_unix_scheduler(self, cron_schedule: str) -> Dict[str, Any]:
        """Set up cron job on Unix-like systems."""
        try:
            # Get current directory as working directory
            current_dir = os.path.abspath(os.curdir)
            
            # Build the command
            script_path = os.path.join(current_dir, "engine_utilities", "etl_scheduler.py")
            cron_cmd = f'cd {current_dir} && {sys.executable} {script_path} --run-job >> {current_dir}/logging/etl_cron.log 2>&1'
            
            # Check if crontab is available
            result = subprocess.run(["which", "crontab"], capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("crontab not found on this system")
            
            # Get existing crontab
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            existing_crontab = result.stdout if result.returncode == 0 else ""
            
            # Check if our job is already in crontab
            job_marker = f"# V7P3R Chess ETL Job"
            if job_marker in existing_crontab:
                # Remove old job
                lines = existing_crontab.splitlines()
                new_lines = []
                skip = False
                for line in lines:
                    if line == job_marker:
                        skip = True
                        continue
                    if skip and line.strip() == "":
                        skip = False
                    if not skip:
                        new_lines.append(line)
                existing_crontab = "\n".join(new_lines)
            
            # Add new job
            new_crontab = existing_crontab.strip() + f"\n\n{job_marker}\n{cron_schedule} {cron_cmd}\n"
            
            # Write to temp file
            with open("/tmp/v7p3r_crontab", "w") as f:
                f.write(new_crontab)
            
            # Install new crontab
            result = subprocess.run(["crontab", "/tmp/v7p3r_crontab"], capture_output=True, text=True, check=True)
            
            logger.info("Unix cron job created successfully")
            return {"status": "success", "platform": "unix", "schedule": cron_schedule}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating Unix cron job: {e.stderr}")
            return {"status": "error", "reason": e.stderr}
        
        except Exception as e:
            logger.error(f"Error setting up Unix scheduler: {e}")
            return {"status": "error", "reason": str(e)}


def parse_args(parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(description='V7P3R Chess Engine ETL Scheduler')
    
    parser.add_argument('--config', default='config/etl_config.yaml',
                        help='Path to ETL configuration file')
    parser.add_argument('--run-job', action='store_true',
                        help='Run ETL job immediately')
    parser.add_argument('--setup-gcp', action='store_true',
                        help='Set up Google Cloud Scheduler for ETL jobs')
    parser.add_argument('--setup-local', action='store_true',
                        help='Set up local scheduler for ETL jobs')
    parser.add_argument('--limit', type=int,
                        help='Limit number of records to process')
    parser.add_argument('--start-date',
                        help='Start date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                        help='End date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--cron',
                        help='Cron schedule expression (e.g., "0 2 * * *")')
    
    return parser, parser.parse_args()


if __name__ == "__main__":
    import sys    
    parser, args = parse_args()
    
    scheduler = ETLScheduler(config_path=args.config)
    
    if args.run_job:
        result = scheduler.run_etl_job(
            limit=args.limit,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if result["status"] != "success":
            sys.exit(1)
            
    elif args.setup_gcp:
        result = scheduler.setup_gcp_scheduler(cron_schedule=args.cron)
        print(json.dumps(result, indent=2))
        
    elif args.setup_local:
        result = scheduler.setup_local_scheduler()
        print(json.dumps(result, indent=2))
        
    else:
        print("No action specified. Use --run-job, --setup-gcp, or --setup-local")
        parser.print_help()
        parser.print_help()
