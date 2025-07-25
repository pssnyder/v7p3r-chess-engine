"""
ETL Monitor for v7p3r Chess Engine Analytics

This module provides monitoring capabilities for the ETL process, tracking:
1. Job status and metrics
2. Resource usage (CPU, memory, disk)
3. Data quality and validation results
4. Performance benchmarks

It can be used to monitor ETL jobs in real-time or to generate reports
on historical ETL job performance.
"""

import os
import json
import yaml
import time
import sqlite3
import logging
import argparse
import datetime
import subprocess
import threading
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging/etl_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("etl_monitor")

class ETLMonitor:
    """
    Monitor for v7p3r Chess Engine ETL jobs.
    
    This class provides monitoring capabilities for ETL jobs, including:
    - Job status tracking
    - Resource usage monitoring
    - Data quality metrics
    - Performance analytics
    """
    
    def __init__(self, config_path: str = "config/etl_config.yaml", 
                analytics_db_path: Optional[str] = None):
        """
        Initialize the ETL monitor.
        
        Args:
            config_path: Path to the ETL configuration file
            analytics_db_path: Path to the analytics database (defaults to config)
        """
        self.config = self._load_config(config_path)
        
        # Set up database connections
        self.analytics_db_path = analytics_db_path or self.config.get('reporting_db', {}).get(
            'path', "metrics/chess_analytics.db")
        
        # Check if analytics db exists
        if not os.path.exists(self.analytics_db_path):
            logger.warning(f"Analytics database not found at {self.analytics_db_path}")
        
        # Monitoring metrics
        self.monitoring_data = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitor_interval = self.config.get('monitoring', {}).get('interval_seconds', 10)
        
        logger.info(f"ETL Monitor initialized with database at {self.analytics_db_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a connection to the analytics database."""
        conn = sqlite3.connect(self.analytics_db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_etl_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of ETL job executions.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job metrics dictionaries
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT * FROM etl_job_metrics 
                ORDER BY start_time DESC 
                LIMIT ?
                """, (limit,))
                
                jobs = [dict(row) for row in cursor.fetchall()]
                
                # Parse timestamps
                for job in jobs:
                    if job.get('start_time'):
                        job['start_time'] = datetime.datetime.fromisoformat(job['start_time'])
                    if job.get('end_time'):
                        job['end_time'] = datetime.datetime.fromisoformat(job['end_time'])
                
                return jobs
                
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Error getting ETL job history: {e}")
            return []
    
    def get_etl_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific ETL job.
        
        Args:
            job_id: ID of the ETL job to retrieve
            
        Returns:
            Dictionary with job details or None if not found
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get job metrics
                cursor.execute("SELECT * FROM etl_job_metrics WHERE job_id = ?", (job_id,))
                job = dict(cursor.fetchone() or {})
                
                if not job:
                    return None
                
                # Parse timestamps
                if job.get('start_time'):
                    job['start_time'] = datetime.datetime.fromisoformat(job['start_time'])
                if job.get('end_time'):
                    job['end_time'] = datetime.datetime.fromisoformat(job['end_time'])
                
                # Get job errors
                cursor.execute("SELECT * FROM etl_job_errors WHERE job_id = ?", (job_id,))
                job['errors'] = [dict(row) for row in cursor.fetchall()]
                
                # Get processing logs
                cursor.execute("SELECT * FROM data_processing_log WHERE job_id = ? LIMIT 1000", (job_id,))
                job['processing_logs'] = [dict(row) for row in cursor.fetchall()]
                
                # Get record counts
                cursor.execute("""
                SELECT COUNT(*) as game_count FROM game_analytics 
                WHERE etl_job_id = ?
                """, (job_id,))
                job['game_count'] = cursor.fetchone()[0]
                
                cursor.execute("""
                SELECT COUNT(*) as move_count FROM move_analytics 
                WHERE etl_job_id = ?
                """, (job_id,))
                job['move_count'] = cursor.fetchone()[0]
                
                return job
                
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Error getting ETL job details: {e}")
            return None
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with CPU, memory, and disk usage metrics
        """
        import psutil
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            
            # Get process info if available
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=0.5)
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_mb": memory_used_mb,
                "disk_percent": disk_percent,
                "disk_used_gb": disk_used_gb,
                "process_cpu_percent": process_cpu,
                "process_memory_mb": process_memory_mb
            }
            
        except ImportError:
            logger.error("psutil module not installed. Install with 'pip install psutil'")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": "psutil module not installed"
            }
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def start_monitoring(self):
        """Start continuous monitoring of system resources."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_data = []
        
        def monitor_resources():
            while self.monitoring_active:
                usage = self.get_resource_usage()
                self.monitoring_data.append(usage)
                time.sleep(self.monitor_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Resource monitoring started with {self.monitor_interval}s interval")
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """
        Stop continuous monitoring and return collected data.
        
        Returns:
            List of resource usage metrics
        """
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return self.monitoring_data
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        logger.info(f"Resource monitoring stopped with {len(self.monitoring_data)} data points")
        return self.monitoring_data
    
    def generate_resource_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a report of resource usage during monitoring.
        
        Args:
            output_file: Path to save the report (defaults to timestamped file)
            
        Returns:
            Path to the generated report file
        """
        if not self.monitoring_data:
            logger.warning("No monitoring data available")
            return ""
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.monitoring_data)
        
        # Generate report file path if not provided
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"metrics/etl_resource_report_{timestamp}.html"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate HTML report
        with open(output_file, 'w') as f:
            f.write("<html><head>")
            f.write("<title>ETL Resource Usage Report</title>")
            f.write("<style>body {font-family: Arial; margin: 20px;}")
            f.write("table {border-collapse: collapse; width: 100%;}")
            f.write("th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}")
            f.write("th {background-color: #f2f2f2;}")
            f.write("</style></head><body>")
            
            f.write("<h1>ETL Resource Usage Report</h1>")
            f.write(f"<p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            f.write(f"<p>Monitoring duration: {len(self.monitoring_data)} data points, ")
            f.write(f"~{len(self.monitoring_data) * self.monitor_interval / 60:.1f} minutes</p>")
            
            # Summary statistics
            f.write("<h2>Summary Statistics</h2>")
            f.write("<table>")
            f.write("<tr><th>Metric</th><th>Mean</th><th>Max</th><th>Min</th></tr>")
            
            for col in ['cpu_percent', 'memory_percent', 'disk_percent', 
                       'process_cpu_percent', 'process_memory_mb']:
                if col in df.columns:
                    f.write(f"<tr><td>{col}</td>")
                    f.write(f"<td>{df[col].mean():.2f}</td>")
                    f.write(f"<td>{df[col].max():.2f}</td>")
                    f.write(f"<td>{df[col].min():.2f}</td></tr>")
            
            f.write("</table>")
            
            # Generate plots
            if 'matplotlib' in sys.modules:
                for metric in ['cpu_percent', 'memory_percent', 'process_memory_mb']:
                    if metric in df.columns:
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(len(df)), df[metric])
                        plt.title(f"{metric} Over Time")
                        plt.xlabel("Time (intervals)")
                        plt.ylabel(metric)
                        plt.grid(True)
                        
                        # Save plot to file
                        plot_file = f"metrics/plot_{metric}_{timestamp}.png"
                        plt.savefig(plot_file)
                        plt.close()
                        
                        # Add to report
                        f.write(f"<h2>{metric} Over Time</h2>")
                        f.write(f"<img src='../{plot_file}' width='800'><br>")
            
            # Raw data
            f.write("<h2>Raw Data</h2>")
            f.write("<table>")
            f.write("<tr>")
            for col in df.columns:
                f.write(f"<th>{col}</th>")
            f.write("</tr>")
            
            for _, row in df.iterrows():
                f.write("<tr>")
                for col in df.columns:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        f.write(f"<td>{value:.2f}</td>")
                    else:
                        f.write(f"<td>{value}</td>")
                f.write("</tr>")
            
            f.write("</table>")
            f.write("</body></html>")
        
        logger.info(f"Resource report generated at {output_file}")
        return output_file
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Get data quality metrics from the analytics database.
        
        Returns:
            Dictionary with data quality metrics
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                metrics = {}
                
                # Total record counts
                cursor.execute("SELECT COUNT(*) FROM game_analytics")
                metrics['total_games'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM move_analytics")
                metrics['total_moves'] = cursor.fetchone()[0]
                
                # Completeness checks
                cursor.execute("""
                SELECT COUNT(*) FROM game_analytics 
                WHERE white_player IS NULL OR black_player IS NULL OR result IS NULL
                """)
                metrics['incomplete_games'] = cursor.fetchone()[0]
                
                cursor.execute("""
                SELECT COUNT(*) FROM move_analytics 
                WHERE move_san IS NULL OR fen_before IS NULL
                """)
                metrics['incomplete_moves'] = cursor.fetchone()[0]
                
                # Time range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM game_analytics")
                min_date, max_date = cursor.fetchone()
                metrics['date_range'] = {
                    'min': min_date,
                    'max': max_date
                }
                
                # Games per engine
                cursor.execute("""
                SELECT white_player, COUNT(*) as game_count 
                FROM game_analytics 
                GROUP BY white_player
                ORDER BY game_count DESC
                """)
                metrics['games_by_engine'] = {
                    row[0]: row[1] for row in cursor.fetchall()
                }
                
                return metrics
                
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            return {"error": str(e)}
        
        except Exception as e:
            logger.error(f"Error getting data quality metrics: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='v7p3r Chess Engine ETL Monitor')
    parser.add_argument('--config', default='config/etl_config.yaml',
                        help='Path to ETL configuration file')
    parser.add_argument('--job-history', action='store_true',
                        help='Show ETL job history')
    parser.add_argument('--job-details', type=str,
                        help='Show details for a specific ETL job ID')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor system resources during ETL execution')
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration to monitor in seconds (default: 300)')
    parser.add_argument('--report', action='store_true',
                        help='Generate a resource usage report')
    parser.add_argument('--quality', action='store_true',
                        help='Show data quality metrics')
    
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    
    args = parse_args()
    
    monitor = ETLMonitor(config_path=args.config)
    
    if args.job_history:
        jobs = monitor.get_etl_job_history()
        if jobs:
            print("ETL Job History:")
            for job in jobs:
                print(f"Job ID: {job['job_id']}")
                print(f"  Started: {job['start_time']}")
                print(f"  Status: {job.get('status', 'unknown')}")
                print(f"  Records: {job.get('records_processed', 0)} processed, {job.get('records_failed', 0)} failed")
                print(f"  Duration: {job.get('total_time_seconds', 0):.2f} seconds")
                print()
        else:
            print("No ETL job history found")
    
    elif args.job_details:
        job = monitor.get_etl_job_details(args.job_details)
        if job:
            print(f"ETL Job Details for {job['job_id']}:")
            print(f"  Started: {job['start_time']}")
            print(f"  Ended: {job.get('end_time', 'N/A')}")
            print(f"  Status: {job.get('status', 'unknown')}")
            print(f"  Records: {job.get('records_processed', 0)} processed, {job.get('records_failed', 0)} failed")
            print(f"  Games: {job.get('game_count', 0)}")
            print(f"  Moves: {job.get('move_count', 0)}")
            print(f"  Duration: {job.get('total_time_seconds', 0):.2f} seconds")
            print(f"  Errors: {len(job.get('errors', []))}")
            
            if job.get('errors'):
                print("\nErrors:")
                for error in job['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error.get('error_type')}: {error.get('message')}")
                if len(job['errors']) > 5:
                    print(f"  ... and {len(job['errors']) - 5} more errors")
        else:
            print(f"No ETL job found with ID {args.job_details}")
    
    elif args.monitor:
        print(f"Monitoring system resources for {args.duration} seconds...")
        monitor.start_monitoring()
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        data = monitor.stop_monitoring()
        print(f"Collected {len(data)} data points")
        
        if args.report:
            report_file = monitor.generate_resource_report()
            print(f"Resource report generated at {report_file}")
    
    elif args.quality:
        metrics = monitor.get_data_quality_metrics()
        if 'error' in metrics:
            print(f"Error getting data quality metrics: {metrics['error']}")
        else:
            print("Data Quality Metrics:")
            print(f"  Total Games: {metrics.get('total_games', 0)}")
            print(f"  Total Moves: {metrics.get('total_moves', 0)}")
            print(f"  Incomplete Games: {metrics.get('incomplete_games', 0)}")
            print(f"  Incomplete Moves: {metrics.get('incomplete_moves', 0)}")
            
            if 'date_range' in metrics:
                print(f"  Date Range: {metrics['date_range'].get('min')} to {metrics['date_range'].get('max')}")
            
            if 'games_by_engine' in metrics:
                print("\nGames by Engine:")
                for engine, count in metrics['games_by_engine'].items():
                    print(f"  {engine}: {count}")
    
    else:
        parser = argparse.ArgumentParser(description='v7p3r Chess Engine ETL Monitor')
        parser.print_help()
