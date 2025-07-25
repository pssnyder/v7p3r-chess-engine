#!/usr/bin/env python3
"""
v7p3r Chess Engine - Unit Testing Suite Launcher

This module provides a comprehensive, configurable unit testing launcher that can run
tests in parallel with robust error handling and detailed reporting. It supports
both monitored and unmonitored testing modes with extensive configuration options.

Features:
- Parallel test execution with configurable thread limits
- Comprehensive error handling and reporting
- Multiple output formats (JSON, XML, text)
- Performance monitoring and memory tracking
- Configurable test selection and exclusion
- CI/CD-ready structured output
- Real-time progress monitoring

Author: v7p3r Testing Suite
Date: 2025-06-22
"""

import os
import sys
import json
import yaml
import time
import threading
import subprocess
import concurrent.futures
import logging
import traceback
import psutil
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestResult:
    """Container for individual test results."""
    test_name: str
    module_name: str
    status: str  # 'passed', 'failed', 'skipped', 'timeout', 'error'
    duration: float
    start_time: str
    end_time: str
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    memory_usage: Optional[float] = None
    assertions_count: Optional[int] = None
    warnings: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class TestSuiteResult:
    """Container for overall test suite results."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    timeouts: int
    total_duration: float
    start_time: str
    end_time: str
    success_rate: float
    average_duration: float
    max_memory_usage: float
    test_results: List[TestResult]
    configuration: Dict[str, Any]
    environment_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'test_results': [asdict(result) for result in self.test_results]
        }


class TestExecutor:
    """Handles execution of individual test modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.test_timeout = config.get('execution', {}).get('test_timeout', 300)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for test execution."""
        logger = logging.getLogger('TestExecutor')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def execute_test_module(self, test_module_path: str, test_name: str) -> TestResult:
        """Execute a single test module and return results."""
        start_time = datetime.now()
        start_time_str = start_time.isoformat()
        
        # Initialize result
        result = TestResult(
            test_name=test_name,
            module_name=os.path.basename(test_module_path),
            status='unknown',
            duration=0.0,
            start_time=start_time_str,
            end_time='',
            memory_usage=0.0
        )
        
        try:
            # Set up environment variables
            env = os.environ.copy()
            env.update(self.config.get('environment', {}).get('test_env_vars', {}))
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute the test
            cmd = [sys.executable, '-m', 'unittest', test_module_path, '-v']
            
            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                env=env,
                cwd=os.path.dirname(os.path.dirname(test_module_path))
            )
            
            # Calculate duration and memory usage
            end_time = datetime.now()
            result.duration = (end_time - start_time).total_seconds()
            result.end_time = end_time.isoformat()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            result.memory_usage = max(0, final_memory - initial_memory)
            
            # Parse test output
            self._parse_test_output(result, process_result)
            
        except subprocess.TimeoutExpired:
            result.status = 'timeout'
            result.error_message = f"Test timed out after {self.test_timeout} seconds"
            result.end_time = datetime.now().isoformat()
            result.duration = self.test_timeout
            
        except Exception as e:
            result.status = 'error'
            result.error_message = str(e)
            result.stack_trace = traceback.format_exc()
            result.end_time = datetime.now().isoformat()
            result.duration = (datetime.now() - start_time).total_seconds()
            
        return result
    
    def _parse_test_output(self, result: TestResult, process_result: subprocess.CompletedProcess):
        """Parse unittest output to extract detailed results."""
        stdout = process_result.stdout
        stderr = process_result.stderr
        
        # Determine status from return code and output
        if process_result.returncode == 0:
            result.status = 'passed'
        else:
            result.status = 'failed'
            result.error_message = stderr or "Test failed"
            result.stack_trace = stderr
        
        # Extract test count and other metrics from output
        lines = stdout.split('\n')
        for line in lines:
            if 'Ran' in line and 'test' in line:
                # Extract test count: "Ran 15 tests in 2.345s"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result.assertions_count = int(parts[1])
                    except (ValueError, IndexError):
                        pass
              # Extract warnings
            if 'warning' in line.lower():
                if result.warnings is not None:
                    result.warnings.append(line.strip())
        
        # Check for specific failure patterns
        if 'FAILED' in stdout or 'ERROR' in stdout:
            result.status = 'failed'
            if not result.error_message:
                result.error_message = "Test failures detected in output"
        
        # Check for skipped tests
        if 'skipped' in stdout.lower():
            if result.status == 'passed' and 'FAILED' not in stdout:
                result.status = 'skipped'


class TestSuiteLauncher:
    """Main class for launching and managing the test suite."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'unit_testing_config.yaml'
        )
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.executor = TestExecutor(self.config)
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available."""
        return {
            'execution': {
                'test_timeout': 300,
                'max_threads': 4,
                'failure_mode': 'continue'
            },
            'output': {
                'verbosity': 'standard',
                'terminal': {'enabled': True, 'colored_output': True},
                'file_logging': {'enabled': True, 'log_format': 'json'}
            },
            'test_selection': {
                'run_all': True,
                'categories': {'engine_utilities': True, 'metrics': True},
                'include_tests': [],
                'exclude_tests': []
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('TestSuiteLauncher')
        
        # Set verbosity level
        verbosity = self.config.get('output', {}).get('verbosity', 'standard')
        if verbosity == 'debug':
            logger.setLevel(logging.DEBUG)
        elif verbosity == 'verbose':
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        # Set up file logging if enabled
        file_logging = self.config.get('output', {}).get('file_logging', {})
        if file_logging.get('enabled', True):
            log_dir = file_logging.get('log_directory', 'testing/results')
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime(file_logging.get('timestamp_format', '%Y%m%d_%H%M%S'))
            log_file = os.path.join(log_dir, f'test_suite_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def discover_tests(self) -> List[Tuple[str, str]]:
        """Discover test modules based on configuration."""
        test_dir = os.path.join(os.path.dirname(__file__), 'unit_test_launchers')
        test_files = []
        
        # Get all test files
        for filename in os.listdir(test_dir):
            if filename.endswith('_testing.py') and not filename.startswith('_'):
                test_path = os.path.join(test_dir, filename)
                test_name = filename[:-3]  # Remove .py extension
                test_files.append((test_path, test_name))
        
        # Filter based on configuration
        return self._filter_tests(test_files)
    
    def _filter_tests(self, test_files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter test files based on configuration."""
        test_selection = self.config.get('test_selection', {})
        
        # If specific tests are included, use only those
        include_tests = test_selection.get('include_tests', [])
        if include_tests:
            return [(path, name) for path, name in test_files 
                   if name in include_tests]
        
        # Filter by categories if run_all is False
        if not test_selection.get('run_all', True):
            categories = test_selection.get('categories', {})
            filtered_files = []
            
            for path, name in test_files:
                should_include = False
                
                # Check category inclusion
                if name.startswith('chess_game') and categories.get('chess_game', False):
                    should_include = True
                elif name.startswith('v7p3r') and categories.get('main_engine', False):
                    should_include = True
                elif name.startswith('metrics') and categories.get('metrics', False):
                    should_include = True
                elif categories.get('engine_utilities', False):
                    # Most other tests are engine utilities
                    should_include = True
                
                if should_include:
                    filtered_files.append((path, name))
            
            test_files = filtered_files
        
        # Exclude specific tests
        exclude_tests = test_selection.get('exclude_tests', [])
        if exclude_tests:
            test_files = [(path, name) for path, name in test_files 
                         if name not in exclude_tests]
        
        return test_files
    
    def run_tests(self) -> TestSuiteResult:
        """Run all discovered tests and return comprehensive results."""
        self.start_time = datetime.now()
        start_time_str = self.start_time.isoformat()
        
        print(f"🚀 Starting v7p3r Chess Engine Unit Test Suite at {start_time_str}")
        print("=" * 80)
        
        # Discover tests
        test_files = self.discover_tests()
        total_tests = len(test_files)
        
        if total_tests == 0:
            print("❌ No tests found to execute!")
            return self._create_empty_result()
        
        print(f"📋 Found {total_tests} test modules to execute")
        
        # Execute tests in parallel
        max_threads = self.config.get('execution', {}).get('max_threads', 4)
        failure_mode = self.config.get('execution', {}).get('failure_mode', 'continue')
        
        print(f"⚙️  Configuration: {max_threads} parallel threads, {failure_mode} mode")
        print("=" * 80)
        
        results = []
        failed_early = False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self.executor.execute_test_module, test_path, test_name): (test_path, test_name)
                for test_path, test_name in test_files
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_test):
                test_path, test_name = future_to_test[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print progress
                    self._print_test_result(result, len(results), total_tests)
                    
                    # Check failure mode
                    if failure_mode == 'stop' and result.status in ['failed', 'error', 'timeout']:
                        print(f"\n🛑 Stopping test suite due to failure in {test_name}")
                        print(f"Error: {result.error_message}")
                        if result.stack_trace:
                            print(f"Stack trace:\n{result.stack_trace}")
                        failed_early = True
                        
                        # Cancel remaining tests
                        for remaining_future in future_to_test:
                            if remaining_future != future:
                                remaining_future.cancel()
                        # Properly shutdown the executor to cancel pending futures
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                        
                except Exception as e:
                    # Handle executor errors
                    error_result = TestResult(
                        test_name=test_name,
                        module_name=os.path.basename(test_path),
                        status='error',
                        duration=0.0,
                        start_time=datetime.now().isoformat(),
                        end_time=datetime.now().isoformat(),
                        error_message=f"Executor error: {str(e)}",
                        stack_trace=traceback.format_exc()
                    )
                    results.append(error_result)
                    self._print_test_result(error_result, len(results), total_tests)
        
        self.end_time = datetime.now()
        self.results = results
        
        # Create comprehensive result summary
        suite_result = self._create_suite_result(results, failed_early)
        
        # Print summary
        self._print_summary(suite_result)
        
        # Save results
        self._save_results(suite_result)
        
        return suite_result
    
    def _print_test_result(self, result: TestResult, current: int, total: int):
        """Print individual test result with progress."""
        status_symbols = {
            'passed': '✅',
            'failed': '❌',
            'skipped': '⏭️',
            'timeout': '⏰',
            'error': '💥'
        }
        
        symbol = status_symbols.get(result.status, '❓')
        progress = f"[{current}/{total}]"
        duration_str = f"{result.duration:.2f}s"
        
        # Color output if enabled
        if self.config.get('output', {}).get('terminal', {}).get('colored_output', True):
            colors = {
                'passed': '\033[92m',  # Green
                'failed': '\033[91m',  # Red
                'skipped': '\033[93m', # Yellow
                'timeout': '\033[95m', # Magenta
                'error': '\033[91m',   # Red
                'reset': '\033[0m'
            }
            color = colors.get(result.status, '')
            reset = colors['reset']
        else:
            color = reset = ''
        
        print(f"{progress} {symbol} {color}{result.test_name}{reset} ({duration_str})")
        
        # Print error details if failed and verbose mode
        if result.status in ['failed', 'error'] and self.config.get('output', {}).get('verbosity') in ['verbose', 'debug']:
            if result.error_message:
                print(f"    💬 {result.error_message}")
    
    def _create_suite_result(self, results: List[TestResult], failed_early: bool) -> TestSuiteResult:
        """Create comprehensive test suite result."""
        # Count results by status
        status_counts = defaultdict(int)
        for result in results:
            status_counts[result.status] += 1
        
        total_tests = len(results)
        passed = status_counts['passed']
        failed = status_counts['failed']
        skipped = status_counts['skipped']
        errors = status_counts['error']
        timeouts = status_counts['timeout']
          # Calculate metrics
        total_duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0.0
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        average_duration = sum(r.duration for r in results) / len(results) if results else 0
        max_memory = max(r.memory_usage or 0 for r in results) if results else 0
        
        # Environment info
        env_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'failed_early': failed_early,
            'total_system_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'available_memory': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'cpu_count': psutil.cpu_count()
        }
        
        return TestSuiteResult(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            timeouts=timeouts,
            total_duration=total_duration,            start_time=self.start_time.isoformat() if self.start_time else '',
            end_time=self.end_time.isoformat() if self.end_time else '',
            success_rate=success_rate,
            average_duration=average_duration,
            max_memory_usage=max_memory,
            test_results=results,
            configuration=self.config,
            environment_info=env_info
        )
    
    def _create_empty_result(self) -> TestSuiteResult:
        """Create empty result when no tests are found."""
        now = datetime.now()
        return TestSuiteResult(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            timeouts=0,
            total_duration=0.0,
            start_time=now.isoformat(),
            end_time=now.isoformat(),
            success_rate=0.0,
            average_duration=0.0,
            max_memory_usage=0.0,
            test_results=[],
            configuration=self.config,
            environment_info={}
        )
    
    def _print_summary(self, result: TestSuiteResult):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 80)
        
        # Overall results
        print(f"🎯 Total Tests: {result.total_tests}")
        print(f"✅ Passed: {result.passed}")
        print(f"❌ Failed: {result.failed}")
        print(f"⏭️  Skipped: {result.skipped}")
        print(f"💥 Errors: {result.errors}")
        print(f"⏰ Timeouts: {result.timeouts}")
        print(f"📈 Success Rate: {result.success_rate:.1f}%")
        print(f"⏱️  Total Duration: {result.total_duration:.2f}s")
        print(f"📊 Average Test Duration: {result.average_duration:.2f}s")
        print(f"💾 Max Memory Usage: {result.max_memory_usage:.1f}MB")
        
        # Failed tests details
        if result.failed > 0 or result.errors > 0 or result.timeouts > 0:
            print("\n🔍 FAILED TESTS:")
            print("-" * 40)
            for test_result in result.test_results:
                if test_result.status in ['failed', 'error', 'timeout']:
                    print(f"❌ {test_result.test_name}: {test_result.error_message}")
        
        # Performance warnings
        if result.max_memory_usage > 500:  # 500MB threshold
            print(f"\n⚠️  WARNING: High memory usage detected ({result.max_memory_usage:.1f}MB)")
        
        if result.average_duration > 30:  # 30 second average threshold
            print(f"⚠️  WARNING: Slow test execution detected (avg: {result.average_duration:.1f}s)")
        
        print("=" * 80)
        
        # Final status
        if result.passed == result.total_tests:
            print("🎉 ALL TESTS PASSED!")
        elif result.success_rate >= 90:
            print("🟢 MOSTLY SUCCESSFUL")
        elif result.success_rate >= 70:
            print("🟡 PARTIAL SUCCESS")
        else:
            print("🔴 SIGNIFICANT FAILURES")
    
    def _save_results(self, result: TestSuiteResult):
        """Save test results to files."""
        file_logging = self.config.get('output', {}).get('file_logging', {})
        if not file_logging.get('enabled', True):
            return
        
        log_dir = file_logging.get('log_directory', 'testing/results')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime(file_logging.get('timestamp_format', '%Y%m%d_%H%M%S'))
        log_format = file_logging.get('log_format', 'json')
        
        # Save in requested format
        if log_format == 'json':
            result_file = os.path.join(log_dir, f'test_results_{timestamp}.json')
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"📄 Results saved to: {result_file}")
        
        elif log_format == 'xml':
            result_file = os.path.join(log_dir, f'test_results_{timestamp}.xml')
            self._save_xml_results(result, result_file)
            print(f"📄 Results saved to: {result_file}")
        
        elif log_format == 'text':
            result_file = os.path.join(log_dir, f'test_results_{timestamp}.txt')
            self._save_text_results(result, result_file)
            print(f"📄 Results saved to: {result_file}")
    
    def _save_xml_results(self, result: TestSuiteResult, filename: str):
        """Save results in XML format (JUnit compatible)."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('testsuite')
        root.set('name', 'v7p3r_Unit_Tests')
        root.set('tests', str(result.total_tests))
        root.set('failures', str(result.failed))
        root.set('errors', str(result.errors))
        root.set('time', str(result.total_duration))
        root.set('timestamp', result.start_time)
        
        for test_result in result.test_results:
            testcase = ET.SubElement(root, 'testcase')
            testcase.set('name', test_result.test_name)
            testcase.set('classname', test_result.module_name)
            testcase.set('time', str(test_result.duration))
            
            if test_result.status == 'failed':
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', test_result.error_message or 'Test failed')
                failure.text = test_result.stack_trace or ''
            elif test_result.status == 'error':
                error = ET.SubElement(testcase, 'error')
                error.set('message', test_result.error_message or 'Test error')
                error.text = test_result.stack_trace or ''
            elif test_result.status == 'skipped':
                ET.SubElement(testcase, 'skipped')
        
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
    
    def _save_text_results(self, result: TestSuiteResult, filename: str):
        """Save results in text format."""
        with open(filename, 'w') as f:
            f.write("v7p3r Chess Engine - Unit Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Execution Time: {result.start_time} to {result.end_time}\n")
            f.write(f"Total Duration: {result.total_duration:.2f} seconds\n")
            f.write(f"Total Tests: {result.total_tests}\n")
            f.write(f"Passed: {result.passed}\n")
            f.write(f"Failed: {result.failed}\n")
            f.write(f"Errors: {result.errors}\n")
            f.write(f"Skipped: {result.skipped}\n")
            f.write(f"Timeouts: {result.timeouts}\n")
            f.write(f"Success Rate: {result.success_rate:.1f}%\n\n")
            
            f.write("Individual Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_result in result.test_results:
                f.write(f"{test_result.test_name}: {test_result.status.upper()}")
                f.write(f" ({test_result.duration:.2f}s)\n")
                
                if test_result.status in ['failed', 'error'] and test_result.error_message:
                    f.write(f"  Error: {test_result.error_message}\n")
                if test_result.warnings:
                    f.write(f"  Warnings: {', '.join(test_result.warnings)}\n")
                f.write("\n")


def main():
    """Main entry point for the test suite launcher."""
    parser = argparse.ArgumentParser(description='v7p3r Chess Engine Unit Test Suite')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--stop-on-fail', action='store_true', help='Stop on first failure')
    parser.add_argument('--include', nargs='+', help='Include specific tests')
    parser.add_argument('--exclude', nargs='+', help='Exclude specific tests')
    parser.add_argument('--timeout', type=int, help='Test timeout in seconds')
    parser.add_argument('--threads', type=int, help='Number of parallel threads')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = TestSuiteLauncher(config_path=args.config)
    
    # Override config with command line arguments
    if args.verbose:
        launcher.config['output']['verbosity'] = 'verbose'
    if args.quiet:
        launcher.config['output']['verbosity'] = 'minimal'
    if args.stop_on_fail:
        launcher.config['execution']['failure_mode'] = 'stop'
    if args.include:
        launcher.config['test_selection']['include_tests'] = args.include
        launcher.config['test_selection']['run_all'] = False
    if args.exclude:
        launcher.config['test_selection']['exclude_tests'] = args.exclude
    if args.timeout:
        launcher.config['execution']['test_timeout'] = args.timeout
    if args.threads:
        launcher.config['execution']['max_threads'] = args.threads
    
    # Run tests
    try:
        result = launcher.run_tests()
        
        # Exit with appropriate code
        if result.failed > 0 or result.errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error in test suite: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()