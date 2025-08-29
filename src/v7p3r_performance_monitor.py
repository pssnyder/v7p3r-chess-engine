#!/usr/bin/env python3
"""
V7P3R Chess Engine V8.3 - Performance Auditing and Waste Detection
Comprehensive analysis of computational efficiency and optimization opportunities
"""

import time
import functools
import sys
import os
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import traceback
import threading
import psutil
import inspect

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class FunctionProfile:
    """Profile data for a single function"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    avg_time: float = 0.0
    total_memory_delta: float = 0.0
    recursive_calls: int = 0
    error_count: int = 0
    last_call_time: float = 0.0
    
    def update(self, execution_time: float, memory_delta: float = 0.0, 
               is_recursive: bool = False, had_error: bool = False):
        """Update profile with new execution data"""
        self.call_count += 1
        self.total_time += execution_time
        self.total_memory_delta += memory_delta
        self.last_call_time = time.time()
        
        if execution_time > self.max_time:
            self.max_time = execution_time
        
        if execution_time < self.min_time:
            self.min_time = execution_time
        
        self.avg_time = self.total_time / self.call_count
        
        if is_recursive:
            self.recursive_calls += 1
        
        if had_error:
            self.error_count += 1


@dataclass
class PerformanceIssue:
    """Represents a detected performance issue"""
    category: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    function_name: str
    description: str
    impact_estimate: str
    recommendation: str
    data: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiling and waste detection"""
    
    def __init__(self):
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.call_stack: List[Tuple[str, float, float]] = []  # (function, start_time, start_memory)
        self.recursion_depth: Dict[str, int] = defaultdict(int)
        self.memory_snapshots: List[Tuple[float, float]] = []  # (time, memory_mb)
        self.performance_issues: List[PerformanceIssue] = []
        
        # Waste detection
        self.redundant_calls: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)
        self.memory_leaks: List[Tuple[str, float, float]] = []  # (function, time, memory_increase)
        self.inefficient_loops: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Process monitoring
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Configuration
        self.slow_function_threshold = 0.1  # seconds
        self.memory_leak_threshold = 1.0    # MB
        self.redundancy_window = 5.0        # seconds
        self.max_recursion_depth = 50
        
        # Statistics
        self.profiling_overhead = 0.0
        self.start_time = time.time()
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance"""
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Track recursion
            self.recursion_depth[func_name] += 1
            is_recursive = self.recursion_depth[func_name] > 1
            
            # Check for excessive recursion
            if self.recursion_depth[func_name] > self.max_recursion_depth:
                self.performance_issues.append(PerformanceIssue(
                    category="RECURSION",
                    severity="CRITICAL",
                    function_name=func_name,
                    description=f"Excessive recursion depth: {self.recursion_depth[func_name]}",
                    impact_estimate="Stack overflow risk, severe performance degradation",
                    recommendation="Implement iterative solution or add recursion depth limits"
                ))
            
            # Add to call stack
            self.call_stack.append((func_name, start_time, start_memory))
            
            # Execute function
            had_error = False
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                had_error = True
                raise
            finally:
                # End profiling
                end_time = time.time()
                end_memory = self.process.memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Update profile
                if func_name not in self.function_profiles:
                    self.function_profiles[func_name] = FunctionProfile(func_name)
                
                self.function_profiles[func_name].update(
                    execution_time, memory_delta, is_recursive, had_error
                )
                
                # Remove from call stack
                self.call_stack.pop()
                self.recursion_depth[func_name] -= 1
                
                # Detect performance issues
                self._detect_issues(func_name, execution_time, memory_delta, args, kwargs)
                
                # Track profiling overhead
                overhead = time.time() - end_time
                self.profiling_overhead += overhead
            
            return result
        
        return wrapper
    
    def _detect_issues(self, func_name: str, execution_time: float, 
                      memory_delta: float, args: Tuple, kwargs: Dict):
        """Detect various performance issues"""
        current_time = time.time()
        
        # Slow function detection
        if execution_time > self.slow_function_threshold:
            self.performance_issues.append(PerformanceIssue(
                category="PERFORMANCE",
                severity="HIGH" if execution_time > 1.0 else "MEDIUM",
                function_name=func_name,
                description=f"Slow execution: {execution_time:.3f}s",
                impact_estimate=f"Function takes {execution_time:.1f}x longer than threshold",
                recommendation="Profile internally for bottlenecks, consider optimization",
                data={"execution_time": execution_time, "args_count": len(args)}
            ))
        
        # Memory leak detection
        if memory_delta > self.memory_leak_threshold:
            self.memory_leaks.append((func_name, current_time, memory_delta))
            self.performance_issues.append(PerformanceIssue(
                category="MEMORY",
                severity="HIGH",
                function_name=func_name,
                description=f"Memory increase: {memory_delta:.1f}MB",
                impact_estimate="Potential memory leak or inefficient allocation",
                recommendation="Review memory allocation and ensure proper cleanup",
                data={"memory_delta": memory_delta}
            ))
        
        # Redundant call detection
        call_signature = self._create_call_signature(func_name, args, kwargs)
        recent_calls = self.redundant_calls[call_signature]
        
        # Remove old calls outside the window
        recent_calls = [
            (call_time, result) for call_time, result in recent_calls
            if current_time - call_time <= self.redundancy_window
        ]
        self.redundant_calls[call_signature] = recent_calls
        
        # Check for redundancy
        if len(recent_calls) > 3:  # More than 3 identical calls in window
            self.performance_issues.append(PerformanceIssue(
                category="REDUNDANCY",
                severity="MEDIUM",
                function_name=func_name,
                description=f"Redundant calls: {len(recent_calls)} identical calls in {self.redundancy_window}s",
                impact_estimate="Wasted computation cycles",
                recommendation="Implement caching or memoization",
                data={"call_count": len(recent_calls), "call_signature": call_signature}
            ))
        
        # Add current call to redundancy tracking
        recent_calls.append((current_time, None))  # Could store result for more analysis
    
    def _create_call_signature(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Create a signature for redundancy detection"""
        # Simple signature based on function name and argument types/lengths
        # More sophisticated signatures could include actual values for deterministic functions
        arg_signature = f"args({len(args)})"
        kwarg_signature = f"kwargs({len(kwargs)})"
        return f"{func_name}:{arg_signature}:{kwarg_signature}"
    
    def analyze_loop_efficiency(self, loop_info: Dict[str, Any]):
        """Analyze loop efficiency (called manually for complex analysis)"""
        func_name = loop_info.get('function', 'unknown')
        iterations = loop_info.get('iterations', 0)
        total_time = loop_info.get('total_time', 0.0)
        
        if iterations > 0 and total_time > 0:
            time_per_iteration = total_time / iterations
            
            # Detect inefficient loops
            if time_per_iteration > 0.001:  # More than 1ms per iteration
                self.inefficient_loops[func_name].append(loop_info)
                
                severity = "HIGH" if time_per_iteration > 0.01 else "MEDIUM"
                self.performance_issues.append(PerformanceIssue(
                    category="LOOP_EFFICIENCY",
                    severity=severity,
                    function_name=func_name,
                    description=f"Slow loop: {time_per_iteration*1000:.1f}ms per iteration",
                    impact_estimate=f"Loop inefficiency: {iterations} iterations, {total_time:.3f}s total",
                    recommendation="Optimize loop body, consider vectorization or algorithm changes",
                    data=loop_info
                ))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_time = time.time()
        total_runtime = current_time - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Function statistics
        function_stats = []
        for profile in self.function_profiles.values():
            if profile.call_count > 0:
                function_stats.append({
                    'name': profile.name,
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'avg_time': profile.avg_time,
                    'max_time': profile.max_time,
                    'time_percentage': (profile.total_time / total_runtime) * 100,
                    'memory_delta': profile.total_memory_delta,
                    'recursive_calls': profile.recursive_calls,
                    'error_count': profile.error_count
                })
        
        # Sort by total time spent
        function_stats.sort(key=lambda x: x['total_time'], reverse=True)
        
        # Memory analysis
        memory_analysis = {
            'baseline_mb': self.baseline_memory,
            'current_mb': current_memory,
            'peak_increase_mb': current_memory - self.baseline_memory,
            'leak_count': len(self.memory_leaks),
            'functions_with_leaks': len(set(leak[0] for leak in self.memory_leaks))
        }
        
        # Issue summary
        issue_summary = Counter(issue.category for issue in self.performance_issues)
        severity_summary = Counter(issue.severity for issue in self.performance_issues)
        
        # Top bottlenecks
        top_bottlenecks = function_stats[:10]  # Top 10 time consumers
        
        # Efficiency metrics
        total_function_time = sum(f['total_time'] for f in function_stats)
        profiling_overhead_percentage = (self.profiling_overhead / total_runtime) * 100
        
        return {
            'runtime_summary': {
                'total_runtime': total_runtime,
                'total_function_time': total_function_time,
                'profiling_overhead': self.profiling_overhead,
                'profiling_overhead_percentage': profiling_overhead_percentage
            },
            'function_statistics': function_stats,
            'memory_analysis': memory_analysis,
            'performance_issues': [
                {
                    'category': issue.category,
                    'severity': issue.severity,
                    'function': issue.function_name,
                    'description': issue.description,
                    'impact': issue.impact_estimate,
                    'recommendation': issue.recommendation,
                    'data': issue.data
                }
                for issue in self.performance_issues
            ],
            'issue_summary': dict(issue_summary),
            'severity_summary': dict(severity_summary),
            'top_bottlenecks': top_bottlenecks,
            'redundancy_analysis': {
                'tracked_signatures': len(self.redundant_calls),
                'total_redundant_groups': sum(
                    1 for calls in self.redundant_calls.values() if len(calls) > 1
                )
            },
            'loop_analysis': {
                'functions_with_loops': len(self.inefficient_loops),
                'total_loop_issues': sum(len(loops) for loops in self.inefficient_loops.values())
            }
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized optimization recommendations"""
        recommendations = []
        
        # High-priority function optimizations
        for profile in sorted(self.function_profiles.values(), 
                            key=lambda p: p.total_time, reverse=True)[:5]:
            if profile.total_time > 1.0:  # Functions taking more than 1 second total
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'FUNCTION_OPTIMIZATION',
                    'target': profile.name,
                    'description': f"Optimize {profile.name} - {profile.total_time:.2f}s total time",
                    'potential_savings': f"Up to {profile.total_time:.1f}s runtime reduction",
                    'approaches': [
                        'Profile internal function calls',
                        'Optimize algorithms and data structures',
                        'Implement caching if appropriate',
                        'Consider parallel processing for independent operations'
                    ]
                })
        
        # Memory optimization recommendations
        memory_increase = self.process.memory_info().rss / 1024 / 1024 - self.baseline_memory
        if memory_increase > 10:  # More than 10MB increase
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'MEMORY_OPTIMIZATION',
                'target': 'Global memory management',
                'description': f"High memory usage increase: {memory_increase:.1f}MB",
                'potential_savings': f"Reduce memory footprint by {memory_increase * 0.3:.1f}MB",
                'approaches': [
                    'Implement automatic cache cleanup',
                    'Use memory pools for frequent allocations',
                    'Profile memory allocation patterns',
                    'Add memory pressure handling'
                ]
            })
        
        # Redundancy elimination
        redundant_functions = [
            sig for sig, calls in self.redundant_calls.items() 
            if len(calls) > 2
        ]
        if redundant_functions:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'REDUNDANCY_ELIMINATION',
                'target': 'Multiple functions',
                'description': f"Eliminate redundant calls in {len(redundant_functions)} function signatures",
                'potential_savings': "10-30% reduction in redundant computation",
                'approaches': [
                    'Implement function-level memoization',
                    'Add result caching for expensive operations',
                    'Restructure algorithms to avoid repeated calculations',
                    'Use lazy evaluation where appropriate'
                ]
            })
        
        return recommendations
    
    def export_profile_data(self, filename: str):
        """Export detailed profiling data for external analysis"""
        report = self.get_performance_report()
        recommendations = self.get_optimization_recommendations()
        
        export_data = {
            'timestamp': time.time(),
            'profiler_version': '1.0',
            'performance_report': report,
            'optimization_recommendations': recommendations,
            'raw_function_profiles': {
                name: {
                    'call_count': p.call_count,
                    'total_time': p.total_time,
                    'avg_time': p.avg_time,
                    'max_time': p.max_time,
                    'min_time': p.min_time if p.min_time != float('inf') else 0,
                    'memory_delta': p.total_memory_delta,
                    'recursive_calls': p.recursive_calls,
                    'error_count': p.error_count
                }
                for name, p in self.function_profiles.items()
            }
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


# Global profiler instance for easy integration
_global_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def profile(func: Callable) -> Callable:
    """Convenience decorator for profiling functions"""
    return get_profiler().profile_function(func)

def analyze_performance() -> Dict[str, Any]:
    """Get performance analysis from global profiler"""
    return get_profiler().get_performance_report()

def get_recommendations() -> List[Dict[str, Any]]:
    """Get optimization recommendations from global profiler"""
    return get_profiler().get_optimization_recommendations()


# Context manager for profiling code blocks
class ProfiledSection:
    """Context manager for profiling arbitrary code blocks"""
    
    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        self.name = name
        self.profiler = profiler or get_profiler()
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = end_time - self.start_time
            memory_delta = end_memory - self.start_memory
            
            # Create or update profile for this section
            if self.name not in self.profiler.function_profiles:
                self.profiler.function_profiles[self.name] = FunctionProfile(self.name)
            
            self.profiler.function_profiles[self.name].update(
                execution_time, memory_delta, False, exc_type is not None
            )
            
            # Detect issues for this section
            self.profiler._detect_issues(self.name, execution_time, memory_delta, (), {})


def profiled_section(name: str):
    """Create a profiled section context manager"""
    return ProfiledSection(name)
