"""
performance_monitor.py

Performance monitoring system for tracking system metrics and identifying bottlenecks.

Features:
- Time tracking for operations
- Metric collection
- Performance reports
- Bottleneck identification
- Resource usage monitoring
"""

import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class PerformanceMonitor:
    """
    Monitors and tracks system performance metrics.
    
    Tracks:
    - Operation execution times
    - Resource usage
    - Error rates
    - Throughput
    """
    
    def __init__(self):
        """
        Initialize performance monitor.
        """
        self.metrics = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.start_times = {}
        self.session_start = datetime.now()
        
        logging.info("Performance Monitor initialized")
    
    def track_time(self, operation_name: str):
        """
        Decorator to track function execution time.
        
        :param operation_name: Name of the operation to track
        :return: Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    self.metrics[operation_name].append(elapsed)
                    self.operation_counts[operation_name] += 1
                    
                    logging.debug(f"{operation_name} completed in {elapsed:.2f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.error_counts[operation_name] += 1
                    logging.error(f"{operation_name} failed after {elapsed:.2f}s: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def start_operation(self, operation_name: str):
        """
        Start tracking an operation.
        
        :param operation_name: Name of the operation
        """
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str):
        """
        End tracking an operation and record the time.
        
        :param operation_name: Name of the operation
        :return: Elapsed time in seconds
        """
        if operation_name in self.start_times:
            elapsed = time.time() - self.start_times[operation_name]
            self.metrics[operation_name].append(elapsed)
            self.operation_counts[operation_name] += 1
            del self.start_times[operation_name]
            
            logging.debug(f"{operation_name} completed in {elapsed:.2f}s")
            return elapsed
        else:
            logging.warning(f"Operation {operation_name} was not started")
            return 0.0
    
    def record_metric(self, operation_name: str, value: float):
        """
        Record a metric value.
        
        :param operation_name: Name of the operation
        :param value: Metric value
        """
        self.metrics[operation_name].append(value)
    
    def record_error(self, operation_name: str):
        """
        Record an error for an operation.
        
        :param operation_name: Name of the operation
        """
        self.error_counts[operation_name] += 1
        logging.warning(f"Error recorded for {operation_name}")
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict]:
        """
        Get statistics for an operation.
        
        :param operation_name: Name of the operation
        :return: Statistics dictionary or None
        """
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return None
        
        times = self.metrics[operation_name]
        
        return {
            'operation': operation_name,
            'count': len(times),
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': statistics.median(times),
            'stdev_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'errors': self.error_counts.get(operation_name, 0),
            'error_rate': self.error_counts.get(operation_name, 0) / len(times) if times else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all operations.
        
        :return: Dictionary of operation statistics
        """
        all_stats = {}
        
        for operation_name in self.metrics.keys():
            stats = self.get_operation_stats(operation_name)
            if stats:
                all_stats[operation_name] = stats
        
        return all_stats
    
    def get_bottleneck(self) -> Optional[Dict]:
        """
        Identify the performance bottleneck.
        
        :return: Bottleneck operation statistics or None
        """
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return None
        
        # Find operation with highest total time
        bottleneck = max(all_stats.items(), key=lambda x: x[1]['total_time'])
        
        return {
            'operation': bottleneck[0],
            'stats': bottleneck[1],
            'reason': f"Highest total execution time: {bottleneck[1]['total_time']:.2f}s"
        }
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        :return: Performance report dictionary
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        all_stats = self.get_all_stats()
        bottleneck = self.get_bottleneck()
        
        # Calculate totals
        total_operations = sum(stats['count'] for stats in all_stats.values())
        total_time = sum(stats['total_time'] for stats in all_stats.values())
        total_errors = sum(self.error_counts.values())
        
        report = {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': session_duration,
            'total_operations': total_operations,
            'total_time_seconds': total_time,
            'total_errors': total_errors,
            'average_operation_time': total_time / total_operations if total_operations > 0 else 0.0,
            'error_rate': total_errors / total_operations if total_operations > 0 else 0.0,
            'bottleneck': bottleneck,
            'operation_stats': all_stats,
            'summary': self._generate_summary(all_stats, bottleneck)
        }
        
        return report
    
    def _generate_summary(self, all_stats: Dict, bottleneck: Optional[Dict]) -> str:
        """
        Generate human-readable summary.
        
        :param all_stats: All operation statistics
        :param bottleneck: Bottleneck information
        :return: Summary text
        """
        if not all_stats:
            return "No operations tracked yet."
        
        summary_lines = [
            "Performance Summary:",
            f"  Total operations: {sum(s['count'] for s in all_stats.values())}",
            f"  Total time: {sum(s['total_time'] for s in all_stats.values()):.2f}s",
            f"  Total errors: {sum(s['errors'] for s in all_stats.values())}",
            ""
        ]
        
        if bottleneck:
            summary_lines.append(f"Bottleneck: {bottleneck['operation']}")
            summary_lines.append(f"  {bottleneck['reason']}")
            summary_lines.append("")
        
        # Top 5 slowest operations
        sorted_ops = sorted(all_stats.items(), 
                          key=lambda x: x[1]['total_time'], 
                          reverse=True)[:5]
        
        summary_lines.append("Top 5 Slowest Operations:")
        for i, (op_name, stats) in enumerate(sorted_ops, 1):
            summary_lines.append(
                f"  {i}. {op_name}: "
                f"{stats['total_time']:.2f}s total, "
                f"{stats['avg_time']:.2f}s avg, "
                f"{stats['count']} calls"
            )
        
        return "\n".join(summary_lines)
    
    def print_report(self):
        """
        Print performance report to console.
        """
        report = self.generate_performance_report()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Session Duration: {report['session_duration_seconds']:.2f}s")
        print(f"Total Operations: {report['total_operations']}")
        print(f"Total Time: {report['total_time_seconds']:.2f}s")
        print(f"Average Operation Time: {report['average_operation_time']:.3f}s")
        print(f"Total Errors: {report['total_errors']}")
        print(f"Error Rate: {report['error_rate']:.2%}")
        print()
        
        if report['bottleneck']:
            print("BOTTLENECK:")
            print(f"  Operation: {report['bottleneck']['operation']}")
            print(f"  Reason: {report['bottleneck']['reason']}")
            print()
        
        print("OPERATION STATISTICS:")
        for op_name, stats in sorted(report['operation_stats'].items(), 
                                    key=lambda x: x[1]['total_time'], 
                                    reverse=True):
            print(f"  {op_name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time']:.2f}s")
            print(f"    Avg: {stats['avg_time']:.3f}s")
            print(f"    Min: {stats['min_time']:.3f}s")
            print(f"    Max: {stats['max_time']:.3f}s")
            if stats['errors'] > 0:
                print(f"    Errors: {stats['errors']} ({stats['error_rate']:.2%})")
            print()
        
        print("=" * 80)
    
    def save_report(self, file_path: str):
        """
        Save performance report to JSON file.
        
        :param file_path: Output file path
        """
        report = self.generate_performance_report()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Performance report saved to {file_path}")
    
    def reset(self):
        """
        Reset all metrics (start new session).
        """
        self.metrics.clear()
        self.operation_counts.clear()
        self.error_counts.clear()
        self.start_times.clear()
        self.session_start = datetime.now()
        
        logging.info("Performance monitor reset")


# Global instance for easy access
_global_monitor = None

def get_monitor() -> PerformanceMonitor:
    """
    Get global performance monitor instance.
    
    :return: Global PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


if __name__ == "__main__":
    # Test the performance monitor
    monitor = PerformanceMonitor()
    
    # Test decorator
    @monitor.track_time("test_operation")
    def test_function():
        time.sleep(0.1)
        return "done"
    
    # Test manual tracking
    for i in range(5):
        monitor.start_operation(f"manual_op_{i}")
        time.sleep(0.05)
        monitor.end_operation(f"manual_op_{i}")
    
    # Test function calls
    for i in range(3):
        test_function()
    
    # Test error tracking
    monitor.record_error("test_operation")
    
    # Generate and print report
    monitor.print_report()
    
    # Save report
    import tempfile
    report_path = tempfile.mktemp(suffix='.json')
    monitor.save_report(report_path)
    print(f"\nReport saved to: {report_path}")
