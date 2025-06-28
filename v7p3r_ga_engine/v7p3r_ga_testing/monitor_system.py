#!/usr/bin/env python3
"""
System monitor for GA training to identify performance issues
"""
import psutil
import time
import sys
import os

def monitor_system():
    """Monitor CPU, memory, and GPU usage"""
    print("=== System Monitor Started ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_temp = "N/A"
            
            # Try to get CPU temperature (may not work on all systems)
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            for entry in entries:
                                if entry.current:
                                    cpu_temp = f"{entry.current:.1f}°C"
                                    break
            except:
                pass
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            # GPU Usage (if nvidia-ml-py available)
            gpu_info = "N/A"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_info = f"GPU: {gpu_util.gpu}% | GPU Mem: {gpu_mem.used/1024**3:.1f}GB | GPU Temp: {gpu_temp}°C"
            except:
                # Try with torch if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3
                        gpu_info = f"GPU Mem: {gpu_mem_alloc:.1f}GB allocated, {gpu_mem_cached:.1f}GB cached"
                except:
                    pass
            
            # Display info
            print(f"\\rCPU: {cpu_percent:5.1f}% | Temp: {cpu_temp} | RAM: {memory_gb:5.1f}GB ({memory_percent:4.1f}%) | {gpu_info}", end="")
            
            # Check for dangerous conditions
            if memory_gb > 20:
                print(f"\\n⚠️  HIGH MEMORY USAGE: {memory_gb:.1f}GB - Consider stopping process!")
            if cpu_percent > 90:
                print(f"\\n⚠️  HIGH CPU USAGE: {cpu_percent:.1f}% - System may be overloaded!")
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\\n\\n=== Monitoring stopped ===")

if __name__ == "__main__":
    monitor_system()
