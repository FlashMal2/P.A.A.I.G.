# situational_awareness.py

from datetime import datetime
import socket
import psutil
import platform
import os
import subprocess

def get_time_of_day():
    now = datetime.now()
    hour = now.hour
    if hour < 6:
        part_of_day = "early morning"
    elif hour < 12:
        part_of_day = "morning"
    elif hour < 17:
        part_of_day = "afternoon"
    elif hour < 21:
        part_of_day = "evening"
    else:
        part_of_day = "late night"
    return {
        "formatted": now.strftime("%A, %I:%M %p"),
        "part_of_day": part_of_day
    }

def get_hostname():
    return socket.gethostname()

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "Unavailable"

def get_system_info():
    uname = platform.uname()
    return {
        "system": uname.system,
        "node_name": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor
    }

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def is_running_on_mobile():
    return os.getenv("KOHANA_CLIENT", "desktop").lower() == "mobile"

def get_gpu_usage():
    """Returns GPU memory and usage percentage using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        usage_percent, mem_used, mem_total = output.strip().split(',')
        return {
            "gpu_utilization": f"{usage_percent.strip()}%",
            "gpu_memory_used": f"{mem_used.strip()} MiB",
            "gpu_memory_total": f"{mem_total.strip()} MiB"
        }
    except Exception as e:
        return {
            "gpu_utilization": "Unavailable",
            "gpu_memory_used": "Unavailable",
            "gpu_memory_total": "Unavailable",
            "error": str(e)
        }

def collect_situation_snapshot():
    return {
        "time": get_time_of_day(),
        "hostname": get_hostname(),
        "ip_address": get_ip_address(),
        "system_info": get_system_info(),
        "cpu_usage": get_cpu_usage(),
        "memory_usage": get_memory_usage(),
        "gpu_status": get_gpu_usage(),
        "client_type": "mobile" if is_running_on_mobile() else "desktop"
    }
