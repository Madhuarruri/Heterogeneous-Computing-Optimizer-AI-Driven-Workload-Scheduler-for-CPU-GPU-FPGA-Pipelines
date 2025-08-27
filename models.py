
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class WorkloadType(str, Enum):
    MATRIX_MULTIPLY = "matrix_multiply"
    FFT_2D = "fft_2d"
    IMAGE_PROCESSING = "image_processing"
    RESNET50 = "resnet50"
    VIDEO_TRANSCODE = "video_transcode"

class OptimizationGoal(str, Enum):
    SPEED = "speed"
    ENERGY = "energy"
    BALANCED = "balanced"

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    organization: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    organization: Optional[str] = None
    created_at: datetime

class TaskCreate(BaseModel):
    workload_type: WorkloadType
    params: Dict[str, Any]
    optimization_goal: OptimizationGoal
    device_preference: Optional[str] = None

class Task(BaseModel):
    id: str
    user_id: str
    workload_type: WorkloadType
    params: Dict[str, Any]
    optimization_goal: OptimizationGoal
    device_chosen: str
    status: TaskStatus
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class RunMetrics(BaseModel):
    task_id: str
    cpu_ms: Optional[float] = None
    gpu_ms: Optional[float] = None
    fpga_ms: Optional[float] = None
    energy_joules_est: Optional[float] = None
    mem_gb: Optional[float] = None
    speedup_vs_cpu: Optional[float] = None
    timestamp: datetime

class DeviceInfo(BaseModel):
    name: str
    type: str
    status: str
    utilization: float
    specs: Dict[str, Any]
