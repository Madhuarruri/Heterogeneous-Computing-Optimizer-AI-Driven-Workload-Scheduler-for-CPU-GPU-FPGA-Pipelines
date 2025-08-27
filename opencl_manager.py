
import pyopencl as cl
import numpy as np
from typing import Dict, Any, Tuple
import os

class OpenCLManager:
    """Manager for OpenCL operations"""

    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.programs = {}
        self._initialize_opencl()

    def _initialize_opencl(self):
        """Initialize OpenCL context and command queue"""
        try:
            # Get available platforms and devices
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            # Prefer GPU devices, fallback to CPU
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    self.device = devices[0]
                    break

            if not self.device:
                # Fallback to CPU
                for platform in platforms:
                    devices = platform.get_devices(cl.device_type.CPU)
                    if devices:
                        self.device = devices[0]
                        break

            if not self.device:
                raise RuntimeError("No suitable OpenCL device found")

            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)

            # Load and compile kernels
            self._load_kernels()

        except Exception as e:
            print(f"OpenCL initialization failed: {e}")
            # Continue without OpenCL acceleration

    def _load_kernels(self):
        """Load and compile OpenCL kernels"""
        kernel_files = {
            "gemm": "gemm.cl",
            "fft": "fft.cl", 
            "image": "image_ops.cl"
        }

        for name, filename in kernel_files.items():
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    source = f.read()
                try:
                    program = cl.Program(self.context, source).build()
                    self.programs[name] = program
                except cl.RuntimeError as e:
                    print(f"Failed to compile {name} kernel: {e}")

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication using OpenCL"""
        if "gemm" not in self.programs:
            # Fallback to NumPy
            return np.dot(A, B)

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions don't match"

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=M*N*4)

        # Execute kernel
        program = self.programs["gemm"]
        kernel = program.gemm_nn

        local_size = (16, 16)
        global_size = (((M + 15) // 16) * 16, ((N + 15) // 16) * 16)

        kernel(self.queue, global_size, local_size,
               A_buf, B_buf, C_buf,
               np.int32(M), np.int32(N), np.int32(K),
               np.float32(1.0), np.float32(0.0))

        # Read result
        C = np.empty((M, N), dtype=np.float32)
        cl.enqueue_copy(self.queue, C, C_buf)

        return C

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the OpenCL device"""
        if not self.device:
            return {"status": "not_available"}

        return {
            "status": "available",
            "name": self.device.name,
            "vendor": self.device.vendor,
            "version": self.device.version,
            "max_compute_units": self.device.max_compute_units,
            "max_work_group_size": self.device.max_work_group_size,
            "global_mem_size": self.device.global_mem_size // (1024**3),  # GB
            "local_mem_size": self.device.local_mem_size // 1024,  # KB
        }
