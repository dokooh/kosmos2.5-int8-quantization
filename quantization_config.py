"""
Configuration file for Kosmos 2.5 quantization with different strategies.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from optimum.onnxruntime.configuration import AutoQuantizationConfig


@dataclass
class QuantizationConfig:
    """Configuration class for different quantization strategies"""
    
    # Model configuration
    model_id: str = "microsoft/kosmos-2.5"
    output_dir: str = "./quantized_kosmos"
    
    # Quantization settings
    quantization_approach: str = "avx512_vnni"  # avx512_vnni, avx512, avx2, arm64
    per_channel: bool = False
    is_static: bool = False
    
    # Performance settings
    force_reload: bool = False
    use_cache: bool = True
    
    # Advanced settings
    optimization_level: str = "all"  # all, basic, extended
    
    def get_quantization_config(self) -> AutoQuantizationConfig:
        """Get the appropriate QuantizationConfig based on approach"""
        
        config_map = {
            "avx512_vnni": AutoQuantizationConfig.avx512_vnni,
            "avx512": AutoQuantizationConfig.avx512,
            "avx2": AutoQuantizationConfig.avx2,
            "arm64": AutoQuantizationConfig.arm64,
        }
        
        if self.quantization_approach not in config_map:
            raise ValueError(f"Unsupported quantization approach: {self.quantization_approach}")
        
        config_func = config_map[self.quantization_approach]
        return config_func(is_static=self.is_static, per_channel=self.per_channel)
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get organized output paths for different components"""
        base_dir = self.output_dir
        return {
            "base": base_dir,
            "onnx": os.path.join(base_dir, "onnx_model"),
            "quantized": os.path.join(base_dir, "quantized"),
            "logs": os.path.join(base_dir, "logs"),
            "benchmarks": os.path.join(base_dir, "benchmarks")
        }


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    "fast_cpu": QuantizationConfig(
        quantization_approach="avx2",
        per_channel=False,
        is_static=False,
        optimization_level="basic"
    ),
    
    "balanced_cpu": QuantizationConfig(
        quantization_approach="avx512_vnni",
        per_channel=False,
        is_static=False,
        optimization_level="all"
    ),
    
    "high_performance_cpu": QuantizationConfig(
        quantization_approach="avx512_vnni",
        per_channel=True,
        is_static=False,
        optimization_level="extended"
    ),
    
    "mobile_arm": QuantizationConfig(
        quantization_approach="arm64",
        per_channel=False,
        is_static=False,
        optimization_level="all"
    ),
    
    "server_optimized": QuantizationConfig(
        quantization_approach="avx512_vnni",
        per_channel=True,
        is_static=False,
        optimization_level="extended"
    )
}


def get_preset_config(preset_name: str) -> QuantizationConfig:
    """Get a preset configuration by name"""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return PRESET_CONFIGS[preset_name]


def list_available_presets() -> Dict[str, str]:
    """List all available preset configurations with descriptions"""
    descriptions = {
        "fast_cpu": "Fast quantization for general CPU usage (AVX2)",
        "balanced_cpu": "Balanced performance and quality (AVX512 VNNI)",
        "high_performance_cpu": "Maximum performance with per-channel quantization",
        "mobile_arm": "Optimized for ARM64 mobile processors",
        "server_optimized": "Server deployment with maximum optimizations"
    }
    
    return descriptions


# Hardware detection utilities
def detect_cpu_capabilities() -> Dict[str, bool]:
    """Detect CPU capabilities for optimal quantization strategy selection"""
    import platform
    import subprocess
    
    capabilities = {
        "avx2": False,
        "avx512": False,
        "vnni": False,
        "arm64": False
    }
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Basic ARM detection
    if any(arch in machine for arch in ['arm', 'aarch64']):
        capabilities["arm64"] = True
        return capabilities
    
    # For Windows/Linux x86_64, try to detect CPU features
    try:
        if system == "windows":
            # Try wmic (Windows Management Instrumentation)
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name,Description"],
                capture_output=True, text=True, timeout=10
            )
            cpu_info = result.stdout.lower()
            
            # Basic heuristics based on CPU name
            if "intel" in cpu_info:
                # Most modern Intel CPUs support AVX2
                capabilities["avx2"] = True
                # Check for newer Intel CPUs that likely support AVX-512
                if any(gen in cpu_info for gen in ["xeon", "i7-", "i9-"]):
                    capabilities["avx512"] = True
                    capabilities["vnni"] = True
        
        elif system == "linux":
            # Check /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = f.read().lower()
                
            if "avx2" in cpu_info:
                capabilities["avx2"] = True
            if "avx512" in cpu_info:
                capabilities["avx512"] = True
            if "vnni" in cpu_info or "avx512_vnni" in cpu_info:
                capabilities["vnni"] = True
                
    except Exception:
        # Fallback: assume AVX2 support for modern x86_64
        if "x86_64" in machine or "amd64" in machine:
            capabilities["avx2"] = True
    
    return capabilities


def recommend_quantization_approach() -> str:
    """Recommend the best quantization approach based on detected hardware"""
    capabilities = detect_cpu_capabilities()
    
    if capabilities["arm64"]:
        return "arm64"
    elif capabilities["vnni"]:
        return "avx512_vnni"
    elif capabilities["avx512"]:
        return "avx512"
    elif capabilities["avx2"]:
        return "avx2"
    else:
        # Conservative fallback
        return "avx2"


def get_recommended_config(model_id: str = "microsoft/kosmos-2.5", output_dir: str = "./quantized_kosmos") -> QuantizationConfig:
    """Get a recommended configuration based on hardware detection"""
    approach = recommend_quantization_approach()
    
    config = QuantizationConfig(
        model_id=model_id,
        output_dir=output_dir,
        quantization_approach=approach,
        per_channel=False,  # Start conservative
        is_static=False     # Dynamic quantization is easier to set up
    )
    
    return config