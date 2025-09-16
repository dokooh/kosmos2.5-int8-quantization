#!/usr/bin/env python3
"""
Quick setup and hardware detection script for Kosmos 2.5 quantization.

This script detects your hardware capabilities and recommends the best
quantization approach for your system.
"""

import platform
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

def check_virtual_environment():
    """Check if virtual environment exists and is activated"""
    venv_path = Path("kosmos-qint8")
    
    if not venv_path.exists():
        print("❌ Virtual environment 'kosmos-qint8' not found")
        print("   Run: python -m venv kosmos-qint8")
        return False
    
    # Check if we're in the virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment exists but not activated")
        system = platform.system().lower()
        if system == "windows":
            print("   Run: kosmos-qint8\\Scripts\\activate")
        else:
            print("   Run: source kosmos-qint8/bin/activate")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    import pkg_resources
    
    required_packages = [
        "torch",
        "transformers", 
        "optimum",
        "onnx", 
        "onnxruntime",
        "pillow",
        "numpy",
        "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Use pkg_resources instead of __import__ to avoid initialization issues
            pkg_resources.get_distribution(package)
            print(f"✅ {package}")
        except (pkg_resources.DistributionNotFound, pkg_resources.RequirementParseError):
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def detect_hardware():
    """Detect hardware capabilities and recommend quantization approach"""
    print("\n" + "="*50)
    print("HARDWARE DETECTION")
    print("="*50)
    
    system = platform.system()
    machine = platform.machine().lower()
    processor = platform.processor()
    
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    print(f"Processor: {processor}")
    
    # Import our detection functions if available
    try:
        from quantization_config import detect_cpu_capabilities, recommend_quantization_approach
        
        capabilities = detect_cpu_capabilities()
        recommended = recommend_quantization_approach()
        
        print(f"\nCPU Capabilities:")
        for feature, supported in capabilities.items():
            status = "✅" if supported else "❌"
            print(f"  {feature.upper()}: {status}")
        
        print(f"\nRecommended quantization approach: {recommended}")
        
        # Provide specific recommendations
        if recommended == "avx512_vnni":
            print("🚀 Your CPU supports the fastest quantization method!")
            print("   Expected speedup: 2-3x")
        elif recommended == "avx512":
            print("⚡ Your CPU supports AVX-512 instructions.")
            print("   Expected speedup: 1.8-2.5x")
        elif recommended == "avx2":
            print("👍 Your CPU supports AVX2 instructions.")
            print("   Expected speedup: 1.5-2x")
        elif recommended == "arm64":
            print("🍎 ARM64 processor detected (Apple Silicon?).")
            print("   Expected speedup: 1.5-2x")
        
    except ImportError:
        print("⚠️  Hardware detection unavailable (missing quantization_config.py)")
        
        # Basic fallback detection
        if "arm" in machine or "aarch64" in machine:
            recommended = "arm64"
            print("ARM processor detected - recommend: arm64")
        else:
            recommended = "avx2"  # Conservative fallback
            print("x86_64 processor detected - recommend: avx2 (conservative)")

def show_next_steps(recommended_approach="avx512_vnni"):
    """Show next steps for quantization"""
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    
    print("\n1. Start quantization with recommended settings:")
    print(f"   python quantize_kosmos.py --quantization_approach {recommended_approach}")
    
    print("\n2. Or use the batch script (Windows):")
    print(f"   run_quantization.bat {recommended_approach}")
    
    print("\n3. Test the quantized model:")
    print("   python test_quantized_model.py --quantized_model_path ./quantized_kosmos/quantized")
    
    print("\n4. Compare with original (if available):")
    print("   python test_quantized_model.py \\")
    print("       --quantized_model_path ./quantized_kosmos/quantized \\")
    print("       --original_model_path ./quantized_kosmos/onnx_model")

def main():
    print("Kosmos 2.5 Quantization Setup Check")
    print("=" * 50)
    
    # Check system requirements
    print("\nSYSTEM REQUIREMENTS:")
    print("-" * 20)
    
    python_ok = check_python_version()
    venv_ok = check_virtual_environment()
    
    if not python_ok or not venv_ok:
        print("\n❌ Please fix the above issues before continuing.")
        return
    
    print("\nDEPENDENCIES:")
    print("-" * 20)
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Hardware detection
    detect_hardware()
    
    # Get recommendation
    try:
        from quantization_config import recommend_quantization_approach
        recommended = recommend_quantization_approach()
    except ImportError:
        recommended = "avx512_vnni"  # Default fallback
    
    # Show next steps
    show_next_steps(recommended)
    
    print("\n✅ Setup check completed!")
    print("\nFor detailed documentation, see: README.md")

if __name__ == "__main__":
    main()