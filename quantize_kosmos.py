#!/usr/bin/env python3
"""
Kosmos 2.5 Model Quantization using ORTQuantizer

This script quantizes the Kosmos 2.5 Hugging Face model using ONNX Runtime's
dynamic quantization to reduce model size and improve inference speed.

Usage:
    python quantize_kosmos.py --model_id microsoft/kosmos-2.5 --output_dir ./quantized_kosmos
"""

import argparse
import os
import time
import psutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from optimum.onnxruntime import ORTQuantizer, ORTModelForVision2Seq
from optimum.onnxruntime.configuration import AutoQuantizationConfig

try:
    from quantization_config import get_recommended_config, recommend_quantization_approach
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def quantize_kosmos_model(
    model_id: str = "microsoft/kosmos-2.5",
    output_dir: str = "./quantized_kosmos",
    quantization_approach: str = "avx512_vnni",
    per_channel: bool = False,
    force_reload: bool = False
):
    """
    Quantize Kosmos 2.5 model using ORTQuantizer
    
    Args:
        model_id: Hugging Face model identifier
        output_dir: Directory to save quantized model
        quantization_approach: Quantization strategy (avx512_vnni, avx512, avx2, arm64)
        per_channel: Whether to use per-channel quantization
        force_reload: Whether to force reloading the model
    """
    
    print(f"Starting quantization of {model_id}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization approach: {quantization_approach}")
    print(f"Per-channel quantization: {per_channel}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track memory and time
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    try:
        # Step 1: Load the original model and convert to ONNX
        print("Step 1: Loading model and converting to ONNX...")
        
        # Load processor for tokenization/preprocessing
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Check if ONNX model already exists
        onnx_model_path = os.path.join(output_dir, "onnx_model")
        
        if os.path.exists(onnx_model_path) and not force_reload:
            print(f"Found existing ONNX model at {onnx_model_path}, loading...")
            ort_model = ORTModelForVision2Seq.from_pretrained(onnx_model_path)
        else:
            print("Converting PyTorch model to ONNX format...")
            # Load and export to ONNX
            ort_model = ORTModelForVision2Seq.from_pretrained(
                model_id, 
                export=True,
                use_cache=False
            )
            # Save ONNX model
            ort_model.save_pretrained(onnx_model_path)
            print(f"ONNX model saved to {onnx_model_path}")
        
        conversion_memory = get_memory_usage()
        conversion_time = time.time() - start_time
        
        print(f"ONNX conversion completed in {conversion_time:.2f}s")
        print(f"Memory usage after conversion: {conversion_memory:.2f} MB (+{conversion_memory - initial_memory:.2f} MB)")
        
        # Step 2: Create quantizer
        print("\nStep 2: Creating quantizer...")
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        
        # Step 3: Configure quantization
        print(f"Step 3: Configuring {quantization_approach} quantization...")
        
        # Select quantization configuration
        if quantization_approach == "avx512_vnni":
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=per_channel)
        elif quantization_approach == "avx512":
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=per_channel)
        elif quantization_approach == "avx2":
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=per_channel)
        elif quantization_approach == "arm64":
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=per_channel)
        else:
            raise ValueError(f"Unsupported quantization approach: {quantization_approach}")
        
        print(f"Quantization config: {qconfig}")
        
        # Step 4: Perform quantization
        print("\nStep 4: Performing quantization...")
        quantization_start = time.time()
        
        quantized_model_path = os.path.join(output_dir, "quantized")
        
        # Quantize the model
        quantizer.quantize(
            save_dir=quantized_model_path,
            quantization_config=qconfig,
        )
        
        quantization_time = time.time() - quantization_start
        final_memory = get_memory_usage()
        total_time = time.time() - start_time
        
        # Save processor alongside quantized model
        processor.save_pretrained(quantized_model_path)
        
        print("\nQuantization completed successfully!")
        print(f"Quantized model saved to: {quantized_model_path}")
        print(f"Total time: {total_time:.2f}s (Conversion: {conversion_time:.2f}s, Quantization: {quantization_time:.2f}s)")
        print(f"Peak memory usage: {final_memory:.2f} MB")
        
        # Step 5: Compare model sizes
        print("\nStep 5: Comparing model sizes...")
        
        def get_folder_size(folder_path):
            """Calculate total size of all files in folder"""
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)  # Convert to MB
        
        if os.path.exists(onnx_model_path):
            original_size = get_folder_size(onnx_model_path)
            quantized_size = get_folder_size(quantized_model_path)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            
            print(f"Original ONNX model size: {original_size:.2f} MB")
            print(f"Quantized model size: {quantized_size:.2f} MB")
            print(f"Size reduction: {original_size - quantized_size:.2f} MB ({(1 - quantized_size/original_size)*100:.1f}%)")
            print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model_path
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Quantize Kosmos 2.5 model using ORTQuantizer")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/kosmos-2.5",
        help="Hugging Face model identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_kosmos",
        help="Directory to save quantized model"
    )
    # Auto-detect default approach if possible
    default_approach = "avx512_vnni"
    if CONFIG_AVAILABLE:
        try:
            default_approach = recommend_quantization_approach()
        except Exception:
            pass
    
    parser.add_argument(
        "--quantization_approach",
        type=str,
        choices=["avx512_vnni", "avx512", "avx2", "arm64"],
        default=default_approach,
        help=f"Quantization strategy (auto-detected: {default_approach})"
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Use per-channel quantization"
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force reload and re-export the model"
    )
    
    args = parser.parse_args()
    
    print("Kosmos 2.5 Model Quantization")
    print("=" * 50)
    
    result = quantize_kosmos_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        quantization_approach=args.quantization_approach,
        per_channel=args.per_channel,
        force_reload=args.force_reload
    )
    
    if result:
        print(f"\n✅ Quantization successful! Model saved to: {result}")
    else:
        print("\n❌ Quantization failed!")


if __name__ == "__main__":
    main()