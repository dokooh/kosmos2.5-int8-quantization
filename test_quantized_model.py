#!/usr/bin/env python3
"""
Test script for verifying quantized Kosmos 2.5 model functionality and performance.

This script loads both the original and quantized models, performs inference tests,
and compares performance metrics including speed and memory usage.

Usage:
    python test_quantized_model.py --quantized_model_path ./quantized_kosmos/quantized
"""

import argparse
import time
import os
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

# Sample test cases for Kosmos 2.5
TEST_CASES = [
    {
        "name": "Simple Description",
        "prompt": "<grounding>Describe this image.",
        "description": "Basic image description task"
    },
    {
        "name": "Object Detection", 
        "prompt": "<grounding>What objects can you see in this image?",
        "description": "Object detection and grounding task"
    },
    {
        "name": "OCR Task",
        "prompt": "<grounding>What text do you see in this image?", 
        "description": "Optical character recognition task"
    }
]


def create_sample_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a sample test image with text and shapes"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some colored rectangles
    draw.rectangle([20, 20, 100, 80], fill='red', outline='black', width=2)
    draw.rectangle([120, 40, 180, 100], fill='blue', outline='black', width=2)
    draw.rectangle([50, 120, 150, 180], fill='green', outline='black', width=2)
    
    # Add some text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        draw.text((30, 190), "Sample Test Image", fill='black', font=font)
        draw.text((30, 210), "For Kosmos 2.5", fill='black', font=font)
    except Exception:
        # Fallback if no font available
        draw.text((30, 190), "Sample Test Image", fill='black')
    
    return img


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_model_inference(
    model: ORTModelForVision2Seq,
    processor: any,
    image: Image.Image,
    prompt: str,
    num_runs: int = 5
) -> Dict[str, float]:
    """Benchmark model inference performance"""
    
    # Warm up
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    _ = model.generate(**inputs, max_new_tokens=10)
    
    # Benchmark
    times = []
    memory_before = get_memory_usage()
    
    for _ in range(num_runs):
        start_time = time.time()
        
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    memory_after = get_memory_usage()
    
    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "memory_delta": memory_after - memory_before
    }


def test_model_functionality(
    model: ORTModelForVision2Seq,
    processor: any,
    test_image: Image.Image
) -> Dict[str, any]:
    """Test model functionality with various prompts"""
    
    results = {}
    
    for test_case in TEST_CASES:
        print(f"Testing: {test_case['name']}")
        
        try:
            inputs = processor(text=test_case["prompt"], images=test_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            results[test_case["name"]] = {
                "success": True,
                "output": generated_text,
                "input_prompt": test_case["prompt"],
                "description": test_case["description"]
            }
            
        except Exception as e:
            results[test_case["name"]] = {
                "success": False,
                "error": str(e),
                "input_prompt": test_case["prompt"],
                "description": test_case["description"]
            }
    
    return results


def compare_models(
    original_model_path: Optional[str],
    quantized_model_path: str,
    test_image: Image.Image,
    num_benchmark_runs: int = 5
) -> Dict[str, any]:
    """Compare original and quantized models"""
    
    print("Loading models for comparison...")
    
    # Load quantized model
    print("Loading quantized model...")
    quantized_processor = AutoProcessor.from_pretrained(quantized_model_path)
    quantized_model = ORTModelForVision2Seq.from_pretrained(quantized_model_path)
    
    results = {
        "quantized": {
            "functionality": test_model_functionality(quantized_model, quantized_processor, test_image),
            "performance": {}
        }
    }
    
    # Benchmark quantized model
    print("Benchmarking quantized model...")
    for test_case in TEST_CASES:
        benchmark_results = benchmark_model_inference(
            quantized_model, quantized_processor, test_image, 
            test_case["prompt"], num_benchmark_runs
        )
        results["quantized"]["performance"][test_case["name"]] = benchmark_results
    
    # Load and test original model if path provided
    if original_model_path and os.path.exists(original_model_path):
        print("Loading original model...")
        original_processor = AutoProcessor.from_pretrained(original_model_path)
        original_model = ORTModelForVision2Seq.from_pretrained(original_model_path)
        
        results["original"] = {
            "functionality": test_model_functionality(original_model, original_processor, test_image),
            "performance": {}
        }
        
        # Benchmark original model
        print("Benchmarking original model...")
        for test_case in TEST_CASES:
            benchmark_results = benchmark_model_inference(
                original_model, original_processor, test_image,
                test_case["prompt"], num_benchmark_runs
            )
            results["original"]["performance"][test_case["name"]] = benchmark_results
    
    return results


def print_comparison_report(results: Dict[str, any]):
    """Print a detailed comparison report"""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)
    
    # Functionality comparison
    print("\nFUNCTIONALITY TEST RESULTS:")
    print("-" * 50)
    
    for test_name in TEST_CASES:
        test_name = test_name["name"]
        print(f"\n{test_name}:")
        
        if "quantized" in results:
            quant_result = results["quantized"]["functionality"][test_name]
            print(f"  Quantized: {'✅ PASS' if quant_result['success'] else '❌ FAIL'}")
            if quant_result["success"]:
                output = quant_result["output"][:100] + "..." if len(quant_result["output"]) > 100 else quant_result["output"]
                print(f"    Output: {output}")
            else:
                print(f"    Error: {quant_result['error']}")
        
        if "original" in results:
            orig_result = results["original"]["functionality"][test_name]
            print(f"  Original:  {'✅ PASS' if orig_result['success'] else '❌ FAIL'}")
    
    # Performance comparison
    print("\n\nPERFORMANCE BENCHMARK RESULTS:")
    print("-" * 50)
    
    for test_name in TEST_CASES:
        test_name = test_name["name"]
        print(f"\n{test_name}:")
        
        if "quantized" in results:
            quant_perf = results["quantized"]["performance"][test_name]
            print(f"  Quantized - Avg: {quant_perf['avg_time']:.3f}s ± {quant_perf['std_time']:.3f}s")
        
        if "original" in results:
            orig_perf = results["original"]["performance"][test_name]
            print(f"  Original  - Avg: {orig_perf['avg_time']:.3f}s ± {orig_perf['std_time']:.3f}s")
            
            # Calculate speedup
            if "quantized" in results:
                speedup = orig_perf['avg_time'] / quant_perf['avg_time']
                print(f"  Speedup: {speedup:.2f}x")
    
    # Model size comparison
    print("\n\nMODEL SIZE COMPARISON:")
    print("-" * 50)
    
    def get_folder_size(folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)  # MB
    
    quantized_path = results.get("quantized_model_path", "")
    if quantized_path and os.path.exists(quantized_path):
        quantized_size = get_folder_size(quantized_path)
        print(f"Quantized model size: {quantized_size:.2f} MB")
    
    original_path = results.get("original_model_path", "")
    if original_path and os.path.exists(original_path):
        original_size = get_folder_size(original_path)
        print(f"Original model size: {original_size:.2f} MB")
        
        if quantized_path and os.path.exists(quantized_path):
            compression_ratio = original_size / quantized_size
            size_reduction = (1 - quantized_size/original_size) * 100
            print(f"Size reduction: {size_reduction:.1f}% (compression ratio: {compression_ratio:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="Test quantized Kosmos 2.5 model")
    parser.add_argument(
        "--quantized_model_path",
        type=str,
        required=True,
        help="Path to the quantized model directory"
    )
    parser.add_argument(
        "--original_model_path",
        type=str,
        help="Path to the original ONNX model directory for comparison"
    )
    parser.add_argument(
        "--test_image_path",
        type=str,
        help="Path to custom test image (optional)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of benchmark runs for averaging"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        help="Path to save detailed test report"
    )
    
    args = parser.parse_args()
    
    print("Kosmos 2.5 Quantized Model Test Suite")
    print("=" * 50)
    
    # Prepare test image
    if args.test_image_path and os.path.exists(args.test_image_path):
        test_image = Image.open(args.test_image_path).convert("RGB")
        print(f"Using custom test image: {args.test_image_path}")
    else:
        test_image = create_sample_image()
        print("Using generated test image")
    
    # Run comparison
    results = compare_models(
        args.original_model_path,
        args.quantized_model_path,
        test_image,
        args.num_runs
    )
    
    # Store paths for size comparison
    results["quantized_model_path"] = args.quantized_model_path
    results["original_model_path"] = args.original_model_path
    
    # Print report
    print_comparison_report(results)
    
    # Save detailed report if requested
    if args.output_report:
        import json
        with open(args.output_report, 'w') as f:
            # Make results JSON serializable
            json_results = {}
            for model_type, data in results.items():
                if isinstance(data, dict):
                    json_results[model_type] = {}
                    for key, value in data.items():
                        if isinstance(value, dict):
                            json_results[model_type][key] = {
                                k: float(v) if isinstance(v, np.number) else v 
                                for k, v in value.items() if k != 'output'  # Skip long text outputs
                            }
                        else:
                            json_results[model_type][key] = value
                else:
                    json_results[model_type] = str(data)
            
            json.dump(json_results, f, indent=2)
        print(f"\nDetailed report saved to: {args.output_report}")
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()