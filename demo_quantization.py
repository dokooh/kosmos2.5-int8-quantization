#!/usr/bin/env python3
"""
Demo script showing the complete quantization workflow for Kosmos 2.5

This script demonstrates:
1. Hardware detection and configuration
2. Model quantization 
3. Basic functionality testing
4. Performance comparison

Usage: python demo_quantization.py
"""

import os
import time
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a step header"""
    print(f"\n🔄 Step {step_num}: {description}")
    print("-" * 50)

def main():
    print_header("Kosmos 2.5 Quantization Demo")
    
    # Check if we're in the right environment
    if not Path("quantize_kosmos.py").exists():
        print("❌ Error: Please run this script from the project directory")
        sys.exit(1)
    
    # Step 1: Hardware Detection
    print_step(1, "Hardware Detection")
    try:
        from quantization_config import detect_cpu_capabilities, recommend_quantization_approach
        
        capabilities = detect_cpu_capabilities()
        recommended = recommend_quantization_approach()
        
        print("Detected CPU capabilities:")
        for feature, supported in capabilities.items():
            status = "✅ Supported" if supported else "❌ Not supported"
            print(f"  {feature.upper()}: {status}")
        
        print(f"\n🎯 Recommended approach: {recommended}")
        
    except ImportError:
        print("⚠️  Using fallback detection")
        recommended = "avx512_vnni"
        print(f"🎯 Default approach: {recommended}")
    
    # Step 2: Model Quantization  
    print_step(2, "Model Quantization")
    
    output_dir = "./demo_quantized_kosmos"
    
    print(f"Starting quantization with approach: {recommended}")
    print(f"Output directory: {output_dir}")
    
    # Import and run quantization
    try:
        from quantize_kosmos import quantize_kosmos_model
        
        start_time = time.time()
        
        result = quantize_kosmos_model(
            model_id="microsoft/kosmos-2.5",
            output_dir=output_dir,
            quantization_approach=recommended,
            per_channel=False,
            force_reload=False
        )
        
        quantization_time = time.time() - start_time
        
        if result:
            print(f"✅ Quantization completed in {quantization_time:.2f} seconds")
            print(f"📁 Model saved to: {result}")
        else:
            print("❌ Quantization failed")
            return
            
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        return
    
    # Step 3: Basic Functionality Test
    print_step(3, "Basic Functionality Test")
    
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import AutoProcessor
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
        draw.text((60, 170), "Test Image", fill='black')
        
        # Load quantized model
        quantized_path = os.path.join(output_dir, "quantized")
        print(f"Loading quantized model from: {quantized_path}")
        
        processor = AutoProcessor.from_pretrained(quantized_path)
        model = ORTModelForVision2Seq.from_pretrained(quantized_path)
        
        # Test inference
        prompt = "<grounding>Describe this image."
        inputs = processor(text=prompt, images=test_image, return_tensors="pt")
        
        print("Running inference...")
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        inference_time = time.time() - start_time
        
        # Decode output
        result_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"✅ Inference completed in {inference_time:.3f} seconds")
        print(f"📝 Generated text: {result_text}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return
    
    # Step 4: Model Size Comparison
    print_step(4, "Model Size Analysis")
    
    def get_folder_size(folder_path):
        """Calculate folder size in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)
    
    try:
        onnx_path = os.path.join(output_dir, "onnx_model")
        quantized_path = os.path.join(output_dir, "quantized")
        
        if os.path.exists(onnx_path):
            original_size = get_folder_size(onnx_path)
            quantized_size = get_folder_size(quantized_path)
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            size_reduction = (1 - quantized_size/original_size) * 100 if original_size > 0 else 0
            
            print(f"📊 Original ONNX model size: {original_size:.2f} MB")
            print(f"📊 Quantized model size: {quantized_size:.2f} MB")
            print(f"📈 Size reduction: {size_reduction:.1f}%")
            print(f"📈 Compression ratio: {compression_ratio:.2f}x")
        else:
            quantized_size = get_folder_size(quantized_path)
            print(f"📊 Quantized model size: {quantized_size:.2f} MB")
            print("ℹ️  Original model size not available for comparison")
            
    except Exception as e:
        print(f"⚠️  Could not calculate model sizes: {e}")
    
    # Summary
    print_header("Demo Summary")
    
    print("✅ Successfully completed quantization demo!")
    print("\n📋 What was accomplished:")
    print(f"   • Detected optimal quantization approach: {recommended}")
    print(f"   • Quantized Kosmos 2.5 model using ORTQuantizer")  
    print(f"   • Verified model functionality with test inference")
    print(f"   • Analyzed model compression metrics")
    
    print("\n🚀 Next steps:")
    print("   • Run comprehensive tests:")
    print(f"     python test_quantized_model.py --quantized_model_path {quantized_path}")
    print("   • Integrate the quantized model into your application")
    print("   • Benchmark performance with your specific use cases")
    
    print(f"\n📁 Quantized model location: {quantized_path}")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()