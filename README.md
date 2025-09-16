# Kosmos 2.5 Model Quantization with ORTQuantizer

This project provides tools to quantize the Microsoft Kosmos 2.5 multimodal model using ONNX Runtime's ORTQuantizer for improved performance and reduced model size.

## Overview

Kosmos 2.5 is a multimodal large language model that can process both text and images. This repository contains scripts to:

1. **Quantize** the model from FP32 to INT8 using dynamic quantization
2. **Test** the quantized model's functionality and performance
3. **Compare** original vs quantized model metrics
4. **Configure** different quantization strategies for various hardware

## Features

- ✅ **Multiple Quantization Strategies**: AVX-512 VNNI, AVX-512, AVX2, ARM64
- ✅ **Automatic Hardware Detection**: Recommends optimal quantization approach
- ✅ **Preset Configurations**: Fast, balanced, high-performance, mobile, server
- ✅ **Comprehensive Testing**: Functionality and performance benchmarks
- ✅ **Memory Monitoring**: Track memory usage during quantization
- ✅ **Size Comparison**: Analyze model compression ratios
- ✅ **Virtual Environment**: Isolated Python environment setup

## Prerequisites

- Python 3.8+ (tested with Python 3.13)
- Windows/Linux/macOS
- At least 8GB RAM (16GB+ recommended)
- CPU with AVX2+ support (for optimal performance)

## Installation

### 1. Clone/Setup the Project

```bash
# Navigate to your project directory
cd c:\SAI\IA\kosmos-quant-int8

# Create virtual environment
python -m venv kosmos-qint8

# Activate virtual environment
# Windows:
kosmos-qint8\Scripts\activate
# Linux/macOS:
source kosmos-qint8/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Quantization

```bash
# Quantize Kosmos 2.5 with default settings
python quantize_kosmos.py

# Use specific quantization approach
python quantize_kosmos.py --quantization_approach avx512_vnni

# Quantize with per-channel quantization for better quality
python quantize_kosmos.py --per_channel

# Custom output directory
python quantize_kosmos.py --output_dir ./my_quantized_model
```

### Using Preset Configurations

```python
from quantization_config import get_preset_config, list_available_presets

# List available presets
presets = list_available_presets()
print(presets)

# Use a preset
config = get_preset_config("balanced_cpu")
```

### Testing the Quantized Model

```bash
# Test quantized model functionality
python test_quantized_model.py --quantized_model_path ./quantized_kosmos/quantized

# Compare with original model
python test_quantized_model.py \
    --quantized_model_path ./quantized_kosmos/quantized \
    --original_model_path ./quantized_kosmos/onnx_model \
    --num_runs 10

# Save detailed report
python test_quantized_model.py \
    --quantized_model_path ./quantized_kosmos/quantized \
    --output_report ./benchmark_report.json
```

## Detailed Usage

### Command Line Arguments

#### `quantize_kosmos.py`

```bash
python quantize_kosmos.py [OPTIONS]

Options:
  --model_id TEXT                 Hugging Face model ID [default: microsoft/kosmos-2.5]
  --output_dir TEXT              Output directory [default: ./quantized_kosmos]
  --quantization_approach TEXT   Quantization strategy [avx512_vnni|avx512|avx2|arm64]
  --per_channel                  Enable per-channel quantization
  --force_reload                 Force model reload and re-export
  --help                         Show help message
```

#### `test_quantized_model.py`

```bash
python test_quantized_model.py [OPTIONS]

Options:
  --quantized_model_path TEXT    Path to quantized model [required]
  --original_model_path TEXT     Path to original model for comparison
  --test_image_path TEXT         Custom test image path
  --num_runs INTEGER             Number of benchmark runs [default: 5]
  --output_report TEXT           Path to save JSON report
  --help                         Show help message
```

### Quantization Strategies

| Strategy | Description | Best For | Requirements |
|----------|-------------|----------|--------------|
| `avx512_vnni` | AVX-512 with VNNI instructions | Modern Intel Xeon, Core i7/i9 | Intel CPUs with AVX-512 VNNI |
| `avx512` | AVX-512 without VNNI | Intel CPUs with AVX-512 | Intel CPUs with AVX-512 |
| `avx2` | AVX2 instructions | Most modern x86-64 CPUs | x86-64 CPUs with AVX2 |
| `arm64` | ARM64 optimization | Apple Silicon, ARM servers | ARM64/AArch64 processors |

### Preset Configurations

| Preset | Strategy | Per-Channel | Best For |
|--------|----------|-------------|----------|
| `fast_cpu` | AVX2 | No | Quick deployment, older CPUs |
| `balanced_cpu` | AVX-512 VNNI | No | Good balance of speed/quality |
| `high_performance_cpu` | AVX-512 VNNI | Yes | Maximum performance |
| `mobile_arm` | ARM64 | No | Apple Silicon, ARM devices |
| `server_optimized` | AVX-512 VNNI | Yes | Server deployment |

## Understanding the Output

### Quantization Process

```
Starting quantization of microsoft/kosmos-2.5
Output directory: ./quantized_kosmos
Quantization approach: avx512_vnni
Per-channel quantization: False
--------------------------------------------------
Step 1: Loading model and converting to ONNX...
Step 2: Creating quantizer...
Step 3: Configuring avx512_vnni quantization...
Step 4: Performing quantization...
Step 5: Comparing model sizes...

✅ Quantization successful! Model saved to: ./quantized_kosmos/quantized
```

### Performance Metrics

The test script provides detailed metrics:

- **Inference Time**: Average, min, max, standard deviation
- **Memory Usage**: Memory delta during inference
- **Model Size**: Original vs quantized size comparison
- **Functionality Tests**: OCR, object detection, image description

### Expected Results

Typical quantization results for Kosmos 2.5:

- **Size Reduction**: 50-75% smaller model size
- **Speed Improvement**: 1.5-3x faster inference
- **Memory Usage**: 40-60% less RAM during inference
- **Quality**: Minimal accuracy loss for most tasks

## Troubleshooting

### Common Issues

#### 1. ONNX Export Errors
```python
# If you get ONNX export errors, try:
python quantize_kosmos.py --force_reload
```

#### 2. Memory Issues
```python
# For systems with limited RAM:
import torch
torch.cuda.empty_cache()  # If using GPU
```

#### 3. CPU Compatibility
```python
# Check CPU capabilities:
from quantization_config import detect_cpu_capabilities, recommend_quantization_approach

print("CPU capabilities:", detect_cpu_capabilities())
print("Recommended approach:", recommend_quantization_approach())
```

#### 4. Model Loading Issues
```python
# Verify model path and files:
import os
model_path = "./quantized_kosmos/quantized"
required_files = ["model.onnx", "config.json", "preprocessor_config.json"]
for file in required_files:
    path = os.path.join(model_path, file)
    print(f"{file}: {'✅' if os.path.exists(path) else '❌'}")
```

### Performance Tuning

#### 1. Optimize for Speed
```bash
python quantize_kosmos.py --quantization_approach avx512_vnni
```

#### 2. Optimize for Quality
```bash
python quantize_kosmos.py --quantization_approach avx512_vnni --per_channel
```

#### 3. Optimize for Size
```bash
python quantize_kosmos.py --quantization_approach avx2
```

## Advanced Usage

### Custom Quantization Configuration

```python
from quantization_config import QuantizationConfig
from quantize_kosmos import quantize_kosmos_model

# Create custom config
config = QuantizationConfig(
    model_id="microsoft/kosmos-2.5",
    output_dir="./custom_output",
    quantization_approach="avx512_vnni",
    per_channel=True,
    is_static=False
)

# Use custom config
result = quantize_kosmos_model(
    model_id=config.model_id,
    output_dir=config.output_dir,
    quantization_approach=config.quantization_approach,
    per_channel=config.per_channel
)
```

### Batch Processing

```python
# Quantize multiple models or configurations
approaches = ["avx512_vnni", "avx512", "avx2"]

for approach in approaches:
    output_dir = f"./quantized_kosmos_{approach}"
    quantize_kosmos_model(
        output_dir=output_dir,
        quantization_approach=approach
    )
```

### Integration Example

```python
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor
from PIL import Image

# Load quantized model
model = ORTModelForVision2Seq.from_pretrained("./quantized_kosmos/quantized")
processor = AutoProcessor.from_pretrained("./quantized_kosmos/quantized")

# Use the model
image = Image.open("test_image.jpg")
prompt = "<grounding>Describe this image."
inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(result)
```

## File Structure

```
kosmos-quant-int8/
├── kosmos-qint8/                 # Virtual environment
├── quantize_kosmos.py            # Main quantization script
├── test_quantized_model.py       # Testing and benchmarking
├── quantization_config.py        # Configuration management
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
└── quantized_kosmos/             # Output directory (created after running)
    ├── onnx_model/              # Original ONNX model
    │   ├── model.onnx
    │   ├── config.json
    │   └── ...
    ├── quantized/               # Quantized model
    │   ├── model.onnx
    │   ├── config.json
    │   └── ...
    ├── logs/                    # Quantization logs
    └── benchmarks/              # Performance reports
```

## Technical Details

### Quantization Process

1. **Model Loading**: Load PyTorch model from Hugging Face
2. **ONNX Export**: Convert to ONNX format with optimizations
3. **Quantization Setup**: Configure ORTQuantizer with target approach
4. **Dynamic Quantization**: Apply INT8 quantization to weights
5. **Validation**: Test model functionality and measure performance

### Supported Tasks

Kosmos 2.5 supports various multimodal tasks:

- **Image Captioning**: Generate descriptions of images
- **Visual Question Answering**: Answer questions about images  
- **OCR**: Extract text from images
- **Grounded Captioning**: Generate captions with bounding boxes
- **Object Detection**: Identify and locate objects

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86-64 with AVX2 | Intel Xeon or Core i7/i9 with AVX-512 |
| RAM | 8GB | 16GB+ |
| Storage | 10GB free space | 20GB+ SSD |
| GPU | Not required | Optional for comparison with GPU models |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of the underlying models and libraries.

## Acknowledgments

- **Microsoft**: For the Kosmos 2.5 model
- **Hugging Face**: For the Optimum library and model hosting
- **ONNX Runtime**: For quantization capabilities
- **PyTorch**: For the underlying ML framework

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the Optimum documentation: https://huggingface.co/docs/optimum
3. Check ONNX Runtime quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

---

**Happy Quantizing! 🚀**