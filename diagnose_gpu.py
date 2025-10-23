#!/usr/bin/env python3
"""
GPU diagnostic script to identify and resolve CUDA issues
"""

import os
import sys
import torch
import subprocess

def check_cuda_environment():
    """Check CUDA environment variables and settings"""
    print("🔍 CUDA Environment Check")
    print("=" * 50)
    
    # Check environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')
    print(f"CUDA_LAUNCH_BLOCKING: {cuda_launch_blocking}")
    
    pytorch_cuda_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
    print(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_cuda_alloc}")

def check_pytorch_cuda():
    """Check PyTorch CUDA availability and devices"""
    print("\n🔍 PyTorch CUDA Check")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1024**3:.1f} GB")
                
                # Test tensor creation
                test_tensor = torch.randn(100, 100).cuda(i)
                print("    ✅ Tensor creation test: PASSED")
                del test_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ Device {i} error: {e}")
    else:
        print("❌ CUDA not available in PyTorch")

def check_gpu_processes():
    """Check what processes are using the GPU"""
    print("\n🔍 GPU Process Check")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Current GPU status:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ nvidia-smi not found")

def test_gpu_operations():
    """Test basic GPU operations"""
    print("\n🔍 GPU Operations Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU tests")
        return
    
    try:
        device = torch.device('cuda:0')
        print(f"Testing device: {device}")
        
        # Test 1: Basic tensor operations
        print("Test 1: Basic tensor creation...")
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)
        print("✅ Tensor creation: PASSED")
        
        # Test 2: Matrix multiplication
        print("Test 2: Matrix multiplication...")
        c = torch.mm(a, b)
        print("✅ Matrix multiplication: PASSED")
        
        # Test 3: Memory operations
        print("Test 3: Memory operations...")
        del a, b, c
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Memory operations: PASSED")
        
        print("🎉 All GPU tests PASSED!")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("This indicates the GPU is busy or has issues")

def suggest_fixes():
    """Suggest potential fixes for GPU issues"""
    print("\n🔧 Suggested Fixes")
    print("=" * 50)
    
    print("1. Check if other processes are using the GPU:")
    print("   nvidia-smi")
    print("   # Kill processes if they're yours: sudo kill -9 <PID>")
    print()
    
    print("2. Reset GPU state:")
    print("   sudo nvidia-smi --gpu-reset")
    print("   # Or restart the system")
    print()
    
    print("3. Check CUDA installation:")
    print("   nvcc --version")
    print("   # Reinstall CUDA if needed")
    print()
    
    print("4. Try different GPU (if available):")
    print("   export CUDA_VISIBLE_DEVICES=4  # Instead of 5")
    print()
    
    print("5. Use CPU fallback:")
    print("   # The script should automatically fall back to CPU")

def main():
    print("🚀 GPU Diagnostic Tool")
    print("=" * 70)
    
    check_cuda_environment()
    check_pytorch_cuda()
    check_gpu_processes()
    test_gpu_operations()
    suggest_fixes()

if __name__ == "__main__":
    main()
