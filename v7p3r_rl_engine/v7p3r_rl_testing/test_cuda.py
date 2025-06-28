#!/usr/bin/env python3
"""
CUDA Configuration Test Script
Tests if CUDA is properly installed and configured for the v7p3r chess engine.
"""

import sys
import os

def test_cuda_availability():
    """Test if CUDA is available and properly configured."""
    print("=" * 60)
    print("CUDA Configuration Test for v7p3r Chess Engine")
    print("=" * 60)
    
    # Test 1: Check if PyTorch is installed
    print("\n1. Testing PyTorch installation...")
    try:
        import torch
        print(f"   ✓ PyTorch installed: {torch.__version__}")
    except ImportError as e:
        print(f"   ✗ PyTorch not found: {e}")
        print("   Install with: pip install torch")
        return False
    
    # Test 2: Check CUDA availability in PyTorch
    print("\n2. Testing CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("   ✓ CUDA is available")
    else:
        print("   ✗ CUDA is not available")
        print("   Possible issues:")
        print("     - NVIDIA GPU not present")
        print("     - CUDA toolkit not installed")
        print("     - PyTorch not compiled with CUDA support")
        print("     - GPU drivers not properly installed")
    
    # Test 3: Check CUDA device count
    if cuda_available:
        print("\n3. Testing CUDA devices...")
        device_count = torch.cuda.device_count()
        print(f"   ✓ CUDA devices found: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_props = torch.cuda.get_device_properties(i)
            memory_gb = device_props.total_memory / (1024**3)
            compute_capability = f"{device_props.major}.{device_props.minor}"
            
            print(f"   Device {i}: {device_name}")
            print(f"     Memory: {memory_gb:.1f} GB")
            print(f"     Compute Capability: {compute_capability}")
            print(f"     Multiprocessors: {device_props.multi_processor_count}")
    
    # Test 4: Test basic CUDA operations
    if cuda_available:
        print("\n4. Testing basic CUDA operations...")
        try:
            # Create a tensor on CPU
            cpu_tensor = torch.randn(100, 100)
            print("   ✓ Created CPU tensor")
            
            # Move to GPU
            gpu_tensor = cpu_tensor.cuda()
            print("   ✓ Moved tensor to GPU")
            
            # Perform GPU computation
            result = torch.matmul(gpu_tensor, gpu_tensor)
            print("   ✓ Performed GPU matrix multiplication")
            
            # Move back to CPU
            cpu_result = result.cpu()
            print("   ✓ Moved result back to CPU")
            
            print("   ✓ All basic CUDA operations successful")
        except Exception as e:
            print(f"   ✗ CUDA operation failed: {e}")
    
    # Test 5: Check CUDA version compatibility
    if cuda_available:
        print("\n5. Testing CUDA version...")
        
        # Get PyTorch version
        pytorch_version = torch.__version__
        print(f"   PyTorch version: {pytorch_version}")
        
        # Try to get system CUDA version
        try:
            import subprocess
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        print(f"   System CUDA version: {line.strip()}")
                        break
            else:
                print("   Could not detect system CUDA version (nvcc not in PATH)")
        except FileNotFoundError:
            print("   Could not detect system CUDA version (nvcc not found)")
    
    # Test 6: Test memory allocation
    if cuda_available:
        print("\n6. Testing GPU memory allocation...")
        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            print(f"   Current allocated memory: {allocated:.1f} MB")
            print(f"   Current cached memory: {cached:.1f} MB")
            
            # Test larger allocation
            test_tensor = torch.randn(1000, 1000, device='cuda')
            new_allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"   After test allocation: {new_allocated:.1f} MB")
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            print("   ✓ Memory allocation test successful")
        except Exception as e:
            print(f"   ✗ Memory allocation failed: {e}")
    
    return cuda_available

def test_ga_cuda_integration():
    """Test CUDA integration with the GA engine components."""
    print("\n" + "=" * 60)
    print("Testing GA Engine CUDA Integration")
    print("=" * 60)
    
    # Test if CUDA accelerator is available
    print("\n1. Testing CUDA accelerator import...")
    try:
        from v7p3r_ga_engine.cuda_accelerator import CUDAAccelerator
        print("   ✓ CUDAAccelerator imported successfully")
        
        accelerator = CUDAAccelerator()
        print(f"   Device: {accelerator.device}")
        print(f"   Using CUDA: {accelerator.use_cuda}")
        
        if accelerator.use_cuda:
            print("   ✓ CUDA accelerator configured correctly")
        else:
            print("   ! CUDA accelerator using CPU fallback")
            
    except ImportError as e:
        print(f"   ✗ Could not import CUDAAccelerator: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error initializing CUDAAccelerator: {e}")
        return False
    
    # Test position evaluator with CUDA
    print("\n2. Testing position evaluator CUDA integration...")
    try:
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        
        # Create a test config
        config = {
            'use_cuda': True,
            'cuda_batch_size': 32,
            'enable_cache': True
        }
        
        evaluator = PositionEvaluator(config)
        print("   ✓ PositionEvaluator created with CUDA config")
        
        if hasattr(evaluator, 'cuda_accelerator') and evaluator.cuda_accelerator.use_cuda:
            print("   ✓ PositionEvaluator using CUDA acceleration")
        else:
            print("   ! PositionEvaluator using CPU fallback")
            
    except Exception as e:
        print(f"   ✗ Error testing PositionEvaluator: {e}")
        return False
    
    return True

def print_recommendations(cuda_available):
    """Print recommendations based on test results."""
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    if cuda_available:
        print("\n✓ CUDA is properly configured!")
        print("\nOptimal GA configuration for your system:")
        print("  use_cuda: true")
        print("  cuda_batch_size: 64  # Start with this, adjust based on GPU memory")
        print("  use_multiprocessing: true  # Enable for faster training")
        print("  max_workers: null  # Auto-detect optimal worker count")
        
        print("\nTo maximize performance:")
        print("  1. Monitor GPU memory usage during training")
        print("  2. Increase cuda_batch_size if you have extra GPU memory")
        print("  3. Increase population_size for better evolution quality")
        print("  4. Use larger positions_count for more accurate fitness evaluation")
        
    else:
        print("\n! CUDA is not available - using CPU fallback")
        print("\nRecommended GA configuration for CPU-only:")
        print("  use_cuda: false")
        print("  use_multiprocessing: true")
        print("  max_workers: 4  # Adjust based on your CPU cores")
        print("  population_size: 10  # Smaller population for faster CPU training")
        print("  positions_count: 30  # Fewer positions for faster evaluation")
        
        print("\nTo enable CUDA:")
        print("  1. Install NVIDIA GPU drivers")
        print("  2. Install CUDA Toolkit from NVIDIA")
        print("  3. Install PyTorch with CUDA support:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    print("Starting CUDA configuration test...\n")
    
    cuda_available = test_cuda_availability()
    ga_integration_ok = test_ga_cuda_integration()
    
    print_recommendations(cuda_available)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"CUDA Available: {'✓' if cuda_available else '✗'}")
    print(f"GA Integration: {'✓' if ga_integration_ok else '✗'}")
    
    if cuda_available and ga_integration_ok:
        print("\n🎉 Your system is ready for CUDA-accelerated GA training!")
    elif cuda_available:
        print("\n⚠️  CUDA is available but GA integration needs attention")
    else:
        print("\n💡 CUDA not available - CPU training will still work")
