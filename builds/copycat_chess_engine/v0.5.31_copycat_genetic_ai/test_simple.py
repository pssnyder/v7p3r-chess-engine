# Simple test script
print("🧪 Testing RL components step by step...")

# Test 1: Basic imports
try:
    import sys
    print(f"✅ Python: {sys.version}")
except Exception as e:
    print(f"❌ Python error: {e}")

# Test 2: PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

# Test 3: Other packages
for pkg in ['numpy', 'chess', 'yaml']:
    try:
        __import__(pkg)
        print(f"✅ {pkg}: imported")
    except Exception as e:
        print(f"❌ {pkg}: {e}")

# Test 4: Our config
try:
    import yaml
    with open("config_test.yaml", encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config: loaded {len(config) if config else 0} sections")
except Exception as e:
    print(f"❌ Config error: {e}")

print("🎯 Component test complete!")
