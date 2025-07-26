# Quick test for RL training system
import yaml
import sys
import torch
import os

def test_rl_system():
    print("🧪 Testing RL Training System...")
    
    # Test 1: Check PyTorch
    try:
        print(f"   🔥 PyTorch version: {torch.__version__}")
        print(f"   🎯 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   📱 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   💻 Using CPU for training")
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False
    
    # Test 2: Check config
    try:
        with open("config.yaml", encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
        print(f"   📋 Config loaded: {len(config)} sections")
        if 'reinforcement_learning' in config:
            print("   ✅ RL config section found")
        else:
            print("   ⚠️ RL config section missing")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False
    
    # Test 3: Check training data
    try:
        training_dir = "training_positions"
        if os.path.exists(training_dir):
            pgn_files = [f for f in os.listdir(training_dir) if f.endswith('.pgn')]
            print(f"   📚 Found {len(pgn_files)} PGN files")
        else:
            print(f"   ❌ Training directory not found: {training_dir}")
            return False
    except Exception as e:
        print(f"   ❌ Training data error: {e}")
        return False
    
    # Test 4: Try to import our RL module
    try:
        from reinforcement_training import ReinforcementTrainer
        print("   ✅ RL Training module imported successfully")
    except Exception as e:
        print(f"   ❌ RL import error: {e}")
        return False
    
    # Test 5: Try to create trainer instance
    try:
        trainer = ReinforcementTrainer()
        print("   ✅ RL Trainer instance created successfully")
        print(f"   🧠 Device: {trainer.device}")
        print(f"   📊 Actor network parameters: {sum(p.numel() for p in trainer.actor.parameters()):,}")
        print(f"   📊 Critic network parameters: {sum(p.numel() for p in trainer.critic.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Trainer creation error: {e}")
        return False
    
    print("\n✅ All RL system tests passed!")
    return True

if __name__ == "__main__":
    success = test_rl_system()
    sys.exit(0 if success else 1)
