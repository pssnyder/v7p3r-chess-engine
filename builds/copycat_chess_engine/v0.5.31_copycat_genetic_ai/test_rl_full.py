# Full RL training pipeline test
print("🚀 Starting Complete RL Training Pipeline Test...")

try:
    from reinforcement_training import ReinforcementTrainer
    
    # Create trainer with test config (smaller parameters)
    trainer = ReinforcementTrainer('config_test.yaml')
    print(f"✅ Trainer initialized on {trainer.device}")
    print(f"   Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}")
    print(f"   Critic params: {sum(p.numel() for p in trainer.critic.parameters()):,}")
    
    # Start actual training (limited scope for testing)
    print("\n🎯 Starting mini training run...")
    trainer.train()
    
    # Check if models were saved
    import os
    if os.path.exists("rl_actor_model.pth"):
        print("✅ Actor model saved successfully")
    if os.path.exists("rl_critic_model.pth"):
        print("✅ Critic model saved successfully")
    if os.path.exists("training_stats.json"):
        print("✅ Training statistics saved successfully")
    
    print("\n🎯 Complete RL training pipeline test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
