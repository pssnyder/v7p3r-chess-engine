# Quick RL training test
print("🚀 Starting Quick RL Training Test...")

try:
    from reinforcement_training import ReinforcementTrainer
    
    # Create trainer with test config
    trainer = ReinforcementTrainer('config_test.yaml')
    print(f"✅ Trainer initialized on {trainer.device}")
    
    # Test evaluation alignment
    from pathlib import Path
    training_dir = Path("training_positions")
    pgn_files = list(training_dir.glob("*.pgn"))[:2]  # Just first 2 files
    
    print(f"📚 Testing with {len(pgn_files)} PGN files...")
    
    # Test validation
    validation_results = trainer.validate_evaluation_alignment(pgn_files)
    print(f"🔍 Validation complete: {len(validation_results)} files tested")
    
    for filename, results in validation_results.items():
        print(f"   📄 {filename}")
        print(f"      Winner avg: {results['winner_avg_score']:.3f}")
        print(f"      Loser avg: {results['loser_avg_score']:.3f}")
        print(f"      Alignment: {'✅' if results['alignment_good'] else '❌'}")
    
    print("🎯 Quick RL test complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
