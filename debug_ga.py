#!/usr/bin/env python3
"""
Debug script to test GA components individually
"""
import yaml
import sys
import os
sys.path.append(os.path.abspath('.'))

def test_stockfish_connection():
    """Test if Stockfish is accessible"""
    print("=== Testing Stockfish Connection ===")
    try:
        with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        stockfish_path = config["stockfish_config"]["stockfish_path"]
        print(f"Stockfish path: {stockfish_path}")
        
        # Check if file exists
        if os.path.exists(stockfish_path):
            print("✅ Stockfish executable found")
        else:
            print("❌ Stockfish executable NOT found")
            return False
            
        # Try to run stockfish
        import subprocess
        try:
            result = subprocess.run([stockfish_path], input="quit\n", 
                                  capture_output=True, text=True, timeout=5)
            if "Stockfish" in result.stdout:
                print("✅ Stockfish runs successfully")
                return True
            else:
                print("❌ Stockfish doesn't respond correctly")
                print(f"Output: {result.stdout}")
                return False
        except Exception as e:
            print(f"❌ Error running Stockfish: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Stockfish: {e}")
        return False

def test_basic_imports():
    """Test if all required modules can be imported"""
    print("\n=== Testing Basic Imports ===")
    try:
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        print("✅ PositionEvaluator imported")
        
        from v7p3r_ga_engine.ruleset_manager import RulesetManager
        print("✅ RulesetManager imported")
        
        from v7p3r_ga_engine.v7p3r_ga import _evaluate_individual_worker
        print("✅ GA evaluation function imported")
        
        from v7p3r_engine.v7p3r_score import v7p3rScore
        print("✅ v7p3rScore imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_evaluator():
    """Test position evaluator components"""
    print("\n=== Testing Position Evaluator ===")
    try:
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        from v7p3r_ga_engine.ruleset_manager import RulesetManager
        
        with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        position_evaluator = PositionEvaluator(
            stockfish_config=config["stockfish_config"],
            use_cuda=config.get("use_cuda", False),
            use_nn_evaluator=config.get("use_neural_evaluator", False),
            nn_model_path=config.get("neural_model_path")
        )
        
        print("✅ Position evaluator created successfully")
        
        # Test loading positions
        positions = position_evaluator.load_positions("random", 3)
        print(f"✅ Loaded {len(positions)} test positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing position evaluator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_evaluation():
    """Test a single individual evaluation"""
    print("\n=== Testing Single Individual Evaluation ===")
    try:
        from v7p3r_ga_engine.v7p3r_ga import _evaluate_individual_worker
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        from v7p3r_ga_engine.ruleset_manager import RulesetManager
        
        with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        ruleset_manager = RulesetManager()
        position_evaluator = PositionEvaluator(
            stockfish_config=config["stockfish_config"],
            use_cuda=config.get("use_cuda", False),
            use_nn_evaluator=config.get("use_neural_evaluator", False),
            nn_model_path=config.get("neural_model_path")
        )
        
        base_ruleset = ruleset_manager.load_ruleset("default_evaluation")
        positions = position_evaluator.load_positions("random", 2)
        
        print("Testing individual evaluation function...")
        args = (0, base_ruleset, positions, position_evaluator)
        
        result = _evaluate_individual_worker(args)
        i, fitness, ruleset = result
        
        print(f"✅ Individual evaluation successful!")
        print(f"   Index: {i}")
        print(f"   Fitness: {fitness}")
        print(f"   Ruleset type: {type(ruleset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing individual evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GA Debug Tests...\n")
    
    # Run tests
    imports_ok = test_basic_imports()
    stockfish_ok = test_stockfish_connection()
    evaluator_ok = test_position_evaluator()
    individual_ok = test_single_evaluation()
    
    print(f"\n=== Debug Summary ===")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Stockfish: {'✅' if stockfish_ok else '❌'}")
    print(f"Position Evaluator: {'✅' if evaluator_ok else '❌'}")
    print(f"Individual Evaluation: {'✅' if individual_ok else '❌'}")
    
    if all([imports_ok, stockfish_ok, evaluator_ok, individual_ok]):
        print("\n🎉 All tests passed! GA should work properly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
