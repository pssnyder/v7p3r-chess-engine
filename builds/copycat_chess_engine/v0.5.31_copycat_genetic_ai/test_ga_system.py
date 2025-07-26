# Test Genetic Algorithm with RL guidance
print("🧬 Testing Genetic Algorithm with RL guidance...")

try:
    from genetic_algorithm import GeneticMoveSelector
    import chess
    
    # Create a test position
    board = chess.Board()
    board.push_san("e4")  # 1. e4
    board.push_san("e5")  # 1... e5
    
    print(f"🏁 Test position: {board.fen()}")
    print(f"📝 Legal moves: {len(list(board.legal_moves))}")
    
    # Initialize genetic selector
    print("\n🧬 Initializing Genetic Algorithm...")
    selector = GeneticMoveSelector("config_test.yaml")
    
    # Select best move
    print("\n🎯 Running genetic algorithm move selection...")
    best_move, stats = selector.select_best_move(board, time_limit=5.0)
    
    print(f"\n✅ GA Move Selection Complete!")
    print(f"   🎯 Best move: {best_move}")
    print(f"   🔥 Fitness: {stats.get('best_fitness', 'N/A'):.3f}")
    print(f"   🧠 RL Score: {stats.get('rl_score', 'N/A'):.3f}")
    print(f"   📊 Eval Score: {stats.get('eval_score', 'N/A'):.3f}")
    print(f"   🕒 Time: {stats.get('selection_time', 'N/A'):.3f}s")
    print(f"   🔄 Generations: {stats.get('generations', 'N/A')}")
    
    # Get performance stats
    perf_stats = selector.get_performance_stats()
    print(f"\n📈 Performance: {perf_stats}")
    
    print("\n🎯 Two-Brain System Test Complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
