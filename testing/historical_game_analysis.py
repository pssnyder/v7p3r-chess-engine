#!/usr/bin/env python3

"""
V7P3R Historical Game Analysis

Test V7P3R's historical performance to identify improvement areas.
This will analyze past games to find patterns of poor moves and suggest improvements.
"""

import sys
import os

# Add the engine tester path for game replay analyzer
engine_tester_path = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester"
sys.path.append(os.path.join(engine_tester_path, 'engine_utilities'))

from game_replay_analyzer import V7P3RGameReplayAnalyzer

def analyze_v7p3r_historical_games():
    """Analyze V7P3R's historical game performance"""
    
    print("V7P3R Historical Game Analysis")
    print("=" * 50)
    
    # Game records directory
    game_records_dir = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records"
    
    print(f"Game records directory: {game_records_dir}")
    
    if not os.path.exists(game_records_dir):
        print(f"❌ Game records directory not found: {game_records_dir}")
        return
    
    try:
        # Initialize the analyzer
        analyzer = V7P3RGameReplayAnalyzer()
        print("✅ Game replay analyzer initialized")
        
        # Analyze games looking for V7P3R as player
        # V7P3R shows up as "V7P3R" in game headers
        results = analyzer.process_game_directory(
            directory=game_records_dir,
            player_name="V7P3R",
            max_games=20,  # Start with 20 games for quick analysis
            analyze_move_ordering=True
        )
        
        if not results or not results.get('weakness_positions'):
            print("❌ No games found with V7P3R as player")
            print("Let me check what player names are available...")
            
            # Let's check a few game directories to see what player names exist
            subdirs = [d for d in os.listdir(game_records_dir) 
                      if os.path.isdir(os.path.join(game_records_dir, d))]
            
            print(f"Available game directories: {subdirs[:5]}...")  # Show first 5
            
            if subdirs:
                # Try to analyze one directory to see player names
                test_dir = os.path.join(game_records_dir, subdirs[0])
                pgn_files = analyzer.find_pgn_files(test_dir)
                
                if pgn_files:
                    print(f"Checking sample PGN file: {pgn_files[0]}")
                    # Read first game to see player names
                    import chess.pgn
                    with open(pgn_files[0], 'r') as f:
                        game = chess.pgn.read_game(f)
                        if game:
                            white = game.headers.get('White', 'Unknown')
                            black = game.headers.get('Black', 'Unknown')
                            print(f"Sample game players: White='{white}', Black='{black}'")
                            
                            # Try with the actual player names found
                            if 'v7p3r' in white.lower() or 'v7p3r' in black.lower():
                                player_name = white if 'v7p3r' in white.lower() else black
                                print(f"Retrying analysis with player name: '{player_name}'")
                                results = analyzer.process_game_directory(
                                    directory=test_dir,
                                    player_name=player_name,
                                    max_games=5
                                )
            return
        
        # Generate comprehensive report
        report = analyzer.generate_weakness_report(results)
        
        # Print the analysis
        analyzer.print_analysis_report(results, report)
        
        # Save results
        timestamp = os.path.basename(__file__).replace('.py', '')
        results_file = f"v7p3r_historical_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            import json
            json.dump({
                'results': results,
                'report': report,
                'metadata': {
                    'analysis_timestamp': str(datetime.now()),
                    'games_directory': game_records_dir,
                    'total_games_processed': results['summary']['games_processed']
                }
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Key insights summary
        summary = results['summary']
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Analyzed {summary['games_processed']} games")
        print(f"   • Found {summary['weaknesses_found']} weak positions")
        print(f"   • Weakness rate: {summary['weakness_rate']:.1f}% of moves")
        
        if report.get('recommendations'):
            print(f"\n📈 TOP IMPROVEMENT AREAS:")
            for i, rec in enumerate(report['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    analyze_v7p3r_historical_games()