#!/usr/bin/env python3
"""
Simple V7P3R Weakness Pattern Analyzer
Lightweight analysis of weakness patterns without engine interaction

This tool analyzes the weakness positions we've already identified to find patterns
in V7P3R's failures without needing to run the engine directly.
"""

import json
import chess
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import os

def load_weakness_data(json_file: str) -> List[Dict]:
    """Load weakness positions from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get("weakness_positions", [])
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def analyze_weakness_patterns(weaknesses: List[Dict]) -> Dict:
    """Analyze patterns in the weakness data"""
    
    # Pattern counters
    position_themes = Counter()
    centipawn_ranges = Counter()
    game_phases = Counter()
    move_types = Counter()
    severity_levels = Counter()
    
    # Tactical pattern analysis
    tactical_misses = 0
    hanging_piece_misses = 0
    material_imbalance_errors = 0
    
    # Move analysis
    bad_moves_by_rank = Counter()
    severe_blunders = []  # >1000cp loss
    
    print(f"🔍 Analyzing {len(weaknesses)} weakness positions...")
    
    for weakness in weaknesses:
        # Extract data
        cp_loss = weakness.get("centipawn_loss", 0)
        themes = weakness.get("position_themes", [])
        game_phase = weakness.get("game_phase", "unknown")
        v7p3r_move = weakness.get("v7p3r_move", "")
        v7p3r_rank = weakness.get("v7p3r_move_rank", 0)
        stockfish_best = weakness.get("stockfish_best_move", "")
        
        # Count themes
        for theme in themes:
            position_themes[theme] += 1
        
        # Categorize by centipawn loss
        if cp_loss >= 1000:
            centipawn_ranges["1000+ (Critical)"] += 1
            severe_blunders.append(weakness)
        elif cp_loss >= 500:
            centipawn_ranges["500-999 (Major)"] += 1
        elif cp_loss >= 200:
            centipawn_ranges["200-499 (Significant)"] += 1
        else:
            centipawn_ranges["<200 (Minor)"] += 1
        
        # Game phase
        game_phases[game_phase] += 1
        
        # Move rank analysis
        if v7p3r_rank == 0:
            bad_moves_by_rank["Not in top 5"] += 1
        else:
            bad_moves_by_rank[f"Rank {v7p3r_rank}"] += 1
        
        # Specific pattern analysis
        if "hanging_piece" in themes:
            hanging_piece_misses += 1
        if "tactics_available" in themes:
            tactical_misses += 1
        if "material_imbalance" in themes:
            material_imbalance_errors += 1
        
        # Analyze move types
        if len(v7p3r_move) == 4:  # Standard UCI format
            from_square = v7p3r_move[:2]
            to_square = v7p3r_move[2:4]
            
            # Detect piece movement patterns (simplified)
            if v7p3r_move[0] in 'abcdefgh' and v7p3r_move[1] in '12345678':
                move_types["Normal move"] += 1
            if len(v7p3r_move) == 5:  # Promotion
                move_types["Promotion"] += 1
    
    return {
        "summary": {
            "total_weaknesses": len(weaknesses),
            "severe_blunders": len(severe_blunders),
            "tactical_misses": tactical_misses,
            "hanging_piece_misses": hanging_piece_misses,
            "material_errors": material_imbalance_errors
        },
        "patterns": {
            "position_themes": dict(position_themes.most_common()),
            "centipawn_ranges": dict(centipawn_ranges),
            "game_phases": dict(game_phases),
            "move_ranks": dict(bad_moves_by_rank),
            "move_types": dict(move_types)
        },
        "severe_blunders": severe_blunders[:10]  # Top 10 worst
    }

def analyze_move_characteristics(weaknesses: List[Dict]) -> Dict:
    """Analyze characteristics of the bad moves V7P3R made"""
    
    move_patterns = {
        "retreating_moves": 0,
        "advancing_moves": 0,
        "lateral_moves": 0,
        "king_moves": 0,
        "piece_drops": 0,
        "passive_moves": 0
    }
    
    positional_errors = {
        "opening_errors": 0,
        "middlegame_tactical": 0,
        "endgame_technique": 0,
        "time_pressure_errors": 0
    }
    
    for weakness in weaknesses:
        move = weakness.get("v7p3r_move", "")
        themes = weakness.get("position_themes", [])
        game_phase = weakness.get("game_phase", "")
        time_pressure = weakness.get("time_pressure", False)
        cp_loss = weakness.get("centipawn_loss", 0)
        
        # Analyze move direction (simplified)
        if len(move) >= 4:
            from_file, from_rank = move[0], move[1]
            to_file, to_rank = move[2], move[3]
            
            # Vertical movement analysis
            if from_rank > to_rank:
                move_patterns["retreating_moves"] += 1
            elif from_rank < to_rank:
                move_patterns["advancing_moves"] += 1
            else:
                move_patterns["lateral_moves"] += 1
        
        # Phase-specific errors
        if game_phase == "opening":
            positional_errors["opening_errors"] += 1
        elif game_phase == "middlegame" and "tactics_available" in themes:
            positional_errors["middlegame_tactical"] += 1
        elif game_phase == "endgame":
            positional_errors["endgame_technique"] += 1
        
        if time_pressure:
            positional_errors["time_pressure_errors"] += 1
    
    return {
        "move_patterns": move_patterns,
        "positional_errors": positional_errors
    }

def generate_improvement_recommendations(analysis: Dict) -> List[str]:
    """Generate specific improvement recommendations based on patterns"""
    recommendations = []
    
    patterns = analysis["patterns"]
    summary = analysis["summary"]
    
    # Tactical recommendations
    if summary["tactical_misses"] > summary["total_weaknesses"] * 0.3:
        recommendations.append("🎯 HIGH PRIORITY: Improve tactical pattern recognition - 30%+ of weaknesses involve missed tactics")
    
    if summary["hanging_piece_misses"] > 10:
        recommendations.append("🔍 CRITICAL: Add hanging piece detection - frequent blind spot")
    
    # Positional recommendations
    if "middlegame" in patterns["game_phases"] and patterns["game_phases"]["middlegame"] > patterns["game_phases"].get("opening", 0):
        recommendations.append("⚡ Focus on middlegame tactical awareness")
    
    # Move ordering recommendations
    not_in_top5 = patterns["move_ranks"].get("Not in top 5", 0)
    if not_in_top5 > summary["total_weaknesses"] * 0.5:
        recommendations.append("📋 URGENT: Improve move ordering - 50%+ of bad moves not even in top 5")
    
    # Severity-based recommendations
    if summary["severe_blunders"] > 5:
        recommendations.append("🚨 CRITICAL: Address severe blunders (1000+ cp) - major calculation errors")
    
    # Theme-specific recommendations
    top_themes = list(patterns["position_themes"].keys())[:3]
    if "hanging_piece" in top_themes:
        recommendations.append("🎯 Implement better piece safety evaluation")
    if "tactics_available" in top_themes:
        recommendations.append("⚡ Increase tactical search depth in complex positions")
    if "material_imbalance" in top_themes:
        recommendations.append("⚖️ Improve material imbalance evaluation")
    
    return recommendations

def print_analysis_report(analysis: Dict, move_analysis: Dict):
    """Print comprehensive analysis report"""
    
    print("\n" + "=" * 80)
    print("🎯 V7P3R WEAKNESS PATTERN ANALYSIS REPORT")
    print("=" * 80)
    
    summary = analysis["summary"]
    patterns = analysis["patterns"]
    
    # Summary statistics
    print(f"📊 SUMMARY STATISTICS")
    print(f"Total weaknesses analyzed: {summary['total_weaknesses']}")
    print(f"Severe blunders (1000+ cp): {summary['severe_blunders']}")
    print(f"Tactical misses: {summary['tactical_misses']}")
    print(f"Hanging piece misses: {summary['hanging_piece_misses']}")
    print(f"Material evaluation errors: {summary['material_errors']}")
    
    # Position themes
    print(f"\n🎯 MOST COMMON WEAKNESS THEMES")
    for theme, count in list(patterns["position_themes"].items())[:8]:
        percentage = (count / summary['total_weaknesses']) * 100
        print(f"  {theme}: {count} ({percentage:.1f}%)")
    
    # Severity breakdown
    print(f"\n📈 SEVERITY BREAKDOWN")
    for severity, count in patterns["centipawn_ranges"].items():
        percentage = (count / summary['total_weaknesses']) * 100
        print(f"  {severity}: {count} ({percentage:.1f}%)")
    
    # Game phase analysis
    print(f"\n🎮 WEAKNESS BY GAME PHASE")
    for phase, count in patterns["game_phases"].items():
        percentage = (count / summary['total_weaknesses']) * 100
        print(f"  {phase}: {count} ({percentage:.1f}%)")
    
    # Move ranking issues
    print(f"\n📋 V7P3R MOVE RANKING ANALYSIS")
    for rank, count in patterns["move_ranks"].items():
        percentage = (count / summary['total_weaknesses']) * 100
        print(f"  {rank}: {count} ({percentage:.1f}%)")
    
    # Move characteristics
    print(f"\n🎯 MOVE PATTERN ANALYSIS")
    move_patterns = move_analysis["move_patterns"]
    for pattern, count in move_patterns.items():
        if count > 0:
            print(f"  {pattern}: {count}")
    
    # Positional errors
    print(f"\n⚠️ POSITIONAL ERROR BREAKDOWN")
    pos_errors = move_analysis["positional_errors"]
    for error_type, count in pos_errors.items():
        if count > 0:
            print(f"  {error_type}: {count}")
    
    # Worst blunders
    print(f"\n🚨 TOP 5 WORST BLUNDERS")
    severe_blunders = sorted(analysis["severe_blunders"], 
                           key=lambda x: x.get("centipawn_loss", 0), 
                           reverse=True)[:5]
    
    for i, blunder in enumerate(severe_blunders, 1):
        print(f"  {i}. {blunder.get('game_id', 'Unknown')} move {blunder.get('move_number', '?')}")
        print(f"     Move: {blunder.get('v7p3r_move', '?')} → Lost {blunder.get('centipawn_loss', 0)}cp")
        print(f"     Best: {blunder.get('stockfish_best_move', '?')}")
        print(f"     Themes: {', '.join(blunder.get('position_themes', [])[:3])}")
    
    # Improvement recommendations
    recommendations = generate_improvement_recommendations(analysis)
    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

def main():
    """Main analysis function"""
    print("🔍 V7P3R Simple Weakness Pattern Analyzer")
    print("Analyzing patterns in identified weaknesses")
    print("=" * 60)
    
    # Load data
    input_file = "v7p3r_parallel_analysis_20251023_171438.json"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    weaknesses = load_weakness_data(input_file)
    
    if not weaknesses:
        print("❌ No weakness data found!")
        return 1
    
    print(f"✅ Loaded {len(weaknesses)} weakness positions")
    
    # Analyze patterns
    analysis = analyze_weakness_patterns(weaknesses)
    move_analysis = analyze_move_characteristics(weaknesses)
    
    # Print report
    print_analysis_report(analysis, move_analysis)
    
    # Save simplified results
    output_data = {
        "analysis_metadata": {
            "input_file": input_file,
            "total_weaknesses": len(weaknesses),
            "analysis_type": "pattern_analysis"
        },
        "weakness_analysis": analysis,
        "move_analysis": move_analysis,
        "recommendations": generate_improvement_recommendations(analysis)
    }
    
    output_file = "v7p3r_weakness_patterns.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Pattern analysis saved to: {output_file}")
    print("✅ Analysis complete!")
    
    return 0

if __name__ == "__main__":
    exit(main())