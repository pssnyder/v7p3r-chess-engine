#!/usr/bin/env python3
"""
V7P3R Enhanced Nudge Database Merger
Merges game-based and puzzle-based nudge databases into a unified,
enhanced nudge system with confidence scoring and tactical metadata.

Handles conflicts, weighting, and creates the final enhanced database
that combines V7P3R's game experience with proven tactical knowledge.

Author: Pat Snyder
Created: September 21, 2025
"""

import json
import os
import hashlib
from typing import Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime


class EnhancedNudgeMerger:
    """
    Merges game-based and puzzle-based nudge databases.
    
    Creates a unified database that combines:
    - Game experience (frequency, evaluation)
    - Tactical knowledge (puzzle perfection)
    - Confidence scoring (hybrid weighting)
    - Tactical classification (offensive/defensive/development)
    """
    
    def __init__(self):
        self.game_nudges = {}
        self.puzzle_nudges = {}
        self.enhanced_nudges = {}
        
        self.merge_stats = {
            'game_positions': 0,
            'puzzle_positions': 0,
            'game_only_positions': 0,
            'puzzle_only_positions': 0,
            'hybrid_positions': 0,
            'total_enhanced_positions': 0,
            'confidence_boosted': 0
        }
        
        print("🔄 Enhanced Nudge Database Merger initialized")
    
    def load_game_nudges(self, game_nudge_file: str) -> bool:
        """Load existing game-based nudge database"""
        try:
            with open(game_nudge_file, 'r') as f:
                self.game_nudges = json.load(f)
            
            self.merge_stats['game_positions'] = len(self.game_nudges)
            print(f"✅ Loaded {len(self.game_nudges)} game-based nudge positions")
            return True
            
        except Exception as e:
            print(f"⚠️  Error loading game nudges: {e}")
            return False
    
    def load_puzzle_nudges(self, puzzle_nudge_file: str) -> bool:
        """Load puzzle-based nudge database"""
        try:
            with open(puzzle_nudge_file, 'r') as f:
                self.puzzle_nudges = json.load(f)
            
            self.merge_stats['puzzle_positions'] = len(self.puzzle_nudges)
            print(f"✅ Loaded {len(self.puzzle_nudges)} puzzle-based nudge positions")
            return True
            
        except Exception as e:
            print(f"⚠️  Error loading puzzle nudges: {e}")
            return False
    
    def calculate_game_confidence(self, frequency: int, eval_score: float) -> float:
        """Calculate confidence score for game-based moves"""
        # Frequency component (0-0.5): more frequent = higher confidence
        freq_component = min(0.5, frequency * 0.1)
        
        # Evaluation component (0-0.5): better eval = higher confidence
        eval_component = min(0.5, max(0, eval_score * 0.1))
        
        return min(1.0, freq_component + eval_component)
    
    def normalize_fen_key(self, fen: str) -> str:
        """Normalize FEN to position key for matching"""
        fen_parts = fen.split()
        simplified_fen = ' '.join(fen_parts[:4])  # Position only, no counters
        return hashlib.md5(simplified_fen.encode()).hexdigest()[:12]
    
    def merge_move_data(self, game_move: Optional[Dict], puzzle_move: Optional[Dict], move_uci: str) -> Dict:
        """Merge game and puzzle data for a single move"""
        
        # Handle puzzle-only moves
        if game_move is None and puzzle_move is not None:
            return {
                'eval': puzzle_move['eval'],
                'frequency': puzzle_move['frequency'],
                'confidence': puzzle_move['confidence'],
                'source': 'puzzle',
                'tactical_info': puzzle_move['tactical_info'],
                'references': puzzle_move['references']
            }
        
        # Handle game-only moves
        if game_move is not None and puzzle_move is None:
            game_confidence = self.calculate_game_confidence(
                game_move['frequency'], 
                game_move['eval']
            )
            
            return {
                'eval': game_move['eval'],
                'frequency': game_move['frequency'],
                'confidence': game_confidence,
                'source': 'game',
                'tactical_info': {
                    'themes': [],
                    'classification': 'development',  # Default for game moves
                    'puzzle_rating': 0,
                    'perfect_sequences': 0
                },
                'references': {
                    'games': game_move.get('games', []),
                    'puzzles': []
                }
            }
        
        # Handle hybrid moves (both game and puzzle data)
        if game_move is not None and puzzle_move is not None:
            game_confidence = self.calculate_game_confidence(
                game_move['frequency'], 
                game_move['eval']
            )
            
            # Hybrid confidence: max of individual confidences with bonus
            hybrid_confidence = min(1.0, max(game_confidence, puzzle_move['confidence']) * 1.1)
            
            # Combine evaluation: weighted average with puzzle bias
            eval_score = (puzzle_move['eval'] * 0.7 + game_move['eval'] * 0.3)
            
            # Combine frequency
            total_frequency = game_move['frequency'] + puzzle_move['frequency']
            
            self.merge_stats['confidence_boosted'] += 1
            
            return {
                'eval': eval_score,
                'frequency': total_frequency,
                'confidence': hybrid_confidence,
                'source': 'hybrid',
                'tactical_info': puzzle_move['tactical_info'],  # Use puzzle tactical info
                'references': {
                    'games': game_move.get('games', []),
                    'puzzles': puzzle_move['references']['puzzles']
                }
            }
        
        # Should not reach here
        return {}
    
    def merge_databases(self) -> Dict:
        """Merge game and puzzle nudge databases"""
        print("🔄 Merging game and puzzle databases...")
        
        # Get all unique position keys
        all_position_keys = set(self.game_nudges.keys()) | set(self.puzzle_nudges.keys())
        
        for position_key in all_position_keys:
            game_position = self.game_nudges.get(position_key)
            puzzle_position = self.puzzle_nudges.get(position_key)
            
            # Determine position source
            if game_position and puzzle_position:
                source_type = 'hybrid'
                self.merge_stats['hybrid_positions'] += 1
                position_fen = puzzle_position['fen']  # Prefer puzzle FEN (more complete)
            elif game_position:
                source_type = 'game'
                self.merge_stats['game_only_positions'] += 1
                position_fen = game_position['fen']
            elif puzzle_position:
                source_type = 'puzzle'
                self.merge_stats['puzzle_only_positions'] += 1
                position_fen = puzzle_position['fen']
            else:
                continue  # Skip if neither exists (shouldn't happen)
            
            # Get all moves for this position
            game_moves = game_position.get('moves', {}) if game_position else {}
            puzzle_moves = puzzle_position.get('moves', {}) if puzzle_position else {}
            all_moves = set(game_moves.keys()) | set(puzzle_moves.keys())
            
            enhanced_moves = {}
            for move_uci in all_moves:
                game_move = game_moves.get(move_uci)
                puzzle_move = puzzle_moves.get(move_uci)
                
                enhanced_move = self.merge_move_data(game_move, puzzle_move, move_uci)
                
                if enhanced_move:
                    enhanced_moves[move_uci] = enhanced_move
            
            # Create enhanced position entry
            if enhanced_moves:
                self.enhanced_nudges[position_key] = {
                    'fen': position_fen,
                    'moves': enhanced_moves,
                    'source_type': source_type
                }
        
        self.merge_stats['total_enhanced_positions'] = len(self.enhanced_nudges)
        
        print(f"✅ Merged into {len(self.enhanced_nudges)} enhanced positions")
        return self.enhanced_nudges
    
    def save_enhanced_database(self, output_file: str):
        """Save enhanced nudge database"""
        with open(output_file, 'w') as f:
            json.dump(self.enhanced_nudges, f, indent=2)
        
        print(f"💾 Enhanced nudge database saved: {output_file}")
    
    def print_merge_statistics(self):
        """Print detailed merge statistics"""
        stats = self.merge_stats
        
        print(f"\\n📊 Enhanced Nudge Database Merge Statistics:")
        print(f"   Input Sources:")
        print(f"     Game positions: {stats['game_positions']}")
        print(f"     Puzzle positions: {stats['puzzle_positions']}")
        
        print(f"   Output Composition:")
        print(f"     Game-only positions: {stats['game_only_positions']}")
        print(f"     Puzzle-only positions: {stats['puzzle_only_positions']}")
        print(f"     Hybrid positions: {stats['hybrid_positions']}")
        print(f"     Total enhanced positions: {stats['total_enhanced_positions']}")
        
        print(f"   Enhancement Benefits:")
        print(f"     Confidence-boosted moves: {stats['confidence_boosted']}")
        
        # Calculate coverage
        if stats['total_enhanced_positions'] > 0:
            game_coverage = (stats['game_only_positions'] + stats['hybrid_positions']) / stats['total_enhanced_positions'] * 100
            puzzle_coverage = (stats['puzzle_only_positions'] + stats['hybrid_positions']) / stats['total_enhanced_positions'] * 100
            hybrid_rate = stats['hybrid_positions'] / stats['total_enhanced_positions'] * 100
            
            print(f"   Coverage Analysis:")
            print(f"     Game coverage: {game_coverage:.1f}%")
            print(f"     Puzzle coverage: {puzzle_coverage:.1f}%")
            print(f"     Hybrid rate: {hybrid_rate:.1f}%")
    
    def analyze_enhanced_database(self):
        """Analyze the enhanced database composition"""
        if not self.enhanced_nudges:
            return
        
        total_moves = sum(len(pos['moves']) for pos in self.enhanced_nudges.values())
        
        source_counts = defaultdict(int)
        confidence_distribution = defaultdict(int)
        tactical_classifications = defaultdict(int)
        
        for position in self.enhanced_nudges.values():
            for move_data in position['moves'].values():
                source_counts[move_data['source']] += 1
                
                # Confidence buckets
                confidence = move_data['confidence']
                if confidence >= 0.9:
                    confidence_distribution['0.9-1.0'] += 1
                elif confidence >= 0.8:
                    confidence_distribution['0.8-0.9'] += 1
                elif confidence >= 0.7:
                    confidence_distribution['0.7-0.8'] += 1
                else:
                    confidence_distribution['<0.7'] += 1
                
                # Tactical classifications
                classification = move_data['tactical_info']['classification']
                tactical_classifications[classification] += 1
        
        print(f"\\n🔍 Enhanced Database Analysis:")
        print(f"   Total moves: {total_moves}")
        print(f"   Move sources:")
        for source, count in sorted(source_counts.items()):
            percentage = count / total_moves * 100
            print(f"     {source}: {count} ({percentage:.1f}%)")
        
        print(f"   Confidence distribution:")
        for bucket, count in sorted(confidence_distribution.items()):
            percentage = count / total_moves * 100
            print(f"     {bucket}: {count} ({percentage:.1f}%)")
        
        print(f"   Tactical classifications:")
        for classification, count in sorted(tactical_classifications.items()):
            percentage = count / total_moves * 100
            print(f"     {classification}: {count} ({percentage:.1f}%)")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V7P3R Enhanced Nudge Database Merger')
    parser.add_argument('--game-nudges', required=True, 
                       help='Game-based nudge database JSON file')
    parser.add_argument('--puzzle-nudges', required=True,
                       help='Puzzle-based nudge database JSON file')
    parser.add_argument('--output', default='v7p3r_enhanced_nudges.json',
                       help='Output file for enhanced nudge database')
    
    args = parser.parse_args()
    
    print("🔄 V7P3R Enhanced Nudge Database Merger")
    print("=" * 45)
    
    start_time = datetime.now()
    
    # Create merger
    merger = EnhancedNudgeMerger()
    
    # Load databases
    game_loaded = merger.load_game_nudges(args.game_nudges)
    puzzle_loaded = merger.load_puzzle_nudges(args.puzzle_nudges)
    
    if not game_loaded and not puzzle_loaded:
        print("❌ Failed to load any nudge databases")
        return
    
    if not game_loaded:
        print("⚠️  No game nudges loaded, using puzzle data only")
    
    if not puzzle_loaded:
        print("⚠️  No puzzle nudges loaded, using game data only")
    
    # Merge databases
    enhanced_db = merger.merge_databases()
    
    if enhanced_db:
        # Save enhanced database
        merger.save_enhanced_database(args.output)
        
        # Print statistics
        merger.print_merge_statistics()
        merger.analyze_enhanced_database()
        
        elapsed = datetime.now() - start_time
        print(f"\\n🎉 Merge complete in {elapsed.total_seconds():.1f} seconds!")
        print(f"📁 Enhanced nudge database: {args.output}")
    else:
        print("❌ Failed to create enhanced database")


if __name__ == "__main__":
    main()