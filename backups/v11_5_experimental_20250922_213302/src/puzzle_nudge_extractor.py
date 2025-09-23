#!/usr/bin/env python3
"""
V7P3R Puzzle Nudge Extractor
Extracts tactical positions and moves from V7P3R's perfect puzzle sequences
to enhance the nudge database with proven tactical knowledge.

This tool scans puzzle analysis results, identifies perfect sequences,
and extracts each position where V7P3R played the correct move.
These become high-confidence nudge entries with tactical metadata.

Author: Pat Snyder
Created: September 21, 2025
"""

import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import chess
from datetime import datetime


@dataclass
class PuzzleNudgeEntry:
    """Represents a nudge entry derived from puzzle analysis"""
    fen: str
    move: str
    puzzle_id: str
    puzzle_rating: int
    themes: List[str]
    confidence: float
    tactical_classification: str
    stockfish_score: int
    stockfish_rank: int


class PuzzleNudgeExtractor:
    """
    Extracts tactical nudge data from V7P3R puzzle analysis results.
    
    Focuses on perfect sequences where V7P3R demonstrated mastery
    of tactical patterns. Each correct position-move pair becomes
    a high-confidence nudge entry with tactical metadata.
    """
    
    def __init__(self):
        self.puzzle_nudges = defaultdict(lambda: defaultdict(list))  # fen -> move -> [entries]
        self.processed_files = set()
        self.stats = {
            'files_processed': 0,
            'perfect_sequences_found': 0,
            'positions_extracted': 0,
            'unique_positions': 0
        }
        
        # Tactical classification mappings
        self.offensive_themes = {
            'crushing', 'sacrifice', 'attack', 'mate', 'mateIn1', 'mateIn2', 'mateIn3',
            'discoveredAttack', 'deflection', 'decoy', 'attraction', 'clearance',
            'doubleCheck', 'exposedKing', 'fork', 'hangingPiece', 'skewer', 'pin',
            'smotheredMate', 'backRankMate', 'xRayAttack', 'zugzwang'
        }
        
        self.defensive_themes = {
            'defense', 'escape', 'interference', 'blocking', 'counterAttack',
            'desperado', 'trappedPiece'
        }
        
        print("🧩 Puzzle Nudge Extractor initialized")
        print(f"   Offensive themes: {len(self.offensive_themes)}")
        print(f"   Defensive themes: {len(self.defensive_themes)}")
    
    def classify_move_tactically(self, themes: List[str]) -> str:
        """Classify move as offensive, defensive, or development"""
        theme_set = set(themes)
        
        # Check for offensive themes
        if theme_set & self.offensive_themes:
            return "offensive"
        
        # Check for defensive themes
        if theme_set & self.defensive_themes:
            return "defensive"
        
        # Default to development for opening/endgame/positional
        return "development"
    
    def create_position_key(self, fen: str) -> str:
        """Create position key (simplified FEN without move counters)"""
        fen_parts = fen.split()
        simplified_fen = ' '.join(fen_parts[:4])  # Position only, no counters
        return hashlib.md5(simplified_fen.encode()).hexdigest()[:12]
    
    def extract_from_puzzle_result(self, puzzle_data: Dict) -> int:
        """Extract nudge entries from a single puzzle result"""
        
        # Only process perfect sequences
        if not puzzle_data.get('perfect_sequence', False):
            return 0
        
        puzzle_id = puzzle_data.get('puzzle_id', 'unknown')
        puzzle_rating = puzzle_data.get('rating', 1000)
        themes = puzzle_data.get('themes', [])
        
        # Classify the overall tactical nature
        tactical_classification = self.classify_move_tactically(themes)
        
        # Extract from position analyses
        position_analyses = puzzle_data.get('position_analyses', [])
        extracted_count = 0
        
        for analysis in position_analyses:
            if not analysis.get('engine_found_solution', False):
                continue
            
            fen = analysis.get('challenge_fen')
            engine_move = analysis.get('engine_move')
            stockfish_score = analysis.get('engine_stockfish_score', 0)
            stockfish_rank = analysis.get('engine_stockfish_rank', 1)
            
            if not fen or not engine_move:
                continue
            
            # Create nudge entry
            entry = PuzzleNudgeEntry(
                fen=fen,
                move=engine_move,
                puzzle_id=puzzle_id,
                puzzle_rating=puzzle_rating,
                themes=themes,
                confidence=1.0,  # Perfect sequences get maximum confidence
                tactical_classification=tactical_classification,
                stockfish_score=stockfish_score,
                stockfish_rank=stockfish_rank
            )
            
            # Store in database structure
            position_key = self.create_position_key(fen)
            self.puzzle_nudges[fen][engine_move].append(entry)
            extracted_count += 1
        
        return extracted_count
    
    def scan_puzzle_analysis_files(self, analysis_directories: List[str]) -> Dict:
        """Scan puzzle analysis files and extract tactical nudges"""
        print(f"📂 Scanning puzzle analysis files...")
        
        total_files = 0
        perfect_sequences = 0
        total_positions = 0
        
        for directory in analysis_directories:
            if not os.path.exists(directory):
                print(f"⚠️  Directory not found: {directory}")
                continue
            
            print(f"   Scanning: {directory}")
            
            # Walk through analysis files
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if ('enhanced_sequence_analysis' in file or 'puzzle_analysis' in file) and file.endswith('.json'):
                        if 'v7p3r' not in file.lower() and 'V7P3R' not in file:
                            continue  # Skip non-V7P3R files
                        
                        file_path = os.path.join(root, file)
                        
                        if file_path in self.processed_files:
                            continue
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            total_files += 1
                            
                            # Process analysis results
                            analysis_results = data.get('analysis_results', [])
                            for puzzle_result in analysis_results:
                                if puzzle_result.get('perfect_sequence', False):
                                    perfect_sequences += 1
                                    positions_extracted = self.extract_from_puzzle_result(puzzle_result)
                                    total_positions += positions_extracted
                            
                            self.processed_files.add(file_path)
                            
                            # Progress update
                            if total_files % 5 == 0:
                                print(f"   📊 {total_files} files, {perfect_sequences} perfect sequences, {total_positions} positions", end='\\r')
                        
                        except Exception as e:
                            print(f"⚠️  Error reading {file}: {e}")
        
        print(f"\\n✅ Scan complete!")
        
        # Update stats
        self.stats.update({
            'files_processed': total_files,
            'perfect_sequences_found': perfect_sequences,
            'positions_extracted': total_positions,
            'unique_positions': len(self.puzzle_nudges)
        })
        
        return self.stats
    
    def build_puzzle_nudge_database(self, min_confidence: float = 0.8) -> Dict:
        """Build final puzzle nudge database"""
        print(f"🏗️  Building puzzle nudge database...")
        print(f"   Min confidence: {min_confidence}")
        
        nudge_db = {}
        
        for position_fen, moves_data in self.puzzle_nudges.items():
            position_moves = {}
            
            for move, entries in moves_data.items():
                if not entries:
                    continue
                
                # Aggregate data from multiple entries
                total_confidence = sum(e.confidence for e in entries)
                avg_confidence = total_confidence / len(entries)
                
                if avg_confidence < min_confidence:
                    continue
                
                # Aggregate themes and ratings
                all_themes = set()
                ratings = []
                puzzle_refs = []
                
                for entry in entries:
                    all_themes.update(entry.themes)
                    ratings.append(entry.puzzle_rating)
                    puzzle_refs.append(entry.puzzle_id)
                
                # Determine classification (most common)
                classifications = [e.tactical_classification for e in entries]
                tactical_classification = max(set(classifications), key=classifications.count)
                
                # Calculate tactical score (higher for better Stockfish ranks)
                stockfish_ranks = [e.stockfish_rank for e in entries]
                avg_rank = sum(stockfish_ranks) / len(stockfish_ranks)
                tactical_score = max(1.0, 6.0 - avg_rank)  # Score 1-5 based on rank
                
                position_moves[move] = {
                    'eval': tactical_score,
                    'frequency': len(entries),
                    'confidence': avg_confidence,
                    'source': 'puzzle',
                    'tactical_info': {
                        'themes': sorted(list(all_themes)),
                        'classification': tactical_classification,
                        'puzzle_rating': sum(ratings) / len(ratings) if ratings else 1000,
                        'perfect_sequences': len(entries)
                    },
                    'references': {
                        'games': [],
                        'puzzles': puzzle_refs[:10]  # Limit to 10 references
                    }
                }
            
            if position_moves:
                position_key = self.create_position_key(position_fen)
                nudge_db[position_key] = {
                    'fen': position_fen,
                    'moves': position_moves
                }
        
        print(f"✅ Built puzzle nudge database with {len(nudge_db)} positions")
        return nudge_db
    
    def save_puzzle_nudge_database(self, nudge_db: Dict, output_file: str):
        """Save puzzle nudge database to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(nudge_db, f, indent=2)
        
        print(f"💾 Puzzle nudge database saved: {output_file}")
        
        # Print summary
        total_moves = sum(len(pos['moves']) for pos in nudge_db.values())
        tactical_classifications = defaultdict(int)
        theme_counts = defaultdict(int)
        
        for position in nudge_db.values():
            for move_data in position['moves'].values():
                tactical_info = move_data['tactical_info']
                tactical_classifications[tactical_info['classification']] += 1
                for theme in tactical_info['themes']:
                    theme_counts[theme] += 1
        
        print(f"📊 Puzzle Nudge Database Summary:")
        print(f"   Positions: {len(nudge_db)}")
        print(f"   Total moves: {total_moves}")
        print(f"   Avg moves per position: {total_moves/len(nudge_db):.1f}")
        print(f"   Tactical classifications:")
        for classification, count in sorted(tactical_classifications.items()):
            print(f"     {classification}: {count}")
        print(f"   Top themes:")
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     {theme}: {count}")
    
    def print_extraction_stats(self):
        """Print extraction statistics"""
        print(f"\\n📈 Extraction Statistics:")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Perfect sequences found: {self.stats['perfect_sequences_found']}")
        print(f"   Positions extracted: {self.stats['positions_extracted']}")
        print(f"   Unique positions: {self.stats['unique_positions']}")
        
        if self.stats['perfect_sequences_found'] > 0:
            avg_positions = self.stats['positions_extracted'] / self.stats['perfect_sequences_found']
            print(f"   Avg positions per sequence: {avg_positions:.1f}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V7P3R Puzzle Nudge Extractor')
    parser.add_argument('--analysis-dirs', nargs='+', required=True, 
                       help='Directories containing puzzle analysis results')
    parser.add_argument('--output', default='v7p3r_puzzle_nudges.json', 
                       help='Output file for puzzle nudge database')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                       help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    print("🧩 V7P3R Puzzle Nudge Extractor")
    print("=" * 40)
    
    start_time = datetime.now()
    
    # Create extractor
    extractor = PuzzleNudgeExtractor()
    
    # Scan analysis files
    stats = extractor.scan_puzzle_analysis_files(args.analysis_dirs)
    
    if stats['positions_extracted'] > 0:
        # Build database
        nudge_db = extractor.build_puzzle_nudge_database(
            min_confidence=args.min_confidence
        )
        
        # Save results
        extractor.save_puzzle_nudge_database(nudge_db, args.output)
        
        # Print stats
        extractor.print_extraction_stats()
        
        elapsed = datetime.now() - start_time
        print(f"\\n🎉 Complete in {elapsed.total_seconds():.1f} seconds!")
        print(f"📁 Puzzle nudge database: {args.output}")
        
    else:
        print("❌ No perfect sequences found in analysis files")


if __name__ == "__main__":
    main()