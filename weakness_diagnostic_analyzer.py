#!/usr/bin/env python3
"""
V7P3R Weakness Diagnostic Analyzer
Deep analysis of V7P3R's decision-making process on identified weakness positions

This tool takes weakness positions from the game replay analyzer and runs them through
V7P3R's internal systems to understand WHY poor moves were selected:

1. Move Ordering Analysis - Which moves were considered first?
2. Evaluation Comparison - How did V7P3R rate its move vs best move?
3. Tactical Detection - Did V7P3R see the tactics in the position?
4. Search Analysis - What depth/time was used for the decision?
5. Heuristic Influence - Which components influenced the move choice?

This diagnostic information will guide targeted improvements to V13.1.
"""

import chess
import chess.engine
import os
import sys
import json
import time
import subprocess
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path for V7P3R analysis
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)

try:
    from analyze_move_ordering import V7P3RMoveAnalyzer
    MOVE_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: V7P3RMoveAnalyzer not available")
    MOVE_ANALYZER_AVAILABLE = False


@dataclass
class WeaknessPosition:
    """Represents a weakness position for analysis"""
    game_id: str
    move_number: int
    fen_before_move: str
    opponent_last_move: str
    v7p3r_move: str
    v7p3r_move_rank: int
    stockfish_top_moves: List[Tuple[str, int]]
    stockfish_best_move: str
    centipawn_loss: int
    position_themes: List[str]
    time_pressure: bool = False
    material_balance: int = 0
    game_phase: str = "middlegame"
    analysis_time: float = 0.0


@dataclass
class V7P3RAnalysisResult:
    """Results from V7P3R analysis of a position"""
    position_fen: str
    v7p3r_chosen_move: str
    v7p3r_evaluation: int  # centipawns
    v7p3r_depth: int
    v7p3r_nodes: int
    v7p3r_time_ms: int
    v7p3r_top_moves: List[Tuple[str, int]]  # (move, evaluation)
    move_ordering: List[str]  # Order moves were considered
    tactical_detected: bool
    tactical_patterns: List[str]
    evaluation_breakdown: Dict[str, float]
    search_info: Dict[str, Any]


@dataclass
class DiagnosticComparison:
    """Comparison between V7P3R and Stockfish analysis"""
    weakness_position: WeaknessPosition
    v7p3r_analysis: V7P3RAnalysisResult
    
    # Comparison metrics
    evaluation_gap: int  # V7P3R eval - Stockfish eval for V7P3R's move
    best_move_ranking: int  # Where Stockfish's best move ranked in V7P3R
    tactical_miss: bool  # Did V7P3R miss tactics that were present?
    move_ordering_issue: bool  # Was best move ordered poorly?
    
    # Diagnostic insights
    primary_failure_mode: str  # evaluation, tactics, ordering, search
    improvement_suggestions: List[str]
    severity: str  # low, medium, high, critical


class WeaknessDiagnosticAnalyzer:
    """Analyzes V7P3R's decision-making on weakness positions"""
    
    def __init__(self, 
                 v7p3r_engine_path: str = r"S:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py",
                 stockfish_path: str = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\Stockfish\stockfish-windows-x86-64-avx2.exe"):
        
        self.v7p3r_engine_path = v7p3r_engine_path
        self.stockfish_path = stockfish_path
        self.move_analyzer = V7P3RMoveAnalyzer() if MOVE_ANALYZER_AVAILABLE else None
        
        # Results storage
        self.diagnostic_results: List[DiagnosticComparison] = []
        self.failure_mode_stats = Counter()
        self.improvement_suggestions = Counter()
        
        # Verify paths
        if not os.path.exists(v7p3r_engine_path):
            print(f"Warning: V7P3R engine not found at {v7p3r_engine_path}")
        if not os.path.exists(stockfish_path):
            print(f"Warning: Stockfish not found at {stockfish_path}")
    
    def load_weakness_positions(self, json_file: str) -> List[WeaknessPosition]:
        """Load weakness positions from analysis JSON file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            positions = []
            for pos_data in data.get("weakness_positions", []):
                position = WeaknessPosition(
                    game_id=pos_data["game_id"],
                    move_number=pos_data["move_number"],
                    fen_before_move=pos_data["fen_before_move"],
                    opponent_last_move=pos_data["opponent_last_move"],
                    v7p3r_move=pos_data["v7p3r_move"],
                    v7p3r_move_rank=pos_data["v7p3r_move_rank"],
                    stockfish_top_moves=pos_data["stockfish_top_moves"],
                    stockfish_best_move=pos_data["stockfish_best_move"],
                    centipawn_loss=pos_data["centipawn_loss"],
                    position_themes=pos_data["position_themes"],
                    time_pressure=pos_data.get("time_pressure", False),
                    material_balance=pos_data.get("material_balance", 0),
                    game_phase=pos_data.get("game_phase", "middlegame"),
                    analysis_time=pos_data.get("analysis_time", 0.0)
                )
                positions.append(position)
            
            print(f"Loaded {len(positions)} weakness positions for diagnostic analysis")
            return positions
            
        except Exception as e:
            print(f"Error loading weakness positions: {e}")
            return []
    
    def analyze_position_with_v7p3r(self, position: WeaknessPosition, time_limit: float = 5.0) -> Optional[V7P3RAnalysisResult]:
        """Analyze a position using V7P3R engine with detailed logging"""
        try:
            # Start V7P3R engine
            engine = chess.engine.SimpleEngine.popen_uci([
                "python", self.v7p3r_engine_path
            ])
            
            # Set up position
            board = chess.Board(position.fen_before_move)
            
            # Configure engine for detailed analysis
            engine.configure({"Hash": 128})  # Set hash size
            
            # Get V7P3R's analysis
            start_time = time.time()
            
            # Get detailed analysis with multiple PV lines
            info = engine.analyse(
                board,
                chess.engine.Limit(time=time_limit),
                multipv=5,
                info=chess.engine.INFO_ALL
            )
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            # Extract analysis results
            if isinstance(info, list):
                main_info = info[0]
            else:
                main_info = info
            
            # Get V7P3R's top moves
            v7p3r_moves = []
            if isinstance(info, list):
                for pv_info in info:
                    if 'pv' in pv_info and pv_info['pv']:
                        move = pv_info['pv'][0].uci()
                        score = pv_info.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                        if score.is_mate():
                            mate_value = score.white().mate()
                            cp_value = 10000 if mate_value and mate_value > 0 else -10000
                        else:
                            cp_value = score.white().score() or 0
                        v7p3r_moves.append((move, cp_value))
            
            # Get move ordering if available (through legal moves order)
            move_ordering = [move.uci() for move in board.legal_moves]
            
            # Check for tactical detection
            tactical_detected = self._check_tactical_detection(board)
            tactical_patterns = self._identify_tactical_patterns(board)
            
            # Create evaluation breakdown (placeholder - would need V7P3R internals)
            evaluation_breakdown = {
                "material": 0.0,
                "position": 0.0,
                "mobility": 0.0,
                "safety": 0.0,
                "tactics": 0.0
            }
            
            # Search info
            search_info = {
                "nodes": main_info.get('nodes', 0),
                "depth": main_info.get('depth', 0),
                "seldepth": main_info.get('seldepth', 0),
                "time": analysis_time,
                "nps": main_info.get('nps', 0)
            }
            
            result = V7P3RAnalysisResult(
                position_fen=position.fen_before_move,
                v7p3r_chosen_move=position.v7p3r_move,
                v7p3r_evaluation=v7p3r_moves[0][1] if v7p3r_moves else 0,
                v7p3r_depth=main_info.get('depth', 0),
                v7p3r_nodes=main_info.get('nodes', 0),
                v7p3r_time_ms=analysis_time,
                v7p3r_top_moves=v7p3r_moves,
                move_ordering=move_ordering,
                tactical_detected=tactical_detected,
                tactical_patterns=tactical_patterns,
                evaluation_breakdown=evaluation_breakdown,
                search_info=search_info
            )
            
            engine.quit()
            return result
            
        except Exception as e:
            print(f"Error analyzing position with V7P3R: {e}")
            return None
    
    def _check_tactical_detection(self, board: chess.Board) -> bool:
        """Check if tactical patterns would be detected in this position"""
        # Look for basic tactical indicators
        has_checks = any(board.gives_check(move) for move in board.legal_moves)
        has_captures = any(board.is_capture(move) for move in board.legal_moves)
        
        # Look for hanging pieces
        has_hanging = False
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                if len(attackers) > len(defenders):
                    has_hanging = True
                    break
        
        return has_checks or has_captures or has_hanging
    
    def _identify_tactical_patterns(self, board: chess.Board) -> List[str]:
        """Identify tactical patterns present in the position"""
        patterns = []
        
        if board.is_check():
            patterns.append("check")
        
        # Look for captures
        if any(board.is_capture(move) for move in board.legal_moves):
            patterns.append("captures_available")
        
        # Look for checks
        if any(board.gives_check(move) for move in board.legal_moves):
            patterns.append("checks_available")
        
        # Look for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                if len(attackers) > len(defenders):
                    patterns.append("hanging_piece")
                    break
        
        return patterns
    
    def compare_analyses(self, weakness: WeaknessPosition, v7p3r_result: V7P3RAnalysisResult) -> DiagnosticComparison:
        """Compare V7P3R analysis with Stockfish to identify failure modes"""
        
        # Find V7P3R's evaluation of its own move
        v7p3r_move_eval = 0
        for move, eval_score in v7p3r_result.v7p3r_top_moves:
            if move == weakness.v7p3r_move:
                v7p3r_move_eval = eval_score
                break
        
        # Find Stockfish's evaluation of V7P3R's move
        stockfish_v7p3r_eval = 0
        for move, eval_score in weakness.stockfish_top_moves:
            if move == weakness.v7p3r_move:
                stockfish_v7p3r_eval = eval_score
                break
        
        # Calculate evaluation gap
        evaluation_gap = v7p3r_move_eval - stockfish_v7p3r_eval
        
        # Find where Stockfish's best move ranks in V7P3R's analysis
        best_move_ranking = 999
        for i, (move, _) in enumerate(v7p3r_result.v7p3r_top_moves):
            if move == weakness.stockfish_best_move:
                best_move_ranking = i + 1
                break
        
        # Check for tactical miss
        tactical_miss = (
            "tactics_available" in weakness.position_themes and 
            not v7p3r_result.tactical_detected
        )
        
        # Check for move ordering issue
        move_ordering_issue = best_move_ranking > 3
        
        # Determine primary failure mode
        primary_failure_mode = self._determine_failure_mode(
            evaluation_gap, best_move_ranking, tactical_miss, weakness.centipawn_loss
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            primary_failure_mode, weakness, v7p3r_result
        )
        
        # Determine severity
        severity = self._determine_severity(weakness.centipawn_loss)
        
        comparison = DiagnosticComparison(
            weakness_position=weakness,
            v7p3r_analysis=v7p3r_result,
            evaluation_gap=evaluation_gap,
            best_move_ranking=best_move_ranking,
            tactical_miss=tactical_miss,
            move_ordering_issue=move_ordering_issue,
            primary_failure_mode=primary_failure_mode,
            improvement_suggestions=suggestions,
            severity=severity
        )
        
        return comparison
    
    def _determine_failure_mode(self, eval_gap: int, best_move_rank: int, tactical_miss: bool, centipawn_loss: int) -> str:
        """Determine the primary failure mode"""
        if tactical_miss and centipawn_loss > 500:
            return "tactical_blindness"
        elif abs(eval_gap) > 300:
            return "evaluation_error"
        elif best_move_rank > 5:
            return "move_ordering"
        elif centipawn_loss > 1000:
            return "search_depth"
        else:
            return "positional_understanding"
    
    def _generate_improvement_suggestions(self, failure_mode: str, weakness: WeaknessPosition, v7p3r: V7P3RAnalysisResult) -> List[str]:
        """Generate specific improvement suggestions based on failure mode"""
        suggestions = []
        
        if failure_mode == "tactical_blindness":
            suggestions.extend([
                "Improve tactical pattern recognition",
                "Increase tactical search depth in tactical positions",
                "Add more tactical pattern templates"
            ])
        elif failure_mode == "evaluation_error":
            suggestions.extend([
                "Recalibrate piece values for this position type",
                "Improve positional evaluation weights",
                "Add position-specific evaluation terms"
            ])
        elif failure_mode == "move_ordering":
            suggestions.extend([
                "Improve move ordering heuristics",
                "Prioritize tactical moves higher",
                "Add killer move heuristic"
            ])
        elif failure_mode == "search_depth":
            suggestions.extend([
                "Increase search depth for critical positions",
                "Improve time management",
                "Add selective deepening"
            ])
        else:
            suggestions.extend([
                "Improve positional understanding",
                "Add strategic pattern recognition",
                "Enhance endgame knowledge"
            ])
        
        # Add theme-specific suggestions
        if "hanging_piece" in weakness.position_themes:
            suggestions.append("Improve hanging piece detection")
        if "material_imbalance" in weakness.position_themes:
            suggestions.append("Improve material imbalance evaluation")
        
        return suggestions
    
    def _determine_severity(self, centipawn_loss: int) -> str:
        """Determine the severity of the weakness"""
        if centipawn_loss >= 1000:
            return "critical"
        elif centipawn_loss >= 300:
            return "high"
        elif centipawn_loss >= 100:
            return "medium"
        else:
            return "low"
    
    def analyze_weakness_batch(self, positions: List[WeaknessPosition], max_positions: Optional[int] = None) -> List[DiagnosticComparison]:
        """Analyze a batch of weakness positions"""
        if max_positions:
            positions = positions[:max_positions]
        
        print(f"🔍 Starting diagnostic analysis of {len(positions)} weakness positions...")
        print("=" * 60)
        
        results = []
        
        for i, position in enumerate(positions, 1):
            print(f"\n📍 Analyzing position {i}/{len(positions)}: {position.game_id} move {position.move_number}")
            print(f"   V7P3R played: {position.v7p3r_move} (lost {position.centipawn_loss}cp)")
            print(f"   Best move: {position.stockfish_best_move}")
            print(f"   Themes: {', '.join(position.position_themes[:3])}")
            
            # Analyze with V7P3R
            v7p3r_result = self.analyze_position_with_v7p3r(position, time_limit=3.0)
            
            if v7p3r_result:
                # Compare analyses
                comparison = self.compare_analyses(position, v7p3r_result)
                results.append(comparison)
                
                # Update statistics
                self.failure_mode_stats[comparison.primary_failure_mode] += 1
                for suggestion in comparison.improvement_suggestions:
                    self.improvement_suggestions[suggestion] += 1
                
                # Print diagnosis
                print(f"   🎯 Primary issue: {comparison.primary_failure_mode}")
                print(f"   📊 V7P3R eval: {v7p3r_result.v7p3r_evaluation}cp, Best move rank: #{comparison.best_move_ranking}")
                print(f"   ⚡ Tactical detected: {v7p3r_result.tactical_detected}, Depth: {v7p3r_result.v7p3r_depth}")
            else:
                print(f"   ❌ Failed to analyze with V7P3R")
            
            # Progress update
            if i % 5 == 0:
                print(f"\n📊 Progress: {i}/{len(positions)} analyzed")
        
        self.diagnostic_results = results
        return results
    
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary"""
        if not self.diagnostic_results:
            print("No diagnostic results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("🎯 V7P3R WEAKNESS DIAGNOSTIC SUMMARY")
        print("=" * 80)
        
        total_positions = len(self.diagnostic_results)
        print(f"Positions analyzed: {total_positions}")
        
        # Severity breakdown
        severity_counts = Counter(result.severity for result in self.diagnostic_results)
        print(f"\n📊 Severity Breakdown:")
        for severity, count in severity_counts.most_common():
            percentage = (count / total_positions) * 100
            print(f"  {severity.upper()}: {count} ({percentage:.1f}%)")
        
        # Failure mode analysis
        print(f"\n🔍 Primary Failure Modes:")
        for failure_mode, count in self.failure_mode_stats.most_common():
            percentage = (count / total_positions) * 100
            print(f"  {failure_mode}: {count} ({percentage:.1f}%)")
        
        # Top improvement suggestions
        print(f"\n💡 Top Improvement Suggestions:")
        for suggestion, count in self.improvement_suggestions.most_common(10):
            print(f"  {suggestion}: {count} positions")
        
        # Critical positions
        critical_positions = [r for r in self.diagnostic_results if r.severity == "critical"]
        if critical_positions:
            print(f"\n🚨 Critical Positions (>1000cp loss):")
            for result in critical_positions[:5]:
                pos = result.weakness_position
                print(f"  {pos.game_id} move {pos.move_number}: {pos.v7p3r_move} (lost {pos.centipawn_loss}cp)")
                print(f"    Issue: {result.primary_failure_mode}")
        
        # Tactical miss analysis
        tactical_misses = [r for r in self.diagnostic_results if r.tactical_miss]
        print(f"\n⚡ Tactical Misses: {len(tactical_misses)} ({len(tactical_misses)/total_positions*100:.1f}%)")
        
        # Move ordering issues
        ordering_issues = [r for r in self.diagnostic_results if r.move_ordering_issue]
        print(f"📋 Move Ordering Issues: {len(ordering_issues)} ({len(ordering_issues)/total_positions*100:.1f}%)")
    
    def save_diagnostic_results(self, output_file: Optional[str] = None):
        """Save diagnostic results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"v7p3r_weakness_diagnostics_{timestamp}.json"
        
        # Prepare serializable data
        serializable_results = []
        for result in self.diagnostic_results:
            result_dict = {
                "weakness_position": {
                    "game_id": result.weakness_position.game_id,
                    "move_number": result.weakness_position.move_number,
                    "fen_before_move": result.weakness_position.fen_before_move,
                    "v7p3r_move": result.weakness_position.v7p3r_move,
                    "stockfish_best_move": result.weakness_position.stockfish_best_move,
                    "centipawn_loss": result.weakness_position.centipawn_loss,
                    "position_themes": result.weakness_position.position_themes
                },
                "v7p3r_analysis": {
                    "v7p3r_chosen_move": result.v7p3r_analysis.v7p3r_chosen_move,
                    "v7p3r_evaluation": result.v7p3r_analysis.v7p3r_evaluation,
                    "v7p3r_depth": result.v7p3r_analysis.v7p3r_depth,
                    "v7p3r_nodes": result.v7p3r_analysis.v7p3r_nodes,
                    "v7p3r_top_moves": result.v7p3r_analysis.v7p3r_top_moves,
                    "tactical_detected": result.v7p3r_analysis.tactical_detected,
                    "tactical_patterns": result.v7p3r_analysis.tactical_patterns
                },
                "diagnosis": {
                    "evaluation_gap": result.evaluation_gap,
                    "best_move_ranking": result.best_move_ranking,
                    "tactical_miss": result.tactical_miss,
                    "move_ordering_issue": result.move_ordering_issue,
                    "primary_failure_mode": result.primary_failure_mode,
                    "improvement_suggestions": result.improvement_suggestions,
                    "severity": result.severity
                }
            }
            serializable_results.append(result_dict)
        
        diagnostic_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "v7p3r_engine_path": self.v7p3r_engine_path,
                "total_positions_analyzed": len(self.diagnostic_results),
                "analysis_version": "1.0_diagnostic"
            },
            "summary_statistics": {
                "failure_modes": dict(self.failure_mode_stats),
                "improvement_suggestions": dict(self.improvement_suggestions.most_common(20)),
                "severity_breakdown": dict(Counter(r.severity for r in self.diagnostic_results))
            },
            "diagnostic_results": serializable_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(diagnostic_data, f, indent=2)
        
        print(f"\n💾 Diagnostic results saved to: {output_file}")
        return output_file


def main():
    """Main execution function for weakness diagnostic analysis"""
    print("🔍 V7P3R Weakness Diagnostic Analyzer")
    print("Deep analysis of V7P3R decision-making on weakness positions")
    print("=" * 60)
    
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='V7P3R Weakness Diagnostic Analyzer')
    parser.add_argument('--input-file', required=True, help='JSON file with weakness positions')
    parser.add_argument('--max-positions', type=int, default=20, help='Maximum positions to analyze')
    parser.add_argument('--output-file', help='Output file for diagnostic results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = WeaknessDiagnosticAnalyzer()
        
        # Load weakness positions
        positions = analyzer.load_weakness_positions(args.input_file)
        if not positions:
            print("No weakness positions loaded!")
            return 1
        
        # Select most severe positions for analysis
        positions = sorted(positions, key=lambda x: x.centipawn_loss, reverse=True)
        
        print(f"Selected top {min(args.max_positions, len(positions))} most severe weaknesses for analysis")
        
        # Analyze positions
        start_time = time.time()
        results = analyzer.analyze_weakness_batch(positions, args.max_positions)
        analysis_time = time.time() - start_time
        
        print(f"\n⏱️  Analysis completed in {analysis_time:.1f} seconds")
        
        # Print diagnostic summary
        analyzer.print_diagnostic_summary()
        
        # Save results
        output_file = analyzer.save_diagnostic_results(args.output_file)
        
        print(f"\n✅ Diagnostic analysis complete!")
        print(f"📊 Analyzed {len(results)} weakness positions")
        print(f"💾 Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Diagnostic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())