#!/usr/bin/env python3
"""
TAL-BOT Heuristic Impact Profiler

Creates a "heat map" of which heuristics actually fire and impact decisions.
Identifies computational overhead vs actual game-changing value.

Focus: Position over material, but FAST. Dynamic piece values starting at zero.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import json

try:
    from vpr import VPREngine
except ImportError:
    # Fallback for import issues
    import importlib.util
    vpr_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vpr.py')
    spec = importlib.util.spec_from_file_location("vpr", vpr_path)
    if spec and spec.loader:
        vpr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vpr_module)
        VPREngine = vpr_module.VPREngine
    else:
        raise ImportError("Could not load VPR module")

@dataclass
class HeuristicMetrics:
    """Track impact of individual heuristics"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    decision_changes: int = 0  # How often it changes the top move
    impact_score: float = 0.0  # Weighted impact on evaluation
    avg_time_per_call: float = 0.0
    
    def calculate_avg_time(self):
        self.avg_time_per_call = self.total_time / max(1, self.call_count)

class ProfiledVPREngine(VPREngine):
    """VPR Engine with heuristic profiling"""
    
    def __init__(self):
        super().__init__()
        self.heuristic_metrics = {}
        self.position_complexity_samples = []
        self.piece_value_samples = []
        self.move_ordering_samples = []
        self.evaluation_samples = []
        
    def _profile_heuristic(self, name: str, func, *args, **kwargs):
        """Profile a heuristic function call"""
        if name not in self.heuristic_metrics:
            self.heuristic_metrics[name] = HeuristicMetrics(name)
        
        metric = self.heuristic_metrics[name]
        metric.call_count += 1
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        metric.total_time += (end_time - start_time)
        return result
    
    def _calculate_chaos_factor(self, board: chess.Board) -> float:
        """Profiled chaos factor calculation"""
        return self._profile_heuristic(
            "chaos_factor",
            super()._calculate_chaos_factor,
            board
        )
    
    def _calculate_piece_true_value_fast(self, board: chess.Board, square: chess.Square, color: chess.Color) -> int:
        """Profiled piece value calculation"""
        def measure_piece_value():
            # Fast dynamic piece value: attacks + mobility
            piece = board.piece_at(square)
            if not piece:
                return 0
            
            start_time = time.perf_counter()
            
            # Base tactical value (attacks this piece can make)
            attacks = len(board.attacks(square))
            
            # Mobility factor (legal moves from this square)
            mobility = 0
            for move in board.legal_moves:
                if move.from_square == square:
                    mobility += 1
            
            # Dynamic value = attacks + mobility (position-based, not material-based)
            true_value = attacks + mobility
            
            end_time = time.perf_counter()
            
            # Sample this calculation
            self.piece_value_samples.append({
                'piece_type': piece.piece_type,
                'square': square,
                'attacks': attacks,
                'mobility': mobility,
                'true_value': true_value,
                'calculation_time': end_time - start_time
            })
            
            return true_value
        
        return self._profile_heuristic(
            "piece_true_value",
            measure_piece_value
        )
    
    def _order_moves_fast(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Profiled move ordering"""
        def measure_move_ordering():
            start_time = time.perf_counter()
            
            move_scores = []
            heuristic_impacts = {
                'captures': 0,
                'checks': 0,
                'piece_values': 0,
                'positional': 0
            }
            
            for move in moves:
                score = 0
                
                # Capture heuristic
                if board.is_capture(move):
                    capture_bonus = 100
                    score += capture_bonus
                    heuristic_impacts['captures'] += 1
                
                # Check heuristic
                board.push(move)
                if board.is_check():
                    check_bonus = 50
                    score += check_bonus
                    heuristic_impacts['checks'] += 1
                board.pop()
                
                # Piece value heuristic
                piece = board.piece_at(move.from_square)
                if piece:
                    piece_bonus = [0, 10, 30, 30, 50, 90, 900][piece.piece_type]
                    score += piece_bonus
                    heuristic_impacts['piece_values'] += 1
                
                move_scores.append((score, move))
            
            # Sort by score
            move_scores.sort(reverse=True, key=lambda x: x[0])
            ordered_moves = [move for _, move in move_scores]
            
            end_time = time.perf_counter()
            
            # Sample this ordering
            self.move_ordering_samples.append({
                'move_count': len(moves),
                'heuristic_impacts': heuristic_impacts,
                'ordering_time': end_time - start_time,
                'top_move_changed': len(moves) > 0 and ordered_moves[0] != moves[0]
            })
            
            return ordered_moves
        
        return self._profile_heuristic(
            "move_ordering",
            measure_move_ordering
        )
    
    def _quick_evaluate(self, board: chess.Board) -> float:
        """Profiled evaluation function"""
        def measure_evaluation():
            start_time = time.perf_counter()
            
            heuristic_components = {
                'material': 0.0,
                'position': 0.0,
                'complexity': 0.0
            }
            
            # Material component (traditional)
            piece_values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 0
            }
            
            material_score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        material_score += value
                    else:
                        material_score -= value
            
            heuristic_components['material'] = abs(material_score)
            
            # Position component (mobility/attacks)
            legal_moves = len(list(board.legal_moves))
            position_score = legal_moves * 2  # Mobility bonus
            heuristic_components['position'] = position_score
            
            # Complexity component (chaos factor)
            if hasattr(self, '_calculate_chaos_factor'):
                complexity_score = self._calculate_chaos_factor(board) * 0.1
                heuristic_components['complexity'] = complexity_score
            else:
                complexity_score = 0
            
            # Final score
            score = material_score + position_score + complexity_score
            
            # Flip for side to move
            if not board.turn:
                score = -score
            
            end_time = time.perf_counter()
            
            # Sample this evaluation
            self.evaluation_samples.append({
                'heuristic_components': heuristic_components,
                'final_score': score,
                'evaluation_time': end_time - start_time
            })
            
            return score
        
        return self._profile_heuristic(
            "evaluation",
            measure_evaluation
        )
    
    def generate_heuristic_report(self) -> Dict[str, Any]:
        """Generate comprehensive heuristic impact report"""
        # Calculate averages for all metrics
        for metric in self.heuristic_metrics.values():
            metric.calculate_avg_time()
        
        # Analyze samples
        analysis = {
            'heuristic_performance': {},
            'piece_value_analysis': self._analyze_piece_values(),
            'move_ordering_analysis': self._analyze_move_ordering(),
            'evaluation_analysis': self._analyze_evaluation(),
            'recommendations': []
        }
        
        # Performance metrics
        total_time = sum(m.total_time for m in self.heuristic_metrics.values())
        for name, metric in self.heuristic_metrics.items():
            time_percentage = (metric.total_time / total_time * 100) if total_time > 0 else 0
            analysis['heuristic_performance'][name] = {
                'call_count': metric.call_count,
                'total_time_ms': metric.total_time * 1000,
                'avg_time_ms': metric.avg_time_per_call * 1000,
                'time_percentage': time_percentage,
                'calls_per_second': metric.call_count / max(metric.total_time, 0.001)
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_piece_values(self) -> Dict[str, Any]:
        """Analyze piece value calculation patterns"""
        if not self.piece_value_samples:
            return {}
        
        # Group by piece type
        by_piece_type = defaultdict(list)
        for sample in self.piece_value_samples:
            by_piece_type[sample['piece_type']].append(sample)
        
        analysis = {}
        for piece_type, samples in by_piece_type.items():
            piece_name = chess.piece_name(piece_type)
            avg_attacks = sum(s['attacks'] for s in samples) / len(samples)
            avg_mobility = sum(s['mobility'] for s in samples) / len(samples)
            avg_true_value = sum(s['true_value'] for s in samples) / len(samples)
            avg_calc_time = sum(s['calculation_time'] for s in samples) / len(samples)
            
            analysis[piece_name] = {
                'sample_count': len(samples),
                'avg_attacks': avg_attacks,
                'avg_mobility': avg_mobility,
                'avg_true_value': avg_true_value,
                'avg_calc_time_ms': avg_calc_time * 1000,
                'value_variance': max(s['true_value'] for s in samples) - min(s['true_value'] for s in samples)
            }
        
        return analysis
    
    def _analyze_move_ordering(self) -> Dict[str, Any]:
        """Analyze move ordering effectiveness"""
        if not self.move_ordering_samples:
            return {}
        
        total_samples = len(self.move_ordering_samples)
        moves_that_changed_order = sum(1 for s in self.move_ordering_samples if s['top_move_changed'])
        
        heuristic_usage = Counter()
        for sample in self.move_ordering_samples:
            for heuristic, count in sample['heuristic_impacts'].items():
                heuristic_usage[heuristic] += count
        
        avg_ordering_time = sum(s['ordering_time'] for s in self.move_ordering_samples) / total_samples
        
        return {
            'total_orderings': total_samples,
            'order_changes': moves_that_changed_order,
            'order_change_percentage': (moves_that_changed_order / total_samples * 100) if total_samples > 0 else 0,
            'heuristic_usage': dict(heuristic_usage),
            'avg_ordering_time_ms': avg_ordering_time * 1000
        }
    
    def _analyze_evaluation(self) -> Dict[str, Any]:
        """Analyze evaluation component contributions"""
        if not self.evaluation_samples:
            return {}
        
        total_samples = len(self.evaluation_samples)
        
        # Average component contributions
        avg_components = defaultdict(float)
        for sample in self.evaluation_samples:
            for component, value in sample['heuristic_components'].items():
                avg_components[component] += value
        
        for component in avg_components:
            avg_components[component] /= total_samples
        
        avg_eval_time = sum(s['evaluation_time'] for s in self.evaluation_samples) / total_samples
        
        return {
            'total_evaluations': total_samples,
            'avg_component_values': dict(avg_components),
            'avg_evaluation_time_ms': avg_eval_time * 1000,
            'component_variance': self._calculate_component_variance()
        }
    
    def _calculate_component_variance(self) -> Dict[str, float]:
        """Calculate variance in evaluation components"""
        if not self.evaluation_samples:
            return {}
        
        component_values = defaultdict(list)
        for sample in self.evaluation_samples:
            for component, value in sample['heuristic_components'].items():
                component_values[component].append(value)
        
        variance = {}
        for component, values in component_values.items():
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance[component] = sum((x - mean) ** 2 for x in values) / len(values)
            else:
                variance[component] = 0
        
        return variance
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations based on profiling"""
        recommendations = []
        
        # Performance recommendations
        perf = analysis['heuristic_performance']
        if 'piece_true_value' in perf and perf['piece_true_value']['time_percentage'] > 50:
            recommendations.append("CRITICAL: Piece value calculation takes >50% of time - needs optimization")
        
        if 'chaos_factor' in perf and perf['chaos_factor']['time_percentage'] > 20:
            recommendations.append("WARNING: Chaos factor calculation is expensive - consider simplification")
        
        # Move ordering recommendations
        if 'move_ordering_analysis' in analysis:
            mo = analysis['move_ordering_analysis']
            if mo.get('order_change_percentage', 0) < 10:
                recommendations.append("INFO: Move ordering rarely changes order - may not be worth the cost")
        
        # Evaluation recommendations
        if 'evaluation_analysis' in analysis:
            eval_analysis = analysis['evaluation_analysis']
            components = eval_analysis.get('avg_component_values', {})
            
            if components.get('material', 0) > components.get('position', 0) * 3:
                recommendations.append("WARNING: Material heavily outweighs position - not achieving position-over-material goal")
            
            if components.get('complexity', 0) < components.get('material', 0) * 0.1:
                recommendations.append("INFO: Complexity factor has minimal impact - consider increasing weight or removing")
        
        return recommendations

def run_heuristic_profiling():
    """Run comprehensive heuristic profiling"""
    print("🔬 TAL-BOT Heuristic Impact Profiler")
    print("=" * 50)
    
    engine = ProfiledVPREngine()
    
    # Test positions for profiling
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  # Complex tactical
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Endgame
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  # Sacrificial opportunity
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"  # Opening complexity
    ]
    
    total_start_time = time.perf_counter()
    
    for i, fen in enumerate(test_positions, 1):
        print(f"\n📍 Testing Position {i}/5")
        board = chess.Board(fen)
        
        # Run search to trigger all heuristics
        start_time = time.perf_counter()
        best_move = engine.search(board, time_limit=2.0, depth=4)
        end_time = time.perf_counter()
        
        print(f"   Best move: {best_move}")
        print(f"   Search time: {(end_time - start_time)*1000:.1f}ms")
        print(f"   Nodes: {engine.nodes_searched}")
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    
    print(f"\n⏱️ Total profiling time: {total_time:.2f}s")
    print("\n🔍 Generating Heat Map Analysis...")
    
    # Generate comprehensive report
    report = engine.generate_heuristic_report()
    
    # Display heat map results
    print("\n" + "="*60)
    print("🔥 HEURISTIC HEAT MAP - IMPACT ANALYSIS")
    print("="*60)
    
    # Performance heat map
    print("\n🎯 PERFORMANCE BREAKDOWN:")
    perf = report['heuristic_performance']
    for name, metrics in sorted(perf.items(), key=lambda x: x[1]['time_percentage'], reverse=True):
        percentage = metrics['time_percentage']
        bar_length = int(percentage / 2)  # Scale to fit
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"{name:20} │{bar}│ {percentage:5.1f}% ({metrics['avg_time_ms']:.3f}ms avg)")
    
    # Piece value analysis
    if 'piece_value_analysis' in report and report['piece_value_analysis']:
        print("\n🏗️ PIECE VALUE DYNAMICS:")
        for piece, data in report['piece_value_analysis'].items():
            print(f"{piece:8}: avg_value={data['avg_true_value']:.1f}, variance={data['value_variance']}, calc_time={data['avg_calc_time_ms']:.3f}ms")
    
    # Move ordering effectiveness
    if 'move_ordering_analysis' in report:
        mo = report['move_ordering_analysis']
        print(f"\n🎲 MOVE ORDERING IMPACT:")
        print(f"   Order changes: {mo['order_change_percentage']:.1f}% of the time")
        print(f"   Avg time: {mo['avg_ordering_time_ms']:.3f}ms")
        print(f"   Heuristic usage: {mo['heuristic_usage']}")
    
    # Evaluation component balance
    if 'evaluation_analysis' in report:
        eval_data = report['evaluation_analysis']
        components = eval_data['avg_component_values']
        print(f"\n⚖️ EVALUATION BALANCE:")
        total_value = sum(components.values())
        for component, value in components.items():
            percentage = (value / total_value * 100) if total_value > 0 else 0
            print(f"   {component:12}: {percentage:5.1f}% (avg: {value:.1f})")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed report
    with open('heuristic_impact_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: heuristic_impact_report.json")
    
    return report

if __name__ == "__main__":
    run_heuristic_profiling()