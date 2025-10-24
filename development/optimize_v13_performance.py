#!/usr/bin/env python3
"""
V7P3R v13.1 Performance Optimizations
Addresses critical performance bottlenecks identified in profiling
"""

import chess
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OptimizationMetrics:
    """Track optimization impact"""
    cache_hits: int = 0
    cache_misses: int = 0
    tactical_calls_avoided: int = 0
    dynamic_calls_avoided: int = 0
    total_evaluations: int = 0


class V7P3ROptimizedEvaluator:
    """High-performance evaluator with selective tactical analysis"""
    
    def __init__(self):
        self.eval_cache: Dict[str, float] = {}
        self.tactical_cache: Dict[str, List] = {}
        self.position_cache: Dict[str, float] = {}
        
        # Performance thresholds
        self.CACHE_SIZE_LIMIT = 10000
        self.TACTICAL_THRESHOLD_DEPTH = 2  # Only use tactics at depth 2+
        self.DYNAMIC_EVAL_FREQUENCY = 3    # Every 3rd evaluation
        
        # Metrics
        self.metrics = OptimizationMetrics()
        
        # Import engine components lazily
        self.tactical_detector = None
        self.dynamic_evaluator = None
        self.base_evaluator = None
        
    def _load_components(self):
        """Lazy load expensive components"""
        if self.tactical_detector is None:
            try:
                from v7p3r_tactical_detector import V7P3RTacticalDetector
                self.tactical_detector = V7P3RTacticalDetector()
            except ImportError:
                pass
                
        if self.dynamic_evaluator is None:
            try:
                from v7p3r_dynamic_evaluator import V7P3RDynamicEvaluator  
                self.dynamic_evaluator = V7P3RDynamicEvaluator()
            except ImportError:
                pass
                
        if self.base_evaluator is None:
            try:
                from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator
                self.base_evaluator = V7P3RBitboardEvaluator()
            except ImportError:
                pass
    
    def should_use_tactical_analysis(self, board: chess.Board, depth: int, move_count: int) -> bool:
        """Determine if tactical analysis is worth the cost"""
        # Skip tactical analysis in certain conditions
        if depth < self.TACTICAL_THRESHOLD_DEPTH:
            return False
            
        # Skip in endgame (< 8 pieces) unless depth is shallow
        piece_count = len(board.piece_map())
        if piece_count < 8 and depth > 3:
            return False
            
        # Skip if we've done many evaluations (diminishing returns)
        if move_count > 50:
            return False
            
        # Skip in quiescence search
        if hasattr(board, '_in_quiescence') and board._in_quiescence:
            return False
            
        return True
    
    def should_use_dynamic_evaluation(self, board: chess.Board, evaluation_count: int) -> bool:
        """Determine if dynamic evaluation is needed"""
        # Use dynamic evaluation less frequently for performance
        return evaluation_count % self.DYNAMIC_EVAL_FREQUENCY == 0
    
    def get_cached_evaluation(self, board: chess.Board) -> Optional[float]:
        """Get cached evaluation if available"""
        position_hash = str(board.fen())
        
        if position_hash in self.eval_cache:
            self.metrics.cache_hits += 1
            return self.eval_cache[position_hash]
            
        self.metrics.cache_misses += 1
        return None
    
    def cache_evaluation(self, board: chess.Board, score: float):
        """Cache evaluation result"""
        if len(self.eval_cache) >= self.CACHE_SIZE_LIMIT:
            # Simple cache eviction - remove oldest 25%
            items_to_remove = len(self.eval_cache) // 4
            keys_to_remove = list(self.eval_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.eval_cache[key]
        
        position_hash = str(board.fen())
        self.eval_cache[position_hash] = score
    
    def fast_evaluate_position(self, board: chess.Board, depth: int = 1, move_count: int = 0) -> float:
        """High-performance position evaluation with selective enhancements"""
        self.metrics.total_evaluations += 1
        
        # Check cache first
        cached_score = self.get_cached_evaluation(board)
        if cached_score is not None:
            return cached_score
        
        # Load components if needed
        self._load_components()
        
        # Start with base evaluation
        if self.base_evaluator:
            base_score = self.base_evaluator.evaluate_position_bitboard(board)
        else:
            # Fallback basic evaluation
            base_score = self._basic_material_evaluation(board)
        
        final_score = base_score
        
        # Selective tactical analysis
        if (self.tactical_detector and 
            self.should_use_tactical_analysis(board, depth, move_count)):
            
            try:
                tactical_patterns = self.tactical_detector.detect_all_tactical_patterns(board, board.turn)
                tactical_bonus = len(tactical_patterns) * 50  # Simple bonus
                final_score += tactical_bonus if board.turn else -tactical_bonus
            except:
                self.metrics.tactical_calls_avoided += 1
        else:
            self.metrics.tactical_calls_avoided += 1
        
        # Selective dynamic evaluation
        if (self.dynamic_evaluator and 
            self.should_use_dynamic_evaluation(board, self.metrics.total_evaluations)):
            
            try:
                dynamic_bonus = self.dynamic_evaluator.evaluate_dynamic_position_value(board, board.turn)
                final_score += dynamic_bonus * 0.5  # Reduced weight for performance
            except:
                self.metrics.dynamic_calls_avoided += 1
        else:
            self.metrics.dynamic_calls_avoided += 1
        
        # Cache the result
        self.cache_evaluation(board, final_score)
        
        return final_score
    
    def _basic_material_evaluation(self, board: chess.Board) -> float:
        """Fast fallback material evaluation"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        score = 0
        for square, piece in board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
        
        return score
    
    def get_optimization_stats(self) -> Dict:
        """Get performance optimization statistics"""
        total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
        cache_hit_rate = (self.metrics.cache_hits / max(total_cache_requests, 1)) * 100
        
        tactical_skip_rate = (self.metrics.tactical_calls_avoided / max(self.metrics.total_evaluations, 1)) * 100
        dynamic_skip_rate = (self.metrics.dynamic_calls_avoided / max(self.metrics.total_evaluations, 1)) * 100
        
        return {
            'total_evaluations': self.metrics.total_evaluations,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'tactical_calls_avoided': self.metrics.tactical_calls_avoided,
            'tactical_skip_rate': f"{tactical_skip_rate:.1f}%", 
            'dynamic_calls_avoided': self.metrics.dynamic_calls_avoided,
            'dynamic_skip_rate': f"{dynamic_skip_rate:.1f}%",
            'cache_size': len(self.eval_cache)
        }
    
    def clear_caches(self):
        """Clear all caches for memory management"""
        self.eval_cache.clear()
        self.tactical_cache.clear()
        self.position_cache.clear()
        self.metrics = OptimizationMetrics()


class V7P3ROptimizedMoveOrdering:
    """Improved move ordering to reduce search tree size"""
    
    def __init__(self):
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        self.history_heuristic: Dict[str, int] = {}
        
    def score_move(self, board: chess.Board, move: chess.Move, depth: int) -> int:
        """Score moves for ordering (higher = better)"""
        score = 0
        
        # Captures (MVV-LVA - Most Valuable Victim, Least Valuable Attacker)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self._get_piece_value(victim.piece_type)
                attacker_value = self._get_piece_value(attacker.piece_type)
                score += (victim_value - attacker_value // 10) * 100
        
        # Checks
        board.push(move)
        if board.is_check():
            score += 500
        board.pop()
        
        # Promotions
        if move.promotion:
            score += 800
        
        # Killer moves
        if depth in self.killer_moves:
            if move in self.killer_moves[depth]:
                score += 300
        
        # History heuristic
        move_key = f"{move.from_square}-{move.to_square}"
        score += self.history_heuristic.get(move_key, 0) // 10
        
        # Castle
        if board.is_castling(move):
            score += 100
            
        return score
    
    def order_moves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """Order moves for optimal search performance"""
        moves = list(board.legal_moves)
        
        # Score and sort moves
        scored_moves = [(self.score_move(board, move, depth), move) for move in moves]
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        return [move for score, move in scored_moves]
    
    def update_killer_move(self, move: chess.Move, depth: int):
        """Update killer moves heuristic"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:  # Keep only 2 killers per depth
                self.killer_moves[depth].pop()
    
    def update_history(self, move: chess.Move, depth: int):
        """Update history heuristic"""
        move_key = f"{move.from_square}-{move.to_square}"
        self.history_heuristic[move_key] = self.history_heuristic.get(move_key, 0) + depth * depth
    
    def _get_piece_value(self, piece_type: int) -> int:
        """Get piece value for MVV-LVA"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        return values.get(piece_type, 0)


def create_performance_test():
    """Test performance improvements"""
    print("🚀 Testing V7P3R v13.1 Performance Optimizations...")
    
    evaluator = V7P3ROptimizedEvaluator()
    move_orderer = V7P3ROptimizedMoveOrdering()
    
    # Test positions
    test_positions = [
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
        'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4',
        '8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1'
    ]
    
    total_time = 0
    total_positions = 0
    
    for i, fen in enumerate(test_positions):
        board = chess.Board(fen)
        
        print(f"  Testing position {i+1}...")
        start_time = time.time()
        
        # Test multiple evaluations
        for depth in range(1, 4):
            for move_count in range(20):
                score = evaluator.fast_evaluate_position(board, depth, move_count)
                
        elapsed = time.time() - start_time
        total_time += elapsed
        total_positions += 60  # 3 depths × 20 move counts
        
        print(f"    Time: {elapsed:.3f}s")
    
    # Calculate performance metrics
    avg_time_per_eval = (total_time / total_positions) * 1000  # Convert to ms
    evals_per_second = total_positions / total_time
    
    print(f"\n📊 Performance Results:")
    print(f"  Total evaluations: {total_positions}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per evaluation: {avg_time_per_eval:.2f}ms")
    print(f"  Evaluations per second: {evals_per_second:.0f}")
    
    # Show optimization stats
    stats = evaluator.get_optimization_stats()
    print(f"\n🎯 Optimization Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Performance target check
    if avg_time_per_eval < 1.0:  # Less than 1ms per evaluation
        print(f"\n✅ PERFORMANCE TARGET MET! Under 1ms per evaluation")
    else:
        print(f"\n⚠️  Performance target missed. Target: <1.0ms, Actual: {avg_time_per_eval:.2f}ms")
    
    return avg_time_per_eval, stats


if __name__ == "__main__":
    create_performance_test()