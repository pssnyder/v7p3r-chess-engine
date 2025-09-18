#!/usr/bin/env python3
"""
V7P3R v11 Phase 3A: Lightweight Defensive Analysis
Fast defensive threat assessment with performance constraints
Author: Pat Snyder
"""

import chess
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class V7P3RLightweightDefense:
    """Fast defensive analysis with performance safeguards"""
    
    def __init__(self):
        self.defense_cache = {}  # Position -> defense score cache
        self.performance_stats = {
            'calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time_ms': 0.0
        }
        
        # Piece values for defense calculation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Defense patterns with fast computation
        self.defense_patterns = {
            'piece_safety': self._evaluate_piece_safety,
            'king_shelter': self._evaluate_king_shelter,
            'piece_coordination': self._evaluate_piece_coordination
        }
    
    def quick_defensive_assessment(self, board: chess.Board) -> float:
        """Fast defensive assessment (target: <2ms)"""
        start_time = time.time()
        self.performance_stats['calls'] += 1
        
        # Create cache key (simplified to avoid FEN overhead)
        cache_key = self._create_position_key(board)
        
        # Check cache first
        if cache_key in self.defense_cache:
            self.performance_stats['cache_hits'] += 1
            return self.defense_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            defense_score = 0.0
            
            # Quick piece safety assessment
            defense_score += self._evaluate_piece_safety(board) * 0.4
            
            # Basic king shelter evaluation
            defense_score += self._evaluate_king_shelter(board) * 0.3
            
            # Simple piece coordination
            defense_score += self._evaluate_piece_coordination(board) * 0.3
            
            # Cache the result (with size limit)
            if len(self.defense_cache) < 1000:  # Prevent memory bloat
                self.defense_cache[cache_key] = defense_score
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats['total_time'] += elapsed_time
            self.performance_stats['avg_time_ms'] = (
                self.performance_stats['total_time'] / self.performance_stats['calls'] * 1000
            )
            
            return defense_score
            
        except Exception as e:
            # Fallback: return neutral score if calculation fails
            return 0.0
    
    def _create_position_key(self, board: chess.Board) -> str:
        """Create simplified position key for caching"""
        # Use more stable position characteristics for better cache hits
        key_parts = []
        
        # Material count by piece type
        material_counts = {}
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, True))
            black_count = len(board.pieces(piece_type, False))
            material_counts[piece_type] = (white_count, black_count)
        
        # King positions (important for defensive analysis)
        white_king = board.king(True)
        black_king = board.king(False)
        
        # Create stable cache key
        key_parts.append(f"mat_{material_counts}")
        key_parts.append(f"wk_{white_king}")
        key_parts.append(f"bk_{black_king}")
        key_parts.append(f"turn_{board.turn}")
        
        return "_".join(key_parts)
    
    def _evaluate_piece_safety(self, board: chess.Board) -> float:
        """Quick evaluation of piece safety"""
        safety_score = 0.0
        
        for color in [True, False]:  # White, Black
            color_multiplier = 1.0 if color == board.turn else -1.0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Check if piece is defended
                    if self._is_piece_defended(board, square, color):
                        safety_score += self.piece_values[piece.piece_type] * 0.01 * color_multiplier
                    
                    # Check if piece is attacked
                    if board.is_attacked_by(not color, square):
                        # Penalty for undefended attacked pieces
                        if not self._is_piece_defended(board, square, color):
                            safety_score -= self.piece_values[piece.piece_type] * 0.02 * color_multiplier
        
        return safety_score / 100.0  # Normalize to reasonable range
    
    def _is_piece_defended(self, board: chess.Board, square: int, color: bool) -> bool:
        """Quick check if piece is defended"""
        return board.is_attacked_by(color, square)
    
    def _evaluate_king_shelter(self, board: chess.Board) -> float:
        """Basic king shelter evaluation"""
        shelter_score = 0.0
        
        for color in [True, False]:
            color_multiplier = 1.0 if color == board.turn else -1.0
            king_square = board.king(color)
            
            if king_square is None:
                continue
            
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Check pawn shelter in front of king
            pawn_shelter = 0
            
            if color:  # White king
                shelter_squares = [
                    chess.square(f, king_rank + 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank < 7
                ]
            else:  # Black king
                shelter_squares = [
                    chess.square(f, king_rank - 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank > 0
                ]
            
            for square in shelter_squares:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shelter += 1
            
            # Bonus for pawn shelter
            shelter_score += pawn_shelter * 0.2 * color_multiplier
            
            # Penalty for open files near king
            open_files_near_king = 0
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file < 8:
                    file_has_pawn = False
                    for rank in range(8):
                        piece = board.piece_at(chess.square(check_file, rank))
                        if piece and piece.piece_type == chess.PAWN:
                            file_has_pawn = True
                            break
                    if not file_has_pawn:
                        open_files_near_king += 1
            
            shelter_score -= open_files_near_king * 0.1 * color_multiplier
        
        return shelter_score
    
    def _evaluate_piece_coordination(self, board: chess.Board) -> float:
        """Simple piece coordination evaluation"""
        coordination_score = 0.0
        
        for color in [True, False]:
            color_multiplier = 1.0 if color == board.turn else -1.0
            
            # Count pieces defending each other
            defended_pieces = 0
            total_pieces = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type != chess.KING:
                    total_pieces += 1
                    if board.is_attacked_by(color, square):
                        defended_pieces += 1
            
            if total_pieces > 0:
                coordination_ratio = defended_pieces / total_pieces
                coordination_score += coordination_ratio * 0.3 * color_multiplier
        
        return coordination_score
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        if self.performance_stats['calls'] > 0:
            cache_hit_rate = (self.performance_stats['cache_hits'] / 
                            self.performance_stats['calls'] * 100)
        else:
            cache_hit_rate = 0.0
        
        return {
            'calls': self.performance_stats['calls'],
            'cache_hit_rate': cache_hit_rate,
            'avg_time_ms': self.performance_stats['avg_time_ms'],
            'cache_size': len(self.defense_cache)
        }
    
    def clear_cache(self):
        """Clear defense cache to free memory"""
        self.defense_cache.clear()
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.performance_stats = {
            'calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time_ms': 0.0
        }


class V7P3RTacticalEscape:
    """Lightweight tactical escape detection"""
    
    def __init__(self):
        self.escape_cache = {}
        self.performance_stats = {
            'calls': 0,
            'escapes_found': 0,
            'total_time': 0.0
        }
        
        # Escape values by piece type
        self.escape_values = {
            chess.PAWN: 10,
            chess.KNIGHT: 32,
            chess.BISHOP: 33,
            chess.ROOK: 50,
            chess.QUEEN: 90,
            chess.KING: 200
        }
    
    def detect_escape_opportunities(self, board: chess.Board, color: bool) -> float:
        """Quick escape opportunity detection (target: <3ms)"""
        start_time = time.time()
        self.performance_stats['calls'] += 1
        
        try:
            escape_bonus = 0.0
            escapes_found = 0
            
            # Check each piece of the given color
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Check if piece is under attack
                    if board.is_attacked_by(not color, square):
                        # Check if piece can escape to safety
                        if self._has_safe_escape_moves(board, square):
                            escape_value = self.escape_values[piece.piece_type]
                            escape_bonus += escape_value * 0.1  # Scale down the bonus
                            escapes_found += 1
            
            self.performance_stats['escapes_found'] += escapes_found
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats['total_time'] += elapsed_time
            
            return escape_bonus
            
        except Exception as e:
            return 0.0
    
    def _has_safe_escape_moves(self, board: chess.Board, square: int) -> bool:
        """Quick check if piece has safe escape moves"""
        piece = board.piece_at(square)
        if not piece:
            return False
        
        # Check up to 5 moves to keep it fast
        move_count = 0
        for move in board.legal_moves:
            if move.from_square == square:
                move_count += 1
                if move_count > 5:  # Limit for performance
                    break
                
                # Quick safety check
                board.push(move)
                is_safe = not board.is_attacked_by(not piece.color, move.to_square)
                board.pop()
                
                if is_safe:
                    return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time_ms = 0.0
        if self.performance_stats['calls'] > 0:
            avg_time_ms = (self.performance_stats['total_time'] / 
                          self.performance_stats['calls'] * 1000)
        
        return {
            'calls': self.performance_stats['calls'],
            'escapes_found': self.performance_stats['escapes_found'],
            'avg_time_ms': avg_time_ms
        }


# Performance monitoring for Phase 3
class Phase3PerformanceMonitor:
    """Real-time performance monitoring during Phase 3 development"""
    
    def __init__(self, baseline_nps: float = 2200):
        self.baseline_nps = baseline_nps
        self.performance_samples = []
        self.alert_threshold = 0.9  # 90% of baseline
        self.critical_threshold = 0.8  # 80% of baseline
    
    def check_performance_regression(self, current_nps: float) -> str:
        """Monitor for performance regressions"""
        self.performance_samples.append(current_nps)
        
        # Keep last 10 samples for trend analysis
        if len(self.performance_samples) > 10:
            self.performance_samples.pop(0)
        
        performance_ratio = current_nps / self.baseline_nps
        
        if performance_ratio < self.critical_threshold:
            return "CRITICAL_REGRESSION"
        elif performance_ratio < self.alert_threshold:
            return "WARNING_REGRESSION"
        else:
            return "PERFORMANCE_OK"
    
    def get_performance_trend(self) -> Dict:
        """Get performance trend analysis"""
        if len(self.performance_samples) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        recent_avg = sum(self.performance_samples[-3:]) / min(3, len(self.performance_samples))
        overall_avg = sum(self.performance_samples) / len(self.performance_samples)
        
        trend = "IMPROVING" if recent_avg > overall_avg else "DECLINING"
        
        return {
            'trend': trend,
            'recent_avg_nps': recent_avg,
            'overall_avg_nps': overall_avg,
            'baseline_ratio': recent_avg / self.baseline_nps,
            'samples_count': len(self.performance_samples)
        }