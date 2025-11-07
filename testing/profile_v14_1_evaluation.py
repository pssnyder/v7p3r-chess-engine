#!/usr/bin/env python3
"""
V7P3R v14.1 Evaluation Profiler

Instruments the bitboard evaluator to measure time spent in each component.
Identifies bottlenecks preventing deeper search.
"""

import chess
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator


class ProfiledEvaluator(V7P3RBitboardEvaluator):
    """Instrumented version of bitboard evaluator"""
    
    def __init__(self, piece_values):
        super().__init__(piece_values)
        self.timings = defaultdict(float)
        self.call_counts = defaultdict(int)
    
    def evaluate_bitboard(self, board: chess.Board, color: chess.Color) -> float:
        """Profiled version of evaluate_bitboard"""
        
        # Convert to bitboards (no timing - setup)
        white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
        white_knights = int(board.pieces(chess.KNIGHT, chess.WHITE))
        black_knights = int(board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishops = int(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = int(board.pieces(chess.BISHOP, chess.BLACK))
        white_rooks = int(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = int(board.pieces(chess.ROOK, chess.BLACK))
        white_queens = int(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = int(board.pieces(chess.QUEEN, chess.BLACK))
        white_king = int(board.pieces(chess.KING, chess.WHITE))
        black_king = int(board.pieces(chess.KING, chess.BLACK))
        
        white_pieces = white_pawns | white_knights | white_bishops | white_rooks | white_queens | white_king
        black_pieces = black_pawns | black_knights | black_bishops | black_rooks | black_queens | black_king
        all_pieces = white_pieces | black_pieces
        
        total_material = self._popcount(all_pieces & ~(white_pawns | black_pawns))
        
        score = 0.0
        
        # 1. MATERIAL
        start = time.perf_counter()
        score += self._popcount(white_pawns) * self.piece_values[chess.PAWN]
        score += self._popcount(white_knights) * self.piece_values[chess.KNIGHT]
        score += self._popcount(white_bishops) * self.piece_values[chess.BISHOP]
        score += self._popcount(white_rooks) * self.piece_values[chess.ROOK]
        score += self._popcount(white_queens) * self.piece_values[chess.QUEEN]
        score -= self._popcount(black_pawns) * self.piece_values[chess.PAWN]
        score -= self._popcount(black_knights) * self.piece_values[chess.KNIGHT]
        score -= self._popcount(black_bishops) * self.piece_values[chess.BISHOP]
        score -= self._popcount(black_rooks) * self.piece_values[chess.ROOK]
        score -= self._popcount(black_queens) * self.piece_values[chess.QUEEN]
        self.timings['material'] += time.perf_counter() - start
        self.call_counts['material'] += 1
        
        # 2. CENTER CONTROL (pawns)
        start = time.perf_counter()
        white_center_pawns = white_pawns & self.CENTER
        black_center_pawns = black_pawns & self.CENTER
        score += self._popcount(white_center_pawns) * 10
        score -= self._popcount(black_center_pawns) * 10
        white_extended_center = white_pawns & self.EXTENDED_CENTER
        black_extended_center = black_pawns & self.EXTENDED_CENTER
        score += self._popcount(white_extended_center) * 5
        score -= self._popcount(black_extended_center) * 5
        self.timings['center_pawns'] += time.perf_counter() - start
        self.call_counts['center_pawns'] += 1
        
        # 3. CENTER CONTROL (pieces in opening)
        if total_material >= 20:
            start = time.perf_counter()
            white_center_pieces = (white_knights | white_bishops) & self.CENTER
            black_center_pieces = (black_knights | black_bishops) & self.CENTER
            score += self._popcount(white_center_pieces) * 15
            score -= self._popcount(black_center_pieces) * 15
            white_extended_pieces = (white_knights | white_bishops) & self.EXTENDED_CENTER
            black_extended_pieces = (black_knights | black_bishops) & self.EXTENDED_CENTER
            score += self._popcount(white_extended_pieces) * 8
            score -= self._popcount(black_extended_pieces) * 8
            self.timings['center_pieces'] += time.perf_counter() - start
            self.call_counts['center_pieces'] += 1
        
        # 4. KNIGHT OUTPOSTS
        start = time.perf_counter()
        white_knight_outposts = white_knights & self.KNIGHT_OUTPOSTS
        black_knight_outposts = black_knights & self.KNIGHT_OUTPOSTS
        score += self._popcount(white_knight_outposts) * 15
        score -= self._popcount(black_knight_outposts) * 15
        self.timings['knight_outposts'] += time.perf_counter() - start
        self.call_counts['knight_outposts'] += 1
        
        # 5. DEVELOPMENT PENALTIES (opening)
        if total_material >= 18:
            start = time.perf_counter()
            white_undeveloped = 0
            black_undeveloped = 0
            
            if white_knights & (1 << 1): white_undeveloped += 1
            if white_knights & (1 << 6): white_undeveloped += 1
            if black_knights & (1 << 57): black_undeveloped += 1
            if black_knights & (1 << 62): black_undeveloped += 1
            if white_bishops & (1 << 2): white_undeveloped += 1
            if white_bishops & (1 << 5): white_undeveloped += 1
            if black_bishops & (1 << 58): black_undeveloped += 1
            if black_bishops & (1 << 61): black_undeveloped += 1
            
            score -= white_undeveloped * 12
            score += black_undeveloped * 12
            self.timings['development'] += time.perf_counter() - start
            self.call_counts['development'] += 1
        
        # 6. ENHANCED CASTLING
        start = time.perf_counter()
        score += self._evaluate_enhanced_castling(board, color)
        self.timings['castling'] += time.perf_counter() - start
        self.call_counts['castling'] += 1
        
        # 7. PASSED PAWNS
        start = time.perf_counter()
        score += self._count_passed_pawns(white_pawns, black_pawns, True) * 20
        score -= self._count_passed_pawns(black_pawns, white_pawns, False) * 20
        self.timings['passed_pawns'] += time.perf_counter() - start
        self.call_counts['passed_pawns'] += 1
        
        # 8. ENDGAME KING DRIVING
        if total_material <= 8:
            start = time.perf_counter()
            black_king_on_edge = black_king & self.EDGES
            white_king_on_edge = white_king & self.EDGES
            score += self._popcount(black_king_on_edge) * 10
            score -= self._popcount(white_king_on_edge) * 10
            self.timings['endgame_king'] += time.perf_counter() - start
            self.call_counts['endgame_king'] += 1
        
        # 9. DRAW PREVENTION
        if board.halfmove_clock > 30:
            start = time.perf_counter()
            draw_penalty = (board.halfmove_clock - 30) * 2.0
            score -= draw_penalty if color == chess.WHITE else -draw_penalty
            self.timings['draw_prevention'] += time.perf_counter() - start
            self.call_counts['draw_prevention'] += 1
        
        # 10. ACTIVITY PENALTIES
        if total_material >= 12:
            start = time.perf_counter()
            white_back_rank_pieces = (white_knights | white_bishops | white_rooks | white_queens) & (self.RANK_1 | self.RANK_2)
            black_back_rank_pieces = (black_knights | black_bishops | black_rooks | black_queens) & (self.RANK_7 | self.RANK_8)
            activity_penalty = (self._popcount(white_back_rank_pieces) - self._popcount(black_back_rank_pieces)) * 3
            score -= activity_penalty if color == chess.WHITE else -activity_penalty
            self.timings['activity'] += time.perf_counter() - start
            self.call_counts['activity'] += 1
        
        return score if color == chess.WHITE else -score
    
    def get_report(self) -> str:
        """Generate profiling report"""
        total_time = sum(self.timings.values())
        
        report = []
        report.append("\n" + "="*80)
        report.append("V7P3R v14.1 EVALUATION PROFILER REPORT")
        report.append("="*80)
        report.append(f"\nTotal evaluation time: {total_time*1000:.2f} ms")
        report.append(f"Total positions evaluated: {self.call_counts.get('material', 0)}")
        report.append(f"Average time per eval: {(total_time/max(1, self.call_counts.get('material', 1)))*1000000:.2f} μs")
        report.append("\n" + "-"*80)
        report.append(f"{'Component':<25} {'Time (ms)':<12} {'%':<8} {'Calls':<10} {'μs/call':<10}")
        report.append("-"*80)
        
        # Sort by time spent
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for component, comp_time in sorted_timings:
            percentage = (comp_time / total_time * 100) if total_time > 0 else 0
            calls = self.call_counts[component]
            avg_time = (comp_time / calls * 1000000) if calls > 0 else 0
            
            report.append(f"{component:<25} {comp_time*1000:>10.3f}  {percentage:>6.1f}%  {calls:>8}  {avg_time:>8.2f}")
        
        report.append("-"*80)
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-"*80)
        
        for component, comp_time in sorted_timings:
            percentage = (comp_time / total_time * 100) if total_time > 0 else 0
            
            if percentage > 20:
                report.append(f"🔴 HIGH IMPACT: {component} ({percentage:.1f}%) - Prime candidate for removal/simplification")
            elif percentage > 10:
                report.append(f"⚠️  MEDIUM IMPACT: {component} ({percentage:.1f}%) - Consider simplifying")
            elif percentage < 2:
                report.append(f"✅ LOW COST: {component} ({percentage:.1f}%) - Negligible impact")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


def create_test_positions() -> List[Tuple[str, str, str]]:
    """Create test suite of positions"""
    positions = []
    
    # Opening positions (10)
    positions.append(("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "opening"))
    positions.append(("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "opening"))
    positions.append(("French Defense", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "opening"))
    positions.append(("Caro-Kann", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "opening"))
    positions.append(("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "opening"))
    positions.append(("Scotch Game", "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3", "opening"))
    positions.append(("King's Indian", "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5", "opening"))
    positions.append(("Queen's Gambit", "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3", "opening"))
    positions.append(("Nimzo-Indian", "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4", "opening"))
    positions.append(("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "opening"))
    
    # Middlegame tactics (20 positions)
    positions.append(("Fork threat", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6", "middlegame"))
    positions.append(("Pin tactic", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "middlegame"))
    positions.append(("Skewer setup", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 7", "middlegame"))
    positions.append(("Discovery attack", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 5", "middlegame"))
    positions.append(("Sacrifice opportunity", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 8", "middlegame"))
    positions.append(("Complex center", "rnbqk2r/ppp2ppp/3p1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 6", "middlegame"))
    positions.append(("Opposite castling", "2kr1b1r/pppq1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQ1PPP/R1B2RK1 w - - 0 9", "middlegame"))
    positions.append(("Closed position", "rnbqk2r/ppp1bppp/3p1n2/4p3/3PP3/2N2N2/PPP1BPPP/R1BQK2R b KQkq d3 0 6", "middlegame"))
    positions.append(("Open position", "r1bqk2r/pp1p1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7", "middlegame"))
    positions.append(("Pawn majority", "r1bqk2r/pp1p1ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq d6 0 7", "middlegame"))
    positions.append(("Weak squares", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 b - - 0 8", "middlegame"))
    positions.append(("Piece imbalance", "r1bq1rk1/pppn1ppp/3p1n2/4p3/1bB1P3/2NP1N2/PPP1QPPP/R1B2RK1 b - - 0 9", "middlegame"))
    positions.append(("Rook on 7th", "r1bq1rk1/1pp2ppp/p1np1n2/4p3/P1B1P3/2NP1N2/1PP2PPP/R1BQR1K1 b - a3 0 10", "middlegame"))
    positions.append(("Central control", "r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6", "middlegame"))
    positions.append(("Pawn storm", "rnbq1rk1/ppp1bppp/3p1n2/4p3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 8", "middlegame"))
    positions.append(("Blockade", "r1bq1rk1/pppn1ppp/3p1n2/4p3/1bB1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 w - - 0 9", "middlegame"))
    positions.append(("Outpost knight", "r1bq1rk1/ppp2ppp/2np1n2/4p3/1bB1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 8", "middlegame"))
    positions.append(("Bishop pair", "r1bq1rk1/ppp2ppp/2np1n2/4pb2/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 w - - 0 9", "middlegame"))
    positions.append(("Material up", "r1bq1rk1/ppp2ppp/2np4/4p3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 10", "middlegame"))
    positions.append(("Passed pawn", "r1bq1rk1/1pp2ppp/p1np4/4P3/2B5/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 11", "middlegame"))
    
    # Endgame positions (20 positions)
    positions.append(("K+P vs K", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", "endgame"))
    positions.append(("K+2P vs K+P", "8/8/4k3/5p2/5P2/8/4P3/4K3 w - - 0 1", "endgame"))
    positions.append(("Rook endgame", "8/8/4k3/5r2/5R2/8/4K3/8 w - - 0 1", "endgame"))
    positions.append(("Queen endgame", "8/8/4k3/5q2/5Q2/8/4K3/8 w - - 0 1", "endgame"))
    positions.append(("Bishop vs Knight", "8/8/4k3/5b2/5N2/8/4K3/8 w - - 0 1", "endgame"))
    positions.append(("Opposite bishops", "8/4k3/8/2b5/2B5/8/4K3/8 w - - 0 1", "endgame"))
    positions.append(("Rook + pawns", "8/4kp2/8/5r2/5R2/8/4KP2/8 w - - 0 1", "endgame"))
    positions.append(("Passed pawn race", "8/5kp1/8/8/8/8/5KP1/8 w - - 0 1", "endgame"))
    positions.append(("King activity", "8/8/4k3/8/8/8/8/4K3 w - - 0 1", "endgame"))
    positions.append(("Pawn breakthrough", "8/4k1pp/8/8/8/8/4K1PP/8 w - - 0 1", "endgame"))
    positions.append(("K+R vs K+P", "8/8/4k3/5p2/8/8/4R3/4K3 w - - 0 1", "endgame"))
    positions.append(("K+Q vs K+P", "8/8/4k3/5p2/8/8/4Q3/4K3 w - - 0 1", "endgame"))
    positions.append(("Fortress", "8/8/8/5k2/5p2/5P2/8/5K2 w - - 0 1", "endgame"))
    positions.append(("Zugzwang", "8/8/8/3k4/3P4/3K4/8/8 b - - 0 1", "endgame"))
    positions.append(("Triangulation", "8/8/4k3/8/8/4K3/8/8 w - - 0 1", "endgame"))
    positions.append(("Lucena", "1K6/1P1k4/8/8/8/8/5r2/5R2 w - - 0 1", "endgame"))
    positions.append(("Philidor", "3k4/R7/3K4/8/8/8/r7/8 b - - 0 1", "endgame"))
    positions.append(("K+B+B vs K", "8/8/8/4k3/8/8/2B1B3/4K3 w - - 0 1", "endgame"))
    positions.append(("K+B+N vs K", "8/8/8/4k3/8/8/2B1N3/4K3 w - - 0 1", "endgame"))
    positions.append(("Stalemate risk", "8/8/8/8/8/3k4/3p4/3K4 w - - 0 1", "endgame"))
    
    return positions


def main():
    """Run profiler on test suite"""
    print("V7P3R v14.1 Evaluation Profiler")
    print("=" * 80)
    print("\nInitializing evaluator...")
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    evaluator = ProfiledEvaluator(piece_values)
    
    print("Creating test positions...")
    positions = create_test_positions()
    
    print(f"Running profiler on {len(positions)} positions...\n")
    
    opening_count = middlegame_count = endgame_count = 0
    
    for name, fen, phase in positions:
        board = chess.Board(fen)
        _ = evaluator.evaluate_bitboard(board, chess.WHITE)
        
        if phase == "opening":
            opening_count += 1
        elif phase == "middlegame":
            middlegame_count += 1
        else:
            endgame_count += 1
    
    print(f"Evaluated: {opening_count} opening, {middlegame_count} middlegame, {endgame_count} endgame")
    print(evaluator.get_report())
    
    # Estimate NPS
    total_time = sum(evaluator.timings.values())
    positions_per_second = len(positions) / total_time if total_time > 0 else 0
    
    print("\nPERFORMANCE ESTIMATE:")
    print("-" * 80)
    print(f"Positions/second: {positions_per_second:,.0f}")
    print(f"Estimated NPS in search: {positions_per_second * 0.7:,.0f} (70% of test NPS)")
    print("\nNote: Actual search NPS will be lower due to move generation overhead")
    print("=" * 80)


if __name__ == "__main__":
    main()
