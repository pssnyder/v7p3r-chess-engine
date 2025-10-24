#!/usr/bin/env python3
"""
V13.x Focused Move Ordering System
Dramatically reduces move volume by focusing only on critical moves

Priority System: Checks, Captures, Attacks, Threats, Position
Based on V12.6 weakness analysis: 75% bad moves not in top 5, 70% tactical misses

Philosophy:
- Most positions have 3-8 critical moves that matter
- Quiet moves only when forced (no captures/checks/threats)
- Separate "waiting moves" list for zugzwang/time situations
- Drastically reduce search tree branching factor
"""

import chess
import sys
import os
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class V13MoveScore:
    """Scoring system for V13.x move prioritization"""
    base_score: int = 0
    tactical_bonus: int = 0
    safety_bonus: int = 0
    development_bonus: int = 0
    total_score: int = 0
    
    def calculate_total(self):
        self.total_score = self.base_score + self.tactical_bonus + self.safety_bonus + self.development_bonus

class V13FocusedMoveOrderer:
    """V13.x focused move ordering system - critical moves only"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320, 
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # V13.x Priority Thresholds - Aggressive Pruning
        self.CRITICAL_MOVE_THRESHOLD = 400    # Moves that must be searched
        self.IMPORTANT_MOVE_THRESHOLD = 200   # Moves worth considering
        self.QUIET_MOVE_THRESHOLD = 100       # Only if nothing else available
        self.MAX_CRITICAL_MOVES = 6          # Maximum moves to search in complex positions
        
        # Performance tracking
        self.pruning_stats = {
            'total_legal_moves': 0,
            'critical_moves_selected': 0,
            'quiet_moves_pruned': 0,
            'pruning_rate': 0.0
        }
    
    def order_moves_v13_focused(self, board: chess.Board, legal_moves: List[chess.Move], 
                                depth: int = 0, tt_move: Optional[chess.Move] = None) -> Tuple[List[chess.Move], List[chess.Move]]:
        """
        V13.x focused move ordering - returns (critical_moves, waiting_moves)
        
        Returns:
            critical_moves: High-priority moves that should be searched
            waiting_moves: Quiet moves for zugzwang/waiting situations
        """
        if len(legal_moves) <= 2:
            return legal_moves, []
        
        self.pruning_stats['total_legal_moves'] += len(legal_moves)
        
        # Detect position characteristics
        position_info = self._analyze_position_v13(board)
        
        # Score all moves
        move_scores = []
        for move in legal_moves:
            score = self._score_move_v13(board, move, position_info)
            move_scores.append((score, move))
        
        # Sort by total score
        move_scores.sort(key=lambda x: x[0].total_score, reverse=True)
        
        # Separate into critical and waiting moves
        critical_moves = []
        waiting_moves = []
        
        # Always include TT move if available
        if tt_move and tt_move in legal_moves:
            critical_moves.append(tt_move)
            # Remove from list to avoid duplicates
            move_scores = [(s, m) for s, m in move_scores if m != tt_move]
        
        for score, move in move_scores:
            if score.total_score >= self.CRITICAL_MOVE_THRESHOLD:
                critical_moves.append(move)
            elif score.total_score >= self.IMPORTANT_MOVE_THRESHOLD and len(critical_moves) < 4:
                # Include important moves but limit to 4 total
                critical_moves.append(move)
            elif score.total_score >= self.QUIET_MOVE_THRESHOLD and len(critical_moves) < 2:
                # Only include lower-tier moves if we have very few critical moves
                critical_moves.append(move)
            else:
                # Everything else goes to waiting list
                waiting_moves.append(move)
        
        # Enforce maximum critical moves limit for complex positions
        if len(critical_moves) > self.MAX_CRITICAL_MOVES:
            # Move excess critical moves to waiting moves
            excess_moves = critical_moves[self.MAX_CRITICAL_MOVES:]
            critical_moves = critical_moves[:self.MAX_CRITICAL_MOVES]
            waiting_moves = excess_moves + waiting_moves
        
        # Ensure we have at least 2 moves to search (unless fewer legal moves)
        while len(critical_moves) < min(2, len(legal_moves)) and waiting_moves:
            critical_moves.append(waiting_moves.pop(0))
        
        # Update stats
        self.pruning_stats['critical_moves_selected'] += len(critical_moves)
        self.pruning_stats['quiet_moves_pruned'] += len(waiting_moves)
        
        if len(legal_moves) > 0:
            self.pruning_stats['pruning_rate'] = len(waiting_moves) / len(legal_moves) * 100
        
        return critical_moves, waiting_moves
    
    def _analyze_position_v13(self, board: chess.Board) -> Dict:
        """Analyze position characteristics for V13.x move ordering"""
        info = {
            'in_check': board.is_check(),
            'hanging_pieces': self._detect_hanging_pieces(board),
            'attacking_pieces': self._detect_attacking_pieces(board),
            'game_phase': self._determine_game_phase(board),
            'material_balance': self._calculate_material_balance(board),
            'king_safety_issues': self._detect_king_safety_issues(board),
            'piece_count': len(board.piece_map())
        }
        return info
    
    def _score_move_v13(self, board: chess.Board, move: chess.Move, position_info: Dict) -> V13MoveScore:
        """V13.x focused move scoring based on weakness analysis"""
        score = V13MoveScore()
        
        # PRIORITY 1: CHECKS (highest priority - V12.6 missed these)
        if self._gives_check_safe(board, move):
            score.base_score = 1000
            score.tactical_bonus = 500
            
            # Bonus for checkmate
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                score.tactical_bonus = 10000
        
        # PRIORITY 2: KING SAFETY (save king from attacks)
        elif self._saves_king_safety(board, move, position_info):
            score.base_score = 800
            score.safety_bonus = 400
        
        # PRIORITY 3: PIECE SAFETY (save hanging pieces - 27.7% miss rate in V12.6)
        elif self._saves_hanging_piece(board, move, position_info):
            score.base_score = 700
            score.safety_bonus = 300
        
        # PRIORITY 4: GOOD CAPTURES (MVV-LVA with safety check)
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                
                # Only good captures get high scores
                if victim_value >= attacker_value:
                    score.base_score = 600
                    score.tactical_bonus = (victim_value - attacker_value) // 10
                    
                    # Check if capture is safe
                    if self._is_capture_safe(board, move):
                        score.safety_bonus = 100
                    else:
                        score.safety_bonus = -200  # Penalize unsafe captures
                else:
                    # Bad captures get very low scores
                    score.base_score = 50
                    score.tactical_bonus = -100
        
        # PRIORITY 5: TACTICAL THREATS (pins, forks, skewers)
        elif self._creates_tactical_threat(board, move):
            score.base_score = 500
            score.tactical_bonus = 200
        
        # PRIORITY 6: PIECE DEVELOPMENT (opening/early middlegame)
        elif self._is_development_move(board, move, position_info):
            if position_info['game_phase'] == 'opening':
                score.base_score = 300  # Reduced from 400
                score.development_bonus = 100  # Reduced from 200
            else:
                score.base_score = 80   # Reduced from 150
                score.development_bonus = 20   # Reduced from 50
        
        # PRIORITY 7: CASTLING
        elif board.is_castling(move):
            score.base_score = 350
            score.safety_bonus = 200
        
        # PRIORITY 8: PAWN PROMOTIONS
        elif move.promotion == chess.QUEEN:
            score.base_score = 300
            score.tactical_bonus = 500
        elif move.promotion:
            score.base_score = 200
            score.tactical_bonus = 100
        
        # PRIORITY 9: CENTER CONTROL (early game)
        elif self._controls_center(board, move) and position_info['game_phase'] == 'opening':
            score.base_score = 80   # Reduced from 120
            score.development_bonus = 20  # Reduced from 80
        
        # EVERYTHING ELSE: Quiet moves (heavily penalized)
        else:
            score.base_score = 5    # Reduced from 10
            # Small bonus for improving piece position
            if self._improves_piece_position(board, move):
                score.development_bonus = 10  # Reduced from 20
        
        # Apply position-specific bonuses/penalties
        self._apply_position_modifiers_v13(score, board, move, position_info)
        
        score.calculate_total()
        return score
    
    def _gives_check_safe(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move gives check and is relatively safe"""
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.is_check()
    
    def _saves_king_safety(self, board: chess.Board, move: chess.Move, position_info: Dict) -> bool:
        """Check if move improves king safety"""
        if not position_info['king_safety_issues']:
            return False
        
        # Simple check: does move get king out of immediate danger
        king_square = board.king(board.turn)
        if king_square and move.from_square == king_square:
            # King move - check if it goes to safer square
            return not self._is_square_attacked(board, move.to_square, not board.turn)
        
        # Check if move blocks attack on king
        return self._blocks_attack_on_king(board, move)
    
    def _saves_hanging_piece(self, board: chess.Board, move: chess.Move, position_info: Dict) -> bool:
        """Check if move saves a hanging piece"""
        if not position_info['hanging_pieces']:
            return False
        
        return move.from_square in position_info['hanging_pieces']
    
    def _is_capture_safe(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if capture is safe (doesn't lose material)"""
        # Simple check: is the destination square defended after capture?
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if our piece on destination is attacked
        attackers = board_copy.attackers(not board.turn, move.to_square)
        return len(attackers) == 0
    
    def _creates_tactical_threat(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates tactical threats (pins, forks, etc.)"""
        board_copy = board.copy()
        board_copy.push(move)
        
        # Look for pieces that can attack multiple targets
        attacking_piece = board_copy.piece_at(move.to_square)
        if not attacking_piece:
            return False
        
        targets = 0
        high_value_targets = 0
        
        for square in chess.SQUARES:
            if board_copy.is_attacked_by(attacking_piece.color, square):
                target = board_copy.piece_at(square)
                if target and target.color != attacking_piece.color:
                    targets += 1
                    if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                        high_value_targets += 1
        
        # Fork: attacking 2+ pieces, or 1+ high-value pieces
        return targets >= 2 or high_value_targets >= 1
    
    def _is_development_move(self, board: chess.Board, move: chess.Move, position_info: Dict) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Knights and bishops from starting squares
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            start_rank = chess.square_rank(move.from_square)
            if piece.color == chess.WHITE and start_rank == 0:
                return True
            elif piece.color == chess.BLACK and start_rank == 7:
                return True
        
        return False
    
    def _controls_center(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move controls center squares"""
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        
        # Check if move attacks center squares
        board_copy = board.copy()
        board_copy.push(move)
        
        for center_sq in center_squares:
            if board_copy.is_attacked_by(board.turn, center_sq):
                return True
        
        return False
    
    def _improves_piece_position(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if quiet move improves piece position"""
        # Very simple heuristic: move towards center
        center_files = [3, 4]  # D and E files (0-indexed)
        center_ranks = [3, 4]  # 0-indexed, so ranks 4-5
        
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        
        return to_file in center_files or to_rank in center_ranks
    
    def _detect_hanging_pieces(self, board: chess.Board) -> Set[chess.Square]:
        """Detect pieces that are hanging (undefended or underdefended)"""
        hanging = set()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = len(board.attackers(not piece.color, square))
                defenders = len(board.attackers(piece.color, square))
                
                if attackers > 0 and attackers > defenders:
                    hanging.add(square)
        
        return hanging
    
    def _detect_attacking_pieces(self, board: chess.Board) -> Set[chess.Square]:
        """Detect pieces that are actively attacking enemy pieces"""
        attacking = set()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # Check if this piece attacks enemy pieces
                for target_sq in chess.SQUARES:
                    if board.is_attacked_by(piece.color, target_sq):
                        target = board.piece_at(target_sq)
                        if target and target.color != piece.color:
                            attacking.add(square)
                            break
        
        return attacking
    
    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine game phase"""
        piece_count = len(board.piece_map())
        
        if piece_count >= 28:
            return 'opening'
        elif piece_count <= 10:
            return 'endgame'
        else:
            return 'middlegame'
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance"""
        white_material = sum(self.piece_values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self.piece_values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        return white_material - black_material
    
    def _detect_king_safety_issues(self, board: chess.Board) -> bool:
        """Detect if king is in danger"""
        king_square = board.king(board.turn)
        if not king_square:
            return False
        
        # Check if king is attacked or will be attacked
        return len(board.attackers(not board.turn, king_square)) > 0
    
    def _is_square_attacked(self, board: chess.Board, square: chess.Square, by_color: chess.Color) -> bool:
        """Check if square is attacked by given color"""
        return len(board.attackers(by_color, square)) > 0
    
    def _blocks_attack_on_king(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move blocks an attack on the king"""
        # Simplified: check if move interposes between attacker and king
        king_square = board.king(board.turn)
        if not king_square:
            return False
        
        # This would need more sophisticated ray-tracing logic
        # For now, just check if move destination is between common attack lines
        return False  # Placeholder
    
    def _apply_position_modifiers_v13(self, score: V13MoveScore, board: chess.Board, 
                                     move: chess.Move, position_info: Dict):
        """Apply position-specific score modifiers"""
        
        # In check: prioritize getting out of check
        if position_info['in_check']:
            if not board.is_capture(move) and not self._gives_check_safe(board, move):
                score.safety_bonus += 100
        
        # Material imbalance: adjust tactics vs positional play
        if abs(position_info['material_balance']) > 300:
            if board.is_capture(move) or self._gives_check_safe(board, move):
                score.tactical_bonus += 50
        
        # Endgame: prioritize king activity and pawn promotion
        if position_info['game_phase'] == 'endgame':
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                score.development_bonus += 80
            elif piece and piece.piece_type == chess.PAWN:
                score.development_bonus += 60
    
    def print_move_ordering_analysis(self, board: chess.Board, legal_moves: List[chess.Move]):
        """Print detailed analysis of V13.x move ordering"""
        critical_moves, waiting_moves = self.order_moves_v13_focused(board, legal_moves)
        
        print(f"\n{'='*60}")
        print(f"🚀 V13.x FOCUSED MOVE ORDERING ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\n📊 PRUNING STATISTICS:")
        print(f"  Total legal moves: {len(legal_moves)}")
        print(f"  Critical moves selected: {len(critical_moves)}")
        print(f"  Waiting moves (quiet): {len(waiting_moves)}")
        print(f"  Pruning rate: {len(waiting_moves)/len(legal_moves)*100:.1f}%")
        print(f"  Search reduction: {(1 - len(critical_moves)/len(legal_moves))*100:.1f}%")
        
        print(f"\n🎯 CRITICAL MOVES (will be searched):")
        for i, move in enumerate(critical_moves[:10], 1):
            score = self._score_move_v13(board, move, self._analyze_position_v13(board))
            categories = self._categorize_move_v13(board, move)
            print(f"  {i:2d}. {move.uci()} ({board.san(move):8s}) {score.total_score:4d} - {', '.join(categories[:2])}")
        
        print(f"\n💤 WAITING MOVES (for zugzwang/time):")
        for i, move in enumerate(waiting_moves[:5], 1):
            print(f"  {i:2d}. {move.uci()} ({board.san(move):8s}) - Quiet")
        
        if len(waiting_moves) > 5:
            print(f"  ... and {len(waiting_moves) - 5} more quiet moves")
        
        # Analysis summary
        position_info = self._analyze_position_v13(board)
        print(f"\n🔍 POSITION ANALYSIS:")
        print(f"  Game phase: {position_info['game_phase']}")
        print(f"  In check: {position_info['in_check']}")
        print(f"  Hanging pieces: {len(position_info['hanging_pieces'])}")
        print(f"  King safety issues: {position_info['king_safety_issues']}")
        print(f"  Material balance: {position_info['material_balance']}")
        
        return critical_moves, waiting_moves
    
    def _categorize_move_v13(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Categorize move for display purposes"""
        categories = []
        
        if self._gives_check_safe(board, move):
            categories.append("Check")
        if board.is_capture(move):
            categories.append("Capture")
        if board.is_castling(move):
            categories.append("Castle")
        if move.promotion:
            categories.append("Promotion")
        if self._is_development_move(board, move, self._analyze_position_v13(board)):
            categories.append("Development")
        if self._creates_tactical_threat(board, move):
            categories.append("Tactical")
        
        if not categories:
            categories.append("Quiet")
        
        return categories


def test_v13_focused_ordering():
    """Test V13.x focused move ordering on various positions"""
    orderer = V13FocusedMoveOrderer()
    
    test_positions = {
        "Opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Middlegame Complex": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        "Tactical Position": "r2qkb1r/pp2nppp/3p1n2/2pP4/2P1P3/2N2N2/PP3PPP/R1BQKB1R w KQq - 0 6",
        "Hanging Piece": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQKBNR w KQkq - 4 4",
        "Endgame": "6k1/8/6K1/8/8/8/r7/8 w - - 0 1"
    }
    
    print("🚀 V13.x FOCUSED MOVE ORDERING - PRUNING TEST")
    print("Fixing V12.6 weaknesses: 75% bad move ordering, 38% weakness rate")
    print("="*80)
    
    total_original = 0
    total_critical = 0
    
    for position_name, fen in test_positions.items():
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        print(f"\n📍 TESTING: {position_name}")
        print(f"FEN: {fen}")
        
        critical_moves, waiting_moves = orderer.print_move_ordering_analysis(board, legal_moves)
        
        total_original += len(legal_moves)
        total_critical += len(critical_moves)
        
        print("\n" + "="*40)
    
    # Overall statistics
    pruning_rate = (1 - total_critical / total_original) * 100
    
    print(f"\n🎯 OVERALL V13.x PRUNING PERFORMANCE:")
    print(f"Total legal moves across all positions: {total_original}")
    print(f"Total critical moves selected: {total_critical}")
    print(f"Average pruning rate: {pruning_rate:.1f}%")
    print(f"Search tree reduction: ~{pruning_rate:.0f}% fewer nodes")
    print(f"Expected speedup: {100/(100-pruning_rate):.1f}x")


if __name__ == "__main__":
    test_v13_focused_ordering()