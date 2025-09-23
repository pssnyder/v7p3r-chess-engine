#!/usr/bin/env python3
"""
V11.5 Balanced Search Implementation
====================================

PROBLEM: Current fast search has poor tactical accuracy (33% vs 87%)
SOLUTION: Balanced approach - tactical analysis for critical positions only

STRATEGY:
1. Fast search for most positions (speed)
2. Full tactical analysis for critical positions (accuracy)
3. Smart detection of when tactics matter

Target: 3,000+ NPS with 75%+ tactical accuracy
"""

def create_balanced_search_replacement():
    """
    Balanced search that uses tactical analysis only when needed
    """
    
    balanced_search_code = '''
    def _is_tactical_position(self, board: chess.Board) -> bool:
        """
        Detect if position requires tactical analysis
        """
        # Always analyze if in check
        if board.is_check():
            return True
        
        # Analyze if there are hanging pieces (pieces under attack)
        for square, piece in board.piece_map().items():
            if piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                defenders = board.attackers(board.turn, square)
                if len(attackers) > len(defenders):
                    return True
        
        # Analyze if there are tactical patterns (pins, forks, etc.)
        # Check for pieces that can potentially be pinned or forked
        piece_squares = [sq for sq, piece in board.piece_map().items() 
                        if piece.color != board.turn and piece.piece_type in [chess.QUEEN, chess.ROOK]]
        
        if len(piece_squares) >= 2:
            # Multiple high-value pieces - potential for tactics
            return True
        
        # Analyze positions with few pieces (endgame tactics)
        total_pieces = len(board.piece_map())
        if total_pieces <= 12:
            return True
        
        return False
    
    def _balanced_recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        V11.5 BALANCED SEARCH - Speed + Tactical Accuracy
        """
        self.nodes_searched += 1
        
        # Time checking every 3000 nodes (balance between speed and responsiveness)
        if hasattr(self, 'search_start_time') and self.nodes_searched % 3000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                return self._evaluate_position(board, depth=0), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            score = self._evaluate_position(board, depth=0)
            return score, None
            
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (self.default_depth - search_depth), None
            else:
                return 0.0, None
        
        # 3. NULL MOVE PRUNING (keep for performance)
        if (search_depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board) and beta - alpha > 1):
            
            board.turn = not board.turn
            null_score, _ = self._balanced_recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # 4. SMART MOVE ORDERING - tactical analysis only when needed
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        # Determine if this position needs tactical analysis
        needs_tactical_analysis = self._is_tactical_position(board)
        
        if needs_tactical_analysis:
            # Use full tactical move ordering
            ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        else:
            # Use fast move ordering
            ordered_moves = self._order_moves_fast(board, legal_moves, tt_move)
        
        # 5. MAIN SEARCH LOOP
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # Simplified search - no LMR for now to maintain stability
            score, _ = self._balanced_recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
            score = -score
            
            board.pop()
            moves_searched += 1
            
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Update killer moves for pruning move
                if not board.is_capture(move) and not board.gives_check(move):
                    self.killer_moves.add_killer(search_depth, move)
                break
        
        # 6. TRANSPOSITION TABLE STORE
        self._store_transposition_table(board, search_depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves_fast(self, board: chess.Board, moves: List[chess.Move], tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        Fast move ordering without tactical analysis
        """
        tt_moves = []
        captures = []
        checks = []
        killers = []
        quiet = []
        
        killer_set = set(self.killer_moves.get_killers(1))  # Just get some killers
        
        for move in moves:
            # 1. TT move first
            if tt_move and move == tt_move:
                tt_moves.append(move)
            # 2. Captures with simple MVV-LVA
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                captures.append((victim_value, move))
            # 3. Checks
            elif board.gives_check(move):
                checks.append(move)
            # 4. Killer moves
            elif move in killer_set:
                killers.append(move)
            # 5. Quiet moves
            else:
                quiet.append(move)
        
        # Sort captures by victim value
        captures.sort(reverse=True, key=lambda x: x[0])
        
        # Combine in order
        ordered = []
        ordered.extend(tt_moves)
        ordered.extend([move for _, move in captures])
        ordered.extend(checks)
        ordered.extend(killers)
        ordered.extend(quiet)
        
        return ordered
    '''
    
    return balanced_search_code

if __name__ == "__main__":
    print("V11.5 Balanced Search Implementation")
    print("====================================")
    print()
    print("CURRENT RESULTS:")
    print("- Fast search: 5,278 NPS, 33% tactical accuracy")
    print("- Original v11.4: 300-600 NPS, 87% tactical accuracy")
    print()
    print("BALANCED APPROACH:")
    print("- Tactical analysis only for critical positions")
    print("- Fast ordering for simple positions")
    print("- Smart position detection")
    print()
    print("DETECTION CRITERIA:")
    print("- In check (always tactical)")
    print("- Hanging pieces detected")
    print("- Multiple high-value pieces (fork potential)")
    print("- Endgame positions (<=12 pieces)")
    print()
    print("TARGET:")
    print("- 3,000+ NPS (6x faster than v11.4)")
    print("- 75%+ tactical accuracy (retain most tactics)")
    print("- Best of both worlds approach")