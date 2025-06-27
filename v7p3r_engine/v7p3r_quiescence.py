


class v7p3rQuiescence:    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None, current_ply: int = 0) -> float:
        self.nodes_searched += 1

        if stop_callback and stop_callback(): # Pass current search depth and nodes
            return 0 # Or appropriate alpha/beta

        # Use self.current_player for perspective
        stand_pat_score = self.evaluate_position_from_perspective(board, self.current_player if maximizing_player else not self.current_player)

        if not maximizing_player: # Minimizing node (opponent's turn from perspective of self.current_player)
            stand_pat_score = -stand_pat_score # Adjust score if eval is always from white's POV

        if maximizing_player: # Current player (the one who initiated the search) is maximizing
            if stand_pat_score >= beta:
                return beta 
            alpha = max(alpha, stand_pat_score)
        else: # Opponent is minimizing (from current player's perspective)
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)

        # Consider only captures and promotions (and maybe checks)
        # Use a fixed quiescence max depth (quiescence is a boolean in config, not a dict)
        max_q_depth = 5  # Fixed reasonable quiescence depth
        if current_ply >= self.depth + max_q_depth: # self.depth is the main search depth
             return stand_pat_score


        capture_moves = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
        if not capture_moves and not board.is_check(): # If no captures and not in check, return stand_pat
             return stand_pat_score
        
        # If in check, all legal moves should be considered to escape check.
        if board.is_check():
            capture_moves = list(board.legal_moves)


        # Order capture moves (e.g., MVV-LVA or simple capture value)
        # For simplicity, not re-ordering here, but could be beneficial.
        # ordered_q_moves = self.order_moves(board, capture_moves, depth=current_ply) # Can reuse order_moves logic

        for move in capture_moves: # Potentially use ordered_q_moves
            board.push(move)
            score = self._quiescence_search(board, alpha, beta, not maximizing_player, stop_callback, current_ply + 1)
            board.pop()

            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break 
            else:
                beta = min(beta, score)
                if alpha >= beta:
                    break
        
        return alpha if maximizing_player else beta