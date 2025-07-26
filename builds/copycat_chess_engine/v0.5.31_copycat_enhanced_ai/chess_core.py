# chess_core.py - V0.5.30 Copycat + Evaluation AI
# Combines neural network move prediction with evaluation-based selection

import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from evaluation_engine import EvaluationEngine

class ChessDataset(Dataset):
    def __init__(self, pgn_path, username=None, max_games=100):
        self.positions = []
        self.moves = []
        games_processed = 0
        
        try:
            pgn = open(pgn_path, encoding='utf-8', errors='ignore')
        except:
            pgn = open(pgn_path, encoding='latin-1', errors='ignore')
            
        while games_processed < max_games:
            try:
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                
                # Determine training strategy based on players
                white_player = game.headers.get("White", "")
                black_player = game.headers.get("Black", "")
                has_v7p3r = "v7p3r" in [white_player, black_player]
                
                # Skip games with no moves
                if not list(game.mainline_moves()):
                    continue
                
                # Training logic:
                # 1. If v7p3r is playing, ONLY train on v7p3r's moves
                # 2. If v7p3r is NOT playing, train on ALL moves (master games knowledge)
                board = game.board()
                for move in game.mainline_moves():
                    collect_move = False
                    
                    if has_v7p3r:
                        # v7p3r is in this game - only collect v7p3r's moves
                        if (board.turn == chess.WHITE and white_player == "v7p3r") or \
                           (board.turn == chess.BLACK and black_player == "v7p3r"):
                            collect_move = True
                    else:
                        # No v7p3r in this game - collect all moves (chess knowledge)
                        collect_move = True
                    
                    if collect_move:
                        self.positions.append(self.board_to_tensor(board))
                        self.moves.append(move.uci())
                    
                    board.push(move)
                
                # Increment games processed counter after successfully processing a game
                games_processed += 1
                    
            except Exception as e:
                # Skip problematic games and continue
                continue
        
        pgn.close()

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        return tensor

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

class ChessAI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256*8*8, num_classes)  # Dynamic output size
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 256*8*8)
        return self.fc(x)
    
    def get_top_moves(self, board, move_to_index, index_to_move, device, top_k=5):
        """Get top-k move candidates with their confidence scores"""
        # Convert board to tensor
        position_tensor = torch.FloatTensor(self.board_to_tensor(board)).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self(position_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(index_to_move)))
        
        candidates = []
        legal_moves = set(move.uci() for move in board.legal_moves)
        
        for i in range(top_k):
            if i < len(top_indices[0]):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                move_uci = index_to_move.get(idx, "unknown")
                
                # Only include legal moves
                if move_uci in legal_moves:
                    try:
                        move_obj = chess.Move.from_uci(move_uci)
                        candidates.append({
                            'move': move_obj,
                            'uci': move_uci,
                            'neural_confidence': prob
                        })
                    except:
                        continue
        
        return candidates
    
    def board_to_tensor(self, board):
        """Convert chess board to tensor format (same as training)"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        return tensor

class CopycatEvaluationAI:
    """
    Enhanced copycat AI that combines neural network move prediction 
    with evaluation engine-based move selection
    """
    
    def __init__(self, model_path, vocab_path, device):
        self.device = device
        
        # Load move vocabulary
        import pickle
        with open(vocab_path, "rb") as f:
            self.move_to_index = pickle.load(f)
        self.index_to_move = {idx: move for move, idx in self.move_to_index.items()}
        
        # Load neural network model
        self.model = ChessAI(len(self.move_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.eval()
        
    def select_best_move(self, board, top_k=5, debug=False):
        """
        Select the best move by:
        1. Getting top-k moves from neural network (copycat preferences)
        2. Evaluating each candidate with the evaluation engine
        3. Selecting the move with the best evaluation score
        """
        
        # Step 1: Get neural network move candidates
        candidates = self.model.get_top_moves(
            board, self.move_to_index, self.index_to_move, self.device, top_k
        )
        
        if not candidates:
            # Fallback to random legal move if no candidates
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return np.random.choice(legal_moves)
            return None
        
        # Step 2: Evaluate each candidate move
        evaluator = EvaluationEngine(board)
        
        for candidate in candidates:
            # Make the move temporarily
            board_copy = board.copy()
            board_copy.push(candidate['move'])
            
            # Evaluate the resulting position
            candidate['eval_score'] = evaluator.evaluate_position()
        
        # Step 3: Select best move based on evaluation
        # Sort by evaluation score (higher is better)
        candidates.sort(key=lambda x: x['eval_score'], reverse=True)
        best_candidate = candidates[0]
        
        if debug:
            print(f"\n🤖 Move Selection Analysis:")
            print(f"   {'Move':<8} {'Neural':<8} {'Eval':<8} {'Final'}")
            print(f"   {'-'*8:<8} {'-'*8:<8} {'-'*8:<8} {'-'*8}")
            for i, c in enumerate(candidates):
                marker = "👑" if i == 0 else "  "
                print(f"{marker} {c['uci']:<8} {c['neural_confidence']:.3f}    {c['eval_score']:.2f}    {'BEST' if i == 0 else ''}")
        
        return best_candidate['move']
    
    def get_move_analysis(self, board, top_k=5):
        """Get detailed analysis of move selection for debugging"""
        candidates = self.model.get_top_moves(
            board, self.move_to_index, self.index_to_move, self.device, top_k
        )
        
        if not candidates:
            return {"error": "No valid candidates found"}
        
        evaluator = EvaluationEngine(board)
        
        for candidate in candidates:
            board_copy = board.copy()
            board_copy.push(candidate['move'])
            candidate['eval_score'] = evaluator.evaluate_position()
        
        candidates.sort(key=lambda x: x['eval_score'], reverse=True)
        
        return {
            "best_move": candidates[0]['move'],
            "analysis": candidates
        }
