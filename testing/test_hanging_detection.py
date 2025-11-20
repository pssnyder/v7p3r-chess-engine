#!/usr/bin/env python3
"""
Debug hanging piece detection
"""

import chess

# Position: White knight on d5, attacked by black pieces, not defended
board = chess.Board("rnbqkbnr/pppppppp/8/3N4/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")

knight_square = chess.D5
piece = board.piece_at(knight_square)

print("Position: Knight on d5")
print(board)
print()

print(f"Piece at d5: {piece}")
print(f"Piece color: {piece.color}")
print()

# Check attacks
attacked_by_black = board.is_attacked_by(chess.BLACK, knight_square)
defended_by_white = board.is_attacked_by(chess.WHITE, knight_square)

print(f"Attacked by BLACK: {attacked_by_black}")
print(f"Defended by WHITE: {defended_by_white}")
print()

# List attackers
black_attackers = board.attackers(chess.BLACK, knight_square)
white_defenders = board.attackers(chess.WHITE, knight_square)

print(f"Black attackers: {[chess.square_name(sq) for sq in black_attackers]}")
print(f"White defenders: {[chess.square_name(sq) for sq in white_defenders]}")
print()

# Check logic
is_hanging = attacked_by_black and not defended_by_white
print(f"Is hanging? {is_hanging}")
print()

# Now test Qxh7 position
print("="*60)
print()

board2 = chess.Board("r1bqkbnr/ppppp1p1/2n2p2/8/1nP5/5N2/PPQP1PPP/RNB1KB1R w KQkq - 0 1")
board2.push(chess.Move.from_uci('c2h7'))

print("Position: After Qxh7")
print(board2)
print()

queen_square = chess.H7
queen = board2.piece_at(queen_square)

print(f"Piece at h7: {queen}")
print(f"Piece color: {queen.color}")
print()

attacked_by_black = board2.is_attacked_by(chess.BLACK, queen_square)
defended_by_white = board2.is_attacked_by(chess.WHITE, queen_square)

print(f"Attacked by BLACK: {attacked_by_black}")
print(f"Defended by WHITE: {defended_by_white}")
print()

black_attackers = board2.attackers(chess.BLACK, queen_square)
white_defenders = board2.attackers(chess.WHITE, queen_square)

print(f"Black attackers: {[chess.square_name(sq) for sq in black_attackers]}")
print(f"White defenders: {[chess.square_name(sq) for sq in white_defenders]}")
print()

is_hanging = attacked_by_black and not defended_by_white
print(f"Is queen hanging? {is_hanging}")
