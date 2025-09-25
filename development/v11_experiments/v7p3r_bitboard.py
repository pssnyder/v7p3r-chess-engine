# This file contains the complete bitboard representation and move generation logic
# for pawns and rooks. This is designed as a single, self-contained example for study.

# -----------------------------------------------------------
# 1. BOARD REPRESENTATION AND CONSTANTS
# -----------------------------------------------------------

# We represent the board as a single 64-bit integer. Each bit
# corresponds to a square on the board, using a little-endian
# rank-file mapping (A1=0, H8=63).

# Mask bitboards for ranks (rows) and files (columns).
# These are constant and never change. We use them to check conditions.
RANK_2_MASK = 0x000000000000FF00
RANK_7_MASK = 0x00FF000000000000
A_FILE_MASK = 0x0101010101010101
H_FILE_MASK = 0x8080808080808080

# Pre-calculated attack tables for sliding pieces.
# This is a key trick to avoid generating moves on the fly for every single square.
# We'll calculate these once and use them for quick lookups.
ROOK_ATTACKS = [0] * 64

# A function to initialize the attack tables.
def init_attack_tables():
    for square in range(64):
        # A bitboard representing the squares the rook on this square could attack
        # on an empty board.
        # This will be used as a base for our move generation.
        ROOK_ATTACKS[square] = generate_rook_attacks(square)

def generate_rook_attacks(square):
    """
    Generates a bitboard of all squares a rook can attack from a given square,
    assuming an empty board. This is a static, pre-calculated bitboard.
    """
    attacks = 0
    rank = square // 8
    file = square % 8

    # Rank attacks (right)
    for i in range(file + 1, 8):
        attacks |= (1 << (rank * 8 + i))
    # Rank attacks (left)
    for i in range(file - 1, -1, -1):
        attacks |= (1 << (rank * 8 + i))
    
    # File attacks (up)
    for i in range(rank + 1, 8):
        attacks |= (1 << (i * 8 + file))
    # File attacks (down)
    for i in range(rank - 1, -1, -1):
        attacks |= (1 << (i * 8 + file))
        
    return attacks


# Call the initialization function once.
init_attack_tables()


# -----------------------------------------------------------
# 2. MOVE GENERATION FUNCTIONS
# -----------------------------------------------------------

def get_white_pawn_moves(white_pawns, occupied_squares, black_pawns):
    """
    Generates a bitboard of all legal moves for white pawns.
    This logic remains the same from our previous conversation.
    """
    
    # 1. Generate single-square forward moves.
    single_moves = (white_pawns << 8) & ~occupied_squares

    # 2. Generate two-square forward moves (only from the starting rank).
    two_moves = ((white_pawns & RANK_2_MASK) << 16) & ~occupied_squares & (single_moves << 8)

    # 3. Generate diagonal attacks.
    left_attacks = (white_pawns << 7) & ~H_FILE_MASK & black_pawns
    right_attacks = (white_pawns << 9) & ~A_FILE_MASK & black_pawns
    
    # 4. Combine all legal moves into a single bitboard.
    all_moves = single_moves | two_moves | left_attacks | right_attacks

    return all_moves

def get_rook_moves(rook_bitboard, occupied_squares, friendly_squares):
    """
    Generates all legal moves for a rook. This is where we solve the
    multi-attack vector and blocking piece problems.
    """
    all_rook_moves = 0
    
    # Iterate through all squares where a rook is present.
    # The '>>' operator is used to find the location of the '1's in the bitboard.
    while rook_bitboard:
        # Isolate the least significant bit (the first rook we find).
        from_square = (rook_bitboard & -rook_bitboard).bit_length() - 1
        
        # Get all potential attacks on an empty board.
        attacks = ROOK_ATTACKS[from_square]
        
        # Intersect with all occupied squares to find the blockers.
        blockers = attacks & occupied_squares
        
        # Remove the blockers and squares beyond them to find the legal moves.
        # This is where the real "magic" happens.
        # We perform four separate operations for each of the four directions a rook can move.
        
        # To the right
        right_blocker = (blockers << 1) & A_FILE_MASK
        if right_blocker:
            attacks &= ~((right_blocker << 1) - 1)
        
        # To the left
        left_blocker = (blockers >> 1) & H_FILE_MASK
        if left_blocker:
            attacks &= (left_blocker << 1) - 1
        
        # Up
        up_blocker = (blockers << 8) & RANK_7_MASK
        if up_blocker:
            attacks &= ~((up_blocker << 8) - 1)
        
        # Down
        down_blocker = (blockers >> 8) & RANK_2_MASK
        if down_blocker:
            attacks &= (down_blocker << 8) - 1
        
        # The resulting 'attacks' bitboard now only contains legal moves.
        # We combine these moves with our master moves bitboard.
        all_rook_moves |= attacks
        
        # Remove the rook we just processed and continue the loop.
        rook_bitboard &= rook_bitboard - 1
    
    return all_rook_moves

def main():
    """
    A simple example demonstrating the move generation functions.
    """
    print("Welcome to the V7P3R Bitboard Engine.")
    print("-----------------------------------")
    
    # Example 1: Pawn move generation
    white_pawns = RANK_2_MASK
    black_pawns = RANK_7_MASK
    occupied = white_pawns | black_pawns
    pawn_moves = get_white_pawn_moves(white_pawns, occupied, black_pawns)
    print(f"\nCalculated all legal starting white pawn moves: {hex(pawn_moves)}")
    
    # Example 2: Rook move generation
    # Imagine a rook on A1 (square 0) with a blocker on A5 (square 32).
    rook_a1 = 1 << 0
    friendly_pieces = rook_a1
    
    # An imaginary blocker (a friendly pawn) on A5
    blocker_a5 = 1 << 32
    occupied_rooks = rook_a1 | blocker_a5
    
    rook_moves = get_rook_moves(rook_a1, occupied_rooks, friendly_pieces)
    print(f"\nCalculated rook moves from A1 blocked by a piece on A5: {hex(rook_moves)}")
    print(f"This bitboard represents moves to A2, A3, and A4.")

if __name__ == "__main__":
    main()
