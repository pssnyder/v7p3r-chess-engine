"""
V7P3R v16.3 Enhanced Opening Repertoire
Deep center-control focused opening book (10-15 moves deep)

WHITE REPERTOIRE:
1. e4 → Italian Game (Giuoco Piano)
2. d4 → Queen's Gambit Declined  
3. Nf3 → King's Indian Attack

BLACK REPERTOIRE:
1. e4 → Sicilian Najdorf
2. d4 → King's Indian Defense
3. Alternatives → French Defense, Caro-Kann

Focus: Rapid development, center control, smooth middlegame transition
Target: Beat C0BR4 v3.2
"""

def get_enhanced_opening_book():
    """
    Returns comprehensive opening repertoire as dict mapping FEN -> [(move_uci, weight)]
    """
    return {
        # ========================================
        # STARTING POSITION
        # ========================================
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
            ("e2e4", 100),  # Primary: Italian Game path
            ("d2d4", 100),  # Secondary: QGD path
            ("g1f3", 50),   # Flexible: KIA path
        ],
        
        # ========================================
        # WHITE: ITALIAN GAME (after 1.e4)
        # ========================================
        
        # 1.e4 - Black's main responses
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
            ("e7e5", 100),  # Main line
            ("c7c5", 90),   # Sicilian
            ("e7e6", 70),   # French
            ("c7c6", 70),   # Caro-Kann
        ],
        
        # 1.e4 e5 - Italian Game setup
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
            ("g1f3", 100),  # Knight development
        ],
        
        # 1.e4 e5 2.Nf3
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
            ("b8c6", 100),  # Normal development
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 - Italian Game
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": [
            ("f1c4", 100),  # Italian Game - Giuoco Piano
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": [
            ("f8c5", 100),  # Giuoco Piano
            ("g8f6", 80),   # Two Knights
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 (Giuoco Piano)
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": [
            ("c2c3", 100),  # Main line - prepare d4
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4": [
            ("g8f6", 100),  # Most common
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 1 5": [
            ("d2d4", 100),  # Center break
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/2P2N2/PP3PPP/RNBQK2R b KQkq - 0 5": [
            ("e5d4", 100),  # Take center
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4
        "r1bqk2r/pppp1ppp/2n2n2/2b5/2BpP3/2P2N2/PP3PPP/RNBQK2R w KQkq - 0 6": [
            ("c3d4", 100),  # Recapture
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4
        "r1bqk2r/pppp1ppp/2n2n2/2b5/2BPP3/5N2/PP3PPP/RNBQK2R b KQkq - 0 6": [
            ("f8b4", 100),  # Pin knight
            ("d7d5", 80),   # Counter-attack
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+
        "r1bqk2r/pppp1ppp/2n2n2/8/1b1PP3/5N2/PP3PPP/RNBQK2R w KQkq - 1 7": [
            ("b1c3", 100),  # Block with development
        ],
        
        # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+ 7.Nc3
        "r1bqk2r/pppp1ppp/2n2n2/8/1b1PP3/2N2N2/PP3PPP/R1BQK2R b KQkq - 2 7": [
            ("b4c3", 80),   # Trade
            ("d7d5", 100),  # Counter
        ],
        
        # ========================================
        # WHITE: QUEEN'S GAMBIT DECLINED (after 1.d4)
        # ========================================
        
        # 1.d4 - Black's main responses
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
            ("d7d5", 100),  # QGD setup
            ("g8f6", 90),   # Indian systems
        ],
        
        # 1.d4 d5 - Queen's Gambit
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": [
            ("c2c4", 100),  # Queen's Gambit
        ],
        
        # 1.d4 d5 2.c4 - Black's choice
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": [
            ("e7e6", 100),  # QGD
            ("c7c6", 70),   # Slav
        ],
        
        # 1.d4 d5 2.c4 e6 (QGD)
        "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3": [
            ("b1c3", 100),  # Develop
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3
        "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3": [
            ("g8f6", 100),  # Normal development
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4": [
            ("c1g5", 100),  # Classical variation
            ("g1f3", 80),   # Flexible
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5
        "rnbqkb1r/ppp2ppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR b KQkq - 3 4": [
            ("f8e7", 100),  # Normal
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7
        "rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq - 4 5": [
            ("g1f3", 100),  # Complete development
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.Nf3
        "rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq - 5 5": [
            ("e8g8", 100),  # Castle
            ("h7h6", 70),   # Ask bishop
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.Nf3 O-O
        "rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R w KQ - 6 6": [
            ("e2e3", 100),  # Solid setup
        ],
        
        # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.Nf3 O-O 6.e3
        "rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ - 0 6": [
            ("b8d7", 100),  # Flexible development
        ],
        
        # ========================================
        # BLACK: SICILIAN NAJDORF (after 1.e4 c5)
        # ========================================
        
        # 1.e4 c5 - White's second move
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
            ("g1f3", 100),  # Open Sicilian
        ],
        
        # 1.e4 c5 2.Nf3 - Black's setup
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
            ("d7d6", 100),  # Najdorf setup
        ],
        
        # 1.e4 c5 2.Nf3 d6 - White pushes
        "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3": [
            ("d2d4", 100),  # Open it up
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 - Take center
        "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3": [
            ("c5d4", 100),  # Capture
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 cxd4
        "rnbqkbnr/pp2pppp/3p4/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4": [
            ("f3d4", 100),  # Recapture
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 - Najdorf
        "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4": [
            ("g8f6", 100),  # Develop knight
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6
        "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5": [
            ("b1c3", 100),  # Develop
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 - NAJDORF!
        "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5": [
            ("a7a6", 100),  # Najdorf move!
        ],
        
        # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 (Najdorf)
        "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6": [
            ("f1e2", 90),   # Classical
            ("c1g5", 80),   # Aggressive
            ("g2g3", 70),   # Fianchetto
        ],
        
        # ========================================
        # BLACK: KING'S INDIAN DEFENSE (after 1.d4 Nf6)
        # ========================================
        
        # 1.d4 Nf6 - White's second
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": [
            ("c2c4", 100),  # Most common
        ],
        
        # 1.d4 Nf6 2.c4 - King's Indian setup
        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": [
            ("g7g6", 100),  # King's Indian
        ],
        
        # 1.d4 Nf6 2.c4 g6
        "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3": [
            ("b1c3", 100),  # Natural development
        ],
        
        # 1.d4 Nf6 2.c4 g6 3.Nc3
        "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3": [
            ("f8g7", 100),  # Fianchetto
        ],
        
        # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7
        "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4": [
            ("e2e4", 100),  # Classical King's Indian
        ],
        
        # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4
        "rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq - 0 4": [
            ("d7d6", 100),  # Solid center
        ],
        
        # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6
        "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5": [
            ("g1f3", 100),  # Complete development
        ],
        
        # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3
        "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 1 5": [
            ("e8g8", 100),  # Castle kingside
        ],
        
        # ========================================
        # BLACK: FRENCH DEFENSE (after 1.e4 e6)
        # ========================================
        
        # 1.e4 e6 - White's second
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
            ("d2d4", 100),  # Main line
        ],
        
        # 1.e4 e6 2.d4 - French structure
        "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2": [
            ("d7d5", 100),  # Central tension
        ],
        
        # 1.e4 e6 2.d4 d5
        "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3": [
            ("b1c3", 100),  # Classical French
            ("e4d5", 70),   # Exchange French
        ],
        
        # 1.e4 e6 2.d4 d5 3.Nc3
        "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3": [
            ("f8b4", 90),   # Winawer
            ("g8f6", 100),  # Classical
        ],
        
        # 1.e4 e6 2.d4 d5 3.Nc3 Nf6 (Classical French)
        "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4": [
            ("c1g5", 100),  # Pin knight
        ],
        
        # ========================================
        # BLACK: CARO-KANN (after 1.e4 c6)
        # ========================================
        
        # 1.e4 c6 - White's second
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
            ("d2d4", 100),  # Main line
        ],
        
        # 1.e4 c6 2.d4 - Caro-Kann structure
        "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2": [
            ("d7d5", 100),  # Challenge center
        ],
        
        # 1.e4 c6 2.d4 d5
        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3": [
            ("b1c3", 100),  # Classical Caro
            ("e4d5", 70),   # Exchange Caro
        ],
        
        # 1.e4 c6 2.d4 d5 3.Nc3
        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3": [
            ("d5e4", 100),  # Classical line
        ],
        
        # 1.e4 c6 2.d4 d5 3.Nc3 dxe4
        "rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4": [
            ("c3e4", 100),  # Recapture
        ],
    }
