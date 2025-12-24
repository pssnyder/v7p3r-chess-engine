# Manual Stockfish Analysis Notes - V7P3R v18.0

## Game 1: EKQVz3G5 - WHITE LOSS vs vanibot (1985)
**Opening:** Dutch Defense: Alapin Variation (A80)
**Time Control:** 600+2 (Rapid)
**Termination:** Checkmate on move 22
**Final Position:** Qxd2# (Black checkmates)

### Move-by-Move Analysis
Let's analyze critical positions with Stockfish.

#### Opening Phase (Moves 1-10)
```
1. d4 f5 2. Qd3 d5 3. Nf3 e6 4. Ne5 Nd7 5. Nc3 Nxe5 6. dxe5 c5 
7. Bf4 g5 8. Be3 d4 9. O-O-O Bd7 10. Bxd4 cxd4
```

**Position after 10...cxd4:**
FEN: r2qkbnr/pp1b3p/4p3/4Pp2/3p4/2NQ4/PPP2PPP/2KR1BNR w - - 0 11

**Position after 15. f4 Nc4:**
FEN: r2qkb1r/pp1b3p/4p3/4Pnp1/2nqpP2/2NQ4/PPP3PP/2KR1BNR w - - 0 16

**Position after 19. b3 Na3+:**
FEN: r2q1b1r/pp1bkp1p/4p3/4Pnp1/3q4/nP1Q4/P1P2PPP/2KR1BNR w - - 0 20

**CRITICAL POSITION - Move 21...Qe3+:**
FEN: r5kr/pp1b1p1p/4p3/4Pnp1/8/1b1qQ3/P1PR1PPP/1K3BNR w - - 0 22

**FINAL POSITION - Move 22...Qxd2#:**
FEN: r5kr/pp1b1p1p/4p3/4Pnp1/8/1b1q4/P1PQ1PPP/1K3BNR w - - 0 23

### Questions for Stockfish Analysis:
1. Was the early queen sortie (2. Qd3) sound?
2. What was White's best defense after 19...Na3+?
3. Could White have avoided the checkmate pattern?
4. At what move did the position become lost?

---

## Stockfish Command Template:
```cmd
cd "S:\Programming\Chess Engines\Tournament Engines\downloaded_engines\stockfish"
stockfish-windows-x86-64-avx2.exe

# Then in Stockfish:
position fen [FEN_STRING_HERE]
go depth 20
```

Ready to analyze positions manually. Let me know which position you'd like to start with.
