"""
PV Stability Diagnostic - Track Principal Variation Changes Across Depths

Uses v17.3's enhanced UCI output to diagnose when/why the engine changes moves.
This will reveal if quiescence search is contradicting regular search decisions.

Key Questions:
1. Does PV change frequently as depth increases?
2. When PV changes, is seldepth (quiescence depth) also jumping?
3. Are evaluation swings correlated with quiescence extensions?
4. Do certain position types show more instability?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
import time
from collections import defaultdict

class PVStabilityAnalyzer:
    """Analyzes PV stability across iterative deepening"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.analysis_results = []
        
    def analyze_position(self, fen, description, max_depth=8):
        """
        Analyze a single position across multiple depths
        
        Args:
            fen: Position to analyze
            description: Human-readable description
            max_depth: Maximum depth to search
        
        Returns:
            Dictionary with depth-by-depth PV data
        """
        print(f"\n{'='*80}")
        print(f"Position: {description}")
        print(f"FEN: {fen}")
        print(f"{'='*80}\n")
        
        board = chess.Board(fen)
        
        depth_data = []
        previous_pv = None
        pv_changes = []
        
        # Search iteratively from depth 1 to max_depth
        for target_depth in range(1, max_depth + 1):
            # Create fresh engine instance for each depth to avoid TT contamination
            self.engine = V7P3REngine()
            self.engine.board = board.copy()
            
            start_time = time.time()
            
            # Perform search to this depth
            best_move = self.engine.search(board.copy(), time_limit=30.0)
            
            elapsed = time.time() - start_time
            
            # Extract full PV from TT
            pv = self._extract_pv_from_tt(board)
            
            # Get statistics
            tt_size = len(self.engine.transposition_table)
            hashfull = min(999, int((tt_size / 50000) * 1000))  # Per mille
            
            # Get score from TT
            key = hash(board.fen())
            score = 0.0
            if key in self.engine.transposition_table:
                score = self.engine.transposition_table[key]['score'] / 100.0
            
            # Calculate NPS
            nps = int(self.engine.nodes_searched / elapsed) if elapsed > 0 else 0
            
            depth_info = {
                'depth': target_depth,
                'seldepth': self.engine.seldepth,
                'score': score,
                'nodes': self.engine.nodes_searched,
                'time_ms': int(elapsed * 1000),
                'nps': nps,
                'hashfull': hashfull,
                'best_move': best_move.uci() if best_move else 'none',
                'pv': pv,
                'pv_string': ' '.join(pv[:6]) if pv else ''  # First 6 moves
            }
            
            # Check for PV change
            if previous_pv and len(pv) > 0 and len(previous_pv) > 0:
                if pv[0] != previous_pv[0]:  # First move changed
                    score_change = score - depth_data[-1]['score']
                    seldepth_change = self.engine.seldepth - depth_data[-1]['seldepth']
                    nodes_ratio = self.engine.nodes_searched / depth_data[-1]['nodes'] if depth_data[-1]['nodes'] > 0 else 0
                    
                    pv_change = {
                        'at_depth': target_depth,
                        'from_move': previous_pv[0],
                        'to_move': pv[0],
                        'score_change': score_change,
                        'seldepth_change': seldepth_change,
                        'nodes_ratio': nodes_ratio
                    }
                    pv_changes.append(pv_change)
                    
                    print(f"\n⚠️  PV CHANGE at depth {target_depth}:")
                    print(f"    {previous_pv[0]} → {pv[0]}")
                    print(f"    Score Δ: {score_change:+.2f}")
                    print(f"    Seldepth Δ: {seldepth_change:+d}")
                    print(f"    Nodes ratio: {nodes_ratio:.2f}x")
            
            # Print UCI-style info line
            print(f"depth {depth_info['depth']:2d} | "
                  f"seldepth {depth_info['seldepth']:2d} | "
                  f"score {int(score * 100):+5d} | "
                  f"nodes {depth_info['nodes']:7d} | "
                  f"nps {depth_info['nps']:7d} | "
                  f"hashfull {depth_info['hashfull']:3d} | "
                  f"pv {depth_info['pv_string']}")
            
            depth_data.append(depth_info)
            previous_pv = pv
        
        # Analysis
        analysis = self._analyze_stability(depth_data, pv_changes, description)
        
        result = {
            'description': description,
            'fen': fen,
            'depth_data': depth_data,
            'pv_changes': pv_changes,
            'analysis': analysis
        }
        
        self.analysis_results.append(result)
        return result
    
    def _extract_pv_from_tt(self, board, max_length=10):
        """Extract PV from transposition table"""
        pv = []
        board_copy = board.copy()
        
        for _ in range(max_length):
            key = hash(board_copy.fen())
            
            if key not in self.engine.transposition_table:
                break
            
            entry = self.engine.transposition_table[key]
            
            if entry['best_move'] is None:
                break
            
            try:
                pv.append(entry['best_move'].uci())
                board_copy.push(entry['best_move'])
                
                if board_copy.is_game_over():
                    break
            except:
                break  # Invalid move in TT
        
        return pv
    
    def _analyze_stability(self, depth_data, pv_changes, description):
        """Analyze PV stability patterns"""
        
        print(f"\n{'-'*80}")
        print("STABILITY ANALYSIS:")
        print(f"{'-'*80}")
        
        analysis = {
            'total_depths': len(depth_data),
            'pv_changes': len(pv_changes),
            'stability_score': 0.0,
            'issues': []
        }
        
        # Calculate stability score (0-100, higher = more stable)
        if len(depth_data) > 1:
            stability_score = 100 * (1 - (len(pv_changes) / (len(depth_data) - 1)))
            analysis['stability_score'] = stability_score
            
            print(f"\nPV Stability Score: {stability_score:.1f}/100")
            
            if stability_score < 50:
                print("  🔴 UNSTABLE: Frequent move changes")
                analysis['issues'].append('high_instability')
            elif stability_score < 75:
                print("  🟡 MODERATE: Some move changes")
                analysis['issues'].append('moderate_instability')
            else:
                print("  🟢 STABLE: Consistent move selection")
        
        # Analyze PV changes
        if pv_changes:
            print(f"\nPV Changes Detected: {len(pv_changes)}")
            
            for i, change in enumerate(pv_changes, 1):
                print(f"\n  Change {i} at depth {change['at_depth']}:")
                print(f"    Move: {change['from_move']} → {change['to_move']}")
                print(f"    Score Δ: {change['score_change']:+.2f}")
                print(f"    Seldepth Δ: {change['seldepth_change']:+d}")
                print(f"    Nodes ratio: {change['nodes_ratio']:.2f}x")
                
                # Diagnose likely cause
                if abs(change['seldepth_change']) >= 3:
                    print(f"    ⚠️  QUIESCENCE IMPACT: Large seldepth change")
                    analysis['issues'].append('quiescence_interference')
                
                if abs(change['score_change']) > 0.5:
                    print(f"    ⚠️  EVALUATION SWING: Significant score change")
                    analysis['issues'].append('evaluation_instability')
                
                if change['nodes_ratio'] > 3.0:
                    print(f"    ⚠️  SEARCH EXPLOSION: Node count tripled")
                    analysis['issues'].append('search_explosion')
        else:
            print("\n✅ NO PV CHANGES: Move remained consistent across all depths")
        
        # Analyze seldepth progression
        seldepths = [d['seldepth'] for d in depth_data]
        depths = [d['depth'] for d in depth_data]
        
        print(f"\nQuiescence Depth Progression:")
        depth_line = "  Depth:    " + ' '.join(f"{d:2d}" for d in depths)
        seldepth_line = "  Seldepth: " + ' '.join(f"{s:2d}" for s in seldepths)
        print(depth_line)
        print(seldepth_line)
        
        # Calculate quiescence overhead
        avg_quiescence_overhead = sum(s - d for s, d in zip(seldepths, depths)) / len(depths)
        print(f"  Average quiescence overhead: {avg_quiescence_overhead:.1f} plies")
        
        if avg_quiescence_overhead > 4:
            print(f"  ⚠️  HIGH QUIESCENCE OVERHEAD: >4 plies on average")
            analysis['issues'].append('excessive_quiescence')
        
        # Analyze score progression
        scores = [d['score'] for d in depth_data]
        score_variance = max(scores) - min(scores)
        print(f"\nScore Progression:")
        print(f"  Scores: {[f'{s:+.2f}' for s in scores]}")
        print(f"  Variance: {score_variance:.2f}")
        
        if score_variance > 1.0:
            print(f"  ⚠️  HIGH SCORE VARIANCE: >{1.0}")
            analysis['issues'].append('high_score_variance')
        
        return analysis
    
    def print_summary_report(self):
        """Print summary of all positions analyzed"""
        
        print("\n" + "="*80)
        print("PV STABILITY SUMMARY REPORT")
        print("="*80)
        
        total_positions = len(self.analysis_results)
        total_changes = sum(len(r['pv_changes']) for r in self.analysis_results)
        
        print(f"\nPositions Analyzed: {total_positions}")
        print(f"Total PV Changes: {total_changes}")
        print(f"Average Changes per Position: {total_changes / total_positions:.1f}")
        
        # Issue frequency
        issue_counts = defaultdict(int)
        for result in self.analysis_results:
            for issue in result['analysis']['issues']:
                issue_counts[issue] += 1
        
        if issue_counts:
            print(f"\nIssue Frequency:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue}: {count}/{total_positions} ({100*count/total_positions:.1f}%)")
        
        # Stability distribution
        stability_scores = [r['analysis']['stability_score'] for r in self.analysis_results]
        avg_stability = sum(stability_scores) / len(stability_scores)
        
        print(f"\nAverage Stability Score: {avg_stability:.1f}/100")
        
        stable_count = sum(1 for s in stability_scores if s >= 75)
        moderate_count = sum(1 for s in stability_scores if 50 <= s < 75)
        unstable_count = sum(1 for s in stability_scores if s < 50)
        
        print(f"  🟢 Stable (≥75):     {stable_count}/{total_positions}")
        print(f"  🟡 Moderate (50-74): {moderate_count}/{total_positions}")
        print(f"  🔴 Unstable (<50):   {unstable_count}/{total_positions}")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print(f"{'='*80}")
        
        if 'quiescence_interference' in issue_counts and issue_counts['quiescence_interference'] > total_positions * 0.3:
            print("\n⚠️  QUIESCENCE INTERFERENCE DETECTED")
            print("   High correlation between seldepth jumps and PV changes.")
            print("   Recommendation: Reduce quiescence search depth or restrict stand-pat.")
        
        if 'excessive_quiescence' in issue_counts and issue_counts['excessive_quiescence'] > total_positions * 0.5:
            print("\n⚠️  EXCESSIVE QUIESCENCE DEPTH")
            print("   Quiescence is extending too deeply (>4 plies average overhead).")
            print("   Recommendation: Reduce MAX_QUIESCENCE_DEPTH from 10 to 4-6.")
        
        if 'evaluation_instability' in issue_counts and issue_counts['evaluation_instability'] > total_positions * 0.3:
            print("\n⚠️  EVALUATION INSTABILITY")
            print("   Large score swings (>0.5) when PV changes.")
            print("   Recommendation: Use simplified evaluation in quiescence search.")
        
        if avg_stability < 60:
            print("\n⚠️  LOW OVERALL STABILITY")
            print("   Engine changes moves frequently as depth increases.")
            print("   Recommendation: Investigate move ordering and TT replacement strategy.")
        
        if unstable_count == 0 and avg_stability > 85:
            print("\n✅ EXCELLENT STABILITY")
            print("   PV remains consistent across depths. Search is working well.")

def main():
    """Run PV stability analysis on critical positions"""
    
    analyzer = PVStabilityAnalyzer()
    
    print("\n" + "="*80)
    print("V7P3R v17.3 PV STABILITY DIAGNOSTIC")
    print("Testing: Does quiescence search contradict regular search?")
    print("="*80)
    
    # Test Position 1: Tactical position (Italian Game)
    analyzer.analyze_position(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        description="Tactical - Italian Game position",
        max_depth=7
    )
    
    # Test Position 2: Discovered Attack (100% puzzle theme success)
    analyzer.analyze_position(
        fen="r1bq1rk1/ppp2ppp/2n5/3np3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 8",
        description="Discovered Attack - Should be stable tactical line",
        max_depth=7
    )
    
    # Test Position 3: Zugzwang (60.5% puzzle success - known weakness)
    analyzer.analyze_position(
        fen="8/8/p7/1p6/1P6/P7/8/k1K5 w - - 0 1",
        description="Zugzwang - Pawn endgame (known weakness area)",
        max_depth=7
    )
    
    # Test Position 4: Complex middlegame with multiple candidate moves
    analyzer.analyze_position(
        fen="r2qkb1r/ppp2ppp/2n5/3pP3/3Pn1b1/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        description="Complex Middlegame - French Defense tactical position",
        max_depth=7
    )
    
    # Test Position 5: Quiet positional position
    analyzer.analyze_position(
        fen="rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
        description="Quiet Positional - Queen's Gambit Declined",
        max_depth=7
    )
    
    # Test Position 6: Forcing tactical sequence (mate threat)
    analyzer.analyze_position(
        fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        description="Forcing Tactics - Back rank mate in 2",
        max_depth=7
    )
    
    # Print summary report
    analyzer.print_summary_report()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
