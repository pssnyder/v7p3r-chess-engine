#!/usr/bin/env python3

"""
V14.3 UCI Logging System

Proper UCI protocol implementation with debugging support:
- Production UCI info messages (always sent)
- Debug info messages (only in debug mode)
- Proper UCI currmove/currmovenumber reporting
- Search progress and statistics
"""

import sys
import time
from typing import Optional, List
import chess

class UCILogger:
    """
    UCI Protocol compliant logging system
    
    Handles:
    - Production UCI info messages
    - Debug logging (enabled via UCI debug option)
    - Search progress reporting
    - Move statistics
    """
    
    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled
        self.search_start_time = None
        self.nodes_searched = 0
        
    def set_debug(self, enabled: bool):
        """Enable/disable debug logging via UCI setoption"""
        self.debug_enabled = enabled
        
    def start_search(self):
        """Mark start of search for timing"""
        self.search_start_time = time.time()
        self.nodes_searched = 0
        
    def update_nodes(self, nodes: int):
        """Update node count"""
        self.nodes_searched = nodes
        
    def info_depth(self, depth: int, score_cp: int, nodes: int, time_ms: int, 
                   nps: int, pv: List[str], selective_depth: Optional[int] = None):
        """
        Standard UCI info depth message (PRODUCTION - always sent)
        
        Args:
            depth: Search depth completed
            score_cp: Score in centipawns  
            nodes: Nodes searched
            time_ms: Time taken in milliseconds
            nps: Nodes per second
            pv: Principal variation moves
            selective_depth: Selective search depth (optional)
        """
        pv_str = " ".join(pv) if pv else ""
        
        info_parts = [
            f"info depth {depth}",
            f"score cp {score_cp}",
            f"nodes {nodes}",
            f"time {time_ms}",
            f"nps {nps}",
        ]
        
        if selective_depth:
            info_parts.append(f"seldepth {selective_depth}")
            
        if pv_str:
            info_parts.append(f"pv {pv_str}")
            
        print(" ".join(info_parts))
        sys.stdout.flush()
        
    def info_currmove(self, move: str, move_number: int):
        """
        UCI currmove info (PRODUCTION - for GUI move display)
        
        Args:
            move: Current move being searched
            move_number: Move number in search order
        """
        print(f"info currmove {move} currmovenumber {move_number}")
        sys.stdout.flush()
        
    def info_string(self, message: str, debug_only: bool = False):
        """
        UCI info string message
        
        Args:
            message: String message to send
            debug_only: If True, only send when debug is enabled
        """
        if debug_only and not self.debug_enabled:
            return
            
        print(f"info string {message}")
        sys.stdout.flush()
        
    def info_score_bound(self, score_cp: int, bound_type: str, depth: int):
        """
        UCI score with bounds (PRODUCTION)
        
        Args:
            score_cp: Score in centipawns
            bound_type: "lowerbound" or "upperbound"
            depth: Search depth
        """
        print(f"info depth {depth} score cp {score_cp} {bound_type}")
        sys.stdout.flush()
        
    def info_nodes(self, nodes: int):
        """Simple nodes update (PRODUCTION)"""
        print(f"info nodes {nodes}")
        sys.stdout.flush()
        
    def info_time(self, time_ms: int):
        """Simple time update (PRODUCTION)"""
        print(f"info time {time_ms}")
        sys.stdout.flush()
        
    def info_nps(self, nps: int):
        """Nodes per second info (PRODUCTION)"""
        print(f"info nps {nps}")
        sys.stdout.flush()
        
    def info_hashfull(self, permille: int):
        """
        Hash table fill percentage (PRODUCTION)
        
        Args:
            permille: Fill level in permille (0-1000, where 1000 = 100%)
        """
        print(f"info hashfull {permille}")
        sys.stdout.flush()
        
    # Debug-only logging methods
    def debug_search_start(self, time_limit: float, target_depth: int):
        """Debug: Search initialization"""
        if self.debug_enabled:
            self.info_string(f"Search started: {time_limit:.1f}s limit, target depth {target_depth}", debug_only=True)
            
    def debug_emergency_stop(self, reason: str, elapsed: float, limit: float):
        """Debug: Emergency time controls"""
        if self.debug_enabled:
            percentage = (elapsed / limit) * 100 if limit > 0 else 0
            self.info_string(f"EMERGENCY STOP: {reason} - {elapsed:.3f}s ({percentage:.1f}% of limit)", debug_only=True)
            
    def debug_iteration_complete(self, depth: int, time_taken: float, nodes: int):
        """Debug: Iteration completion"""
        if self.debug_enabled:
            nps = int(nodes / max(time_taken, 0.001))
            self.info_string(f"Depth {depth} complete: {time_taken:.3f}s, {nodes} nodes, {nps} nps", debug_only=True)
            
    def debug_time_allocation(self, target: float, max_time: float, time_limit: float):
        """Debug: Time management"""
        if self.debug_enabled:
            self.info_string(f"Time allocation: {target:.1f}s target, {max_time:.1f}s max ({time_limit:.1f}s limit)", debug_only=True)
            
    def debug_game_phase(self, phase: str, material: int, moves: int):
        """Debug: Game phase detection"""
        if self.debug_enabled:
            self.info_string(f"Game phase: {phase} (material: {material}, moves: {moves})", debug_only=True)
            
    def debug_move_ordering(self, total_moves: int, tt_moves: int, captures: int, checks: int):
        """Debug: Move ordering statistics"""
        if self.debug_enabled:
            self.info_string(f"Move ordering: {total_moves} total ({tt_moves} TT, {captures} captures, {checks} checks)", debug_only=True)
            
    def debug_search_stats(self, nodes: int, tt_hits: int, cache_hits: int, cutoffs: int):
        """Debug: Search statistics"""
        if self.debug_enabled:
            tt_rate = (tt_hits / max(nodes, 1)) * 100
            cache_rate = (cache_hits / max(nodes, 1)) * 100
            cutoff_rate = (cutoffs / max(nodes, 1)) * 100
            self.info_string(f"Stats: TT {tt_rate:.1f}%, Cache {cache_rate:.1f}%, Cutoffs {cutoff_rate:.1f}%", debug_only=True)
            
    def debug_quiescence_depth(self, max_depth: int, positions: int):
        """Debug: Quiescence search info"""
        if self.debug_enabled:
            self.info_string(f"Quiescence: max depth {max_depth}, {positions} positions", debug_only=True)
            
    def debug_evaluation_cache(self, size: int, hit_rate: float):
        """Debug: Evaluation cache performance"""
        if self.debug_enabled:
            self.info_string(f"Eval cache: {size} entries, {hit_rate:.1f}% hit rate", debug_only=True)
            
    # Search progress reporting
    def report_search_progress(self, current_depth: int, best_move: str, best_score: int, 
                             elapsed: float, nodes: int, pv: List[str]):
        """
        Complete search progress report (PRODUCTION)
        Combines multiple UCI info types for comprehensive reporting
        """
        elapsed_ms = int(elapsed * 1000)
        nps = int(nodes / max(elapsed, 0.001))
        
        # Main depth info with PV
        self.info_depth(current_depth, best_score, nodes, elapsed_ms, nps, pv)
        
        # Current best move info
        if best_move:
            self.info_string(f"Best move: {best_move} (depth {current_depth})")
            
    def final_search_summary(self, best_move: str, final_depth: int, total_time: float, 
                           total_nodes: int, time_limit: float):
        """
        Final search summary (mix of PRODUCTION and DEBUG)
        """
        time_percentage = (total_time / time_limit) * 100 if time_limit > 0 else 0
        nps = int(total_nodes / max(total_time, 0.001))
        
        # Production info
        self.info_string(f"Search complete: depth {final_depth}, {total_time:.3f}s")
        
        # Debug info  
        if self.debug_enabled:
            self.info_string(f"Final stats: {total_nodes} nodes, {nps} nps, {time_percentage:.1f}% time used", debug_only=True)
            
            if time_percentage > 90:
                self.info_string(f"WARNING: High time usage {time_percentage:.1f}%", debug_only=True)
                
    def bestmove(self, move: str, ponder_move: Optional[str] = None):
        """
        Send UCI bestmove (PRODUCTION - required)
        
        Args:
            move: Best move found
            ponder_move: Move to ponder (optional)
        """
        if ponder_move:
            print(f"bestmove {move} ponder {ponder_move}")
        else:
            print(f"bestmove {move}")
        sys.stdout.flush()
        
    def readyok(self):
        """UCI readyok response (PRODUCTION - required)"""
        print("readyok")
        sys.stdout.flush()
        
    def uciok(self):
        """UCI uciok response (PRODUCTION - required)"""
        print("uciok")
        sys.stdout.flush()
        
    def id_name(self, name: str):
        """UCI engine name (PRODUCTION - required)"""
        print(f"id name {name}")
        sys.stdout.flush()
        
    def id_author(self, author: str):
        """UCI engine author (PRODUCTION - required)"""
        print(f"id author {author}")
        sys.stdout.flush()
        
    def option(self, name: str, option_type: str, default=None, min_val=None, max_val=None):
        """
        UCI option declaration (PRODUCTION - required)
        
        Args:
            name: Option name
            option_type: "check", "spin", "combo", "button", "string"
            default: Default value
            min_val: Minimum value (for spin)
            max_val: Maximum value (for spin)
        """
        option_parts = [f"option name {name} type {option_type}"]
        
        if default is not None:
            option_parts.append(f"default {default}")
        if min_val is not None:
            option_parts.append(f"min {min_val}")
        if max_val is not None:
            option_parts.append(f"max {max_val}")
            
        print(" ".join(option_parts))
        sys.stdout.flush()