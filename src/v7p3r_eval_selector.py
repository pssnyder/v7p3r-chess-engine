#!/usr/bin/env python3
"""
V7P3R Evaluation Profile Selector

Smart profile builder that selects evaluation modules based on position context.

This is the "brain" of the modular evaluation system - it decides which
evaluations to run based on time pressure, material balance, game phase,
and tactical considerations.

Author: Pat Snyder
Created: 2025-12-26 (v18.2 Modular Evaluation System - Day 3)
"""

from typing import List, Set
from dataclasses import dataclass
import chess

from v7p3r_position_context import (
    PositionContext, GamePhase, MaterialBalance, TacticalFlags
)
from v7p3r_eval_modules import (
    EvaluationModule, MODULE_REGISTRY, get_module, is_module_relevant,
    get_desperate_modules, get_emergency_modules
)


class EvaluationProfile:
    """Named evaluation profile with module list"""
    
    DESPERATE = "DESPERATE"         # Down material, tactics only
    EMERGENCY = "EMERGENCY"         # Time pressure, essentials only
    FAST = "FAST"                   # Fast time control, skip expensive
    TACTICAL = "TACTICAL"           # Tactical position, emphasize tactics
    ENDGAME = "ENDGAME"            # Endgame, focus on technique
    COMPREHENSIVE = "COMPREHENSIVE" # Full evaluation


@dataclass
class SelectedProfile:
    """
    Profile selection result with reasoning.
    
    This is what gets returned to the search engine.
    """
    name: str                          # Profile name (DESPERATE, EMERGENCY, etc.)
    modules: List[EvaluationModule]    # Active modules for this profile
    module_count: int                  # Number of active modules
    reason: str                        # Why this profile was chosen
    estimated_cost_ms: float           # Estimated evaluation time per node


class EvaluationProfileSelector:
    """
    Selects optimal evaluation profile based on position context.
    
    Priority order:
    1. DESPERATE: Down 300+ cp (tactics only, recover material)
    2. EMERGENCY: Time pressure < 30s (essentials only)
    3. FAST: Fast time control < 5s/move (skip expensive modules)
    4. TACTICAL: High tactical activity (emphasize tactics)
    5. ENDGAME: Endgame phase (technique focus)
    6. COMPREHENSIVE: Default full evaluation
    
    Each profile filters MODULE_REGISTRY based on context.
    """
    
    def select_profile(self, context: PositionContext) -> SelectedProfile:
        """
        Main entry point: Select evaluation profile for this position.
        
        Args:
            context: Position context from PositionContextCalculator
            
        Returns:
            SelectedProfile with modules and reasoning
        """
        # PRIORITY 1: DESPERATE (down material - tactical recovery)
        if context.material_diff_cp < -300:  # Down 3+ pawns
            return self._build_desperate_profile(context)
        
        # PRIORITY 2: EMERGENCY (time pressure)
        if context.time_pressure:  # < 30 seconds
            return self._build_emergency_profile(context)
        
        # PRIORITY 3: FAST (fast time control)
        if context.use_fast_profile:  # < 5s per move
            return self._build_fast_profile(context)
        
        # PRIORITY 4: TACTICAL (high tactical activity)
        if self._is_tactical_position(context):
            return self._build_tactical_profile(context)
        
        # PRIORITY 5: ENDGAME (endgame technique)
        if context.game_phase in {GamePhase.ENDGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX}:
            return self._build_endgame_profile(context)
        
        # DEFAULT: COMPREHENSIVE (full evaluation)
        return self._build_comprehensive_profile(context)
    
    def _build_desperate_profile(self, context: PositionContext) -> SelectedProfile:
        """
        DESPERATE profile: Down significant material, need tactical recovery.
        
        Strategy:
        - ONLY tactical modules (captures, checks, threats)
        - Skip ALL strategic evaluations
        - Goal: Find forcing moves to regain material
        
        Modules: 10 (material, hanging, captures, checks, pins, SEE, traps, safety)
        """
        modules = get_desperate_modules()
        
        # Filter by context (some modules may not be relevant)
        modules = [m for m in modules if is_module_relevant(m, context)]
        
        material_deficit = abs(context.material_diff_cp)
        
        return SelectedProfile(
            name=EvaluationProfile.DESPERATE,
            modules=modules,
            module_count=len(modules),
            reason=f"Down {material_deficit}cp - tactical recovery mode",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_emergency_profile(self, context: PositionContext) -> SelectedProfile:
        """
        EMERGENCY profile: Time pressure, minimize computation.
        
        Strategy:
        - Absolute essentials only
        - No move generation, no expensive checks
        - Fast enough to avoid time forfeit
        
        Modules: 5 (material, PST, basic king safety, hanging, safety)
        """
        modules = get_emergency_modules()
        
        return SelectedProfile(
            name=EvaluationProfile.EMERGENCY,
            modules=modules,
            module_count=len(modules),
            reason=f"Time pressure ({context.time_remaining:.1f}s remaining)",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_fast_profile(self, context: PositionContext) -> SelectedProfile:
        """
        FAST profile: Fast time control, skip expensive modules.
        
        Strategy:
        - Include important evaluations
        - Skip HIGH cost modules (SEE, full mobility)
        - Target: 4.0+ depth in blitz
        
        Modules: 12-18 (essentials + important, skip expensive)
        """
        # Get all relevant modules
        all_modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Filter: Skip HIGH cost modules
        from v7p3r_eval_modules import EvaluationCost
        modules = [m for m in all_modules if m.cost != EvaluationCost.HIGH]
        
        return SelectedProfile(
            name=EvaluationProfile.FAST,
            modules=modules,
            module_count=len(modules),
            reason=f"Fast time control ({context.time_per_move:.1f}s/move)",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_tactical_profile(self, context: PositionContext) -> SelectedProfile:
        """
        TACTICAL profile: High tactical activity, emphasize tactics.
        
        Strategy:
        - Include all tactical modules
        - Include strategic if relevant
        - Emphasize king safety, hanging pieces, threats
        
        Modules: 18-22 (tactical focus + strategic context)
        """
        # Get all relevant modules
        all_modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Ensure tactical modules are included
        tactical_names = [
            "hanging_pieces", "check_threats", "pins_forks_skewers",
            "see_evaluation", "trapped_pieces", "back_rank_threats",
            "king_safety_complex", "capture_priority"
        ]
        
        # Add tactical modules that aren't already included
        for name in tactical_names:
            module = get_module(name)
            if module and is_module_relevant(module, context):
                if module not in all_modules:
                    all_modules.append(module)
        
        tactical_flags = ", ".join([f.value for f in context.tactical_flags])
        
        return SelectedProfile(
            name=EvaluationProfile.TACTICAL,
            modules=all_modules,
            module_count=len(all_modules),
            reason=f"Tactical position ({tactical_flags or 'multiple threats'})",
            estimated_cost_ms=self._estimate_cost(all_modules)
        )
    
    def _build_endgame_profile(self, context: PositionContext) -> SelectedProfile:
        """
        ENDGAME profile: Endgame technique and precision.
        
        Strategy:
        - Endgame-specific modules (king centralization, opposition, square of pawn)
        - Skip opening/middlegame modules (development, center control)
        - Emphasize king activity and pawn races
        
        Modules: 10-15 (endgame technique focus)
        """
        # Get all relevant modules (phase filtering already applied)
        modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Ensure endgame modules are prioritized
        endgame_names = [
            "king_centralization", "opposition", "square_of_pawn", 
            "endgame_tables", "passed_pawns"
        ]
        
        for name in endgame_names:
            module = get_module(name)
            if module and is_module_relevant(module, context):
                if module not in modules:
                    modules.insert(0, module)  # Prioritize at front
        
        return SelectedProfile(
            name=EvaluationProfile.ENDGAME,
            modules=modules,
            module_count=len(modules),
            reason=f"{context.game_phase.value} - endgame technique",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_comprehensive_profile(self, context: PositionContext) -> SelectedProfile:
        """
        COMPREHENSIVE profile: Full evaluation, no restrictions.
        
        Strategy:
        - Include all relevant modules
        - Use for long time controls and complex middlegames
        - Maximum accuracy, don't worry about speed
        
        Modules: 20-28 (filtered by relevance)
        """
        # Get all relevant modules
        modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        return SelectedProfile(
            name=EvaluationProfile.COMPREHENSIVE,
            modules=modules,
            module_count=len(modules),
            reason=f"Full evaluation ({context.game_phase.value})",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _is_tactical_position(self, context: PositionContext) -> bool:
        """
        Detect if position has high tactical activity.
        
        Indicators:
        - King exposed (attack opportunities)
        - Multiple tactical flags active
        - Material imbalance (tactics to convert/recover)
        
        Returns:
            True if tactical profile should be used
        """
        # King safety issues = tactical
        if context.king_safety_critical:
            return True
        
        # Multiple tactical flags = tactical
        if len(context.tactical_flags) >= 2:
            return True
        
        # Material imbalance (but not desperate) = tactical conversion
        if context.material_balance in {MaterialBalance.ADVANTAGE, MaterialBalance.WINNING}:
            # We're winning, use tactics to convert
            return True
        
        if context.material_balance in {MaterialBalance.SLIGHT_ADVANTAGE}:
            # Small advantage, tactical opportunities
            if context.tactical_flags:
                return True
        
        return False
    
    def _estimate_cost(self, modules: List[EvaluationModule]) -> float:
        """
        Estimate total evaluation cost per node.
        
        Based on module cost metadata:
        - NEGLIGIBLE: 0.05ms
        - LOW: 0.2ms
        - MEDIUM: 1.0ms
        - HIGH: 3.0ms
        
        Args:
            modules: List of active modules
            
        Returns:
            Estimated milliseconds per node evaluation
        """
        from v7p3r_eval_modules import EvaluationCost
        
        cost_map = {
            EvaluationCost.NEGLIGIBLE: 0.05,
            EvaluationCost.LOW: 0.2,
            EvaluationCost.MEDIUM: 1.0,
            EvaluationCost.HIGH: 3.0
        }
        
        total = sum(cost_map.get(m.cost, 0.5) for m in modules)
        return round(total, 2)
    
    def get_dynamic_threefold_threshold(self, context: PositionContext) -> int:
        """
        Calculate dynamic threefold repetition threshold based on material balance.
        
        Philosophy:
        - Equal position: Never accept draw (0cp threshold)
        - Slight advantage: Very reluctant (10cp)
        - Advantage: Somewhat reluctant (15cp)
        - Winning: Only avoid if crushing (25cp)
        - Crushing: Avoid repetition unless forced (50cp)
        
        This replaces the fixed 100cp threshold that caused v18.2 draw issues.
        
        Args:
            context: Position context
            
        Returns:
            Threshold in centipawns (avoid repetition if eval > threshold)
        """
        if context.material_balance == MaterialBalance.EQUAL:
            return 0  # Never accept draw from equal position
        
        elif context.material_balance == MaterialBalance.SLIGHT_ADVANTAGE:
            return 10  # Very aggressive, avoid draws
        
        elif context.material_balance == MaterialBalance.ADVANTAGE:
            return 15  # Still aggressive
        
        elif context.material_balance == MaterialBalance.WINNING:
            return 25  # Only avoid if truly winning
        
        elif context.material_balance == MaterialBalance.CRUSHING:
            return 50  # Can afford to repeat if not completely crushing
        
        return 0  # Default: never accept draw


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_evaluation_profile(context: PositionContext) -> SelectedProfile:
    """
    Convenience function: Select profile for position context.
    
    Args:
        context: Position context from PositionContextCalculator
        
    Returns:
        SelectedProfile with active modules
    """
    selector = EvaluationProfileSelector()
    return selector.select_profile(context)


def get_threefold_threshold(context: PositionContext) -> int:
    """
    Convenience function: Get dynamic threefold threshold.
    
    Args:
        context: Position context
        
    Returns:
        Threshold in centipawns
    """
    selector = EvaluationProfileSelector()
    return selector.get_dynamic_threefold_threshold(context)


# =============================================================================
# DEBUGGING & ANALYSIS
# =============================================================================

def print_profile_details(profile: SelectedProfile):
    """Print detailed profile information"""
    print(f"\n=== Evaluation Profile: {profile.name} ===")
    print(f"Reason: {profile.reason}")
    print(f"Modules: {profile.module_count}")
    print(f"Estimated cost: {profile.estimated_cost_ms:.2f}ms per node")
    print(f"\nActive modules:")
    
    by_criticality = {}
    for module in profile.modules:
        crit = module.criticality.value
        if crit not in by_criticality:
            by_criticality[crit] = []
        by_criticality[crit].append(module.name)
    
    for crit in ["essential", "important", "situational", "optional"]:
        if crit in by_criticality:
            print(f"\n  {crit.upper()}:")
            for name in sorted(by_criticality[crit]):
                print(f"    - {name}")


if __name__ == "__main__":
    # Test profile selection on different positions
    import chess
    from v7p3r_position_context import PositionContextCalculator
    
    calculator = PositionContextCalculator()
    selector = EvaluationProfileSelector()
    
    test_positions = [
        ("Starting position", chess.Board(), 300.0, 10.0),
        ("Time pressure", chess.Board(), 15.0, 2.0),
        ("Down a queen", chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"), 300.0, 10.0),
        ("Endgame", chess.Board("8/8/8/8/8/3r4/4P3/4K2R w - - 0 1"), 300.0, 10.0),
        ("King exposed", chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), 300.0, 10.0),
    ]
    
    for name, board, time_rem, time_per in test_positions:
        print(f"\n{'='*60}")
        print(f"Position: {name}")
        print(f"FEN: {board.fen()}")
        
        context = calculator.calculate(board, time_rem, time_per)
        profile = selector.select_profile(context)
        
        print_profile_details(profile)
        
        threshold = selector.get_dynamic_threefold_threshold(context)
        print(f"\nThreefold threshold: {threshold}cp")
