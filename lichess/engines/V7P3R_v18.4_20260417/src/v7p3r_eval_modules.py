#!/usr/bin/env python3
"""
V7P3R Evaluation Module Registry

Metadata-driven evaluation components with selective activation.

This module defines ALL evaluation components from v18.2, each with:
- Cost: NEGLIGIBLE, LOW, MEDIUM, HIGH (node evaluation overhead)
- Criticality: ESSENTIAL, IMPORTANT, SITUATIONAL, OPTIONAL
- Required pieces: What must be on board for module to be relevant
- Required phases: When module should be active
- Dependencies: Other modules that must run first

Author: Pat Snyder
Created: 2025-12-26 (v18.2 Modular Evaluation System - Day 2)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set, List, Callable, Optional
import chess

from v7p3r_position_context import GamePhase, PositionContext, MaterialBalance


class EvaluationCost(Enum):
    """Computational cost per node evaluation"""
    NEGLIGIBLE = "negligible"  # < 0.1ms, O(1) lookups
    LOW = "low"                # 0.1-0.5ms, simple board scans
    MEDIUM = "medium"          # 0.5-2ms, move generation or complex logic
    HIGH = "high"              # > 2ms, heavy analysis (SEE, mobility, etc.)


class EvaluationCriticality(Enum):
    """How important module is to engine strength"""
    ESSENTIAL = "essential"        # Always needed (material, basic tactics)
    IMPORTANT = "important"        # Needed for competitive play (king safety, PST)
    SITUATIONAL = "situational"    # Helpful in specific positions (bishop pair, passed pawns)
    OPTIONAL = "optional"          # Nice-to-have (knight outposts, rook on 7th)


@dataclass
class EvaluationModule:
    """
    Metadata for a single evaluation component.
    
    Each module is a self-contained evaluation that can be toggled on/off
    based on position context.
    """
    name: str                           # Unique identifier
    description: str                    # Human-readable purpose
    cost: EvaluationCost               # Computational overhead
    criticality: EvaluationCriticality # Strategic importance
    
    # Activation conditions (when module is RELEVANT)
    required_pieces: Set[chess.PieceType] = None  # Must have these pieces
    required_phases: Set[GamePhase] = None        # Active in these phases
    skip_when_desperate: bool = False             # Skip when down material
    skip_in_time_pressure: bool = False           # Skip when < 30s
    
    # Dependencies
    depends_on: List[str] = None  # Other modules that must run first
    
    def __post_init__(self):
        """Initialize optional fields"""
        if self.required_pieces is None:
            self.required_pieces = set()
        if self.required_phases is None:
            self.required_phases = set()
        if self.depends_on is None:
            self.depends_on = []


# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULE_REGISTRY: List[EvaluationModule] = [
    
    # -------------------------------------------------------------------------
    # ESSENTIAL MODULES (Always needed)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="material_counter",
        description="Basic material counting (P=100, N=320, B=330, R=500, Q=900)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases=set(GamePhase),  # All phases
    ),
    
    EvaluationModule(
        name="piece_square_tables",
        description="Positional bonuses for piece placement (PST)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases=set(GamePhase),
    ),
    
    # -------------------------------------------------------------------------
    # DESPERATE MODULES (Only when down material - TACTICAL RECOVERY)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="hanging_pieces",
        description="Detect undefended pieces (captures without recapture)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,  # KEEP when desperate!
    ),
    
    EvaluationModule(
        name="capture_priority",
        description="Prioritize recaptures and material-winning captures",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,
    ),
    
    EvaluationModule(
        name="check_threats",
        description="Evaluate check-giving moves and mate threats",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,
    ),
    
    EvaluationModule(
        name="pins_forks_skewers",
        description="Tactical pattern detection (pins, forks, discovered attacks)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,
    ),
    
    # -------------------------------------------------------------------------
    # KING SAFETY (Critical in middlegame)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="king_safety_basic",
        description="Pawn shield and basic king exposure",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,  # Skip if down material
    ),
    
    EvaluationModule(
        name="king_safety_complex",
        description="Attack patterns, tropism, storm detection",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="king_centralization",
        description="King activity bonus in endgame",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE},
    ),
    
    # -------------------------------------------------------------------------
    # PAWN STRUCTURE
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="passed_pawns",
        description="Passed pawn bonuses (distance to promotion, king proximity)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="doubled_pawns",
        description="Penalty for doubled/tripled pawns",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="isolated_pawns",
        description="Penalty for isolated pawns (no friendly pawns on adjacent files)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="backward_pawns",
        description="Penalty for backward pawns (cannot advance safely)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.OPTIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="pawn_chains",
        description="Bonus for connected pawn chains",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    # -------------------------------------------------------------------------
    # PIECE-SPECIFIC EVALUATIONS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="bishop_pair",
        description="Bonus for having both bishops (powerful in open positions)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.BISHOP},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="knight_outposts",
        description="Bonus for knights on strong outpost squares",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.OPTIONAL,
        required_pieces={chess.KNIGHT},
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="rook_on_7th",
        description="Bonus for rook on 7th rank (attacking enemy pawns)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.ROOK},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="rook_on_open_file",
        description="Bonus for rook on open/semi-open file",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.ROOK},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="queen_mobility",
        description="Queen activity and mobility evaluation",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.QUEEN},
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # MOBILITY & ACTIVITY
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="piece_mobility",
        description="Count legal moves for all pieces (slow, accurate)",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="piece_activity",
        description="Simplified mobility (attacked squares, no move gen)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.SITUATIONAL,
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # POSITIONAL CONCEPTS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="center_control",
        description="Control of central squares (e4, d4, e5, d5)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="space_advantage",
        description="Territorial control (squares controlled in opponent's half)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.OPTIONAL,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="development",
        description="Piece development bonus (pieces off back rank)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.OPENING},
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # ENDGAME SPECIALIZATIONS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="opposition",
        description="King opposition in pawn endgames",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_SIMPLE},
        required_pieces={chess.PAWN},
    ),
    
    EvaluationModule(
        name="square_of_pawn",
        description="Can king catch passed pawn? (rule of square)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX},
        required_pieces={chess.PAWN},
    ),
    
    EvaluationModule(
        name="endgame_tables",
        description="Theoretical endgame knowledge (KQ vs K, KR vs K, etc.)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases={GamePhase.ENDGAME_SIMPLE},
    ),
    
    # -------------------------------------------------------------------------
    # ADVANCED TACTICAL
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="see_evaluation",
        description="Static Exchange Evaluation (capture sequences)",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,  # Keep for tactical recovery
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="trapped_pieces",
        description="Detect pieces with no escape squares",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.SITUATIONAL,
        skip_when_desperate=False,  # Important for tactics
    ),
    
    EvaluationModule(
        name="back_rank_threats",
        description="Back rank mate detection and prevention",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.ROOK, chess.QUEEN},
        skip_when_desperate=False,
    ),
    
    # -------------------------------------------------------------------------
    # SAFETY & STABILITY
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="move_safety_checker",
        description="Pre-move validation (hanging pieces, legality, repetition)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,  # Always check safety
    ),
    
    EvaluationModule(
        name="repetition_detector",
        description="Avoid threefold repetition unless desperate",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
    ),
]


# =============================================================================
# MODULE SELECTION HELPERS
# =============================================================================

def get_module(name: str) -> Optional[EvaluationModule]:
    """Get module by name from registry"""
    for module in MODULE_REGISTRY:
        if module.name == name:
            return module
    return None


def is_module_relevant(module: EvaluationModule, context: PositionContext) -> bool:
    """
    Determine if module should be active for this position.
    
    Args:
        module: Module to check
        context: Current position context
        
    Returns:
        True if module is relevant, False to skip
    """
    # Check piece requirements
    if module.required_pieces:
        if not module.required_pieces.intersection(context.piece_types):
            return False  # Required pieces not on board
    
    # Check phase requirements
    if module.required_phases:
        if context.game_phase not in module.required_phases:
            return False  # Not in required phase
    
    # DESPERATE MODE: Skip non-tactical modules when down material
    if module.skip_when_desperate:
        if context.material_diff_cp < -300:  # Down 3+ pawns
            return False  # Skip strategic evaluations, focus on tactics
    
    # TIME PRESSURE: Skip expensive modules
    if module.skip_in_time_pressure:
        if context.time_pressure or context.use_fast_profile:
            return False  # Too slow for time pressure
    
    return True


def get_active_modules(context: PositionContext) -> List[EvaluationModule]:
    """
    Get all relevant modules for this position context.
    
    Returns modules in dependency order (dependencies first).
    
    Args:
        context: Current position context
        
    Returns:
        List of active modules sorted by dependencies
    """
    active = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
    
    # TODO: Sort by dependencies (topological sort)
    # For now, just return in registry order
    
    return active


def get_desperate_modules() -> List[EvaluationModule]:
    """
    Get minimal tactical module set for desperate positions.
    
    When down significant material, ONLY evaluate:
    - Material counting
    - Tactical opportunities (captures, checks, threats)
    - Safety checks
    
    Skip ALL strategic evaluations (pawn structure, mobility, etc.)
    """
    desperate_names = [
        "material_counter",
        "hanging_pieces",
        "capture_priority",
        "check_threats",
        "pins_forks_skewers",
        "see_evaluation",
        "trapped_pieces",
        "back_rank_threats",
        "move_safety_checker",
        "repetition_detector",
    ]
    
    return [get_module(name) for name in desperate_names if get_module(name)]


def get_emergency_modules() -> List[EvaluationModule]:
    """
    Get minimal module set for time pressure (< 30s or < 5s per move).
    
    Only essential evaluations, skip all expensive computations.
    """
    emergency_names = [
        "material_counter",
        "piece_square_tables",
        "king_safety_basic",
        "hanging_pieces",
        "move_safety_checker",
    ]
    
    return [get_module(name) for name in emergency_names if get_module(name)]


# =============================================================================
# STATISTICS & DEBUGGING
# =============================================================================

def print_module_summary():
    """Print registry statistics"""
    print(f"\n=== V7P3R Evaluation Module Registry ===")
    print(f"Total modules: {len(MODULE_REGISTRY)}")
    
    by_criticality = {}
    by_cost = {}
    
    for module in MODULE_REGISTRY:
        # Count by criticality
        crit = module.criticality.value
        by_criticality[crit] = by_criticality.get(crit, 0) + 1
        
        # Count by cost
        cost = module.cost.value
        by_cost[cost] = by_cost.get(cost, 0) + 1
    
    print(f"\nBy Criticality:")
    for crit, count in sorted(by_criticality.items()):
        print(f"  {crit}: {count}")
    
    print(f"\nBy Cost:")
    for cost, count in sorted(by_cost.items()):
        print(f"  {cost}: {count}")
    
    print(f"\nDesperate Profile: {len(get_desperate_modules())} modules")
    print(f"Emergency Profile: {len(get_emergency_modules())} modules")


if __name__ == "__main__":
    # Test module registry
    print_module_summary()
