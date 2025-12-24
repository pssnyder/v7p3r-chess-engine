# V18.2 Modular Evaluation System - Implementation Design
**Status**: Active Development  
**Created**: December 23, 2025  
**Target**: v18.2 Release (Week of Dec 23-30)  

---

## Executive Summary

### Problem Statement
Tournament data reveals **evaluation overhead compounds in faster time controls:**
- v18.0: **+56 ELO in rapid, -85 ELO in blitz** (140 ELO swing!)
- Depth decline: 3.977-3.994 in long blitz games (should be 4.0+)
- MoveSafetyChecker overhead: Running expensive checks regardless of position relevance
- Adding more heuristics will worsen performance without selective evaluation

### Solution: Per-Search Evaluation Profiles
**Calculate position context ONCE before search, use throughout entire tree:**

```python
# BEFORE SEARCH (O(1) time):
context = calculate_position_context(board)
# → time_pressure, game_phase, material_imbalance, piece_types, tactical_flags

# SELECT PROFILE (O(1) lookup):
eval_profile = select_evaluation_profile(context)
# → Only includes relevant modules for this position

# SEARCH (use selected profile for all nodes):
score = search(board, depth, profile=eval_profile)
# → No per-node overhead, no unnecessary checks
```

**Benefits**:
- ✅ **Constant-time context calculation** (once per root search)
- ✅ **Skip irrelevant evaluations** (no bishop pair check if no bishops)
- ✅ **Time-pressure adaptation** (automatic fast mode)
- ✅ **Scalable** (add heuristics without per-node cost)

---

## Part 1: Position Context System

### Module: `v7p3r_position_context.py`

#### Purpose
Calculate position characteristics **ONCE** before search, persist through entire tree.

#### Position Context Data Structure

```python
from dataclasses import dataclass
from enum import Enum
from typing import Set
import chess

class GamePhase(Enum):
    """Unified game phase classification"""
    OPENING = "opening"              # Move < 12, pieces ≥ 12
    MIDDLEGAME_COMPLEX = "mg_complex"  # Material 1300-2500cp, pieces 7-11
    MIDDLEGAME_SIMPLE = "mg_simple"    # Material 1300-2500cp, pieces 4-6
    ENDGAME_COMPLEX = "eg_complex"     # Material < 1300cp, pieces 3-6
    ENDGAME_SIMPLE = "eg_simple"       # Material < 800cp, pieces ≤ 2

class MaterialBalance(Enum):
    """Material imbalance classification"""
    EQUAL = "equal"              # |diff| < 100cp
    SLIGHT_ADVANTAGE = "slight"  # 100-300cp
    ADVANTAGE = "advantage"      # 300-500cp
    WINNING = "winning"          # 500-900cp
    CRUSHING = "crushing"        # > 900cp

class TacticalFlags(Enum):
    """Binary tactical indicators"""
    KING_EXPOSED = "king_exposed"        # King has ≤2 pawn shield
    PIECES_HANGING = "pieces_hanging"    # Undefended pieces exist
    CHECKS_AVAILABLE = "checks_available" # Can give check
    PINS_PRESENT = "pins_present"        # Pin opportunities
    FORKS_PRESENT = "forks_present"      # Fork opportunities
    BACK_RANK_WEAK = "back_rank_weak"   # Back rank mate threat

@dataclass
class PositionContext:
    """
    Immutable position characteristics calculated once per search.
    
    This context is passed to ALL evaluation modules and persists
    through the entire search tree (not recalculated per node).
    """
    # Time management
    time_remaining: float        # Seconds left on clock
    time_per_move: float         # Allocated time for this move
    time_pressure: bool          # < 30 seconds remaining
    
    # Game phase
    game_phase: GamePhase        # Single authoritative phase
    move_number: int             # Ply count
    
    # Material
    material_balance: MaterialBalance  # Who's winning materially
    material_diff_cp: int        # Centipawn difference (+ = we're winning)
    total_material: int          # Combined material on board
    
    # Piece inventory
    piece_types: Set[chess.PieceType]  # {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}
    white_pieces: int            # Count of white pieces
    black_pieces: int            # Count of black pieces
    
    # Positional flags
    queens_on_board: bool        # At least one queen present
    bishops_on_board: bool       # At least one bishop present
    opposite_bishops: bool       # Each side has 1+ bishops (bishop pair relevant)
    rooks_on_board: bool         # At least one rook present
    
    # Tactical indicators
    tactical_flags: Set[TacticalFlags]  # Active tactical themes
    king_safety_critical: bool   # King exposure detected
    
    # Endgame specifics
    pawn_endgame: bool          # Only kings + pawns
    pure_piece_endgame: bool    # No pawns, only pieces
    theoretical_draw: bool       # Known drawn material (K vs K, etc)
    
    # Search context
    depth_target: int           # Planned search depth based on time
    use_fast_profile: bool      # Force fast evaluation (time pressure)
```

#### Context Calculation Logic

```python
class PositionContextCalculator:
    """
    Calculates position context once before search.
    
    Design Principles:
    - O(1) or O(n) where n = 64 squares (board scan)
    - No recursive analysis
    - No move generation (too expensive)
    - Cache-friendly (single object creation)
    """
    
    def __init__(self):
        # Material values for quick calculation
        self.MATERIAL_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
    
    def calculate(self, board: chess.Board, time_remaining: float, 
                  time_per_move: float) -> PositionContext:
        """
        Main entry point: Calculate all position characteristics.
        
        Time Complexity: O(64) - single board scan
        Space Complexity: O(1) - fixed-size dataclass
        """
        # Material calculation (O(64))
        material_info = self._calculate_material(board)
        
        # Piece inventory (O(64))
        piece_info = self._calculate_piece_inventory(board)
        
        # Game phase (O(1) - uses material_info)
        game_phase = self._determine_game_phase(
            board, material_info, piece_info
        )
        
        # Tactical flags (O(64) - simple board scan, no move gen)
        tactical_flags = self._detect_tactical_flags(board, piece_info)
        
        # Time pressure detection (O(1))
        time_pressure = time_remaining < 30.0
        use_fast_profile = time_pressure or time_per_move < 5.0
        
        # Depth target based on time (O(1))
        depth_target = self._calculate_depth_target(
            time_per_move, game_phase, time_pressure
        )
        
        return PositionContext(
            # Time
            time_remaining=time_remaining,
            time_per_move=time_per_move,
            time_pressure=time_pressure,
            
            # Phase
            game_phase=game_phase,
            move_number=board.fullmove_number,
            
            # Material
            material_balance=material_info['balance'],
            material_diff_cp=material_info['diff_cp'],
            total_material=material_info['total'],
            
            # Pieces
            piece_types=piece_info['types'],
            white_pieces=piece_info['white_count'],
            black_pieces=piece_info['black_count'],
            
            # Flags
            queens_on_board=chess.QUEEN in piece_info['types'],
            bishops_on_board=chess.BISHOP in piece_info['types'],
            opposite_bishops=piece_info['opposite_bishops'],
            rooks_on_board=chess.ROOK in piece_info['types'],
            
            # Tactical
            tactical_flags=tactical_flags,
            king_safety_critical=TacticalFlags.KING_EXPOSED in tactical_flags,
            
            # Endgame
            pawn_endgame=piece_info['pawn_endgame'],
            pure_piece_endgame=piece_info['pure_piece_endgame'],
            theoretical_draw=material_info['theoretical_draw'],
            
            # Search
            depth_target=depth_target,
            use_fast_profile=use_fast_profile
        )
    
    def _calculate_material(self, board: chess.Board) -> dict:
        """Calculate material counts and balance (O(64))"""
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.MATERIAL_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        diff_cp = white_material - black_material
        if not board.turn:  # Black to move
            diff_cp = -diff_cp
        
        # Determine balance category
        abs_diff = abs(diff_cp)
        if abs_diff < 100:
            balance = MaterialBalance.EQUAL
        elif abs_diff < 300:
            balance = MaterialBalance.SLIGHT_ADVANTAGE
        elif abs_diff < 500:
            balance = MaterialBalance.ADVANTAGE
        elif abs_diff < 900:
            balance = MaterialBalance.WINNING
        else:
            balance = MaterialBalance.CRUSHING
        
        # Theoretical draw detection
        total = white_material + black_material
        theoretical_draw = (
            total == 0 or  # K vs K
            total <= 330   # K+B vs K or K+N vs K
        )
        
        return {
            'diff_cp': diff_cp,
            'total': total,
            'balance': balance,
            'theoretical_draw': theoretical_draw
        }
    
    def _calculate_piece_inventory(self, board: chess.Board) -> dict:
        """Count pieces and determine endgame types (O(64))"""
        piece_types = set()
        white_count = 0
        black_count = 0
        white_bishops = 0
        black_bishops = 0
        has_pawns = False
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_types.add(piece.piece_type)
                if piece.color == chess.WHITE:
                    white_count += 1
                    if piece.piece_type == chess.BISHOP:
                        white_bishops += 1
                else:
                    black_count += 1
                    if piece.piece_type == chess.BISHOP:
                        black_bishops += 1
                
                if piece.piece_type == chess.PAWN:
                    has_pawns = True
        
        return {
            'types': piece_types,
            'white_count': white_count,
            'black_count': black_count,
            'opposite_bishops': white_bishops > 0 and black_bishops > 0,
            'pawn_endgame': piece_types == {chess.PAWN},
            'pure_piece_endgame': len(piece_types) > 0 and not has_pawns
        }
    
    def _determine_game_phase(self, board: chess.Board, 
                              material_info: dict, piece_info: dict) -> GamePhase:
        """
        Unified game phase detection (single source of truth).
        
        Logic:
        1. Opening: move < 12 AND pieces ≥ 12
        2. Endgame: material < 1300cp OR (pieces ≤ 4 AND no queens)
        3. Middlegame: everything else
        4. Complex vs Simple: based on piece count
        """
        move_num = board.fullmove_number
        total_material = material_info['total']
        total_pieces = piece_info['white_count'] + piece_info['black_count']
        has_queens = chess.QUEEN in piece_info['types']
        
        # Opening
        if move_num < 12 and total_pieces >= 12:
            return GamePhase.OPENING
        
        # Endgame
        if total_material < 1300 or (total_pieces <= 4 and not has_queens):
            if total_pieces <= 2:
                return GamePhase.ENDGAME_SIMPLE
            else:
                return GamePhase.ENDGAME_COMPLEX
        
        # Middlegame
        if total_pieces <= 6:
            return GamePhase.MIDDLEGAME_SIMPLE
        else:
            return GamePhase.MIDDLEGAME_COMPLEX
    
    def _detect_tactical_flags(self, board: chess.Board, 
                               piece_info: dict) -> Set[TacticalFlags]:
        """
        Quick tactical flag detection (no move generation).
        
        Note: These are HINTS for evaluation selection, not full tactical analysis.
        Full tactical checks done by selected evaluation modules.
        """
        flags = set()
        
        # King exposure (simple pawn shield check)
        our_king = board.king(board.turn)
        if our_king:
            king_rank = chess.square_rank(our_king)
            king_file = chess.square_file(our_king)
            
            # Count pawn shield (squares in front of king)
            pawn_shield_count = 0
            if board.turn == chess.WHITE and king_rank < 2:
                # Check squares in front
                for file_offset in [-1, 0, 1]:
                    check_file = king_file + file_offset
                    if 0 <= check_file <= 7:
                        check_square = chess.square(check_file, king_rank + 1)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                            pawn_shield_count += 1
            elif board.turn == chess.BLACK and king_rank > 5:
                for file_offset in [-1, 0, 1]:
                    check_file = king_file + file_offset
                    if 0 <= check_file <= 7:
                        check_square = chess.square(check_file, king_rank - 1)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                            pawn_shield_count += 1
            
            if pawn_shield_count <= 2:
                flags.add(TacticalFlags.KING_EXPOSED)
        
        # Checks available (queen or rook near enemy king)
        enemy_king = board.king(not board.turn)
        if enemy_king and chess.QUEEN in piece_info['types']:
            flags.add(TacticalFlags.CHECKS_AVAILABLE)
        
        return flags
    
    def _calculate_depth_target(self, time_per_move: float, 
                                game_phase: GamePhase, 
                                time_pressure: bool) -> int:
        """
        Determine target search depth based on available time.
        
        Fast profiles can search deeper due to lower per-node cost.
        """
        if time_pressure:
            return 4  # Emergency mode
        elif time_per_move < 5.0:
            return 5  # Blitz fast mode
        elif time_per_move < 15.0:
            return 6  # Blitz/rapid normal
        elif time_per_move < 60.0:
            return 7  # Rapid deep search
        else:
            return 8  # Long time control
```

---

## Part 2: Evaluation Module System

### Module: `v7p3r_eval_modules.py`

#### Module Metadata Structure

```python
from dataclasses import dataclass
from typing import Callable, Set, List
from enum import Enum

class ModuleCost(Enum):
    """Computational cost categories"""
    NEGLIGIBLE = "negligible"  # < 0.001ms (material counting, PST lookup)
    LOW = "low"                # < 0.01ms (simple bonuses, basic checks)
    MEDIUM = "medium"          # < 0.1ms (tactical detection, complex evaluation)
    HIGH = "high"              # < 1ms (deep analysis, move generation)

class ModuleCriticality(Enum):
    """How critical is module for accurate evaluation"""
    ESSENTIAL = "essential"    # Always needed (material, PST)
    IMPORTANT = "important"    # Needed in most positions
    SITUATIONAL = "situational" # Only relevant in specific scenarios
    OPTIONAL = "optional"       # Nice-to-have, skip in time pressure

@dataclass
class EvaluationModule:
    """
    Metadata for a single evaluation component.
    
    Modules are selected based on position context and activated
    for the duration of the search.
    """
    # Identification
    name: str
    description: str
    
    # Function
    evaluate: Callable[[chess.Board, PositionContext], int]
    
    # Activation conditions
    required_phases: Set[GamePhase]      # When to activate
    required_pieces: Set[chess.PieceType] # Piece requirements
    required_flags: Set[TacticalFlags]   # Tactical requirements (OR logic)
    skip_if_theoretical_draw: bool       # Skip in K vs K, etc
    
    # Performance
    cost: ModuleCost
    criticality: ModuleCriticality
    time_pressure_skip: bool  # Can skip in time pressure
    
    # Dependencies
    depends_on: List[str]  # Other modules this requires (for ordering)
```

#### Core Evaluation Modules

```python
class EvaluationModules:
    """Registry of all evaluation modules with metadata"""
    
    @staticmethod
    def material_counting(board: chess.Board, context: PositionContext) -> int:
        """
        Basic material counting (ESSENTIAL, NEGLIGIBLE cost).
        
        Already calculated in context, just return it.
        """
        return context.material_diff_cp
    
    @staticmethod
    def piece_square_tables(board: chess.Board, context: PositionContext) -> int:
        """
        Position-based piece values (ESSENTIAL, NEGLIGIBLE cost).
        
        Cached PST lookups based on game phase.
        """
        score = 0
        pst_table = PST_TABLES[context.game_phase]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = pst_table[piece.piece_type][square]
                if piece.color == board.turn:
                    score += value
                else:
                    score -= value
        
        return score
    
    @staticmethod
    def king_centralization(board: chess.Board, context: PositionContext) -> int:
        """
        King centralization bonus (IMPORTANT, LOW cost).
        
        Only relevant in endgames.
        """
        if context.game_phase not in {GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE}:
            return 0  # Should never be called due to module selection
        
        our_king = board.king(board.turn)
        king_file = chess.square_file(our_king)
        king_rank = chess.square_rank(our_king)
        
        # Distance from center (3.5, 3.5)
        file_dist = abs(king_file - 3.5)
        rank_dist = abs(king_rank - 3.5)
        center_dist = file_dist + rank_dist
        
        # Bonus: 70cp (center) to 0cp (corner)
        return int(70 - (center_dist * 10))
    
    @staticmethod
    def passed_pawn_exponential(board: chess.Board, context: PositionContext) -> int:
        """
        Passed pawn bonus with exponential scaling (IMPORTANT, MEDIUM cost).
        
        Skip if no pawns on board.
        """
        if chess.PAWN not in context.piece_types:
            return 0  # Should never be called
        
        score = 0
        # Implementation here (v18.1 logic)
        return score
    
    @staticmethod
    def bishop_pair_bonus(board: chess.Board, context: PositionContext) -> int:
        """
        Bishop pair bonus (SITUATIONAL, LOW cost).
        
        Only relevant if bishops on board and opposite-colored.
        """
        if not context.opposite_bishops:
            return 0  # Should never be called
        
        # Check if current side has bishop pair
        our_bishops = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.BISHOP and piece.color == board.turn:
                our_bishops += 1
        
        return 50 if our_bishops >= 2 else 0
    
    @staticmethod
    def hanging_piece_detection(board: chess.Board, context: PositionContext) -> int:
        """
        Detect undefended pieces (IMPORTANT, HIGH cost).
        
        Part of MoveSafetyChecker, expensive due to attack/defend calculation.
        Skip in time pressure unless king safety critical.
        """
        if context.time_pressure and not context.king_safety_critical:
            return 0  # Skip in time pressure
        
        # Full implementation here (v18.0 logic)
        return 0  # Placeholder
    
    # ... 25+ more modules

# Module Registry
MODULE_REGISTRY = {
    'material': EvaluationModule(
        name='material',
        description='Basic material counting',
        evaluate=EvaluationModules.material_counting,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX, 
                        GamePhase.MIDDLEGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX, 
                        GamePhase.ENDGAME_SIMPLE},
        required_pieces=set(),  # Always applicable
        required_flags=set(),
        skip_if_theoretical_draw=False,
        cost=ModuleCost.NEGLIGIBLE,
        criticality=ModuleCriticality.ESSENTIAL,
        time_pressure_skip=False,
        depends_on=[]
    ),
    
    'pst': EvaluationModule(
        name='pst',
        description='Piece-square table positional values',
        evaluate=EvaluationModules.piece_square_tables,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX, 
                        GamePhase.MIDDLEGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX, 
                        GamePhase.ENDGAME_SIMPLE},
        required_pieces=set(),
        required_flags=set(),
        skip_if_theoretical_draw=False,
        cost=ModuleCost.NEGLIGIBLE,
        criticality=ModuleCriticality.ESSENTIAL,
        time_pressure_skip=False,
        depends_on=['material']  # Needs material to interpolate
    ),
    
    'king_centralization': EvaluationModule(
        name='king_centralization',
        description='Endgame king activity bonus',
        evaluate=EvaluationModules.king_centralization,
        required_phases={GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE},
        required_pieces=set(),  # Always have king
        required_flags=set(),
        skip_if_theoretical_draw=True,  # Doesn't matter in K vs K
        cost=ModuleCost.LOW,
        criticality=ModuleCriticality.IMPORTANT,
        time_pressure_skip=False,
        depends_on=[]
    ),
    
    'bishop_pair': EvaluationModule(
        name='bishop_pair',
        description='Bishop pair advantage bonus',
        evaluate=EvaluationModules.bishop_pair_bonus,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        required_pieces={chess.BISHOP},  # Must have bishops
        required_flags=set(),
        skip_if_theoretical_draw=False,
        cost=ModuleCost.LOW,
        criticality=ModuleCriticality.SITUATIONAL,
        time_pressure_skip=True,  # Can skip
        depends_on=[]
    ),
    
    'hanging_pieces': EvaluationModule(
        name='hanging_pieces',
        description='Undefended piece detection (MoveSafetyChecker)',
        evaluate=EvaluationModules.hanging_piece_detection,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE, 
                        GamePhase.ENDGAME_COMPLEX},
        required_pieces=set(),
        required_flags=set(),  # Useful even without specific flags
        skip_if_theoretical_draw=True,
        cost=ModuleCost.HIGH,
        criticality=ModuleCriticality.IMPORTANT,
        time_pressure_skip=True,  # Skip unless king exposed
        depends_on=[]
    ),
    
    # ... 25+ more modules
}
```

---

## Part 3: Evaluation Profile Selector

### Module: `v7p3r_eval_selector.py`

```python
class EvaluationProfileSelector:
    """
    Selects evaluation modules based on position context.
    
    This is the core "smart selector" that determines which
    evaluation modules to activate for a given search.
    """
    
    def __init__(self, module_registry: dict):
        self.module_registry = module_registry
        
        # Pre-built profiles for common scenarios
        self.profiles = {
            'emergency': self._build_emergency_profile(),
            'fast': self._build_fast_profile(),
            'tactical': self._build_tactical_profile(),
            'endgame': self._build_endgame_profile(),
            'comprehensive': self._build_comprehensive_profile()
        }
    
    def select_profile(self, context: PositionContext) -> List[EvaluationModule]:
        """
        Main selection logic: Choose evaluation modules for this position.
        
        Returns: Ordered list of modules to execute during search
        """
        # Emergency mode: Ultra-fast, minimal evaluation
        if context.time_remaining < 10.0:
            return self.profiles['emergency']
        
        # Time pressure: Fast profile
        if context.use_fast_profile:
            return self.profiles['fast']
        
        # Endgame: Specialized endgame modules
        if context.game_phase in {GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE}:
            return self._customize_endgame_profile(context)
        
        # Tactical positions: Add tactical modules
        if context.king_safety_critical or TacticalFlags.CHECKS_AVAILABLE in context.tactical_flags:
            return self._customize_tactical_profile(context)
        
        # Default: Comprehensive evaluation
        return self._customize_comprehensive_profile(context)
    
    def _build_emergency_profile(self) -> List[EvaluationModule]:
        """
        Emergency profile (< 10 seconds): ONLY essentials.
        
        Modules: material, PST
        Expected speed: 40x faster than comprehensive
        """
        return [
            self.module_registry['material'],
            self.module_registry['pst']
        ]
    
    def _build_fast_profile(self) -> List[EvaluationModule]:
        """
        Fast profile (time pressure or blitz): Essential + cheap modules.
        
        Modules: material, PST, basic positional bonuses
        Expected speed: 10-20x faster than comprehensive
        """
        return [
            self.module_registry['material'],
            self.module_registry['pst'],
            # Add other cheap modules
        ]
    
    def _customize_endgame_profile(self, context: PositionContext) -> List[EvaluationModule]:
        """
        Endgame profile: King activity, pawn structure, piece coordination.
        
        Customized based on:
        - Pawn endgame vs piece endgame
        - Theoretical draw positions
        - Material imbalance
        """
        modules = [
            self.module_registry['material'],
            self.module_registry['pst'],
            self.module_registry['king_centralization'],
        ]
        
        # Add pawn evaluation if pawns present
        if chess.PAWN in context.piece_types:
            modules.append(self.module_registry['passed_pawns'])
        
        # Skip complex evaluation in theoretical draws
        if not context.theoretical_draw:
            # Add piece coordination, etc
            pass
        
        return self._order_modules(modules)
    
    def _customize_tactical_profile(self, context: PositionContext) -> List[EvaluationModule]:
        """
        Tactical profile: King safety, hanging pieces, threats.
        
        Customized based on:
        - King exposure
        - Checks available
        - Material imbalance (forcing play)
        """
        modules = [
            self.module_registry['material'],
            self.module_registry['pst'],
        ]
        
        # Add tactical modules
        if context.king_safety_critical:
            modules.append(self.module_registry['king_safety_attackers'])
            modules.append(self.module_registry['hanging_pieces'])
        
        if TacticalFlags.CHECKS_AVAILABLE in context.tactical_flags:
            modules.append(self.module_registry['check_detection'])
        
        return self._order_modules(modules)
    
    def _customize_comprehensive_profile(self, context: PositionContext) -> List[EvaluationModule]:
        """
        Comprehensive profile: All relevant modules for position.
        
        This is the "smart" version that skips irrelevant modules:
        - No bishop pair check if no bishops
        - No passed pawn check if no pawns
        - etc.
        """
        modules = []
        
        for module_name, module in self.module_registry.items():
            # Check if module is relevant for this position
            if self._is_module_relevant(module, context):
                modules.append(module)
        
        return self._order_modules(modules)
    
    def _is_module_relevant(self, module: EvaluationModule, 
                           context: PositionContext) -> bool:
        """
        Determine if module should be activated for this position.
        
        Checks:
        1. Game phase match
        2. Required pieces present
        3. Tactical flags (if any)
        4. Theoretical draw skip
        5. Time pressure skip
        """
        # Phase check
        if context.game_phase not in module.required_phases:
            return False
        
        # Piece requirement check
        if module.required_pieces and not module.required_pieces.issubset(context.piece_types):
            return False
        
        # Tactical flag check (OR logic - any flag matches)
        if module.required_flags:
            if not any(flag in context.tactical_flags for flag in module.required_flags):
                return False
        
        # Theoretical draw skip
        if module.skip_if_theoretical_draw and context.theoretical_draw:
            return False
        
        # Time pressure skip
        if context.time_pressure and module.time_pressure_skip:
            # Exception: Keep if criticality is ESSENTIAL or king safety critical
            if module.criticality != ModuleCriticality.ESSENTIAL:
                if not (context.king_safety_critical and 'safety' in module.name):
                    return False
        
        return True
    
    def _order_modules(self, modules: List[EvaluationModule]) -> List[EvaluationModule]:
        """
        Order modules by dependencies and cost.
        
        Ensures dependencies run first, then sorts by cost (cheap first).
        """
        # Topological sort by dependencies
        ordered = []
        processed = set()
        
        def add_module(module):
            if module.name in processed:
                return
            # Add dependencies first
            for dep_name in module.depends_on:
                if dep_name in self.module_registry:
                    add_module(self.module_registry[dep_name])
            ordered.append(module)
            processed.add(module.name)
        
        for module in modules:
            add_module(module)
        
        return ordered
```

---

## Part 4: Integration with Search

### Modified `v7p3r.py` Search Loop

```python
class V7P3RChessEngine:
    def __init__(self, board: chess.Board, use_fast_evaluator: bool = True):
        self.board = board
        
        # NEW: Modular evaluation system
        self.context_calculator = PositionContextCalculator()
        self.profile_selector = EvaluationProfileSelector(MODULE_REGISTRY)
        
        # OLD: Keep for migration (will be removed)
        if use_fast_evaluator:
            self.fast_evaluator = V7P3RFastEvaluator(board)
        else:
            self.bitboard_evaluator = V7P3RScoringCalculationBitboard(board)
    
    def find_best_move(self, time_limit: float = 5.0) -> chess.Move:
        """
        Main search entry point.
        
        CHANGED: Calculate context ONCE, select profile ONCE, use throughout search.
        """
        # Calculate position context (O(64) - single board scan)
        context = self.context_calculator.calculate(
            self.board, 
            time_remaining=time_limit,  # TODO: Get from time manager
            time_per_move=time_limit
        )
        
        # Select evaluation profile (O(1) - lookup or O(n) filtering)
        eval_profile = self.profile_selector.select_profile(context)
        
        # Iterative deepening with selected profile
        best_move = None
        for depth in range(1, context.depth_target + 1):
            score, move = self._recursive_search(
                self.board, depth, -99999, 99999, time_limit,
                context=context,        # PASS CONTEXT
                eval_profile=eval_profile  # PASS PROFILE
            )
            best_move = move
        
        # Threefold repetition check (use context for threshold)
        if best_move and self._would_cause_threefold(self.board, best_move):
            # NEW: Dynamic threshold based on material balance
            threshold = self._get_threefold_threshold(context)
            current_eval = self._evaluate_position_modular(self.board, context, eval_profile)
            
            if current_eval > threshold:
                # Avoid threefold, find alternative
                # ... (existing logic)
                pass
        
        return best_move
    
    def _evaluate_position_modular(self, board: chess.Board, 
                                   context: PositionContext,
                                   eval_profile: List[EvaluationModule]) -> int:
        """
        NEW: Modular evaluation using selected profile.
        
        Replaces calls to fast_evaluator.evaluate() or bitboard_evaluator.evaluate()
        """
        score = 0
        
        # Execute each module in profile
        for module in eval_profile:
            score += module.evaluate(board, context)
        
        return score
    
    def _get_threefold_threshold(self, context: PositionContext) -> int:
        """
        Dynamic threefold threshold based on position.
        
        - Crushing advantage: 50cp (very aggressive)
        - Winning: 25cp (aggressive)
        - Advantage: 15cp (very aggressive)
        - Slight advantage: 10cp (fight for wins)
        - Equal: 0cp (accept draws)
        """
        if context.material_balance == MaterialBalance.CRUSHING:
            return 50
        elif context.material_balance == MaterialBalance.WINNING:
            return 25
        elif context.material_balance == MaterialBalance.ADVANTAGE:
            return 15
        elif context.material_balance == MaterialBalance.SLIGHT_ADVANTAGE:
            return 10
        else:
            return 0  # Accept draws when equal
    
    def _recursive_search(self, board: chess.Board, depth: int, 
                         alpha: int, beta: int, time_limit: float,
                         context: PositionContext,
                         eval_profile: List[EvaluationModule]) -> Tuple[int, Optional[chess.Move]]:
        """
        Alpha-beta search with modular evaluation.
        
        CHANGED: Pass context and profile throughout tree (no recalculation)
        """
        # ... existing search logic ...
        
        # Leaf node evaluation
        if depth == 0:
            return self._evaluate_position_modular(board, context, eval_profile), None
        
        # ... rest of search ...
```

---

## Part 5: Migration Strategy

### Phase 1: Parallel Evaluation (Week 1)
**Goal**: Validate modular system matches existing evaluation

1. **Day 1-2**: Implement infrastructure
   - `v7p3r_position_context.py`
   - `v7p3r_eval_modules.py`
   - `v7p3r_eval_selector.py`

2. **Day 3-4**: Migrate core modules
   - Material counting
   - PST
   - King centralization
   - Passed pawns
   - Bishop pair

3. **Day 5**: Parallel testing
   - Run both old and new evaluation on same positions
   - Compare scores (should be identical)
   - Profile performance

### Phase 2: Switch to Modular (Week 2)
**Goal**: Replace old evaluation with modular system

1. **Day 6-7**: Integration
   - Connect to search loop
   - Remove old evaluator calls
   - Test 25 games vs v18.2 baseline

2. **Day 8**: Performance validation
   - Blitz tournament: 25 games vs v17.1
   - Rapid tournament: 25 games vs v17.1
   - Verify depth improvement

3. **Day 9**: Tuning
   - Adjust profile thresholds
   - Fine-tune module selection
   - Optimize hot paths

4. **Day 10**: Release
   - Final testing
   - Documentation
   - Deploy v18.2 modular

### Success Criteria
- ✅ **Blitz performance**: ≥50% vs v17.1 (vs current 38%)
- ✅ **Rapid performance**: ≥58% vs v17.1 (maintain v18.0 rapid strength)
- ✅ **Depth stability**: ≥4.0 in all games (vs 3.977 current)
- ✅ **Draw rate**: ≤35% (vs 55-67% current)
- ✅ **Code quality**: Cleaner, more maintainable

---

## Part 6: Expected Benefits

### Performance Improvements
| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Blitz Win Rate | 38% | 50%+ | +32% relative |
| Rapid Win Rate | 58% | 60%+ | +3% relative |
| Depth (Blitz Long Games) | 3.98 | 4.2+ | +5% |
| Draw Rate | 55-67% | 30-35% | -45% relative |
| Per-Node Eval Cost | Baseline | -30% | Faster |

### Scalability
- ✅ Can add new heuristics without performance penalty
- ✅ Automatic time pressure adaptation
- ✅ Position-appropriate evaluation selection
- ✅ Foundation for ML-based component selection (v19.0 DESC)

### Code Quality
- ✅ Single source of truth for game phase
- ✅ Self-documenting module registry
- ✅ Easy to test modules in isolation
- ✅ Clear cost/benefit documentation

---

**Status**: Design Complete, Ready for Implementation  
**Next Action**: Begin Phase 1 implementation (position context calculator)  
**Target Date**: v18.2 release by December 30, 2025  

