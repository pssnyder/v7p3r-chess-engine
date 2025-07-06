# V7P3R Engine Logic Debugging Plan

## Date: July 6, 2025
## Purpose: Systematic identification and resolution of engine logic flaws

## Critical Issues Identified

### 1. **MAJOR: Perspective Evaluation Inconsistency**
**Location:** `v7p3r_score.py` lines 151-168  
**Issue:** The `evaluate_position_from_perspective()` method has a fundamental flaw:
- Line 165: `self.score_dataset['evaluation'] = self.evaluate_position(board)`
- This always calls the general `evaluate_position()` regardless of the perspective requested
- This causes the wrong evaluation to be stored and potentially returned

**Expected Behavior:** Should store the perspective-specific evaluation, not the general one
**Impact:** HIGH - This could cause completely inverted evaluations (positive when should be negative)

### 2. **MAJOR: Ruleset Configuration Loading Issues**
**Location:** `v7p3r_config.py` lines 105-109  
**Issue:** Configuration is trying to load from `custom_rulesets.json` but we have individual files like `default_ruleset.json`
**Expected Behavior:** Should load from the correct ruleset files in `configs/rulesets/`
**Impact:** HIGH - Empty rulesets in game configs indicate this is failing

### 3. **POTENTIAL: Inconsistent Evaluation Calls in Search**
**Location:** `v7p3r_search.py` multiple locations  
**Issue:** Mix of `evaluate_position()` vs `evaluate_position_from_perspective()` calls
**Expected Behavior:** Consistent perspective handling throughout search
**Impact:** MEDIUM - Could cause evaluation inconsistencies during search

### 4. **POTENTIAL: Game Display Evaluation**
**Location:** `v7p3r_play.py` line 354  
**Issue:** Uses `evaluate_position()` for display - should always show from White's perspective
**Expected Behavior:** Should consistently show White's perspective for game display
**Impact:** MEDIUM - Could confuse evaluation display

### 5. **MINOR: Logging Directory Path**
**Location:** Multiple files  
**Issue:** Logging setup still points to `project_root/logging` instead of parent directory
**Expected Behavior:** Should log to `S:\Maker Stuff\Programming\V7P3R Chess Engine\logging`
**Impact:** LOW - Performance issue with git, not game logic

## Investigation Plan

### Phase 1: Core Evaluation Logic (HIGH PRIORITY)
1. **Fix perspective evaluation storage bug** in `v7p3r_score.py`
2. **Verify evaluation consistency** across all scoring methods
3. **Test with known positions** to ensure correct perspective handling

### Phase 2: Configuration System (HIGH PRIORITY) 
1. **Fix ruleset loading** in `v7p3r_config.py`
2. **Verify ruleset values** are properly loaded and applied
3. **Test configuration snapshots** in game files

### Phase 3: Search Evaluation Consistency (MEDIUM PRIORITY)
1. **Audit all evaluation calls** in `v7p3r_search.py`
2. **Ensure consistent perspective handling** throughout search tree
3. **Verify minimax evaluation propagation**

### Phase 4: Display and Logging (MEDIUM PRIORITY)
1. **Standardize evaluation display** to White's perspective
2. **Fix logging directory paths**
3. **Verify PGN evaluation consistency**

### Phase 5: Comprehensive Testing (ALL PRIORITIES)
1. **Create test positions** with known evaluations
2. **Test perspective consistency** across all modules
3. **Verify configuration loading** with different rulesets
4. **Test complete game flow** with proper evaluation tracking

## Test Cases to Create

### Evaluation Perspective Tests
- **Position 1:** White clearly winning (should be positive from White's perspective)
- **Position 2:** Black clearly winning (should be negative from White's perspective)  
- **Position 3:** Checkmate for White (should be highly positive)
- **Position 4:** Checkmate for Black (should be highly negative)

### Configuration Tests
- **Test 1:** Load default_ruleset.json and verify all values
- **Test 2:** Create custom ruleset and verify loading
- **Test 3:** Verify game config snapshots contain complete ruleset

### Search Consistency Tests
- **Test 1:** Verify minimax returns correct perspective
- **Test 2:** Check evaluation consistency during search tree traversal
- **Test 3:** Verify book move evaluations match search evaluations

## Implementation Strategy

1. **One issue at a time** - Fix and test each issue individually
2. **Regression testing** - Ensure fixes don't break existing functionality
3. **Logging verification** - Use logs to verify each fix
4. **Incremental validation** - Test after each change to ensure progress

## Success Metrics

- ✅ Evaluations show correct perspective (negative for Black advantage, positive for White)
- ✅ Rulesets properly loaded and appear in game configuration files
- ✅ Consistent evaluation calls throughout engine
- ✅ Proper evaluation display in terminal and PGN files
- ✅ Clean logging to parent directory

## Next Steps

1. Start with **Phase 1** - Fix the critical perspective evaluation bug
2. Create simple test to verify the fix works correctly
3. Move to **Phase 2** - Fix ruleset loading 
4. Continue systematically through remaining phases

This plan prioritizes the most critical issues first and provides a systematic approach to debugging and fixing the engine logic flaws.
