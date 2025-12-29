:orphan:

# Configuration System Refactoring Plan

## Executive Summary

**Problem:** The current configuration system has defaults specified in multiple locations (YAML files and Python factory functions), leading to duplication and inconsistencies.

**Solution:** Adopt a "single source of truth" approach where defaults live exclusively in Python factory functions, and YAML files contain only experiment-specific overrides.

**Impact:** Improved maintainability, eliminates duplication, clearer separation of concerns, better IDE support for developers.

---

## Current State Analysis

### Configuration Flow

```
YAML Files → Hydra Composition → Pydantic Validation → Factory Functions → Runtime Objects
```

### Problem: Duplicate Defaults

Defaults are currently specified in TWO places:

#### Example: DQN Agent

**YAML** (`configs/agent/dqn.yaml`):
```yaml
learning_rate: 1e-3
gamma: 0.99
epsilon_decay_steps: 50000
target_network_frequency: 1000
```

**Python Factory** (`src/myriad/agents/rl/dqn.py:254`):
```python
def make_agent(
    action_space: Space,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_decay_steps: int = 10000,  # ⚠️ INCONSISTENT!
    target_network_frequency: int = 500,  # ⚠️ INCONSISTENT!
) -> Agent:
```

### Identified Inconsistencies

| Parameter | YAML Value | Factory Value | Status |
|-----------|------------|---------------|--------|
| `epsilon_decay_steps` | 50000 | 10000 | **5x different** |
| `target_network_frequency` | 1000 | 500 | **2x different** |

**Impact:** Users calling factory functions directly get different behavior than users using YAML configs.

---

## Target Architecture

### Design Principle

**Single Source of Truth:** All defaults live in Python factory functions. YAML files contain ONLY overrides for specific experiments.

### Benefits

1. **No Duplication** - Change defaults in one place only
2. **Type Safety** - Python type hints provide compile-time checks
3. **IDE Support** - Autocomplete, docs, and jump-to-definition for defaults
4. **Programmatic Access** - Direct factory calls work correctly
5. **Clear Intent** - YAML files show "what's different" for each experiment
6. **Maintainability** - Less code to keep in sync

### Configuration Responsibilities

| Layer | Responsibility | Example |
|-------|---------------|---------|
| **Python Factory Functions** | Define defaults | `learning_rate: float = 1e-3` |
| **Pydantic Models** | Validate types and structure | `learning_rate: float` |
| **YAML Base Configs** | Document available parameters (no values) | Comments only |
| **YAML Experiment Configs** | Override defaults for specific experiments | `learning_rate: 3e-4` |
| **CLI Overrides** | Quick overrides for testing/sweeps | `agent.learning_rate=3e-4` |

---

## Refactoring Steps

### Phase 1: Audit and Fix Inconsistencies

**Goal:** Ensure YAML and factory defaults match before refactoring.

**Tasks:**

1. Create audit script to compare YAML and factory defaults
2. Document all inconsistencies
3. Decide on correct defaults (consult papers/benchmarks)
4. Update factory functions to match chosen defaults
5. Update YAML files to match chosen defaults
6. Add test to prevent future drift

**Files to Audit:**
- `configs/agent/dqn.yaml` ↔ `src/myriad/agents/rl/dqn.py`
- `configs/agent/pqn.yaml` ↔ `src/myriad/agents/rl/pqn.py`
- `configs/agent/pid.yaml` ↔ `src/myriad/agents/classical/pid.py`
- `configs/env/cartpole_control.yaml` ↔ `src/myriad/envs/cartpole/tasks/control.py`
- `configs/env/cartpole_sysid.yaml` ↔ `src/myriad/envs/cartpole/tasks/sysid.py`
- `configs/env/ccas_ccar_*.yaml` ↔ `src/myriad/envs/ccas_ccar/*.py`

**Estimated Time:** 2-3 hours

---

### Phase 2: Refactor Base YAML Configs

**Goal:** Convert base configs from "defaults" to "parameter documentation."

**Tasks:**

1. Update `configs/agent/dqn.yaml` to remove defaults, add comments
2. Update `configs/agent/pqn.yaml` to remove defaults, add comments
3. Update `configs/agent/pid.yaml` to remove defaults, add comments
4. Update `configs/env/*.yaml` to remove defaults, add comments
5. Add header to each file explaining the new pattern

**Example Before** (`configs/agent/dqn.yaml`):
```yaml
# @package agent
name: dqn
batch_size: 64
learning_rate: 1e-3
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 50000
target_network_frequency: 1000
tau: 1.0
```

**Example After** (`configs/agent/dqn.yaml`):
```yaml
# @package agent
# DQN Agent Configuration
#
# Defaults are defined in src/myriad/agents/rl/dqn.py:make_agent()
# Only specify parameters here to override defaults for this experiment.
#
# Available parameters:
#   learning_rate (float): Adam learning rate [default: 1e-3]
#   gamma (float): Discount factor [default: 0.99]
#   epsilon_start (float): Initial exploration rate [default: 1.0]
#   epsilon_end (float): Final exploration rate [default: 0.05]
#   epsilon_decay_steps (int): Steps to decay epsilon [default: 10000]
#   target_network_frequency (int): Target network update interval [default: 500]
#   tau (float): Target update rate (1.0 = hard update) [default: 1.0]

name: dqn

# Experiment-specific overrides (uncomment to customize):
# learning_rate: 3e-4
# epsilon_decay_steps: 100000
```

**Estimated Time:** 2 hours

---

### Phase 3: Create Experiment-Specific Configs

**Goal:** Create named configs for common experimental variations.

**Tasks:**

1. Create `configs/agent/dqn_fast_exploration.yaml` (short decay)
2. Create `configs/agent/dqn_slow_exploration.yaml` (long decay)
3. Create `configs/agent/dqn_high_lr.yaml` (aggressive learning)
4. Update docs to reference these examples

**Example** (`configs/agent/dqn_fast_exploration.yaml`):
```yaml
# @package agent
# DQN with fast exploration decay for simple control tasks

name: dqn
epsilon_decay_steps: 10000  # 5x faster than default
target_network_frequency: 500  # More frequent updates
```

**Usage:**
```bash
python scripts/train.py agent=dqn_fast_exploration
```

**Estimated Time:** 1 hour

---

### Phase 4: Update Documentation

**Goal:** Document the new configuration pattern clearly.

**Tasks:**

1. Update `docs/contributing/configuration.md`:
   - Add "Configuration Philosophy" section
   - Explain single source of truth principle
   - Update examples to show override pattern
   - Add "Finding Defaults" section pointing to factory functions

2. Update `docs/user-guide/running_experiments.md`:
   - Emphasize override pattern
   - Show how to find defaults in code

3. Create `docs/contributing/adding_parameters.md`:
   - How to add new parameters to agents/envs
   - Where to define defaults (factory functions only)
   - How to update Pydantic models (type validation)
   - Example of full parameter addition workflow

**Estimated Time:** 3 hours

---

### Phase 5: Add Validation Tests

**Goal:** Prevent future inconsistencies through automated testing.

**Tasks:**

1. Create `tests/config/test_yaml_configs.py`:
   - Test that YAML configs are valid
   - Test that YAML configs only contain known parameters
   - Warn if YAML specifies same value as factory default (likely redundant)

2. Add to CI pipeline

**Example Test:**
```python
def test_yaml_configs_are_minimal():
    """Warn if YAML configs contain redundant default values."""
    from myriad.agents.rl import dqn
    import inspect

    # Load YAML
    yaml_config = load_yaml("configs/agent/dqn.yaml")

    # Get factory defaults
    factory_sig = inspect.signature(dqn.make_agent)

    # Warn about redundancy
    for key, value in yaml_config.items():
        if key in factory_sig.parameters:
            factory_default = factory_sig.parameters[key].default
            if value == factory_default:
                warnings.warn(
                    f"YAML specifies {key}={value} which matches factory default. "
                    f"Consider removing from YAML."
                )
```

**Estimated Time:** 2 hours

---

### Phase 6: Optional - Auto-Generate Docs

**Goal:** Generate parameter reference docs from factory function signatures.

**Tasks:**

1. Create `scripts/generate_param_docs.py`:
   - Extract signatures from all factory functions
   - Generate markdown tables with parameters, types, and defaults
   - Write to `docs/api-reference/parameters.md`

2. Add to pre-commit hook or CI

**Example Output:**
```markdown
## DQN Agent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `float` | `1e-3` | Adam learning rate |
| `gamma` | `float` | `0.99` | Discount factor |
| `epsilon_decay_steps` | `int` | `10000` | Steps to decay epsilon |

**Source:** `src/myriad/agents/rl/dqn.py:make_agent()`
```

**Estimated Time:** 3 hours

---

## Migration Guide for Users

### For YAML Users (No Changes Required)

Current YAML configs will continue to work. The refactoring is backward-compatible.

### For Programmatic Users

**Before:**
```python
from myriad.agents.rl.dqn import make_agent
from myriad.core.spaces import Discrete

# May get different behavior than YAML due to inconsistent defaults
agent = make_agent(action_space=Discrete(2))
```

**After (Same Code, But Now Consistent):**
```python
from myriad.agents.rl.dqn import make_agent
from myriad.core.spaces import Discrete

# Now guaranteed to match YAML behavior
agent = make_agent(action_space=Discrete(2))
```

### For Contributors

**Before:**
When adding a new parameter:
1. Add to factory function with default
2. Add to Pydantic model
3. Add to YAML config with same default ⚠️ (easy to forget or mismatch)

**After:**
When adding a new parameter:
1. Add to factory function with default ✅ (single source of truth)
2. Add to Pydantic model (type validation only)
3. Optionally add comment in YAML documenting availability

---

## Rollout Plan

### Week 1: Audit and Align
- Complete Phase 1 (audit and fix inconsistencies)
- Create PR with fixes and consistency test

### Week 2: Refactor Base Configs
- Complete Phase 2 (refactor base YAML configs)
- Complete Phase 3 (create experiment configs)
- Create PR with updated configs

### Week 3: Documentation
- Complete Phase 4 (update docs)
- Create PR with documentation updates

### Week 4: Validation (Optional)
- Complete Phase 5 (add validation tests)
- Complete Phase 6 (auto-generate docs)
- Create PR with tooling

---

## Success Criteria

1. ✅ Zero inconsistencies between YAML and factory defaults
2. ✅ All base YAML configs clearly document they're override-only
3. ✅ Documentation explains single source of truth principle
4. ✅ CI tests prevent future inconsistencies
5. ✅ All existing experiments continue to work unchanged

---

## Future Enhancements

### Consider After Refactoring

1. **Structured Config Generation:**
   - Use `dataclasses` with defaults for agent params
   - Generate both Pydantic models and YAML templates from dataclasses

2. **Config Presets System:**
   - `configs/presets/dqn/` directory with curated experiment configs
   - `atari.yaml`, `control.yaml`, `sysid.yaml` presets per agent

3. **Parameter Search Syntax:**
   - Sugar for common patterns: `agent.learning_rate=log_uniform(1e-4,1e-2)`
   - Integrated with W&B sweeps

4. **Config Validation Levels:**
   - Strict mode: only allow parameters that exist in factory signature
   - Permissive mode: allow extra fields (current behavior with Pydantic `extra: "allow"`)

---

## Related Files

- **Configuration Schemas:** `src/myriad/configs/default.py`
- **Configuration Docs:** `docs/contributing/configuration.md`
- **Factory Registry:** `src/myriad/agents/__init__.py`, `src/myriad/envs/__init__.py`
- **Factory Functions:** `src/myriad/agents/rl/dqn.py`, `src/myriad/envs/cartpole/tasks/control.py`, etc.
- **Base Configs:** `configs/agent/*.yaml`, `configs/env/*.yaml`, `configs/run/*.yaml`
- **Training Scripts:** `scripts/train.py`, `scripts/train_sweep.py`
