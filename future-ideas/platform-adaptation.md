# Engineering Brief: Platform Adaptation for Active SysID & Multi-Task Physics

**Objective:** Evolve the existing JAX-based RL platform from a specific biological control environment into a general-purpose **"Massively Parallel Digital Twin Engine."**
**Goal:** Support Active System Identification (SysID), Model-Based RL (MPC), and Physics-Informed Learning (PINNs) within a unified codebase without forking.

---

## 1. Core Architecture: The "Three-Layer" Pattern

To support multiple distinct objectives (Control vs. Discovery) using the same underlying physics, we must decouple the biological dynamics from the RL task logic.

### Layer A: Pure Stateless Physics (The "Ground Truth")

* **Refactor:** Extract the core ODE/Dynamics solver into a standalone, pure functional module (e.g., `physics.py`).
* **Requirement:** It must be a stateless function: `next_state = f(state, action, params)`.
* **Constraint:** No `self` references or class attributes. All biological parameters (growth rates, delays) must be passed as explicit arguments.
* **Why:** This allows direct access by **MPC planners** and **Neural ODEs** (Layer C) without instantiating the full RL environment.

### Layer B: The Task Wrappers (The "Context")

Implement interchangeable wrappers that define *how* the agent interacts with the physics. Both wrappers must share the same API for the Learner.

1. **`ControlTask` Wrapper:**
* **Obs:** Standard state (e.g., protein concentration).
* **Reward:** Tracking error (MSE).


2. **`SysIDTask` Wrapper:**
* **Obs:** Augmented state: `[Physical_State, Belief_Mean, Belief_Variance]`.
* **Reward:** Information Gain (e.g., Trace of Fisher Information Matrix or Entropy Reduction).
* **Logic:** Includes an internal estimator (e.g., Kalman Filter or RNN) updated at every step to track belief.



### Layer C: The Learner (The "Agent")

* Standard PPO/RL loops remain largely unchanged.
* The Learner interacts with the generic `Env.step()` interface, agnostic to whether it is performing Control or SysID.

---

## 2. Essential Design Patterns for Future-Proofing

Implement the following patterns now to accommodate future roadmap items (Multi-Agent, Federated Learning, Async behavior) without major refactors later.

### A. The "Sensor" Abstraction (Observation Composition)

* **Concept:** Do not hardcode observations in the Physics layer.
* **Pattern:** Inject a `sensor_fn(state, key)` into the Task Wrapper.
* **Use Case:** Allows easy swapping between "Perfect Sensors," "Noisy Sensors," or "Delayed/Bandwidth-Constrained Sensors" (for Federated Learning simulations).

### B. The "Interaction" Hook (Pre-Physics)

* **Concept:** Allow environments to affect each other before the physics step.
* **Pattern:** Insert an `interaction_fn(actions) -> effective_actions` transformation before the `vmap(physics)`.
* **Use Case:** Default is identity (pass-through). Future implementations can use this for **Optical Cross-Talk** (convolving actions across the batch) or resource contention.

### C. The "Residual Pathway" (For PINNs/Grey-Box Models)

* **Concept:** When using Neural Networks to learn dynamics, do not learn from scratch.
* **Pattern:** The Network class should accept the **Layer A Physics Function** as an argument.
* `Output = Physics_Known(x) + NeuralNet_Residual(x)`


* **Why:** Drastically improves convergence for System ID and ensures physical consistency.

### D. Time-Masking for Asynchronicity

* **Concept:** JAX is synchronous; biology is not.
* **Pattern:** Add `time_elapsed` and `next_event_time` to the State struct. Use `jax.lax.select` (masking) to freeze updates for environments that are "waiting" for a cell cycle event.

---

## 3. Implementation Guardrails

### Numerical Stability

* **Clamping:** Hard-code safety clamps in Layer A (e.g., `max(concentration, 1e-9)`) to prevent NaN propagation in gradients.
* **Nondimensionalization:** Ensure inputs to Neural Nets are normalized (ideally via a LogWrapper) to handle the vast scale differences in biological parameters.

### Vectorization Strategy

* **Domain Randomization:** Parameters (e.g., `k_growth`) must be sampled per-environment at reset time and passed through the step function.
* **Jacobian Computation:** For the SysID reward, use `jax.jacfwd` efficiently. If full Hessian computation is too slow for 100k envs, implement a Trace approximation.

---

## 4. Performance & Testing Strategy

Standard Python profiling is insufficient for JAX's async dispatch.

### Benchmarking Protocol

* **Tooling:** Use `pytest-benchmark` (local) and `CodSpeed` (CI).
* **Requirement:** All timing tests **must** use `.block_until_ready()` on the output tensor.

### Test Hierarchy

1. **Micro-Benchmarks:** Latency of Layer A (Pure Physics) to catch mathematical inefficiencies.
2. **Throughput Benchmarks:** Steps Per Second (SPS) of the full `vmap` loop (including Reward/Belief calculations).
3. **Compilation Monitors:** Measure time-to-first-step to detect recompilation bugs (e.g., static argument leakage).

---

## 5. Documentation & Deliverables

* **Tooling:** Use **MkDocs + Material** theme.
* **Framing:** Document the system as a generic "High-Throughput Digital Twin Engine," using the Biological Mother Machine as the primary "Hard Mode" case study.
