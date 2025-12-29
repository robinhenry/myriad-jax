# Myriad Development Roadmap

**Vision**: Provide the computational backend for high-throughput control experiments, enabling researchers to develop and validate algorithms that leverage massive parallel data streams—from microfluidic mother machines to digital twin populations.

**Mission**: Accelerate scientific discovery by allowing researchers to test experimental designs and algorithms in silico before deploying to real hardware, avoiding wasted lab time on approaches that don't work. Focus on systems where stochasticity and parameter uncertainty are fundamental.

**Positioning**: Complement existing RL libraries (Gymnasium, Brax, Gymnax) by specializing in **population-scale system identification and control** for stochastic scientific systems. Support both algorithm development/validation and direct sim-to-real transfer workflows.

---

## Core Principles

1. **Lab Relevance First**: Target problems where simulation results transfer to real experimental hardware
2. **Collaborative Ecosystem**: Build on excellent libraries like Gymnasium and Brax, don't reinvent
3. **Rigorous Stochasticity**: Exact SSA/Gillespie, asynchronous events—noise is the signal
4. **Glass-Box Dynamics**: Expose underlying physics for gradient-based analysis and active learning
5. **Population Paradigm**: Design for controlling distributions and learning from heterogeneity
6. **Proof Over Promises**: Validate all performance claims with published benchmarks
7. **Scientific Rigor**: Match or exceed published baselines, document all hyperparameters
8. **Community First**: Reduce friction for new users, make contributions easy
9. **Humble Innovation**: Acknowledge what others do well, focus on where we add unique value

---

## Current State Assessment

### ✓ Production-Ready
- Core architecture (three-layer pattern)
- DQN agent (fully tested)
- CartPole environment (control + SysID)
- Gene circuit environment (stress test)
- Configuration system (Hydra + Pydantic)
- Testing infrastructure (438 tests)
- Documentation (complete API + guides)

### ⚠ Needs Work
- Performance benchmarks (claims unvalidated)
- Algorithm coverage (only 2 RL agents)
- Environment diversity (only 2 environments)
- Model checkpointing (not implemented)
- PQN tuning (marked as "needs work")

### ✗ Missing
- Continuous control algorithms (PPO, SAC, TD3)
- Biological environment suite
- Published validation results
- Tutorial paper/comprehensive notebook

---

## Phase 1: Critical Path (Months 1-2)

**Goal**: Prove the platform works and is production-ready

### 1.1 Performance Benchmarking ⚡

**Why**: "Massively parallel" needs proof, not just claims

**Tasks**:
- [ ] Create `benchmarks/` directory
- [ ] Benchmark script: steps/second vs. num_envs (100, 1K, 10K, 100K)
- [ ] Wall-clock comparison: Myriad vs. serial Gymnasium on CartPole
- [ ] GPU memory profiling vs. number of environments
- [ ] Comparison to Brax (if applicable)
- [ ] Add benchmark plots to README
- [ ] Create `examples/10_performance_demo.py` (100K parallel envs)

**Success Metric**: Published graphs showing >1M steps/second at 100K envs

---

### 1.2 Model Checkpointing & Resume 💾

**Why**: Can't run serious experiments without saving/loading models

**Tasks**:
- [ ] Implement `orbax-checkpoint` integration
- [ ] Save full agent state (network params + optimizer state)
- [ ] Load pretrained agents for evaluation
- [ ] Support mid-training checkpoint resume
- [ ] Add `examples/11_checkpoint_resume.py`
- [ ] Test checkpoint compatibility across JAX versions
- [ ] Document serialization limitations (if any remain)

**Success Metric**: Train for 1M steps, checkpoint, resume, verify identical results

---

### 1.3 PPO Implementation 🤖

**Why**: Most popular RL algorithm, needed for continuous control

**Tasks**:
- [ ] Implement PPO agent (`src/myriad/agents/rl/ppo.py`)
- [ ] Support both discrete and continuous action spaces
- [ ] GAE (Generalized Advantage Estimation)
- [ ] Clipped surrogate objective
- [ ] Value function loss + entropy bonus
- [ ] Comprehensive tests (following `test_dqn.py` pattern)
- [ ] Config file (`configs/agent/ppo.yaml`)
- [ ] Validation on CartPole (match published baselines)

**Success Metric**: PPO solves CartPole in <500K steps, matches literature performance

---

### 1.4 Fix PQN Tuning

**Why**: Second RL algorithm is marked as "needs work"

**Tasks**:
- [ ] Debug PQN on CartPole
- [ ] Hyperparameter sweep (learning rate, lambda, epochs)
- [ ] Document final working config
- [ ] Add validation test
- [ ] Update project board status

**Success Metric**: PQN reliably solves CartPole-control

---

## Phase 2: Adoption Enablers (Months 3-4)

**Goal**: Make it easy for new users to get value quickly

### 2.1 Expand Environment Suite 🌍

**Why**: "Only 2 environments" signals toy project

**Priority Additions**:

#### Classic Control (Continuous)
- [ ] **Pendulum** (`envs/pendulum/`)
  - Continuous action space
  - Control + SysID tasks
  - Test against Gymnasium baseline

- [ ] **Mountain Car** (`envs/mountain_car/`)
  - Sparse reward challenge
  - Control + SysID tasks

- [ ] **Acrobot** (`envs/acrobot/`)
  - Underactuated system
  - Closer to real robotics

#### Biological Systems (Our Niche!)
- [ ] **Biochemical Oscillator** (`envs/oscillator/`)
  - Synthetic oscillator (e.g., repressilator)
  - Stochastic dynamics
  - Control + SysID tasks

- [ ] **Toggle Switch** (`envs/toggle_switch/`)
  - Bi-stable genetic circuit
  - Parameter identification challenge

**Implementation Pattern**: Follow CartPole structure
- `physics.py`: Pure dynamics
- `control.py`: Control task
- `sysid.py`: SysID task
- `tests/`: Physics + task tests

**Success Metric**: 5-7 total environments (2 continuous control + 2-3 bio systems)

---

### 2.2 Comprehensive Tutorial 📚

**Why**: Reduces time-to-value for new users

**Tasks**:
- [ ] Create `tutorials/03_full_workflow.ipynb`
  - Train DQN on CartPole from scratch
  - Visualize learning curves
  - Compare to PID baseline
  - Switch to SysID variant
  - Scale to 10K parallel envs
  - Show performance speedup
- [ ] Update existing tutorials (01, 02)
- [ ] Add tutorial to docs
- [ ] Create accompanying blog post (Medium/Substack)

**Success Metric**: New user can complete tutorial in <30 minutes

---

### 2.3 Validation Experiments 📊

**Why**: Scientific credibility requires validated baselines

**Tasks**:
- [ ] Create `experiments/validation/` directory
- [ ] **DQN on CartPole**: Match Mnih et al. performance
  - Document hyperparameters
  - Include random seeds
  - Plot learning curves
- [ ] **Gene Circuit**: Match CDC 2025 paper results
  - Reproduce paper experiments
  - Document deviations (if any)
- [ ] **PID Controller**: Match analytical solutions
  - Validate against control theory
- [ ] Add validation plots to documentation

**Success Metric**: All baselines within 10% of published results

---

### 2.4 Tutorial Paper Submission

**Why**: Papers get citations, citations get users

**Tasks**:
- [ ] Write JOSS (Journal of Open Source Software) paper
  - Statement of need
  - Software description
  - Example usage
  - Comparison to alternatives
- [ ] Create companion arXiv preprint (optional, more detail)
- [ ] Submit to JOSS
- [ ] Announce on Twitter, Reddit (r/MachineLearning)

**Success Metric**: Paper accepted, cited in bibliography

---

## Phase 3: Algorithm Expansion (Months 5-6)

**Goal**: Cover standard algorithm baselines

### 3.1 Continuous Control Algorithms

#### SAC (Soft Actor-Critic)
- [ ] Implement `agents/rl/sac.py`
- [ ] Off-policy, continuous actions
- [ ] Maximum entropy objective
- [ ] Automatic temperature tuning
- [ ] Tests + config
- [ ] Validate on Pendulum

#### TD3 (Twin Delayed DDPG)
- [ ] Implement `agents/rl/td3.py`
- [ ] Simpler alternative to SAC
- [ ] Clipped double Q-learning
- [ ] Delayed policy updates
- [ ] Tests + config
- [ ] Validate on Pendulum

**Success Metric**: SAC and TD3 solve Pendulum, match published baselines

---

### 3.2 Derivative-Free Optimization

**Why**: Useful for SysID when gradients fail

- [ ] **CEM** (Cross-Entropy Method): `agents/optimization/cem.py`
- [ ] **CMA-ES** (Covariance Matrix Adaptation): `agents/optimization/cma_es.py`
- [ ] Validate on parameter identification tasks

**Success Metric**: CEM/CMA-ES solve CartPole SysID task

---

## Phase 4: Ecosystem & Polish (Month 6+)

**Goal**: Quality of life improvements and community growth

### 4.1 Environment Wrappers

- [ ] **Frame stacking**: Observation history for partial observability
- [ ] **Action repeat**: Skip frames for faster learning
- [ ] **Reward normalization**: Running mean/std
- [ ] **Domain randomization wrapper**: Automated parameter sampling
- [ ] Document wrapper API

---

### 4.2 Video Gallery & Visualizations 🎥

**Why**: Eye candy attracts users

- [ ] Create `docs/gallery/` with trained agent videos
- [ ] Animated learning curves (GIFs)
- [ ] Side-by-side comparisons (Random vs. PID vs. DQN vs. PPO)
- [ ] Gene circuit fluorescence traces
- [ ] Embed in documentation

---

### 4.3 Community Infrastructure 🤝

- [ ] "Good first issue" labels on GitHub
- [ ] Environment contribution template
- [ ] Agent contribution template
- [ ] CONTRIBUTING.md with checklist
- [ ] Hall of fame for contributors
- [ ] Consider Discord/Slack for real-time help

---

## Phase 5: Lab-to-Sim Integration (Ongoing)

**Goal**: Bridge the gap between in-silico and in-vitro experiments

### 5.1 Expanded Environment Suite

Add environments that mirror real experimental systems:

#### Biological Systems
- [ ] **Metabolic Network**: E. coli glycolysis control
- [ ] **Toggle Switch**: Bi-stable genetic circuit (Gardner et al. 2000)
- [ ] **Oscillator**: Repressilator or synthetic oscillator
- [ ] **Cell Signaling**: MAPK cascade, bistability

#### Chemical Systems
- [ ] **CSTR (Continuous Stirred Tank Reactor)**: Uncertain kinetics
- [ ] **Batch Reactor**: Time-varying parameter estimation
- [ ] **Bioreactor**: Fed-batch culture with growth dynamics

---

### 5.2 High-Throughput Experimental Tools

Features that support real lab integration:

- [ ] **Real-Time Policy Deployment**: Export trained policies for hardware control
- [ ] **Data Import Pipeline**: Load experimental trajectories for validation
- [ ] **Experimental Design Metrics**: Fisher information, D-optimality, mutual information
- [ ] **Uncertainty Quantification**: Bayesian parameter inference, confidence regions
- [ ] **Hardware-in-Loop Testing**: Interface with microfluidic control systems
- [ ] **Gillespie Visualization**: Molecular event traces and population histograms

---

### 5.3 Community & Outreach

Build connections with experimental researchers:

#### Scientific Community
- [ ] Present at **COMBINE** (computational biology)
- [ ] Workshop at **IWBDA** (Bio-Design Automation)
- [ ] Present at **ACC** (American Control Conference)
- [ ] Submit tutorial paper to **Bioinformatics** or **PLOS Comp Bio**

#### Real-World Validation
- [ ] Collaborate with mother machine labs for validation
- [ ] Case study: Trained policy deployed on real hardware
- [ ] Document sim-to-real transfer gaps and solutions

#### Open Science
- [ ] Post preprint on **bioRxiv**
- [ ] Share trained policies and experimental datasets
- [ ] Create video tutorials for experimental biologists

---

## Success Metrics

### Short-term (6 months)
- [ ] 100+ GitHub stars
- [ ] 10+ external contributors
- [ ] 5+ published benchmark results
- [ ] JOSS paper accepted
- [ ] 1000+ PyPI downloads/month
- [ ] 1+ experimental lab expressing interest in using Myriad

### Medium-term (12 months)
- [ ] 500+ GitHub stars
- [ ] 50+ citations (paper + repo)
- [ ] Featured in computational biology/systems newsletters
- [ ] 3+ research papers using Myriad
- [ ] 5000+ PyPI downloads/month
- [ ] 1+ successful sim-to-real transfer case study
- [ ] Active collaboration with experimental lab

### Long-term (24 months)
- [ ] 1000+ GitHub stars
- [ ] Standard tool for high-throughput control experiments
- [ ] Adopted by 10+ research labs (computational + experimental)
- [ ] Featured on Papers with Code
- [ ] 10,000+ PyPI downloads/month
- [ ] Trained policies deployed on real hardware (mother machine, bioreactor, etc.)
- [ ] Community-contributed real-world validated environments

---

## Anti-Patterns to Avoid

- ❌ **Feature creep**: Stay focused on bio systems + active learning
- ❌ **Over-engineering**: Keep APIs simple
- ❌ **Stale docs**: Update docs with every feature
- ❌ **Unvalidated claims**: Benchmark everything
- ❌ **Poor onboarding**: First 5 minutes are critical
- ❌ **Breaking changes**: Maintain backward compatibility
- ❌ **Ignoring issues**: Respond to GitHub issues within 48 hours

---

## Decision Log

Track major architectural decisions here:

### 2024-12-29: High-Throughput Control Paradigm
**Decision**: Position Myriad as the computational backend for high-throughput control experiments, emphasizing lab-to-sim integration and mother machine inspiration
**Rationale**:
- Differentiates from pure RL libraries (Gymnasium, Brax) without dismissing them
- Aligns with real experimental capabilities (mother machines, microfluidics)
- Emphasizes population-level learning and system ID as core, not afterthoughts
- Supports TWO workflows:
  1. **Algorithm validation**: Test approaches in silico before deploying to hardware
  2. **Direct transfer**: Train policies in simulation, deploy to hardware
**Key Changes**:
- Vision: "Computational backend for high-throughput control"
- Mission: "Test experimental designs and algorithms in silico before deploying to real hardware"
- Positioning: General-first (system ID + stochastic + population), biology as flagship
- Tone: Humble and collaborative with existing tools
- Clarification: Not about replacing experiments, but making them more efficient
**Status**: Approved

### 2024-12-29: Three-Layer Architecture Emphasis
**Decision**: Maintain strict separation of Physics, Task, and Learner layers
**Rationale**: Enables reuse of same physics for control, SysID, and planning tasks
**Status**: Ongoing (enforced in CLAUDE.md)

### Future decisions...

---

## Revision History

- **v1.0** (2024-12-29): Refined positioning based on README and documentation updates
  - Shifted to "high-throughput control" paradigm
  - Emphasized mother machine inspiration and lab-to-sim integration
  - Updated success metrics to include real-world validation
  - Added Phase 5: Lab-to-Sim Integration
  - Documented positioning decisions in Decision Log

- **v0.1** (2024-12-29): Initial roadmap based on strategic assessment
  - Core phases defined (Critical Path, Adoption Enablers, Algorithm Expansion)
  - Initial positioning as bio-focused platform
  - Community principles established
