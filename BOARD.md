# AION Project Board

## Todo

### Implement Ccas-Ccar-v1 environment

  - defaultExpanded: true
    ```md
    dod: random & PQN performances match paper results
    ```

### Add support for evaluation-only runs

  - defaultExpanded: true
  - steps:
      - [ ] Use "epsilon_eval" for PQN
    ```md
    dod: a script/CLI that can be run to evaluate the performance of an algorithm, skipping training. Should support:
    - a policy that doesn't require training
    - loading a saved model from a checkpoint file
    ```

### Add support for stacking observation frames

  - defaultExpanded: true
  ```md
  DoD: a new parameter is available to set the number of frames to stack. The frames are unstacked in DQN and PQN.
  ```

### Add option to regularly save model checkpoints

  - defaultExpanded: true
  ```md
  DoD: a new parameter `checkpoint_frequency` (# of eval steps) that can be used by the algos that need training to save themselves.
  The evaluation-only run config supports specifying a model checkpoint (TBD how: filename? run ID?)
  ```


## In Progress

### Test PQN on cartpole-v1

  - defaultExpanded: true
  - steps:
      - [ ] Use parameters currently not included (`eval_epsilon`)
      - [ ] Check that rewards are properly scaled
      - [ ] Potentially move `rollout_steps` and `batch_size` to agent config
      - [ ] Debug why it's slow & make parameters more intuitive
    ```md
    DoD: check that performance matches results from original implementation.
    ```

## Done
