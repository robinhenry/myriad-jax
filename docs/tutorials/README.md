# Tutorial Development Guide

## Execution Workflow

**All tutorials must be executed locally before committing.**

The documentation build does NOT re-execute notebooks in CI. This is because:
- Some tutorials require GPU for reasonable execution times
- Local execution ensures you see accurate outputs before publishing
- Faster CI builds (notebooks can be expensive to run)

### How to Execute Tutorials Locally

1. **Navigate to tutorial directory:**
   ```bash
   cd docs/tutorials/basics
   ```

2. **Execute notebook in-place:**
   ```bash
   jupyter nbconvert --to notebook --execute 03_parallel_training.ipynb --inplace
   ```

   Or use Jupyter Lab/Notebook UI to run all cells.

3. **Verify outputs look correct** (plots, videos, metrics)

4. **Commit the executed `.ipynb` file** with outputs

### CI Validation

- CI checks that all notebooks have outputs (validates you executed locally)
- CI does NOT re-execute notebooks (configured in `docs/conf.py:nb_execution_mode = "cache"`)
- If you forget to execute, the validation workflow will fail with instructions

### GPU Requirements

Some tutorials require GPU for reasonable execution times:
- `basics/03_parallel_training.ipynb` - Parallel PQN training with varying batch sizes

Run these on a machine with GPU before committing.

## Tips

- **Re-execute after code changes:** If you modify Myriad code that affects tutorial behavior, re-run affected tutorials
- **Clear outputs first:** To ensure clean re-execution, you can clear outputs before running:
  ```bash
  jupyter nbconvert --clear-output --inplace <notebook>.ipynb
  jupyter nbconvert --to notebook --execute <notebook>.ipynb --inplace
  ```
- **Check file sizes:** Videos and plots can make notebooks large. Ensure generated assets are reasonable size.
- **Test locally before pushing:** Run `cd docs && sphinx-build -W --keep-going -b html . _build/html` to verify docs build correctly with your changes.
