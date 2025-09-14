# Pipeline profiling

The training pipeline can be profiled to identify performance bottlenecks in
feature extraction, model fitting, and evaluation.

## Generating profiles

Use the helper script to run the pipeline under `cProfile` and emit reports:

```bash
python scripts/profile_pipeline.py DATA.csv OUTPUT_DIR --open
```

This creates several `.prof` files inside `OUTPUT_DIR/profiles` including
`feature_extraction.prof`, `model_fit.prof`, `evaluation.prof` and a
`full_pipeline.prof` capturing the entire run.

## Reading profiles

Profiles can be inspected with [snakeviz](https://jiffyclub.github.io/snakeviz/):

```bash
snakeviz OUTPUT_DIR/profiles/model_fit.prof
```

The browser interface shows call graphs and cumulative timings to guide
optimisation efforts.
