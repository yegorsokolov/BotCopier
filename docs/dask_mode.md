# Dask Mode

BotCopier can operate in a parallel, out-of-core mode powered by [Dask](https://www.dask.org/).
Enable it by passing the `--dask` flag to any command that relies on
`botcopier.data.loading._load_logs`.

## Memory and Parallelism

Dask uses multiple worker processes to execute computations. The number of
workers and the memory available to each worker can be controlled with the
`DASK_WORKERS` and `DASK_MEMORY_LIMIT` environment variables respectively:

```bash
export DASK_WORKERS=2      # number of parallel workers
export DASK_MEMORY_LIMIT="1GB"  # per-worker memory cap
```

These settings allow feature extraction to scale beyond a single machine's
memory limits. Results are computed lazily and realised only at model-fit time
when `.compute()` is invoked.
