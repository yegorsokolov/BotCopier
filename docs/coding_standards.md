# Coding Standards

The project uses [`pre-commit`](https://pre-commit.com/) to enforce style and static analysis.
The following tools run automatically:

- `black` for formatting
- `isort` for import ordering
- `ruff` for linting
- `mypy` for type checking

Run all checks locally before committing:
```bash
pre-commit run --all-files
```
