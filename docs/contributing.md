# Contributing

Thank you for your interest in improving BotCopier. This guide explains how to
set up a development environment, follow the project standards, and validate
changes before opening a pull request.

## Development Workflow

1. Fork the repository and create a feature branch.
2. Install the development dependencies described in
   [Getting Started](getting_started.md).
3. Keep the documentation up to date. When you add a new module or CLI command,
   document it and expose the API using the mkdocstrings directives in
   ``docs/api.md``.

## Coding Standards

* Review the [coding standards](coding_standards.md) document for naming,
  structure, and typing expectations.
* Prefer small, focused commits with descriptive messages.
* Run ``ruff`` or ``flake8`` locally if you have them installed; the CI will
  enforce linting on every push.
* Keep notebooks output-free. The ``nbstripout`` hook is configured in
  ``.pre-commit-config.yaml`` and should be run before pushing changes.

## Testing Checklist

Before submitting a pull request:

1. Run the automated tests:
   ```bash
   pytest
   ```
2. Execute representative smoke tests against the sample data to ensure the
   Typer CLI still succeeds end-to-end:
   ```bash
   botcopier train notebooks/data ./artifacts --model-type logreg --random-seed 7
   botcopier evaluate notebooks/data/predictions.csv notebooks/data/trades_raw.csv --window 900
   ```
3. Build the documentation and ensure no warnings are emitted:
   ```bash
   mkdocs build --strict
   ```
4. Use ``pre-commit`` to apply formatting and static analysis:
   ```bash
   pre-commit run --all-files
   ```

## Opening a Pull Request

* Provide a concise summary of the change, referencing any related issues.
* Include screenshots or terminal output when altering dashboards or CLI
  behaviour.
* Double-check that ``.github/workflows/docs.yml`` succeeds locally if your
  changes touch the documentation configuration.

## Documentation and notebooks

* Update ``docs/getting_started.md`` and ``docs/notebooks.md`` when onboarding
  flows change.
* Add or adjust notebooks under ``notebooks/`` when introducing new CLI
  workflows so readers can reproduce the behaviour interactively.
* Use ``mkdocs serve`` during development to preview documentation changes and
  ``mkdocs build --strict`` before pushing.

Please open an issue for major changes so we can discuss the approach and align
on deliverables before large investments of time.
