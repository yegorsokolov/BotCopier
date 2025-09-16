# Contributing to BotCopier

Development guidelines, coding standards, and testing expectations live in the
[MkDocs site](docs/contributing.md). Highlights:

- Install the project in editable mode together with the tooling listed in
  [docs/getting_started.md](docs/getting_started.md).
- Use the Typer CLI (`botcopier`) for smoke tests; run
  `botcopier train` and `botcopier evaluate` against the sample data before
  opening a pull request.
- Run `pre-commit run --all-files` so formatting, static analysis, and
  `nbstripout` all pass locally.
- Build the documentation with `mkdocs build --strict` to catch warnings early.

Issues and feature proposals can be discussed in GitHub before large changes.
