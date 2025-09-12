# Getting Started

Follow these steps to set up the BotCopier project locally.

## Installation
1. Clone the repository and change into the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Training runs snapshot the environment to `dependencies.txt` in their output
   directories. Reinstall from this file to reproduce the exact package
   versions:
   ```bash
   pip install -r dependencies.txt
   ```

## Documentation
Build the documentation locally with:
```bash
mkdocs serve
```
This launches a local server with live reload.

## Running Tests
Execute the test suite to ensure everything is working:
```bash
pytest
```
