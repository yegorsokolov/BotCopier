#!/bin/sh
# Example cron script to promote models and publish the best one.

REPO_DIR="/path/to/BotCopier"
MODELS_DIR="$REPO_DIR/models"
BEST_DIR="$MODELS_DIR/best"
FILES_DIR="/path/to/MT4/MQL4/Files"

python "$REPO_DIR/scripts/promote_best_models.py" "$MODELS_DIR" "$BEST_DIR" --files-dir "$FILES_DIR"
