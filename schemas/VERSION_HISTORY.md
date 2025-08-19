# Version History

## 1
- Initial schema version. Observer messages prepend this version byte and consumers validate it.
- Logs and metric records include a `schema_version` field with consumers warning on mismatches.

## 2
- Added `exit_reason` and `duration_sec` fields to trade logs for improved exit analysis.
