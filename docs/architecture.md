# System Architecture

The system consists of modular components for ingesting data, training models and deploying strategies.

```mermaid
flowchart TB
    logs[Log Ingestion] --> collector[OTel Collector]
    collector --> pipeline[Processing Pipeline]
    pipeline --> registry[(Model Registry)]
    registry --> deploy[Strategy Deployment]
```

See the [Data Flow](data_flow.md) page for a step-by-step walkthrough from ingestion to deployment.
