# System Architecture

BotCopier is organised as a set of cooperating services that ingest market
data, produce enriched features, train candidate models, and finally deploy the
most promising strategy. The following diagrams provide both high-level and
operational views using Mermaid and PlantUML so the topology can be explored in
multiple notations.

## High-level Pipeline (Mermaid)

```mermaid
flowchart TB
    subgraph ingest[Data Ingestion]
        logs[Exchange Logs]
        metrics[Metrics Stream]
    end
    subgraph enrich[Feature Engineering]
        cache[Feature Cache]
        registry[(Feature Registry)]
    end
    subgraph train[Training & Evaluation]
        trainer[AutoML Controller]
        scoring[Risk & Metric Scoring]
    end
    subgraph deploy[Deployment]
        registryModels[(Model Registry)]
        orchestrator[Strategy Orchestrator]
        traders[Live Trading Bots]
    end

    logs --> ingest
    metrics --> ingest
    ingest --> enrich
    enrich --> train
    train --> registryModels
    registryModels --> orchestrator --> traders
```

## Observability and Feedback Loops (Mermaid)

```mermaid
flowchart LR
    subgraph observability[Observability Stack]
        prom[Prometheus Metrics]
        otel[OpenTelemetry Collector]
        grafana[Grafana Dashboards]
    end
    trader[Trading Bot] --> prom
    trader --> otel
    otel --> grafana
    prom --> grafana
    grafana -->|Alerts| oncall[On-call Rotation]
    oncall -->|Feedback| trainer[AutoML Controller]
```

These feedback loops ensure that human operators can intervene quickly if a
strategy drifts away from acceptable behaviour. The [Data Flow](data_flow.md)
page provides an even more detailed walkthrough from ingestion to deployment.

## Deployment Topology (PlantUML)

```plantuml
@startuml
skinparam componentStyle rectangle
actor Trader
actor Analyst

rectangle "Inference Plane" {
  component "REST API" as Rest
  component "gRPC Stream" as Grpc
  component "Prediction Router" as Router
}

rectangle "Control Plane" {
  component "Model Registry" as Registry
  component "Pipeline Scheduler" as Scheduler
  component "Feature Store" as FeatureStore
}

Trader --> Router
Router --> Rest
Router --> Grpc
Rest --> Registry
Grpc --> Registry
Scheduler --> Registry
Scheduler --> FeatureStore
Analyst --> Scheduler
FeatureStore --> Router : feature snapshots
@enduml
```

This diagram highlights the control plane components that coordinate model
lifecycle management separately from the low-latency inference services. The
PlantUML and Mermaid variants use the same naming so teams familiar with either
notation can collaborate without friction.
