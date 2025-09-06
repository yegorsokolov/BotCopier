#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>
#include "proto/log_service.grpc.pb.h"
#include "proto/trade_event.pb.h"
#include "proto/metric_event.pb.h"

#include <queue>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

namespace {
class GrpcClient {
 public:
  void Init(const std::string& trade_host, int trade_port,
            const std::string& metric_host, int metric_port) {
    trade_stub_ = tbot::LogService::NewStub(
        grpc::CreateChannel(trade_host + ":" + std::to_string(trade_port),
                             grpc::InsecureChannelCredentials()));
    metric_stub_ = tbot::LogService::NewStub(
        grpc::CreateChannel(metric_host + ":" + std::to_string(metric_port),
                             grpc::InsecureChannelCredentials()));
  }

  void EnqueueTrade(const std::string& payload) {
    std::lock_guard<std::mutex> lock(mu_);
    trade_queue_.push(payload);
  }

  void EnqueueMetric(const std::string& payload) {
    std::lock_guard<std::mutex> lock(mu_);
    metric_queue_.push(payload);
  }

  void Flush() {
    FlushTrades();
    FlushMetrics();
  }

  int TradeQueueDepth() {
    std::lock_guard<std::mutex> lock(mu_);
    return trade_queue_.size();
  }

  int MetricQueueDepth() {
    std::lock_guard<std::mutex> lock(mu_);
    return metric_queue_.size();
  }

  int TradeRetryCount() { return trade_retry_count_.load(); }
  int MetricRetryCount() { return metric_retry_count_.load(); }

 private:
  void FlushTrades() {
    std::lock_guard<std::mutex> lock(mu_);
    int backoff_ms = 10;
    while (!trade_queue_.empty()) {
      const std::string payload = trade_queue_.front();
      tbot::TradeEvent req;
      req.set_payload(payload);
      google::protobuf::Empty resp;
      grpc::ClientContext ctx;
      grpc::Status status = trade_stub_->LogTrade(&ctx, req, &resp);
      if (!status.ok()) {
        trade_retry_count_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = std::min(backoff_ms * 2, 1000);
        continue;
      }
      trade_queue_.pop();
      backoff_ms = 10;
    }
  }

  void FlushMetrics() {
    std::lock_guard<std::mutex> lock(mu_);
    int backoff_ms = 10;
    while (!metric_queue_.empty()) {
      const std::string payload = metric_queue_.front();
      tbot::MetricEvent req;
      req.set_payload(payload);
      google::protobuf::Empty resp;
      grpc::ClientContext ctx;
      grpc::Status status = metric_stub_->LogMetrics(&ctx, req, &resp);
      if (!status.ok()) {
        metric_retry_count_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = std::min(backoff_ms * 2, 1000);
        continue;
      }
      metric_queue_.pop();
      backoff_ms = 10;
    }
  }

  std::unique_ptr<tbot::LogService::Stub> trade_stub_;
  std::unique_ptr<tbot::LogService::Stub> metric_stub_;
  std::queue<std::string> trade_queue_;
  std::queue<std::string> metric_queue_;
  std::mutex mu_;
  std::atomic<int> trade_retry_count_{0};
  std::atomic<int> metric_retry_count_{0};
};

GrpcClient g_client;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) void grpc_client_init(const char* trade_host, int trade_port,
                                         const char* metric_host, int metric_port) {
  g_client.Init(trade_host ? trade_host : "", trade_port,
                metric_host ? metric_host : "", metric_port);
}

__attribute__((visibility("default"))) void grpc_enqueue_trade(const char* payload) {
  if (payload)
    g_client.EnqueueTrade(payload);
}

__attribute__((visibility("default"))) void grpc_enqueue_metric(const char* payload) {
  if (payload)
    g_client.EnqueueMetric(payload);
}

__attribute__((visibility("default"))) void grpc_flush() {
  g_client.Flush();
}

__attribute__((visibility("default"))) int grpc_trade_queue_depth() {
  return g_client.TradeQueueDepth();
}

__attribute__((visibility("default"))) int grpc_metric_queue_depth() {
  return g_client.MetricQueueDepth();
}

__attribute__((visibility("default"))) int grpc_trade_retry_count() {
  return g_client.TradeRetryCount();
}

__attribute__((visibility("default"))) int grpc_metric_retry_count() {
  return g_client.MetricRetryCount();
}

}
