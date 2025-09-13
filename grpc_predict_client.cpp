#include <grpcpp/grpcpp.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "proto/predict.grpc.pb.h"

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  int port = 50052;
  std::vector<double> features;
  if (argc > 1) {
    std::stringstream ss(argv[1]);
    std::string item;
    while (std::getline(ss, item, ',')) {
      features.push_back(std::stod(item));
    }
  }
  auto channel = grpc::CreateChannel(host + ":" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  tbot::PredictService::Stub stub(channel);
  tbot::PredictRequest req;
  for (double f : features) req.add_features(f);
  tbot::PredictResponse resp;
  grpc::ClientContext ctx;
  grpc::Status status = stub.Predict(&ctx, req, &resp);
  if (!status.ok()) {
    std::cerr << "RPC failed: " << status.error_message() << std::endl;
    return 1;
  }
  std::cout << resp.prediction() << std::endl;
  return 0;
}
