// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/training/runner/training_runner.h"

#include "gtest/gtest.h"

#include "test/training/runner/data_loader.h"
#include "test/training/runner/training_util.h"

namespace onnxruntime {
namespace training {
namespace test {

constexpr auto k_original_model_path = "./testdata/test_training_model.onnx";
constexpr auto k_backward_model_path = "./testdata/temp_backward_model.onnx";

constexpr auto k_output_directory = "./training_runner_test_output";

TEST(TrainingRunnerTest, Basic) {
  TrainingRunner::Parameters params{};
  params.model_path = k_original_model_path;
  params.model_with_training_graph_path = k_backward_model_path;
  params.output_dir = k_output_directory;
  params.is_perf_test = false;
  params.batch_size = 1;
  params.eval_batch_size = 1;
  params.num_train_steps = 1;
  params.display_loss_steps = 10;
  params.fetch_names = {"predictions"};
  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  TrainingRunner runner{params};

  auto status = runner.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<std::string> tensor_names{
      "X", "labels"};
  std::vector<TensorShape> tensor_shapes{
      {1, 784}, {1, 10}};
  std::vector<ONNX_NAMESPACE::TensorProto_DataType> tensor_types{
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT};
  
  auto data_set = std::make_shared<RandomDataSet>(1, tensor_names, tensor_shapes, tensor_types);
  auto data_loader = std::make_shared<SingleDataLoader>(data_set, tensor_names);

  status = runner.Run(data_loader, data_loader);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // TODO currently skipping load and evaluation of saved model, ideally that would be enabled
  status = runner.EndTraining(data_loader, false);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
