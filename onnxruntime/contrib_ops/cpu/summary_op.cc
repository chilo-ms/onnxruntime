// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "summary_op.h"
#include "core/providers/cpu/tensor/utils.h"

#include "tensorboard/compat/proto/summary.pb.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    SummaryScalar,
    9,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("S", DataTypeImpl::GetTensorType<std::string>()),
    SummaryScalarOp);

ONNX_CPU_OPERATOR_KERNEL(
    SummaryHistogram,
    9,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("S", DataTypeImpl::GetTensorType<std::string>()),
    SummaryHistogramOp);

ONNX_CPU_OPERATOR_KERNEL(
    SummaryMerge,
    9,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::GetTensorType<std::string>()),
    SummaryMergeOp);

SummaryScalarOp::SummaryScalarOp(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttrs("tags", tags_).IsOK(), "Attribute 'tags' must be specified and must be a tensor of strings.");
}

Status SummaryScalarOp::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  auto input_type = input.DataType();

  if (input_type == DataTypeImpl::GetType<float>())
    return ComputeImpl<float>(*context, input);
  else if (input_type == DataTypeImpl::GetType<double>())
    return ComputeImpl<double>(*context, input);

  ORT_THROW("SummaryScalar operator does not support ", input_type, " yet");
}

template <typename T>
Status SummaryScalarOp::ComputeImpl(OpKernelContext& context, const Tensor& input) const {
  ORT_RETURN_IF(static_cast<size_t>(input.Shape().Size()) != tags_.size(), "tags and input must have the same size");

  const T* input_data = input.Data<T>();

  tensorboard::Summary summary;
  for (size_t i = 0; i < tags_.size(); i++) {
    tensorboard::Summary::Value* summary_value = summary.add_value();
    summary_value->set_tag(tags_[i]);
    summary_value->set_simple_value(float(input_data[i]));
  }

  Tensor& output = *context.Output(0, {});
  *output.MutableData<std::string>() = summary.SerializeAsString();
  return Status::OK();
}

static std::vector<double> InitDefaultHistogramBuckets() {
  // Use the same default histogram buckets as Tensorflow
  // Buckets grow by 10% from 1e-12 to 1e20
  std::vector<double> buckets;
  double v = 1e-12;
  while (v < 1e20) {
    buckets.push_back(v);
    v *= 1.1;
  }
  buckets.push_back(std::numeric_limits<double>::max());

  // Copy (-buckets, 0, buckets) to limits.
  std::vector<double> bucket_limits(buckets.size() * 2 + 1, 0.0);
  for (size_t i = 0; i < buckets.size(); i++)
    bucket_limits[i] = -buckets[buckets.size() - 1 - i];
  bucket_limits[buckets.size()] = 0;
  for (size_t i = 0; i < buckets.size(); i++)
    bucket_limits[buckets.size() + i] = buckets[i];
  return bucket_limits;
}

static const std::vector<double>& DefaultHistogramBuckets() {
  static std::vector<double> bucket_limits = InitDefaultHistogramBuckets();
  return bucket_limits;
}

class Histogram final {
 public:
  Histogram() : bucket_limits_(DefaultHistogramBuckets()), buckets_(bucket_limits_.size(), 0.0) {}
  explicit Histogram(const std::vector<double> limits) : bucket_limits_(limits), buckets_(limits.size(), 0.0) {}

  void Add(double value) {
    if (value < min_)
      min_ = value;
    if (value > max_)
      max_ = value;
    num_++;
    sum_ += value;
    sum_squares_ += (value * value);

    int64_t bucket = std::upper_bound(bucket_limits_.begin(), bucket_limits_.end(), value) - bucket_limits_.begin();
    buckets_[bucket] += 1.0;
  }

  void SerializeToProto(tensorboard::HistogramProto& histogram) const {
    histogram.set_max(max_);
    histogram.set_min(min_);
    histogram.set_num(num_);
    histogram.set_sum(sum_);
    histogram.set_sum_squares(sum_squares_);

    for (size_t i = 0; i < buckets_.size(); i++) {
      if (buckets_[i] == 0.0 && (histogram.bucket_size() > 0 && histogram.bucket(histogram.bucket_size() - 1) == 0.0)) {
        // Merge neighboring empty buckets into one empty bucket, by expanding the existing bucket's limit.
        histogram.set_bucket_limit(histogram.bucket_limit_size() - 1, bucket_limits_[i]);
      } else {
        histogram.add_bucket(buckets_[i]);
        histogram.add_bucket_limit(bucket_limits_[i]);
      }
    }

    // Ensure we have at least one bucket.
    if (histogram.bucket_size() == 0) {
      histogram.add_bucket(0.0);
      histogram.add_bucket_limit(std::numeric_limits<double>::max());
    }
  }

 private:
  double min_ = std::numeric_limits<double>::max();
  double max_ = -std::numeric_limits<double>::max();
  double num_ = 0.0;
  double sum_ = 0.0;
  double sum_squares_ = 0.0;

  std::vector<double> bucket_limits_;
  std::vector<double> buckets_;
};

SummaryHistogramOp::SummaryHistogramOp(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttr("tag", &tag_).IsOK(), "Attribute 'tag' must be specified and must be a string.");
}

Status SummaryHistogramOp::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  auto input_type = input.DataType();

  if (input_type == DataTypeImpl::GetType<float>())
    return ComputeImpl<float>(*context, input);
  else if (input_type == DataTypeImpl::GetType<double>())
    return ComputeImpl<double>(*context, input);

  ORT_THROW("SummaryHistogramOp operator does not support ", input_type, " yet");
}

template <typename T>
Status SummaryHistogramOp::ComputeImpl(OpKernelContext& context, const Tensor& input) const {
  const T* input_data = input.Data<T>();
  const int64_t input_size = input.Shape().Size();

  Histogram histogram;
  for (int i = 0; i < input_size; i++) {
    double value = static_cast<double>(input_data[i]);
    ORT_RETURN_IF(std::isnan(value), "SummaryHistogram input contains a NaN value");
    ORT_RETURN_IF(std::isinf(value), "SummaryHistogram input contains an infinite value");
    histogram.Add(value);
  }

  tensorboard::Summary summary;
  tensorboard::Summary::Value* summary_value = summary.add_value();
  summary_value->set_tag(tag_);
  histogram.SerializeToProto(*summary_value->mutable_histo());

  Tensor& output = *context.Output(0, {});
  *output.MutableData<std::string>() = summary.SerializeAsString();
  return Status::OK();
}

SummaryMergeOp::SummaryMergeOp(const OpKernelInfo& info) : OpKernel(info) {
}

Status SummaryMergeOp::Compute(OpKernelContext* context) const {
  tensorboard::Summary summary;
  std::unordered_set<std::string> tags;

  for (int i = 0; i < context->InputCount(); i++) {
    const Tensor& input = *context->Input<Tensor>(i);
    ORT_RETURN_IF_NOT(input.DataType() == DataTypeImpl::GetType<std::string>(), "SummaryMerge input must be a string");
    const std::string* input_data = input.Data<std::string>();

    tensorboard::Summary input_summary;
    ORT_RETURN_IF_NOT(input_summary.ParseFromString(*input_data), "SummaryMerge failed to parse input tensor as a serialized Summary proto");

    for (int v = 0; v < input_summary.value_size(); v++) {
      const auto& summary_value = input_summary.value(v);
      const std::string& tag = summary_value.tag();
      if (!tag.empty() && !tags.insert(tag).second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SummaryMerge inputs contain duplicate tag: ", tag);
      }
      *summary.add_value() = summary_value;
    }
  }

  // Serialize merged Summary output
  Tensor& output = *context->Output(0, {});
  *output.MutableData<std::string>() = summary.SerializeAsString();
  return Status::OK();
}

}  // namespace contrib
};  // namespace onnxruntime
