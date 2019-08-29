// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/function.h"
#include "core/mlas/inc/mlas.h"

#ifdef MICROSOFT_INTERNAL
#include "core/graph/contrib_ops/internal_schema_defs.h"
#endif

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void globalPoolTypeShapeInference(ONNX_NAMESPACE::InferenceContext& ctx);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

void NchwcPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNchwcDomain);
  schema.SinceVersion(1);
  schema.SetDoc(R"DOC(For internal use.)DOC");
  schema.Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"));
  schema.Attr("kernel_shape", "", AttributeProto::INTS);
  schema.Attr("dilations", "", AttributeProto::INTS, OPTIONAL);
  schema.Attr("strides", "", AttributeProto::INTS, OPTIONAL);
  schema.Attr("pads", "", AttributeProto::INTS, OPTIONAL);
  schema.Attr("ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0));
  schema.Input(0, "X", "", "T");
  schema.Output(0, "Y", "", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors");
  schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::convPoolShapeInference(ctx, true, true, 0, 1);
  });
}

void NchwcGlobalPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNchwcDomain);
  schema.SinceVersion(1);
  schema.SetDoc(R"DOC(For internal use.)DOC");
  schema.Input(0, "X", "", "T");
  schema.Output(0, "Y", "", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors");
  schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::globalPoolTypeShapeInference(ctx);
  });
}

void RegisterNchwcSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderInput)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to float/quantized tensors")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderOutput)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr(
          "channels",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to float/quantized tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);

        // Update the output shape with the actual number of channels.
        auto channels = getAttribute(ctx, "channels", 0);
        if (channels <= 0) {
          fail_shape_inference("invalid channel count");
        }
        auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        if (output_shape->dim_size() < 2) {
          fail_shape_inference("tensor rank too small");
        }
        auto* channels_dim = output_shape->mutable_dim(1);
        channels_dim->clear_dim_param();
        channels_dim->set_dim_value(channels);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Conv)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "dilations",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "pads",
          "",
          AttributeProto::INTS, OPTIONAL)
      .Attr(
          "group",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "activation",
          "",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "activation_params",
          "",
          AttributeProto::FLOATS,
          OPTIONAL)
      .Input(0, "X", "", "T")
      .Input(1, "W", "", "T")
      .Input(2, "B", "", "T", OpSchema::Optional)
      .Input(3, "Sum", "", "T", OpSchema::Optional)
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxPool)
      .FillUsing(NchwcPoolOpSchemaGenerator)
      .Attr(
          "storage_order",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(AveragePool)
      .FillUsing(NchwcPoolOpSchemaGenerator)
      .Attr(
          "count_include_pad",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalMaxPool)
      .FillUsing(NchwcGlobalPoolOpSchemaGenerator);

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalAveragePool)
      .FillUsing(NchwcGlobalPoolOpSchemaGenerator);
}

void RegisterContribSchemas() {
  // Register removed experimental ops for backward compatibility.
  // Experimental operators do not have version history. However, RS5 takes bunch of experimental operators
  // as production ops. In order to maintain backward compatibility when the experimental ops are removed from ONNX
  // they need to be added in onnxruntime as contrib ops.
  // ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance
  static const char* Affine_ver1_doc = R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(1)
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ParametricSoftplus_ver1_doc = R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(1)
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ImageScaler_ver1_doc =
      R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(1)
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* Crop_ver1_doc =
      R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(1)
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS, OPTIONAL)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.");

  static const char* ThresholdedRelu_ver1_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise. )DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ThresholdedRelu)
      .SinceVersion(1)
      .SetDoc(ThresholdedRelu_ver1_doc)
      .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
      .Input(0, "X", "Input tensor", "T")
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* DynamicSlice_ver1_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` inputs to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(1)
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GivenTensorFill)
      .SinceVersion(1)
      .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
      .Output(0, "X", "The filled tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .Attr("values", "", AttributeProto::FLOATS, OPTIONAL)
      .Attr("shape", "", AttributeProto::INTS, OPTIONAL)
      .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL)
      .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getAttribute("shape") != nullptr) {
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
          return;
        }
        // The type constraints above do not allow for input_as_shape
        // and may need to be fixed.
        if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
          return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
          ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
          for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
              fail_shape_inference(
                  "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
          }
          updateOutputShape(ctx, 0, shape);
        }
      });

  static const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Scale)
      .SinceVersion(1)
      .Input(0, "input", "Input data to be scaled", "T")
      .Output(0, "output", "Output data after scaling", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .SetDoc(Scale_ver1_doc)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(GRUUnit)
      .SinceVersion(1)
      .SetDoc(GRUUnit_ver1_doc)
      .Attr("drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length.",
            AttributeProto::INT, OPTIONAL)
      .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
      .Input(
          1,
          "gates",
          "Unactivated gate outputs from forget, update, "
          "and output gates, pre-activation.",
          "T")
      .Input(
          2,
          "seq_lengths",
          "Array of sequence lengths.  "
          "len(seq_lengths) should equal batch size N.",
          "T")
      .Input(3, "t", "The timestep for this operation.", "T")
      .Output(
          0,
          "hidden",
          "The new GRU hidden state calculated by this op.",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  static const char* ATen_ver1_doc = R"DOC(
Experimental allowing ATen operations to be accessed directly from Caffe2
to allow for quick prototyping when ONNX is missing standard versions of
and op)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ATen)
      .SinceVersion(1)
      .AllowUncheckedAttributes()
      .SetDoc(ATen_ver1_doc)
      .Input(0, "input", "Arbitrary input", "T", OpSchema::Variadic)
      .Output(0, "output", "Arbitrary output", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(bool)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(float16)",
           "tensor(float)",
           "tensor(double)"},
          "Constrain output types to bool, int32, int64, float16, float, double tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GivenTensorFill)
      .SinceVersion(10)
      .Deprecate()
      .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
      .Output(0, "X", "The filled tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .Attr("values", "", AttributeProto::FLOATS, OPTIONAL)
      .Attr("shape", "", AttributeProto::INTS, OPTIONAL)
      .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL)
      .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getAttribute("shape") != nullptr) {
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
          return;
        }
        // The type constraints above do not allow for input_as_shape
        // and may need to be fixed.
        if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
          return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
          ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
          for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
              fail_shape_inference(
                  "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
          }
          updateOutputShape(ctx, 0, shape);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Scale)
      .SinceVersion(10)
      .Deprecate()
      .Input(0, "input", "Input data to be scaled", "T")
      .Output(0, "output", "Output data after scaling", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .SetDoc(Scale_ver1_doc)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(GRUUnit)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(GRUUnit_ver1_doc)
      .Attr("drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length.",
            AttributeProto::INT, OPTIONAL)
      .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
      .Input(
          1,
          "gates",
          "Unactivated gate outputs from forget, update, "
          "and output gates, pre-activation.",
          "T")
      .Input(
          2,
          "seq_lengths",
          "Array of sequence lengths.  "
          "len(seq_lengths) should equal batch size N.",
          "T")
      .Input(3, "t", "The timestep for this operation.", "T")
      .Output(
          0,
          "hidden",
          "The new GRU hidden state calculated by this op.",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ATen)
      .SinceVersion(10)
      .Deprecate()
      .AllowUncheckedAttributes()
      .SetDoc(ATen_ver1_doc)
      .Input(0, "input", "Arbitrary input", "T", OpSchema::Variadic)
      .Output(0, "output", "Arbitrary output", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(bool)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(float16)",
           "tensor(float)",
           "tensor(double)"},
          "Constrain output types to bool, int32, int64, float16, float, double tensors.");

  ONNX_OPERATOR_SCHEMA(MeanVarianceNormalization)
      .SinceVersion(1)
      .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
      .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT, static_cast<int64_t>(1))
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_OPERATOR_SCHEMA(ScaledTanh)
      .SinceVersion(1)
      .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "input", "Input tensor", "T")
      .Output(
          0,
          "output",
          "The scaled hyperbolic tangent values of the input tensor "
          "computed element-wise",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        auto* output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        if (ONNX_NAMESPACE::hasNInputShapes(ctx, 1)) {
          const auto& input_shape =
              ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank =
              input_shape.dim_size();
          if (input_rank != 4)
            fail_shape_inference("Input's shape must be 4-D");

          // parse necessary attributes for futher processing
          std::vector<int64_t> border;
          bool border_present =
              getRepeatedAttribute(ctx, "border", border);
          if (!border_present || border.size() != 4)
            fail_shape_inference(
                "'Border' attribute must be present and must contain exactly 4 values - "
                "(left_border, top_border, right_border, bottom_border)");

          std::vector<int64_t> scale;
          bool scale_present =
              getRepeatedAttribute(ctx, "scale", scale);
          if (scale_present && scale.size() != 2)
            fail_shape_inference("'Scale' must contain exactly 2 values - (height, width)");

          // actual shape inference processing
          // [N, C] can be copied over from the input as is
          *output_shape->mutable_dim(static_cast<int>(0)) = input_shape.dim(static_cast<int>(0));
          *output_shape->mutable_dim(static_cast<int>(1)) = input_shape.dim(static_cast<int>(1));

          // process 'H' and 'W'
          if (!input_shape.dim(static_cast<int>(2)).has_dim_value() ||
              !input_shape.dim(static_cast<int>(3)).has_dim_value()) {
            // either height and width input has symbolic dims, so can't proceed further
            // add two dims as placeholders for output_H and output_W and return
            output_shape->add_dim();
            output_shape->add_dim();
            return;
          }

          int64_t H = input_shape.dim(static_cast<int>(2)).dim_value();
          int64_t W = input_shape.dim(static_cast<int>(3)).dim_value();

          int64_t left_border = border[0],
                  top_border = border[1],
                  right_border = border[2],
                  bottom_border = border[3];

          if (H < top_border + bottom_border)
            fail_shape_inference("Input's height (", H,
                                 ") needs to be greater than or equal to "
                                 "the top_border (",
                                 top_border, ") + bottom_border (", bottom_border, ")");

          if (W < left_border + right_border)
            fail_shape_inference("Input's width (", W,
                                 ") needs to be greater than or equal to "
                                 "the left_border (",
                                 left_border, ") + right_border (", right_border, ")");

          int64_t bottom_limit = H - bottom_border;
          int64_t right_limit = W - right_border;

          // scale = (height, width)
          if (!scale.empty()) {
            bottom_limit = top_border + scale[0];
            right_limit = left_border + scale[1];

            if (H < bottom_limit)
              fail_shape_inference("Input's height (", H, ") needs to be greater than or equal to the top_border (", top_border, ") + scale[0] (", scale[0], ")");

            if (W < right_limit)
              fail_shape_inference("Input's width (", W, ") needs to be greater than or equal to the left_border (", left_border, ") + scale[1] (", scale[1], ")");
          }

          auto* h_output_dim = output_shape->add_dim();
          h_output_dim->set_dim_value(bottom_limit - top_border);

          auto* w_output_dim = output_shape->add_dim();
          w_output_dim->set_dim_value(right_limit - left_border);

        } else {
          // Rank Inference at the very least
          // (We know that the output is going to be 4-D)
          for (int i = 0; i < 4; ++i) {
            output_shape->add_dim();
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_OPERATOR_SCHEMA(ScaledTanh)
      .SinceVersion(10)
      .Deprecate()
      .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
      .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
      .Input(0, "input", "Input tensor", "T")
      .Output(
          0,
          "output",
          "The scaled hyperbolic tangent values of the input tensor "
          "computed element-wise",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  // End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance

  ONNX_CONTRIB_OPERATOR_SCHEMA(SampleOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(
Sample echo operator.)DOC");

  // register schemas for more operators here
  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxpoolWithMask)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr("pads",
            "",
            AttributeProto::INTS, OPTIONAL)
      .Attr(
          "storage_order",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "strides", "", AttributeProto::INTS, OPTIONAL)
      .Input(
          0,
          "X",
          "",
          "T")
      .Input(1, "M", "mask", "tensor(int32)")
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input0 and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvTransposeWithDynamicPads)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC()DOC")
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr("output_padding",
            "",
            AttributeProto::INTS,
            OPTIONAL)
      .Attr(
          "dilations",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "group",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Input(
          0,
          "X",
          "",
          "T")
      .Input(
          1,
          "W",
          "",
          "T")
      .Input(2, "Pads", "", "tensor(int64)", OpSchema::Optional)
      .Input(3, "B", "", "T", OpSchema::Optional)
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedConv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.)DOC")
      .Attr(
          "auto_pad",
          "",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr(
          "kernel_shape",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "dilations",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "strides",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "pads",
          "",
          AttributeProto::INTS,
          OPTIONAL)
      .Attr(
          "group",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .Attr(
          "activation",
          "",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "activation_params",
          "",
          AttributeProto::FLOATS,
          OPTIONAL)
      .Input(
          0,
          "X",
          "",
          "T")
      .Input(
          1,
          "W",
          "",
          "T")
      .Input(
          2,
          "B",
          "",
          "T",
          OpSchema::Optional)
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedGemm)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.)DOC")
      .Input(
          0,
          "A",
          "Input tensor A. "
          "The shape of A should be (M, K) if transA is 0, "
          "or (K, M) if transA is non-zero.",
          "T")
      .Input(
          1,
          "B",
          "Input tensor B. "
          "The shape of B should be (K, N) if transB is 0, "
          "or (N, K) if transB is non-zero.",
          "T")
      .Input(
          2,
          "C",
          "Input tensor C. "
          "The shape of C should be unidirectional broadcastable to (M, N).",
          "T")
      .Output(0, "Y", "Output tensor of shape (M, N).", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)",
           "tensor(float)",
           "tensor(double)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(int32)",
           "tensor(int64)"},
          "Constrain input and output types to float/int tensors.")
      .Attr(
          "transA",
          "Whether A should be transposed",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "transB",
          "Whether B should be transposed",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "alpha",
          "Scalar multiplier for the product of input tensors A * B.",
          AttributeProto::FLOAT,
          1.0f)
      .Attr(
          "beta",
          "Scalar multiplier for input tensor C.",
          AttributeProto::FLOAT,
          1.0f)
      .Attr(
          "activation",
          "",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "leaky_relu_alpha",
          "",
          AttributeProto::FLOAT,
          OPTIONAL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (hasNInputShapes(ctx, 2)) {
          auto transAAttr = ctx.getAttribute("transA");
          bool transA =
              transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
          auto transBAttr = ctx.getAttribute("transB");
          bool transB =
              transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
          auto& first_input_shape = getInputShape(ctx, 0);
          auto& second_input_shape = getInputShape(ctx, 1);
          if (first_input_shape.dim_size() != 2)
            fail_shape_inference("First input does not have rank 2");
          if (second_input_shape.dim_size() != 2)
            fail_shape_inference("Second input does not have rank 2");
          updateOutputShape(
              ctx,
              0,
              {first_input_shape.dim(transA ? 1 : 0),
               second_input_shape.dim(transB ? 0 : 1)});
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ExpandDims)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Input(1, "axis", "Specified axis to insert a dimension", "tensor(int32)")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::all_tensor_types(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        const int rank = input_shape.dim_size();
        const ONNX_NAMESPACE::TensorProto* axis_initializer = ctx.getInputData(1);
        if (!axis_initializer)
          return;
        const int axis = axis_initializer->int32_data()[0];
        if (axis > rank || axis < -rank - 1) {
          fail_shape_inference("Input axis is invalid: ", axis);
        }
        int pos = axis >= 0 ? axis : rank + axis - 1;
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        for (int i = 0; i < pos; ++i) {
          output_shape.add_dim();
          *(output_shape.mutable_dim(i)) = input_shape.dim(i);
        }
        output_shape.add_dim();
        output_shape.mutable_dim(pos)->set_dim_value(1);
        for (int i = pos + 1; i < rank + 1; ++i) {
          output_shape.add_dim();
          *(output_shape.mutable_dim(i)) = input_shape.dim(i - 1);
        }
        updateOutputShape(ctx, 0, output_shape);
      })
      .SetDoc(R"DOC(ExpandDims echo operator.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);

  static const char* Tokenizer_ver1_doc = R"DOC(
  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by matching and removing separators.
  "separators" is a list of strings which are regular expressions. "tokenexp" is a single regular expression.
  Let's assume "separators" is [" "] and consider an example.
  If input is
  ["Hello World", "I love computer science !"] whose shape is [2],
  then the output would be
 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]
 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.
 If "separators" contains a single empty string, the Tokenizer will enter into character tokenezation mode. This means all strings
 will be broken part into individual characters.
 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.
  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
If tokenizer removes the entire content of [C]-input, it will produce [[]].
I.e. the output shape should be [C][0] or [N][C][0] if input shape was [N][C].
If the tokenizer receives empty input of [0] then the output is [0] if empty input
of [N, 0] then [N, 0].
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Tokenizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "Strings to tokenize", "T")
      .Output(0, "Y", "Tokenized strings", "T")
      .TypeConstraint(
          "T",
          {"tensor(string)"},
          "Input/Output is a string tensor")
      .Attr(
          "mark",
          "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03).",
          AttributeProto::INT)
      .Attr(
          "pad_value",
          "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers.",
          AttributeProto::STRING)
      .Attr(
          "tokenexp",
          "An optional string. Token's regular expression in basic POSIX format"
          " (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09_03)."
          " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only of"
          " 'tokenexp' and 'separators' should be set.",
          AttributeProto::STRING,
          OPTIONAL)
      .Attr(
          "separators",
          "an optional list of strings attribute that contains a list of separators - regular expressions to match separators"
          " Two consecutive segments in X connected by a separator would be divided into two tokens."
          " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
          " the corresponding output would be [\"Hello\", \"World!\"]. To achieve character-level tokenization,"
          " one should set the 'separators' to [\"\"], which contains an empty string.",
          AttributeProto::STRINGS,
          OPTIONAL)
      .Attr(
          "mincharnum",
          "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored",
          AttributeProto::INT)
      .SetDoc(Tokenizer_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        auto& input_shape = getInputShape(ctx, 0);
        auto& dims = input_shape.dim();
        if (dims.size() < 1 || dims.size() > 2) {
          fail_shape_inference("Input dimensions are either [C] or [N][C] allowed");
        }

        int64_t size = 1;
        for (auto& dim : dims) {
          if (dim.has_dim_value()) {
            size *= dim.dim_value();
          }
        }

        if (size > 0) {
          for (auto& dim : dims) {
            *output_shape.add_dim() = dim;
          }
          // Add the last unknown dimension
          // only if the input is not empty
          output_shape.add_dim();
        } else if (size == 0) {
          if (dims.size() == 2) {
            *output_shape.add_dim() = dims[0];
          }
          output_shape.add_dim()->set_dim_value(0);
        }
        updateOutputShape(ctx, 0, output_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceSumInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.)DOC")
      .Input(0, "data", "An input tensor.", "T1")
      .Output(0, "reduced", "Reduced output tensor.", "T2")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
      .TypeConstraint("T2",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output data type to 32-bit integer tensor."
                      "T2 must be tensor(uint32) when T1 is tensor(uint8),"
                      "or must be tensor(int32) when T1 is tensor(int8).")
      .Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.",
          AttributeProto::INTS)
      .Attr(
          "keepdims",
          "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
          AttributeProto::INT);

  ONNX_CONTRIB_OPERATOR_SCHEMA(MurmurHash3)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.)DOC")
      .Input(0, "X", "An input tensor to hash.", "T1")
      .Output(0, "Y", "32-bit hash value.", "T2")
      .TypeConstraint("T1", {"tensor(uint32)", "tensor(int32)", "tensor(string)"}, "Constrain input type to unsigned or signed 32-bit integer tensor, or string tensor. It should be utf-8 encoded if using unicode.")
      .TypeConstraint("T2", {"tensor(uint32)", "tensor(int32)"}, "Constrain output type to unsigned and signed 32-bit integer tensor.")
      .Attr(
          "seed",
          "Seed for the hashing algorithm, unsigned 32-bit integer, default to 0.",
          AttributeProto::INT,
          (int64_t)0LL)
      .Attr(
          "positive",
          "If value is 1, output type is uint32_t, else int32_t. Default value is 1.",
          AttributeProto::INT,
          (int64_t)1LL)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // type inference
        auto positive_attr = ctx.getAttribute("positive");
        bool is_positive =
            positive_attr ? (static_cast<int>(positive_attr->i()) == 1 ? true : false) : true /* default value if attribute not present */;
        auto output_data_type = ctx.getOutputType(0)->mutable_tensor_type();
        if (is_positive) {
          output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32);
        } else {
          output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
        }

        // Shape inference
        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherND)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .Attr(
          "axis",
          "The number of batch dims. The gather of indexing starts from dimension of data[axis:]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "data", "Tensor of rank r >= 1.", "T")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Output(0, "output", "Tensor of rank q-1+r-indices[-1].", "T")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain input and output types to any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indice type to int32 or int64")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 2)) {
          return;
        }
        auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
        auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
        auto data_rank = data_shape.dim_size();
        auto indices_rank = indices_shape.dim_size();
        auto axis = ctx.getAttribute("axis");
        int64_t axis_data = axis ? static_cast<int>(axis->i()) : 0;
        if (data_rank < 1 || indices_rank < 1) {
          fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
        }
        auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value() + axis_data;
        if (last_indice_dimension > data_rank) {
          fail_shape_inference("last dimension of indices must not be larger and rank of data tensor");
        }
        for (int i = 0; i < indices_rank - 1; ++i) {
          *ctx.getOutputType(0)
               ->mutable_tensor_type()
               ->mutable_shape()
               ->add_dim() = indices_shape.dim(i);
        }
        for (int i = static_cast<int>(last_indice_dimension); i < data_rank; ++i) {
          *ctx.getOutputType(0)
               ->mutable_tensor_type()
               ->mutable_shape()
               ->add_dim() = data_shape.dim(i);
        }
      })
      .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]]
)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherNDGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .Attr(
          "axis",
          "The number of batch dims. The gather of indexing starts from dimension of data[axis+1:]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "shape", "The shape of source data input of GatherND.", "T1")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Input(2, "update", "The gradient of the output.", "T")
      .Output(0, "output", "Tensor graident of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indice type to int32 or int64")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain shape type to int64");

  ONNX_CONTRIB_OPERATOR_SCHEMA(WordConvEmbedding)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "embedding_size",
          "Integer representing the embedding vector size for each word."
          "If not provide, use the fileter size of conv weight",
          AttributeProto::INT,
          OPTIONAL)
      .Attr(
          "conv_window_size",
          "This operator applies convolution to word from left to right with window equal to conv_window_size and stride to 1."
          "Take word 'example' for example, with conv_window_size equal to 2, conv is applied to [ex],[xa], [am], [mp]..."
          "If not provide, use the first dimension of conv kernal shape.",
          AttributeProto::INT,
          OPTIONAL)
      .Attr(
          "char_embedding_size",
          "Integer representing the embedding vector size for each char."
          "If not provide, use the char embedding size of embedding vector.",
          AttributeProto::INT,
          OPTIONAL)
      .Input(0, "Sequence", "Specify batchs of sequence words to embedding", "T")
      .Input(1, "W", "Specify weights of conv", "T1")
      .Input(2, "B", "Specify bias of conv", "T1")
      .Input(3, "C", "Specify embedding vector of char", "T1")
      .Output(0, "Y", "output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(int32)"},
          "Constrain to tensor(int32).")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain to tensor(float).")
      .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Pad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "mode",
          "Three modes: `constant`(default) - pads with a given constant value, "
          "`reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis, "
          "`edge` - pads with the edge values of array",
          AttributeProto::STRING,
          std::string("constant"))
      .Input(0, "data", "Input tensor.", "T")
      .Input(
          1,
          "pads",
          "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
          "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
          "`pads` should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]. "
          "`pads` format (1D example) should be as follow [x1_begin, x2_begin,...,x1_end, x2_end,...], "
          "where xi_begin is the number of pixels added at the beginning of axis `i` and "
          "xi_end, the number of pixels added at the end of axis `i`.",
          "tensor(int64)")
      .Input(
          2,
          "value",
          "(Optional) A scalar or rank 1 tensor containing a single value to be filled if the mode chosen is `constant` (by default it is 0.0).",
          "T",
          OpSchema::Optional)
      .Output(0, "output", "Tensor after padding.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        // Shape inference needs the input data shape
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        const auto input_rank = input_shape.dim_size();

        // Infer output shape if 'pads' tensor is available
        const auto* pads_initializer = ctx.getInputData(1);
        if (nullptr != pads_initializer) {
          const auto& pads_shape = ctx.getInputType(1)->tensor_type().shape();
          if ((pads_initializer->dims_size() != 1 &&
               pads_initializer->dims_size() != 2) ||
              (pads_initializer->dims_size() == 2 &&
               pads_shape.dim(static_cast<int>(0)).dim_value() != 1) ||
              pads_initializer->data_type() != ONNX_NAMESPACE::TensorProto::INT64)
            fail_shape_inference(
                "'pads' input must be a 1D (shape: [input_rank]) "
                "or 2D tensor (shape: [1, input_rank]) of type int64");

          // make a copy of the returned const vector - may have to resize
          // this in next step
          std::vector<int64_t> pads_data;
          if (pads_initializer->has_raw_data())
            return;
          else
            pads_data.insert(
                pads_data.end(),
                pads_initializer->int64_data().begin(),
                pads_initializer->int64_data().end());

          // fill with zeros if needed to reach appropriate size
          if (pads_data.size() != 2 * static_cast<size_t>(input_rank))
            pads_data.resize(2 * input_rank, 0);

          const auto& output_shape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
            const auto& input_dim = input_shape.dim(static_cast<int>(i));
            auto* output_dim = output_shape->add_dim();
            if (input_dim.has_dim_value()) {
              output_dim->set_dim_value(
                  input_dim.dim_value() + pads_data[i] + pads_data[i + input_rank]);
            } else if (pads_data[i] + pads_data[i + input_rank] == 0) {
              *output_dim = input_dim;
            }
          }
        } else {
          // Infer output shapes' rank in any case
          auto* output_shape_0 = getOutputShape(ctx, 0);
          for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
            output_shape_0->add_dim();
          }
        }
        return;
      })
      .SetDoc(R"DOC(
            Given `data` tensor, pads, mode, and value.
            Example:
            Insert 0 pads to the beginning of the second dimension.
            data = [
                    [1.0, 1.2],
                    [2.3, 3.4],
                    [4.5, 5.7],
                    ]
            pads = [0, 2, 0, 0]
            output = [
                    [
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 2.3, 3.4],
                    [0.0, 0.0, 4.5, 5.7],
                    ],
                    ]
            )DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Unique)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "x", "A 1-D input tensor that is to be processed.", "T")
      .Output(0, "y",
              "A 1-D tensor of the same type as 'x' "
              "containing all the unique values in 'x' sorted "
              "in the same order that they occur in the input 'x'",
              "T")
      .Output(1, "idx",
              "A 1-D INT64 tensor of the same size as 'x' "
              "containing the indices for each value in 'x' "
              "in the output 'uniques'",
              "tensor(int64)")
      .Output(2, "counts",
              "A 1-D INT64 tensor containing the "
              "the count of each element "
              "of 'uniques' in the input 'x'",
              "tensor(int64)")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Input can be of any tensor type.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::INT64);
        ONNX_NAMESPACE::updateOutputElemType(ctx, 2, ONNX_NAMESPACE::TensorProto::INT64);

        // Shape inference

        // shape of output 'uniques' and 'counts'
        // depends on actual input data, but the rank is always 1
        ctx.getOutputType(0)
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim();

        ctx.getOutputType(2)
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim();

        // if the input shape doesn't exist, further shape inference is not possible
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        // 'idx' output has same shape as input
        ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 1);

        return;
      })
      .SetDoc(R"DOC(
              Finds all the unique values (deduped list) present in the given input tensor.
              This operator returns 3 outputs.
              The first output tensor 'uniques' contains all of the unique elements of the input,
              sorted in the same order that they occur in the input.
              The second output tensor 'idx' is the same size as the input and it contains the index
              of each value of the input in 'uniques'.
              The third output tensor 'counts' contains the count of each element of 'uniques' in the input.
              Example:
                input_x = [2, 1, 1, 3, 4, 3]
                output_uniques = [2, 1, 3, 4]
                output_idx = [0, 1, 1, 2, 3, 2]
                output_counts = [1, 2, 2, 1]
              )DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(CropAndResize)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "mode",
          "The pooling method. Two modes are supported: 'bilinear' and 'nearest'. "
          "Default is 'bilinear'.",
          AttributeProto::STRING,
          std::string("bilinear"))
      .Attr(
          "extrapolation_value",
          "Value used for extrapolation, when applicable. "
          "Default is 0.0f. ",
          AttributeProto::FLOAT,
          0.f)
      .Input(
          0,
          "X",
          "Input data tensor from the previous operator; "
          "4-D feature map of shape (N, C, H, W), "
          "where N is the batch size, C is the number of channels, "
          "and H and W are the height and the width of the data.",
          "T1")
      .Input(
          1,
          "rois",
          "RoIs (Regions of Interest) to pool over; rois is "
          "2-D input of shape (num_rois, 4) given as "
          "[[y1, x1, y2, x2], ...]. "
          "The RoIs' coordinates are normalized in the coordinate system of the input image. "
          "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.",
          "T1")
      .Input(
          2,
          "batch_indices",
          "1-D tensor of shape (num_rois,) with each element denoting "
          "the index of the corresponding image in the batch.",
          "T2")
      .Input(
          3,
          "crop_size",
          "1-D tensor of 2 elements: [crop_height, crop_width]. "
          "All cropped image patches are resized to this size. Both crop_height and crop_width need to be positive.",
          "T2")
      .Output(
          0,
          "Y",
          "RoI pooled output, 4-D tensor of shape "
          "(num_rois, C, crop_height, crop_width). The r-th batch element Y[r-1] "
          "is a pooled feature map corresponding to the r-th RoI X[r-1].",
          "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int32)"},
          "Constrain types to int tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (!hasNInputShapes(ctx, 4)) {
          return;
        }
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        auto& input_shape = getInputShape(ctx, 0);
        auto& rois_shape = getInputShape(ctx, 1);
        auto& batch_index_shape = getInputShape(ctx, 2);
        auto& crop_size_shape = getInputShape(ctx, 3);

        if (input_shape.dim_size() != 4) {
          fail_shape_inference("first input tensor has wrong dimension");
        }
        if (rois_shape.dim_size() != 2) {
          fail_shape_inference("rois input tensor has wrong dimension");
        }
        if (batch_index_shape.dim_size() != 1) {
          fail_shape_inference("batch_indices shape input tensor has wrong dimension");
        }
        if (crop_size_shape.dim_size() != 1) {
          fail_shape_inference("crop_size shape input tensor has wrong dimension");
        }
      })
      .SetDoc(R"DOC(
        Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling
        (possibly with aspect ratio change) to a common output size specified by crop_height and crop_width.
        Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.
        The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to
        a fixed size = [crop_height, crop_width]. The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth].
        The resizing is corner aligned.)DOC");

  // Register the NCHWc schemas if supported by the platform.
  if (MlasNchwcGetBlockSize() > 1) {
    RegisterNchwcSchemas();
  }

  // TODO: push this to ONNX
  static const char* reduction_doc =
      "Type of reduction to apply to loss: none, sum, mean(default). "
      "'none': the output is the loss for each sample in the batch."
      "'sum': the output will be summed. "
      "'mean': the sum of the output will be divided by the batch_size.";

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropy)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, N-D input of shape (-1, num_classes).", "T")
      .Input(1, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "Y", "loss.", "T")
      .Output(1, "probability", "softmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropy)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "probability", "normalized exponential probabilities, N-D input of shape (-1, num_classes).", "T")
      .Input(2, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodAllReduce)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Input(0, "input", "tensor to be reduced", "T")
      .Output(0, "output", "reduced tensor", "T")
      .Output(1, "ready", "true when reduced tensor is ready", "B")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain to bool tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodBarrier)
      .SetDomain(kOnnxDomain)
      .SetDoc("Waits for one or more async Horovod operators to complete")
      .SinceVersion(9)
      .Input(0, "input", "input tensor", "T")
      .Input(1, "input_ready", "one or more bool tensors to wait on", "B", OpSchema::Variadic)
      .Output(0, "output", "output tensor", "T")
      .Output(1, "output_ready", "output tensor is ready", "B")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, (N+1)-D input of shape (-1, num_classes).", "T")
      .Input(1, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(2, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "Y", "loss.", "T")
      .Output(1, "probability", "softmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropy)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "probability", "normalized exponential probabilities, (N+1)-D input of shape (batch_size).", "T")
      .Input(2, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(3, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropout)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TrainableDropout")
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "ratio",
             "The ratio of random dropout, with value in [0, 1]. If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "output", "The output.", "T")
      .Output(1, "mask", "The output mask.", "T2", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain output 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        if (ctx.getNumOutputs() == 2) {
          updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropoutGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TrainableDropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask tensor of the dropout. ", "T2")
      .Input(2, "ratio",
             "The ratio of random dropout, with value in [0, 1]. If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "dx", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeEncoder)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "uncompressed output", "T")
      .Output(1, "Y1", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeDecoder)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Input(0, "X1", "dummy input for late decoding", "T")
      .Input(1, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SinGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Sin")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Sin output's grad", "T")
      .Output(0, "dX", "Sin input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"X_1"}, "Cos", {"X"}},
           {{"dX"}, "Mul", {"X_1", "dY"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(TanhGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Tanh")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Tanh output's grad", "T")
      .Output(0, "dX", "Tanh input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           {{"Squared_output"}, "Mul", {"X", "X"}},
           {{"Tanh_Grad"}, "Sub", {"One", "Squared_output"}},
           {{"dX"}, "Mul", {"dY", "Tanh_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SqrtGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Sqrt")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Sqrt output's grad", "T")
      .Output(0, "dX", "Sqrt input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One_half", 0.5f),
           {{"Sqrt_Grad"}, "Div", {"One_half", "X"}},
           {{"dX"}, "Mul", {"dY", "Sqrt_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ErfGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Erf")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Erf output's grad", "T")
      .Output(0, "dX", "Erf input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Two_sqrt_pi", static_cast<float>(M_2_SQRTPI)),
           {{"Square_x"}, "Mul", {"X", "X"}},
           {{"Neg_Square_x"}, "Neg", {"Square_x"}},
           {{"Exp_Neg_Square_x"}, "Exp", {"Neg_Square_x"}},
           {{"Erf_Grad"}, "Mul", {"Two_sqrt_pi", "Exp_Neg_Square_x"}},
           {{"dX"}, "Mul", {"dY", "Erf_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReshapeGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Reshape")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Reshape output's grad", "T")
      .Output(0, "dX", "REshape input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"x_shape"}, "Shape", {"X"}},
           {{"dX"}, "Reshape", {"dY", "x_shape"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(PowGradient)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Pow")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "Exponent", "Input tensor", "T")
      .Input(2, "dY", "Reshape output's grad", "T")
      .Output(0, "dX", "Pow input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           {{"p_minus_one"}, "Sub", {"Exponent", "One"}},
           {{"X_Pow_p_minus_one"}, "Pow", {"X", "p_minus_one"}},
           {{"a_X_Pow_p_minus_one"}, "Mul", {"X_Pow_p_minus_one", "Exponent"}},
           {{"dX"}, "Mul", {"a_X_Pow_p_minus_one", "dY"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryScalar)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryScalar")
      .Attr("tags", "The tags corresponding to each input scalar.", AttributeProto::STRINGS)
      .Input(0, "input", "The scalar tensor to summarize as simple values.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input type to float tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryHistogram)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryHistogram")
      .Attr("tag", "The tag corresponding to the histogram data.", AttributeProto::STRING)
      .Input(0, "input", "The scalar tensor to produce a histogram over.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input type to float tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryMerge)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryMerge")
      .Input(0, "input", "One or more serialized Tensorboard Summary tensors to merge into a single Summary.", "S", OpSchema::Variadic)
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryText)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryText")
      .Attr("tag", "The tag corresponding to the text data.", AttributeProto::STRING)
      .Input(0, "input", "The string tensor to render in the Tensorboard Text dashboard.", "S")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Gelu)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gelu")
      .Input(0, "X", "The input data as Tensor.", "T")
      .Output(0, "Y", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Sqrt_two", static_cast<float>(M_SQRT2)),
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One_half", 0.5f),
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           {{"X_1"}, "Mul", {"X", "One_half"}},
           {{"X_2"}, "Div", {"X", "Sqrt_two"}},
           {{"X_3"}, "Erf", {"X_2"}},
           {{"X_4"}, "Add", {"X_3", "One"}},
           {{"Y"}, "Mul", {"X_1", "X_4"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(GeluGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("GeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Sqrt_two", static_cast<float>(M_SQRT2)),
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One_half", 0.5f),
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Two_sqrt_pi", static_cast<float>(M_2_SQRTPI)),
           {{"X_1"}, "Mul", {"X", "One_half"}},
           {{"X_2"}, "Div", {"X", "Sqrt_two"}},
           {{"X_3"}, "Erf", {"X_2"}},
           {{"X_4"}, "Add", {"X_3", "One"}},
           {{"X_5_grad"}, "Mul", {"dY", "X_4"}},
           {{"X_6_grad"}, "Mul", {"X_5_grad", "One_half"}},
           {{"X_7"}, "Mul", {"X_2", "X_2"}},
           {{"X_8"}, "Neg", {"X_7"}},
           {{"X_9"}, "Exp", {"X_8"}},
           {{"X_10_grad"}, "Mul", {"Two_sqrt_pi", "X_9"}},
           {{"X_11_grad"}, "Mul", {"dY", "X_1"}},
           {{"X_12_grad"}, "Mul", {"X_11_grad", "X_10_grad"}},
           {{"X_13"}, "Div", {"One", "Sqrt_two"}},
           {{"X_14_grad"}, "Mul", {"X_12_grad", "X_13"}},
           {{"dX"}, "Sum", {"X_14_grad", "X_6_grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalization)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalization")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "scale", "Scale tensor.", "T")
      .Input(2, "B", "Bias tensor.", "T")
      .Output(0, "Y", "Output data tensor.", "T")
      .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
      .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain mean and inv_std_var to be float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        auto saved_mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
        auto saved_inv_std_var_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
          axis = axis_proto->i();
        }
        if (axis < 0) {
          axis += input_ndim;
        }
        for (int i = 0; i < input_ndim; ++i) {
          if (i != axis) {
            auto dim = saved_mean_shape->add_dim();
            dim->CopyFrom(input_shape.dim(i));
          } else {
            auto dim = saved_mean_shape->add_dim();
            dim->set_dim_value(1);
          }
        }
        saved_inv_std_var_shape->CopyFrom(*saved_mean_shape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalizationGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalizationGrad")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .AllowUncheckedAttributes()
      .Input(0, "Y_grad", "The gradient tensor from output.", "T")
      .Input(1, "X", "Input data tensor from the forward path", "T")
      .Input(2, "scale", "Scale tensor.", "T")
      .Input(3, "mean", "mean of X.", "U")
      .Input(4, "inv_std_var", "inverse std variance of X.", "U")
      .Output(0, "X_grad", "Gradient of the input.", "T")
      .Output(1, "scale_grad", "Gradient of the scale.", "T")
      .Output(2, "bias_grad", "Gradient of the bias.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain except mean and inv_std_var to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Group)
      .SetDomain(kOnnxDomain)
      .SetDoc("if all the inputs are available, the output will be true")
      .SinceVersion(9)
      .Input(0, "input_tensors", "list of dependency tensors", "T", OpSchema::Variadic)
      .Output(0, "done", "all the dependency tensors are ready", "B")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool");

  static const char* TransposeMatMul_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(TransposeMatMul)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TransposeMatMul")
      .Input(0, "A", "N-dimensional matrix A", "T")
      .Input(1, "B", "N-dimensional matrix B", "T")
      .Attr(
          "transA",
          "Whether A should be transposed on the last two dimensions before doing multiplication",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "transB",
          "Whether B should be transposed on the last two dimensions before doing multiplication",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Output(0, "Y", "Matrix multiply results", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .SetDoc(TransposeMatMul_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        auto transAAttr = ctx.getAttribute("transA");
        bool transa = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transb = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        int input1Idx = 0;
        int input2Idx = 1;
        if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
          return;
        }

        const auto shape0_raw = getInputShape(ctx, input1Idx);
        const auto shape1_raw = getInputShape(ctx, input2Idx);

        if (shape0_raw.dim_size() == 0 || shape1_raw.dim_size() == 0) {
          fail_shape_inference("Input tensors of wrong rank (0).");
        }

        // numpy transpose on a vector does not change anything.
        if (shape0_raw.dim_size() == 1) {
          transa = false;
        }
        if (shape1_raw.dim_size() == 1) {
          transb = false;
        }

        ONNX_NAMESPACE::TensorShapeProto shape0, shape1;
        auto rank0 = shape0_raw.dim_size();
        if (rank0 == 1) {
          // for vector input, transa does not make impact on the dim.
          shape0 = shape0_raw;
        } else {
          for (int i = 0; i < rank0 - 2; ++i) {
            *shape0.add_dim() = shape0_raw.dim(i);
          }
          *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 1 : rank0 - 2);
          *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 2 : rank0 - 1);
        }

        auto rank1 = shape1_raw.dim_size();
        if (rank1 == 1) {
          // for vector input, transb does not make impact on the dim.
          shape1 = shape1_raw;
        } else {
          for (int i = 0; i < rank1 - 2; ++i) {
            *shape1.add_dim() = shape1_raw.dim(i);
          }
          *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 1 : rank1 - 2);
          *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 2 : rank1 - 1);
        }

        ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

        // First promote each shape to at least rank-2. This logic is
        // specific to matmul, not generic broadcasting.
        {
          if (shape0.dim_size() == 1) {
            shapeL.add_dim()->set_dim_value(1);
            *shapeL.add_dim() = shape0.dim(0);
          } else {
            *shapeL.mutable_dim() = shape0.dim();
          }
          if (shape1.dim_size() == 1) {
            *shapeR.add_dim() = shape1.dim(0);
            shapeR.add_dim()->set_dim_value(1);
          } else {
            *shapeR.mutable_dim() = shape1.dim();
          }
        }

        // Check for compatible matrix multiply dimensions
        {
          auto dimL = shapeL.dim(shapeL.dim_size() - 1);
          auto dimR = shapeR.dim(shapeR.dim_size() - 2);
          if (dimL.has_dim_value() && dimR.has_dim_value() &&
              dimL.dim_value() != dimR.dim_value()) {
            fail_shape_inference("Incompatible dimensions for matrix multiplication");
          }
        }

        ONNX_NAMESPACE::TensorShapeProto resultShape;

        // Now call out to generic multidimensional broadcasting for
        // the broadcastable prefixes.
        {
          ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
          for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
            *prefixShapeL.add_dim() = shapeL.dim(i);
          }
          for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
            *prefixShapeR.add_dim() = shapeR.dim(i);
          }
          bidirectionalBroadcastShapeInference(
              prefixShapeL, prefixShapeR, resultShape);
        }

        // Back to matmul-specific. Add the trailing dimensions back in.
        {
          if (shape0.dim_size() != 1) {
            *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
          }
          if (shape1.dim_size() != 1) {
            *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
          }
        }
        updateOutputShape(ctx, 0, resultShape);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsFinite)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("IsFinite")
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Constrain the output to a boolean tensor.")
      .Input(
          0,
          "X",
          "The input tensor.",
          "T")
      .Output(
          0,
          "Y",
          "The output tensor. Its shape is the same as the input.",
          "T1");
#ifdef MICROSOFT_INTERNAL
  // register internal ops
  RegisterInternalSchemas();
#endif
}
}  // namespace contrib
}  // namespace onnxruntime
