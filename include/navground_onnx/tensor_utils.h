/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#ifndef NAVGROUND_ONNX_TENSOR_UTILS_H_
#define NAVGROUND_ONNX_TENSOR_UTILS_H_

#include "navground/core/buffer.h"
#include "navground_onnx/export.h"
#include <onnxruntime_cxx_api.h>

namespace navground::onnx {

NAVGROUND_ONNX_EXPORT
Ort::Value make_tensor(const core::Buffer &buffer);

} // namespace navground::onnx

#endif // NAVGROUND_ONNX_TENSOR_UTILS_H_
