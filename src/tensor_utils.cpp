/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#include "navground_onnx/tensor_utils.h"

namespace navground::onnx {

Ort::Value make_tensor(const core::Buffer &buffer) {
  const auto sshape = buffer.get_shape();
  const size_t size = buffer.size();
  std::vector<int64_t> shape(sshape.size());
  // 1 batch
  // shape[0] = 1;
  std::copy(sshape.begin(), sshape.end(), shape.begin());

  return std::visit(
      [&shape, size](auto &&arg) {
        using Q = std::remove_reference_t<decltype(arg[0])>;
        using T = std::remove_const_t<Q>;
        T * data = const_cast<T *>(&(arg[0]));
        const Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<T>(
            memory_info, data, size, shape.data(), shape.size());
      },
      buffer.get_data_container());
}

} // namespace navground::onnx
