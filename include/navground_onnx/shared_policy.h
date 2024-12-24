/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#ifndef NAVGROUND_ONNX_SHARED_POLICY_H_
#define NAVGROUND_ONNX_SHARED_POLICY_H_

#include <memory>
#include <optional>
#include <vector>

#include "navground_onnx/export.h"
#include "navground_onnx/policy.h"

namespace navground::onnx {

// all behaviors in share the same policy and config
struct NAVGROUND_ONNX_EXPORT SharedPolicy final: public Policy {

  SharedPolicy(const ControlActionConfig &action_config,
               const DefaultObservationConfig &observation_config,
               const std::filesystem::path &path);
  core::Twist2 get_cmd(const core::Behavior &behavior,
                       ng_float_t time_step) override;
  int64_t get_number_of_batches() const override;
  void leave(const core::Behavior &behavior);
  static std::shared_ptr<SharedPolicy>
  join(const core::Behavior &behavior, const ControlActionConfig &action_config,
       const DefaultObservationConfig &observation_config,
       const std::filesystem::path &path);

private:
  std::vector<core::Behavior *> _behaviors;
  std::optional<size_t> index_of(const core::Behavior &behavior);
  static std::vector<std::shared_ptr<SharedPolicy>> _policies;
};

// void deinit_policy(core::Behavior &behavior);

} // namespace navground::onnx

#endif // NAVGROUND_ONNX_SHARED_POLICY_H_
