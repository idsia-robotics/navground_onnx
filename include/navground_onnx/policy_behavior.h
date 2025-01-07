/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#ifndef NAVGROUND_ONNX_POLICY_BEHAVIOR_H_
#define NAVGROUND_ONNX_POLICY_BEHAVIOR_H_

#include "navground/core/behavior.h"
#include "navground/core/states/sensing.h"
#include "navground_onnx/export.h"
#include "navground_onnx/policy.h"
#include <filesystem>
#include <memory>

namespace navground::onnx {

class NAVGROUND_ONNX_EXPORT PolicyBehavior : public core::Behavior {
public:
  explicit PolicyBehavior(
      std::shared_ptr<core::Kinematics> kinematics = nullptr,
      ng_float_t radius = 0,
      const std::filesystem::path &path = std::filesystem::path(""))
      : core::Behavior(kinematics, radius), action_config(),
        observation_config(), _policy_path(std::filesystem::absolute(path)),
        _shared(false), _env_state() {}

  core::Twist2 compute_cmd_internal(ng_float_t time_step) override;

  core::EnvironmentState *get_environment_state() override {
    return &_env_state;
  }

  std::filesystem::path get_policy_path(bool relative = true) const {
    if (relative) {
      return std::filesystem::relative(_policy_path);
    } else {
      return _policy_path;
    }
  }

  std::string get_policy_path_as_string() const {
    return get_policy_path().string();
  }

  void set_policy_path_as_string(const std::string &value) {
    set_policy_path(std::filesystem::path(value));
  }

  void set_policy_path(const std::filesystem::path &value) {
    _policy_path = std::filesystem::absolute(value);
  }

  bool get_shared() const { return _shared; }

  void set_shared(bool value) { _shared = value; }

  ControlActionConfig action_config;
  DefaultObservationConfig observation_config;

  static const std::string type;

  void prepare();

  Policy *get_policy() const { return _policy.get(); }

  ~PolicyBehavior();

private:
  std::filesystem::path _policy_path;
  bool _shared;
  std::shared_ptr<Policy> _policy;
  core::SensingState _env_state;
};

} // namespace navground::onnx

#endif // NAVGROUND_ONNX_POLICY_BEHAVIOR_H_
