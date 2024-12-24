/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#include "navground_onnx/shared_policy.h"
#include <algorithm>

namespace navground::onnx {

int64_t SharedPolicy::get_number_of_batches() const {
  return _behaviors.size();
}

std::optional<size_t> SharedPolicy::index_of(const core::Behavior &behavior) {
  const auto i = std::find(_behaviors.begin(), _behaviors.end(), &behavior);
  if (i != _behaviors.end()) {
    return i - _behaviors.begin();
  }
  return std::nullopt;
}

core::Twist2 SharedPolicy::get_cmd(const core::Behavior &behavior,
                                   ng_float_t time_step) {
  auto index = index_of(behavior);
  if (!index) {
    throw std::runtime_error(
        "Behavior does not belongs to this group of shared policies");
  }
  // std::cout << "SharedPolicy::get_cmd " << *index << std::endl;
  if (*index == 0) {
    if (!_initialized) {
      prepare(*(_behaviors.at(0)));
    }
    size_t i = 0;
    for (const auto &behavior : _behaviors) {
      _ego_state.update(*behavior, i);
      _target_state.update(*behavior, i);
      i++;
    }
    if (observation_config.flat) {
      i = 0;
      auto out = flat_buffer_interator();
      for (const auto &behavior : _behaviors) {
        flatten(out, get_sensing(*behavior)->get_buffers());
        flatten(out, _state_buffers, i);
        i++;
      }
    }
    run();
  }
  return _action.get_cmd(behavior, time_step, 2 * (*index));
}

std::vector<std::shared_ptr<SharedPolicy>> SharedPolicy::_policies = {};

SharedPolicy::SharedPolicy(const ControlActionConfig &action_config,
                           const DefaultObservationConfig &observation_config,
                           const std::filesystem::path &path)
    : Policy(action_config, observation_config, path), _behaviors() {}

std::shared_ptr<SharedPolicy>
SharedPolicy::join(const core::Behavior &behavior, const ControlActionConfig &action_config,
     const DefaultObservationConfig &observation_config,
     const std::filesystem::path &path) {
  auto i = std::find_if(
      _policies.begin(), _policies.end(),
      [&action_config, &observation_config, &path](const auto &policy) {
        return (policy->action_config == action_config &&
                policy->observation_config == observation_config &&
                policy->path == path);
      });
  std::shared_ptr<SharedPolicy> policy;
  if (i == _policies.end()) {
    policy =
        std::make_shared<SharedPolicy>(action_config, observation_config, path);
    _policies.push_back(policy);
  } else {
    policy = *i;
  }
  policy->_behaviors.push_back(const_cast<core::Behavior *>(&behavior));
  return policy;
}

void SharedPolicy::leave(const core::Behavior &behavior) {
  const auto i = std::find(_behaviors.begin(), _behaviors.end(), &behavior);
  if (i != _behaviors.end()) {
    _behaviors.erase(i);
    if (_behaviors.size() == 0) {
      _policies.erase(std::remove_if(_policies.begin(), _policies.end(),
                                     [this](const auto &policy) {
                                       return policy.get() == this;
                                     }),
                      _policies.end());
    }
  }
}

} // namespace navground::onnx
