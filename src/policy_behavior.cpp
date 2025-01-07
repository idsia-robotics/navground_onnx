/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#include "navground_onnx/policy_behavior.h"
#include "navground/core/property.h"
#include "navground_onnx/shared_policy.h"

namespace navground::onnx {

PolicyBehavior::~PolicyBehavior() {
  if (auto shared_policy = std::dynamic_pointer_cast<SharedPolicy>(_policy)) {
    shared_policy->leave(*this);
  }
}

void PolicyBehavior::prepare() {
  if (!_policy) {
    if (get_shared()) {
      _policy = SharedPolicy::join(*this, action_config, observation_config,
                                   _policy_path);
    } else {
      _policy = std::make_shared<Policy>(action_config, observation_config,
                                         _policy_path);
      _policy->prepare(*this);
    }
  }
}

core::Twist2 PolicyBehavior::compute_cmd_internal(ng_float_t time_step) {
  if (!_policy) {
    prepare();
    if (get_shared()) {
      return core::Twist2();
    }
  }
  return feasible_twist(_policy->get_cmd(*this, time_step));
}

const std::string PolicyBehavior::type = register_type<PolicyBehavior>(
    "CppPolicy",
    {
        {"shared",
         core::Property::make(
             &PolicyBehavior::get_shared, &PolicyBehavior::set_shared, false,
             "Whether to share the policy with similar agents")},
        {"policy_path",
         core::Property::make(&PolicyBehavior::get_policy_path_as_string,
                              &PolicyBehavior::set_policy_path_as_string,
                              std::string(""), "Path to the onnx model")},
        {"use_acceleration_action",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->action_config.use_acceleration_action;
             },
             [](PolicyBehavior *b, bool value) {
               b->action_config.use_acceleration_action = value;
             },
             false, "Whether actions are accelerations.")},
        {"max_acceleration", core::Property::make<ng_float_t, PolicyBehavior>(
                                 [](const PolicyBehavior *b) -> bool {
                                   return b->action_config.max_acceleration;
                                 },
                                 [](PolicyBehavior *b, ng_float_t value) {
                                   b->action_config.max_acceleration = value;
                                 },
                                 10, "The upper bound of the acceleration.")},
        {"max_angular_acceleration",
         core::Property::make<ng_float_t, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->action_config.max_angular_acceleration;
             },
             [](PolicyBehavior *b, ng_float_t value) {
               b->action_config.max_angular_acceleration = value;
             },
             false, "The upper bound of the angular acceleration.")},
        {"include_target_distance",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->observation_config.include_target_distance;
             },
             [](PolicyBehavior *b, bool value) {
               b->observation_config.include_target_distance = value;
             },
             false,
             "Whether to include the target distance in the observations")},
        {"include_target_direction",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->observation_config.include_target_direction;
             },
             [](PolicyBehavior *b, bool value) {
               b->observation_config.include_target_direction = value;
             },
             false,
             "Whether to include the target direction in the observations")},
        {"include_target_speed",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->observation_config.include_target_speed;
             },
             [](PolicyBehavior *b, bool value) {
               b->observation_config.include_target_speed = value;
             },
             false, "Whether to include the target speed in the observations")},
        {"include_velocity",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->observation_config.include_velocity;
             },
             [](PolicyBehavior *b, bool value) {
               b->observation_config.include_velocity = value;
             },
             false,
             "Whether to include the current velocity in the observations")},
        {"include_angular_speed",
         core::Property::make<bool, PolicyBehavior>(
             [](const PolicyBehavior *b) -> bool {
               return b->observation_config.include_angular_speed;
             },
             [](PolicyBehavior *b, bool value) {
               b->observation_config.include_angular_speed = value;
             },
             false,
             "Whether to include the current angular speed in the "
             "observations")},
        {"flat", core::Property::make<bool, PolicyBehavior>(
                     [](const PolicyBehavior *b) -> bool {
                       return b->observation_config.flat;
                     },
                     [](PolicyBehavior *b, bool value) {
                       b->observation_config.flat = value;
                     },
                     false, "Whether to flatten the observations")},
    });

} // namespace navground::onnx
