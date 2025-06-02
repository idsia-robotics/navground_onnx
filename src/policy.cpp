/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#include "navground_onnx/policy.h"
#include "navground_onnx/tensor_utils.h"
#include "navground_onnx/io_utils.h"
#include <algorithm>

namespace navground::onnx {

core::SensingState *get_sensing(const core::Behavior &behavior) {
  return dynamic_cast<core::SensingState *>(
      const_cast<core::Behavior &>(behavior).get_environment_state());
}

template <typename T>
static T *add_buffer(std::map<std::string, core::Buffer> &buffers,
                     const std::string &key, const core::BufferShape &shape) {
  auto buffer = core::Buffer(core::BufferDescription::make<T>(shape));
  auto data = buffer.get_data<T>();
  // std::cout << "add buffer " << key << " " << buffer.size() << std::endl;
  T *data_ptr = const_cast<T *>(&((*data)[0]));
  buffers.emplace(key, std::move(buffer));
  return data_ptr;
}

void flatten(FlatBufferIterator &out,
             const std::map<std::string, core::Buffer> &buffers, int index) {
  for (const auto &[_, buffer] : buffers) {
    // std::cout << _ << std::endl;
    auto size = buffer.size();
    if (index >= 0) {
      const auto shape = buffer.get_shape();
      size /= shape[0];
    }
    std::visit(
        [&out, index, size](auto &&arg) {
          if (index >= 0) {
            std::copy(std::begin(arg) + index * size,
                      std::begin(arg) + (index + 1) * size, out);
          } else {
            std::copy(std::begin(arg), std::end(arg), out);
          }
        },
        buffer.get_data_container());
    out += size;
  }
}

static std::vector<ng_float_t>
compute_wheels(ng_float_t *values, ng_float_t max_value, size_t offset = 0) {
  return {values[offset] * max_value, values[offset + 1] * max_value};
}

static core::Twist2 compute_value(ng_float_t *longitudinal,
                                  ng_float_t *transversal, ng_float_t *angular,
                                  ng_float_t max_value,
                                  ng_float_t max_angular_value,
                                  size_t offset = 0) {
  core::Twist2 twist;
  twist.frame = core::Frame::relative;
  if (angular) {
    twist.angular_speed = max_angular_value * (angular[offset]);
  }
  if (longitudinal) {
    twist.velocity[0] = max_value * (longitudinal[offset]);
  }
  if (transversal) {
    twist.velocity[1] = max_value * (transversal[offset]);
  }
  return twist;
}

core::Twist2 Action::get_cmd(const core::Behavior &behavior,
                             ng_float_t time_step, size_t offset = 0) const {
  if (wheels) {
    if (is_acceleration) {
      const auto accs = compute_wheels(wheels, max_acceleration, offset);
      auto speeds = behavior.get_wheel_speeds();
      speeds[0] += accs[0] * time_step;
      speeds[1] += accs[1] * time_step;
      return behavior.twist_from_wheel_speeds(speeds);
    }
    const auto speeds = compute_wheels(wheels, max_speed, offset);
    return behavior.twist_from_wheel_speeds(speeds);
  }
  if (is_acceleration) {
    const auto acc =
        compute_value(longitudinal, transversal, angular, max_acceleration,
                      max_angular_acceleration, offset);
    const auto twist = behavior.get_twist(core::Frame::relative);
    return core::Twist2(twist.velocity + time_step * acc.velocity,
                        twist.angular_speed + time_step * acc.angular_speed,
                        core::Frame::relative);
  }
  return compute_value(longitudinal, transversal, angular, max_speed,
                       max_angular_speed, offset);
}

void EgoState::update(const core::Behavior &behavior, size_t index = 0) {
  if (longitudinal_speed) {
    // TODO(Jerome): There should be a better way.
    const auto v = to_relative(behavior.get_velocity(), behavior.get_pose());
    longitudinal_speed[index] = v[0];
    if (trasversal_speed) {
      trasversal_speed[index] = v[1];
    }
  }
  if (angular_speed) {
    angular_speed[index] = behavior.get_angular_speed();
  }
  if (radius) {
    radius[index] = behavior.get_radius();
  }
}

void TargetState::update(const core::Behavior &behavior, size_t index = 0) {
  if (distance) {
    const auto value = behavior.get_target_distance();
    distance[index] = std::min(value, max_distance);
    if (distance_valid) {
      distance_valid[index] = value ? 1 : 0;
    }
  }
  if (direction) {
    auto e = behavior.get_target_direction(core::Frame::relative);
    if (e) {
      direction[2 * index] = (*e)[0];
      direction[2 * index + 1] = (*e)[1];
    } else {
      direction[2 * index] = 0;
      direction[2 * index + 1] = 0;
    }
    if (direction_valid) {
      direction_valid[index] = e ? 1 : 0;
    }
  }
  if (speed) {
    speed[index] = std::min(behavior.get_target_speed(), max_speed);
  }
  if (angular_speed) {
    angular_speed[index] =
        std::min(behavior.get_target_angular_speed(), max_angular_speed);
  }
}

core::Twist2 Policy::get_cmd(const core::Behavior &behavior,
                             ng_float_t time_step) {
  if (!_initialized) {
    prepare(behavior);
  }
  _ego_state.update(behavior);
  _target_state.update(behavior);
  if (observation_config.flat) {
    auto out = flat_buffer_interator();
    flatten(out, get_sensing(behavior)->get_buffers());
    flatten(out, _state_buffers);
  }
  run();
  return _action.get_cmd(behavior, time_step);
}

FlatBufferIterator Policy::flat_buffer_interator() const {
  auto data = const_cast<std::valarray<ng_float_t> *>(
      _flat_buffer.get_data<ng_float_t>());
  return std::begin(*data);
}

void Policy::run() {
  if (!_initialized) {
    std::cerr << "Initialize the policy before running it!" << std::endl;
    return;
  }
  _session->Run(Ort::RunOptions{nullptr}, _input_names.data(), _inputs.data(),
                _inputs.size(), _output_names.data(), _outputs.data(),
                _outputs.size());
}

Policy::Policy(const ControlActionConfig &action_config,
               const DefaultObservationConfig &observation_config,
               const std::filesystem::path &path)
    : action_config(action_config), observation_config(observation_config),
      path(path), _action(), _target_state(), _ego_state(), _state_buffers(),
      _initialized(false),
      _env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default"),
      _session(nullptr) {
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  sessionOptions.SetIntraOpNumThreads(1);
  SuppressStdErr s;
  _session = std::make_unique<Ort::Session>(_env, path.c_str(), sessionOptions);
}

int64_t Policy::get_number_of_batches() const { return 1; }

void Policy::prepare(const core::Behavior &behavior) {
  int64_t batches = get_number_of_batches();
  // std::cout << "Policy::prepare " << batches << std::endl;
  _target_state.max_speed = behavior.get_max_speed();
  _target_state.max_angular_speed = behavior.get_angular_speed();
  _target_state.max_distance = behavior.get_horizon();
  auto *out_ptr =
      add_buffer<ng_float_t>(_output_buffers, "action", {batches, 2});
  _action.longitudinal = out_ptr;
  _action.angular = out_ptr + 1;
  _action.max_speed = behavior.get_max_speed();
  _action.max_angular_speed = behavior.get_max_angular_speed();
  _action.is_acceleration = action_config.use_acceleration_action;
  _action.max_acceleration = action_config.max_acceleration;
  _action.max_angular_acceleration = action_config.max_angular_acceleration;

  if (observation_config.include_target_direction) {
    _target_state.direction = add_buffer<ng_float_t>(
        _state_buffers, "ego_target_direction", {batches, 2});
    if (observation_config.include_target_direction_validity) {
      _target_state.direction_valid = add_buffer<uint8_t>(
          _state_buffers, "ego_target_direction_valid", {batches, 1});
    }
  }
  if (observation_config.include_target_distance) {
    _target_state.distance = add_buffer<ng_float_t>(
        _state_buffers, "ego_target_distance", {batches, 1});
    if (observation_config.include_target_distance_validity) {
      _target_state.distance_valid = add_buffer<uint8_t>(
          _state_buffers, "ego_target_distance_valid", {batches, 1});
    }
  }
  if (observation_config.include_velocity) {
    // TODO: complete DOF
    _ego_state.longitudinal_speed =
        add_buffer<ng_float_t>(_state_buffers, "ego_velocity", {batches, 1});
  }
  if (observation_config.include_angular_speed) {
    _ego_state.angular_speed = add_buffer<ng_float_t>(
        _state_buffers, "ego_angular_speed", {batches, 1});
  }
  if (observation_config.include_radius) {
    _ego_state.radius =
        add_buffer<ng_float_t>(_state_buffers, "ego_radius", {batches, 1});
  }
  if (observation_config.include_target_speed) {
    _target_state.speed = add_buffer<ng_float_t>(
        _state_buffers, "ego_target_speed", {batches, 1});
  }
  if (observation_config.include_target_angular_speed) {
    _target_state.angular_speed = add_buffer<ng_float_t>(
        _state_buffers, "ego_target_angular_speed", {batches, 1});
  }
  const auto _env_state = get_sensing(behavior);
  if (observation_config.flat) {
    int64_t obs_size = 0;
    for (const auto &[_, buffer] : _env_state->get_buffers()) {
      obs_size += buffer.size();
    }
    for (const auto &[_, buffer] : _state_buffers) {
      obs_size += buffer.size() / batches;
    }
    _flat_buffer = core::Buffer(
        core::BufferDescription::make<ng_float_t>({batches, obs_size}));
    _inputs.emplace_back(make_tensor(_flat_buffer));
    _input_names.push_back("observation");
  } else {
    for (const auto &[key, buffer] : _env_state->get_buffers()) {
      _inputs.emplace_back(make_tensor(buffer));
      _input_names.push_back(key.c_str());
    }
    for (const auto &[key, buffer] : _state_buffers) {
      _inputs.emplace_back(make_tensor(buffer));
      _input_names.push_back(key.c_str());
    }
  }
  for (const auto &[key, buffer] : _output_buffers) {
    _outputs.emplace_back(make_tensor(buffer));
    _output_names.push_back(key.c_str());
  }
  _initialized = true;
}

} // namespace navground::onnx
