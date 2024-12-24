/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#ifndef NAVGROUND_ONNX_POLICY_H_
#define NAVGROUND_ONNX_POLICY_H_

#include "navground/core/behavior.h"
#include "navground/core/buffer.h"
#include "navground/core/states/sensing.h"
#include "navground/core/types.h"
#include "navground_onnx/export.h"
#include <filesystem>
#include <limits>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <vector>

namespace navground::onnx {

using FlatBufferIterator =
    decltype(std::begin(std::declval<std::valarray<ng_float_t> &>()));

struct NAVGROUND_ONNX_EXPORT ControlActionConfig {
  ng_float_t max_acceleration;
  ng_float_t max_angular_acceleration;
  bool use_acceleration_action;
  bool fix_orientation;
  bool use_wheels;

  auto tie() const {
    return std::tie(max_acceleration, max_angular_acceleration,
                    use_acceleration_action, fix_orientation, use_wheels);
  }

  bool operator==(const ControlActionConfig &other) const {
    return tie() == other.tie();
  }

  ControlActionConfig()
      : max_acceleration(10), max_angular_acceleration(100),
        use_acceleration_action(false), fix_orientation(false),
        use_wheels(false) {}
};

struct NAVGROUND_ONNX_EXPORT DefaultObservationConfig {
  bool flat;
  int history;
  bool include_target_distance;
  bool include_target_distance_validity;
  ng_float_t max_target_distance;
  bool include_target_direction;
  bool include_target_direction_validity;
  bool include_velocity;
  bool include_angular_speed;
  bool include_radius;
  bool include_target_speed;
  bool include_target_angular_speed;
  float max_radius;

  auto tie() const {
    return std::tie(flat, history, include_target_distance,
                    include_target_distance_validity, max_target_distance,
                    include_target_direction, include_target_direction_validity,
                    include_velocity, include_angular_speed, include_radius,
                    include_target_speed, include_target_angular_speed,
                    max_radius);
  }

  bool operator==(const DefaultObservationConfig &other) const {
    return tie() == other.tie();
  }

  DefaultObservationConfig()
      : flat(false), history(1), include_target_distance(false),
        include_target_distance_validity(false),
        max_target_distance(std::numeric_limits<ng_float_t>::infinity()),
        include_target_direction(true),
        include_target_direction_validity(false), include_velocity(false),
        include_angular_speed(false), include_radius(false),
        include_target_speed(false), include_target_angular_speed(false),
        max_radius(std::numeric_limits<ng_float_t>::infinity()) {}
};

struct NAVGROUND_ONNX_EXPORT Action {
  ng_float_t *wheels;
  ng_float_t *longitudinal;
  ng_float_t *transversal;
  ng_float_t *angular;
  ng_float_t max_speed;
  ng_float_t max_angular_speed;
  ng_float_t max_acceleration;
  ng_float_t max_angular_acceleration;
  bool is_acceleration;

  core::Twist2 get_cmd(const core::Behavior &behavior, ng_float_t time_step,
                       size_t index) const;
};

struct NAVGROUND_ONNX_EXPORT EgoState {
  ng_float_t *longitudinal_speed;
  ng_float_t *trasversal_speed;
  ng_float_t *angular_speed;
  ng_float_t *radius;

  void update(const core::Behavior &behavior, size_t index);
};

struct NAVGROUND_ONNX_EXPORT TargetState {
  ng_float_t *distance;
  uint8_t *distance_valid;
  ng_float_t *direction;
  uint8_t *direction_valid;
  ng_float_t *speed;
  ng_float_t *angular_speed;
  ng_float_t max_speed;
  ng_float_t max_angular_speed;
  ng_float_t max_distance;

  void update(const core::Behavior &behavior, size_t index);
};

NAVGROUND_ONNX_EXPORT
core::SensingState *get_sensing(const core::Behavior &behavior);

NAVGROUND_ONNX_EXPORT
void flatten(FlatBufferIterator &out,
             const std::map<std::string, core::Buffer> &buffers,
             int index = -1);

// all behaviors in share the same policy and config
struct NAVGROUND_ONNX_EXPORT Policy {

  ControlActionConfig action_config;
  DefaultObservationConfig observation_config;
  std::filesystem::path path;

  Policy(const ControlActionConfig &action_config,
         const DefaultObservationConfig &observation_config,
         const std::filesystem::path &path);
  virtual core::Twist2 get_cmd(const core::Behavior &behavior,
                               ng_float_t time_step);
  virtual int64_t get_number_of_batches() const;
  void run();

  virtual ~Policy() = default;

  void prepare(const core::Behavior &);

protected:
  FlatBufferIterator flat_buffer_interator() const;
  Action _action;
  TargetState _target_state;
  EgoState _ego_state;
  std::map<std::string, core::Buffer> _state_buffers;
  bool _initialized;

private:
  core::Buffer _flat_buffer;
  std::vector<Ort::Value> _inputs;
  std::vector<const char *> _input_names;
  std::map<std::string, core::Buffer> _output_buffers;
  std::vector<Ort::Value> _outputs;
  std::vector<const char *> _output_names;
  Ort::Env _env;
  std::unique_ptr<Ort::Session> _session;
  std::vector<const std::map<std::string, core::Buffer> *> _input_buffers;
};

} // namespace navground::onnx

#endif // NAVGROUND_ONNX_POLICY_H_
