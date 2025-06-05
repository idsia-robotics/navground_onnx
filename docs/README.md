# navground_onnx

This package contains a c++ implementation of `navground_learning` PolicyBehavior (called `CppPolicy`) that depends only on `navground_core` and `onnxruntime`.

## Install

1. [Install](https://idsia-robotics.github.io/navground/stable/installation/github_release.html) or [build](https://idsia-robotics.github.io/navground/stable/installation/from_source.html) `navground_core` from source following the instructions and source the navground setup script
2. [Setup](https://idsia-robotics.github.io/navground/stable/installation/setup_dev.html) navground (depending on the operating system and on how you installed navground).
2. [Install](https://onnxruntime.ai/docs/install) `onnxruntime`.
3. Build `navground_onnx`:
  - using colcon:
    ```console
    $ colcon build --packages-select navground_onnx
    ```
  - using cmake:
    ```
    $ cmake -DCMAKE_BUILD_TYPE=Release <path to navground_onnx>
    $ cmake --build .
    ```
## First steps

Verify that the behavior is installed by running
```console
$ navground info --behaviors CppPolicy --properties --description
Installed components
====================
Behaviors
---------
CppPolicy
    flat: false (bool)
      Whether to flatten the observations
    include_angular_speed: false (bool)
      Whether to include the current angular speed in the observations
    include_target_direction: false (bool)
      Whether to include the target direction in the observations
    include_target_distance: false (bool)
      Whether to include the target distance in the observations
    include_target_speed: false (bool)
      Whether to include the target speed in the observations
    include_velocity: false (bool)
      Whether to include the current velocity in the observations
    max_acceleration: 10 (float)
      The upper bound of the acceleration.
    max_angular_acceleration: 0 (float)
      The upper bound of the angular acceleration.
    policy_path:  (str)
      Path to the onnx model
    shared: false (bool)
      Whether to share the policy with similar agents
    use_acceleration_action: false (bool)
      Whether actions are accelerations.
```
which prints the list of properties exposed by `CppPolicy` and their default values.

## Usage

Follow the [navground documentation](https://idsia-robotics.github.io/navground/stable/index.html) to configure the behavior from YAML, e.g., 
```yaml
...
behavior:
  type: CppPolicy
  policy_path: 'policy.onnx'
  shared: true
  ...
```
or from C++.

If `shared` is set, the same onnx model is shared between all agents/behaviors that have the same configuration and inference happens *in parallel*, therefore reducing inference costs significantly (e.g., by about factor 5 for crossing with 20 agents (45 us vs 200 us), which in turn reduces the total simulation cost by factor 3 (70 us vs 225 us)). Note that the onnx model finalizes its initialization when the first inference is requested for the first agent that is sharing the policy.

If `shared` is not set, each behavior instantiates its own copy of the onnx policy and perform inference independently.

## Examples

The directory `examples` contains few examples of experiments configured to use `CppPolicy`. The policy have been trained in the corresponding [navground_learning tutorials](https://idsia-robotics.github.io/navground_learning/latest/tutorials/index.html).

Execute them using the navground CLI, like:

```console
$ navground run examples/crossing/experiment.yaml --chdir
Performing experiment ...
Experiment done
Duration: 0.0931841 s
```


## Acknowledgement and disclaimer

The work was supported in part by [REXASI-PRO](https://rexasi-pro.spindoxlabs.com) H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement no. 101070028.

<img src="https://rexasi-pro.spindoxlabs.com/wp-content/uploads/2023/01/Bianco-Viola-Moderno-Minimalista-Logo-e1675187551324.png"  width="300">

The work has been partially funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.


