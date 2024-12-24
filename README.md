# navground_onnx

This package contains a c++ implementation of `navground_learning` PolicyBehavior that depends only on `navground_core` and `onnxruntime`.

## Install

1. Build `navground_core`
2. Install `onnxruntime`
3. colcon build --packages-select navground_onnx

## Usage

```yaml
...
behavior:
  type: CppPolicy
  policy_path: 'policy.onnx'
  shared: true
  ...
```

If `shared` is set, the same onnx model is shared between all agents/behaviors that have the same configuration and inference happens *in parallel*, therefore reducing inference costs significantly (e.g., by about factor 5 for crossing with 20 agents (45 us vs 200 us), which in turn reduces the total simulation cost by factor 3 (70 us vs 225 us)). Note that the onnx model finalizes its initialization when the first inference is requested for the first agent that is sharing the policy.

If `shared` is not set, each behavior instantiates its own copy of the onnx policy and perform inference independently.
