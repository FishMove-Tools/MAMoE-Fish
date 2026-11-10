# Method and interfaces

## Feedback composition

For fish i, each expert gives a categorical distribution over five primitive actions. The gate produces normalized nonnegative weights. The final distribution is:

`p(a | o, task) = sum_k w_k(o, task) p_k(a | o_feedback)`.

The implementation mixes probabilities, not joint torques or sampled actions. Training samples a categorical action; the rollout utility selects its largest-probability action. Gate weights are recomputed from feedback at decision boundaries.

## Observation contract

Zero-based slices in the 35-component observation:

| Slice | Quantity | Scaling |
|---|---|---|
| 0:2 | Target position error in body frame | ×4 |
| 2:4 | Cycle-averaged body-frame velocity | ×20 |
| 4 | Cycle yaw rate | ×3 |
| 5:7 | Sine/cosine of heading error | direct |
| 7:9 | Body-frame reference velocity | ×20 |
| 9:11 | Joint angles | divide by 25°, 35° |
| 11:13 | Joint angular velocity | ×0.2 |
| 13:18 | Previous action | five-way one-hot |
| 18:21 | Fish role | three-way one-hot |
| 21:25 | Two neighbors' relative positions | body frame, m |
| 25:29 | Relative cycle velocities | body frame, ×20 |
| 29:31 | Relative threat position | body frame, m |
| 31 | Threat active flag | 0 or 1 |
| 32:35 | Task | cruise/escape/speed one-hot |

Entries are clipped to [-10, 10]. Experts receive concatenated slices `0:5` and `9:29`. The gate receives all 35 entries. The centralized critic receives the concatenated three-fish observations and predicts three values.

## Primitive interface

| ID | Primitive | Joint reference |
|---|---|---|
| 0 | Continuous | Smooth, phase-shifted oscillation over 0.5 s |
| 1 | Intermittent | 0.3 s oscillation, then 0.2 s neutral reference |
| 2 | Glide | Neutral reference throughout |
| 3 | Left turn | Quintic rise/recovery bending |
| 4 | Right turn | Opposite-signed bending |

All primitives return to neutral at their boundaries. Neutral reference still invokes the position actuator. The numerical waveform in `physics.py` is the authoritative definition.

## Learning and transfer

Source-only teacher labels initialize actors. Source reinforcement learning trains experts and gate together. Target gate adaptation freezes the expert tensors and trains the gate and a reset critic. PPO uses the categorical primitive distribution, centralized value estimates, clipped policy updates and GAE. Target tasks have their own rewards and criteria, defined by the environment implementation.

`--adaptation full` updates the full actor. MLP alternatives and the residual adapter provide separate controls. Expert counts and widths can be configured; the released K=3 actors use width 68 and gate width 32.
