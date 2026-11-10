# Versions and numerical interfaces

The project name is **MAMoE-Fish**. Historical file/class names containing FAMoE are preserved where they define compatibility interfaces.

| Component | Released definition |
|---|---|
| Core | Supplied FAMoE_code sources; portable bootstrap is the packaging change |
| Cruise / speed | Original v19 task definitions |
| Escape actor | Seed 8101, fast_pursuit_v2_25mmps continuation |
| Escape execution | Matched `escape_environment.py`; 25 mm/s nominal pursuit |
| Source / target training runner | Original base `environment.py`; 3 mm/s nominal pursuit |
| Checkpoint packaging | Actor tensors and minimal configuration extracted from trusted source checkpoints |

The current escape actor was continued from an earlier escape gate. That historical initializer is not included, so exact replay of its training history is outside this compact package. The training implementation is included; source initialization additionally requires the currently withheld teacher dataset. Do not merge data from distinct escape definitions as if they were one protocol.

The MuJoCo model constrains each fish to planar translation, yaw and two joint angles. It uses MuJoCo's native fluid approximation; inter-fish resolved wakes are not computed by this environment. CFD is a separate prescribed-motion analysis; its datasets are not included in this release.

Website illustrations explain the architecture and task geometry. They are not runtime recordings. The framework's frozen-policy symbol refers to deployment; source learning and target adaptation use the update rules in METHOD.md.
