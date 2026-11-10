# Training and evaluation

Run commands from the repository root after installing requirements. Training writes to `model/runs/`; use a new `--name` for each experiment. The source teacher dataset is not included in this release. Source training requires a compatible teacher archive at `model/development/teacher_data.npz`; inference with the released actors does not require it.

## Source task

```bash
python code/train_capacity.py --kind moe --experts 3 --width 68 --task cruise --stage source --adaptation full --seed 8101 --updates 250 --workers 3 --horizon 64 --warm-steps 2000 --lr 0.00003 --eval-every 25 --name source_example
```

The resulting checkpoint is `model/runs/source_example/moe_k3/seed_8101/best.pt`, with adjacent `configuration.json`. Keep those files together for target initialization.

## Independent target branches

```bash
python code/train_capacity.py --kind moe --experts 3 --width 68 --task speed --stage target --adaptation gate --source model/runs/source_example/moe_k3/seed_8101/best.pt --seed 8101 --updates 250 --lr 0.001 --eval-every 25 --name speed_example
python code/train_capacity.py --kind moe --experts 3 --width 68 --task escape --stage target --adaptation gate --source model/runs/source_example/moe_k3/seed_8101/best.pt --seed 8101 --updates 250 --lr 0.001 --eval-every 25 --name escape_example
```

Both commands use the base environment. The second command trains its original 3 mm/s pursuit task; it does not replay the later 25 mm/s continuation history. The released escape checkpoint is executed using `escape_environment.py`, as documented in VERSIONS.md.

## Operational details

- `--updates`, `--workers` and `--horizon` set the interaction budget. A team step advances all three fish for 0.5 s.
- `--warm-steps` applies to source initialization. Target training disables teacher warm-start.
- `--uniform-init` optionally zeroes the target gate output layer.
- `--resume` requires an existing compatible run; implementation hashes and arguments are checked.
- Validation prioritizes successes, then the task-specific primary metric. Checkpoint update 0 is eligible.
- Training checkpoints use Python/PyTorch serialized experiment state. Load only trusted files. Public actor checkpoints are loaded with `weights_only=True` by the demo.

The release validation checks command parsing and a short source/target training execution. Full 250-update training has not been rerun during packaging. Long training requires a suitable compute budget.
