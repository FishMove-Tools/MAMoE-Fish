# Release verification

Checked locally on 2026-11-10 using Python 3.12 and the versions pinned in requirements.txt in an existing environment.

- Primitive endpoint neutrality, three actor checkpoint distributions, a MuJoCo physics cycle and visual mesh loading: passed.
- Cruise, current escape and speed demos: four team steps each (2 s), with recorded state/action/gate arrays.
- Demo trajectory plotting: passed.
- Source training: one update, one worker, horizon two, one teacher warm-start step; validation and checkpoint writing completed.
- Speed target training: one gate-only update from that source checkpoint; validation and checkpoint writing completed. The training routine checks the frozen-parameter hash.
- Local CFD data audit completed before packaging; these data are withheld and this audit is not a downloadable data release.
- Website: homepage and all main sections loaded in the local browser; framework and task assets displayed.

These are software packaging checks. Full training, independent scientific evaluation and a fresh-environment installation have not been rerun for this release. Generated smoke-run outputs are excluded from the published files.

Run `python tools/verify_release.py`, `python tests/smoke.py` to repeat the corresponding checks.
