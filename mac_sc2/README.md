# SC2 agent package

The maintained playable-agent pipeline is organised by responsibility:

- `contracts/`: versioned observation and action contracts.
- `architectures/`: PyTorch model definitions only.
- `data/`: raw replay streaming and reconstruction.
- `runtime/`: live SC2 encoding, legality checks, candidate generation, and execution.
- `training/`: fine-tuning lifecycles and checkpoint save policy.
- `evaluation/`: live-match launch, literal result capture, and replay recording.
- `scripts/`: thin command-line entry points only; no policy implementation lives here.
- `legacy/`: historical prototypes, excluded from the playable-agent pipeline.
- `artifacts/`: checkpoints, results, and replays (not source code).

New work must import package modules (for example
`mac_sc2.runtime.placement_candidates`) and execute scripts with
`python -m mac_sc2.scripts.<name>`.

Top-level implementation modules are intentionally prohibited.  A new model,
contract, replay reader, live executor, trainer, or evaluator belongs in its
corresponding responsibility directory.
