# SC2 agent package

The maintained playable-agent pipeline is organised by responsibility:

- `contracts/`: versioned observation and action contracts.
- `architectures/`: PyTorch model definitions only.
- `data/`: raw replay streaming and reconstruction.
- `runtime/`: live SC2 encoding, legality checks, candidate generation, and execution.
- `scripts/`: thin command-line training, validation, and live-run entry points.
- `artifacts/`: checkpoints, results, and replays (not source code).

New work must import package modules (for example
`mac_sc2.runtime.placement_candidates`) and execute scripts with
`python -m mac_sc2.scripts.<name>`.
