# SC2 agent package

The maintained playable-agent pipeline is organised by responsibility:

- `contracts/`: versioned observation and action contracts.
- `architectures/`: PyTorch model definitions only.
- `data/`: raw replay streaming and reconstruction.
- `runtime/`: live SC2 encoding, legality checks, candidate generation, and execution.
- `training/`: fine-tuning lifecycles and checkpoint save policy.
- `evaluation/`: live-match launch, literal result capture, and replay recording.
- `artifacts/`: checkpoints, results, and replays (not source code).

The one maintained policy is compositional MTL. Its macro tuple, placement,
and repair heads retain their own executable contracts and learned
initializers, while joint fine-tuning stores them in one versioned checkpoint.
Training is permitted only from the compatible raw-replay manifests and is
evaluated through saved live SC2 replays with literal game results.

New work must import package modules (for example
`mac_sc2.runtime.placement_candidates`). There are no standalone script entry
points: callers construct the data, training, and evaluation configuration in
code.

Top-level implementation modules are intentionally prohibited.  A new model,
contract, replay reader, live executor, trainer, or evaluator belongs in its
corresponding responsibility directory.
