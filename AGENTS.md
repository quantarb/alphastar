# StarCraft II agent rules

This workspace exists to produce an AI that can be watched playing StarCraft II.
Treat a model as useful only when it can be loaded by a live SC2 runner and
issue valid game actions.

## Non-negotiable delivery rule

Do **not** start or continue training a model unless all of the following are
already true:

1. A compatible `python-sc2` runner exists for that checkpoint format.
2. Every model output has an executable, patch-valid interpretation.
3. A replay or live-match command is available before training begins.
4. The checkpoint is saved at the requested early evaluation interval.

Offline-only classifiers, vocabulary experiments, shadow-only predictors, and
models missing unit-selection or target execution are research artifacts. Do
not train them unless the user explicitly asks for research rather than a
playable agent.

## Required training loop

For every playable-agent training run:

1. Train directly from raw replays on demand; do not create shard datasets
   unless the user explicitly requests them.
2. Use both players' trajectories in the base pass, with correct event/player
   ownership. Command-event ownership must use `event.player.pid`, not the
   zero-based command `pid`.
3. Fine-tune only on the winner's trajectory when the user requests
   winner-only fine-tuning.
4. Save one overwrite-only checkpoint every **200 games** (or a smaller user
   requested interval). Never accumulate unbounded checkpoint files.
5. At the first checkpoint, immediately launch the exact saved checkpoint in
   a real SC2 match while training continues from that snapshot.
6. Record the literal SC2 `Result.Victory`, `Result.Defeat`, or `Result.Tie`
   and the replay path. Never infer gameplay success from classification
   accuracy.

## Patch and action validity

- The installed playable client is SC2 4.9.2 (`Base97563`).
- A model run in this client may emit only actions executable by its runner on
  4.9.2.
- Patch/race MTL is allowed only if every patch/race task still has a complete
  executable decoder. A task-local ability vocabulary by itself is not a
  playable agent.
- Do not claim that a cross-patch head is playable in 4.9.2.

## Training–prediction action contract

Every trainable policy must use one versioned `ActionSpec` shared by replay
extraction, model labels, checkpoint metadata, and the live SC2 decoder.

1. The live decoder defines the only semantic action tuples that may be
   trained. A tuple consists of the required actor/selection role, command
   family, payload or ability role, target kind, and queue semantics.
2. The replay extractor must discard and count commands outside that contract;
   it must never silently keep labels that the runner cannot execute.
3. A checkpoint must store the action-contract hash. The runner must refuse
   to load a checkpoint whose hash differs from its own contract.
4. Before any training starts, automatically validate that every permitted
   tuple has a patch-valid 4.9.2 decoder and legality predicate.
5. Extending the action space is always ordered as: implement decoder and
   legality guard -> update ActionSpec -> validate -> train. Never train a
   head first and promise decoder support later.
6. Classification metrics only cover the permitted live action contract; they
   are not evidence of strategic or tactical game strength.

## Communication

- Lead with live game evidence, not offline metrics.
- Say plainly when a model is not runnable or when a match was lost.
- Do not launch obsolete or known-corrupt checkpoints as demonstrations.
- If a pipeline bug invalidates labels, stop using those checkpoints, preserve
  them only in Trash if needed for forensic comparison, and retrain before
  claiming progress.
