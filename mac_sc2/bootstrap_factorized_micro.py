#!/usr/bin/env python3
"""Create one runnable tactical checkpoint from an existing macro checkpoint.

This is intentionally a *transfer initialization*, not a claim that the new
heads have learned micro.  Training may begin only after replay combat labels
are verified against SC2 replay observations.
"""
import argparse
from pathlib import Path

import torch

from factorized_micro_policy import FactorizedMicroPolicy, checkpoint_metadata
from multirace_general_policy import MultiRaceGeneralMacroPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    macro_data = torch.load(args.macro_checkpoint, map_location="cpu", weights_only=False)
    macro = MultiRaceGeneralMacroPolicy()
    macro.load_state_dict(macro_data["state_dict"])
    model = FactorizedMicroPolicy(macro.shared, freeze_backbone=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "macro_checkpoint": str(args.macro_checkpoint),
        "backbone_frozen": True,
        "trained_micro_examples": 0,
        **checkpoint_metadata(),
    }, output)
    print(f"saved transfer-initialized runnable micro checkpoint: {output}")


if __name__ == "__main__":
    main()
