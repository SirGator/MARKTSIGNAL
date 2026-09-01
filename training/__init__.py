"""MARKTSIGNAL training — offline, separate from the production system.

This package contains everything needed to train EconomyEncoder V1:
    - modeling/:      training-local model, tokenizer and serialization code
    - scenarios.py:   parametric economic scenario generator
    - ood_tests.py:   out-of-distribution test data
    - data.py:        legacy template-based data (kept for compatibility)
    - pipeline.py:    pretraining, score training, checkpoints and resume
    - cli.py:         command-line entry point (python -m training train)

The runtime system never imports from here. The only connection
between training and production is the saved checkpoint (.pt file).
"""

from .modeling.tokenizer import BPETokenizer
from .scenarios import EconomicScenario, generate_parametric, generate_counterexample_groups
from .ood_tests import OODTest, all_ood_tests, ood_by_category
from .data import (
    DatasetLoadError,
    FrozenDataset,
    TrainingExample,
    generate_all,
    generate_training_examples,
    generate_counterexamples,
    load_frozen_dataset_for_training,
)

_PIPELINE_EXPORTS = {
    "TrainingConfig",
    "build_vocabulary",
    "build_tokenizer",
    "pretrain",
    "train_scores",
    "predict_scores",
    "save_checkpoint",
    "load_checkpoint",
    "load_training_checkpoint",
}


def __getattr__(name: str):
    """Resolve Torch-backed training helpers only when they are requested."""

    if name not in _PIPELINE_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(".pipeline", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "BPETokenizer",
    "DatasetLoadError",
    "EconomicScenario",
    "FrozenDataset",
    "OODTest",
    "TrainingConfig",
    "TrainingExample",
    "all_ood_tests",
    "build_tokenizer",
    "build_vocabulary",
    "generate_all",
    "generate_counterexample_groups",
    "generate_counterexamples",
    "generate_parametric",
    "generate_training_examples",
    "load_checkpoint",
    "load_frozen_dataset_for_training",
    "load_training_checkpoint",
    "ood_by_category",
    "predict_scores",
    "pretrain",
    "save_checkpoint",
    "train_scores",
]
