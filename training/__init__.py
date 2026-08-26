"""MARKTSIGNAL training — offline, separate from the production system.

This package contains everything needed to train EconomyEncoder V1:
    - tokenizer.py:   BPE tokenizer (via src/models/tokenizer.py)
    - scenarios.py:   parametric economic scenario generator
    - ood_tests.py:   out-of-distribution test data
    - data.py:        legacy template-based data (kept for compatibility)
    - pipeline.py:    pretraining (MLM) and score training (SmoothL1Loss)
    - cli.py:         command-line entry point (python -m training train)

The production system in src/ never imports from here. The only connection
between training and production is the saved checkpoint (.pt file).
"""

from src.models.tokenizer import BPETokenizer
from .scenarios import EconomicScenario, generate_parametric, generate_counterexample_groups
from .ood_tests import OODTest, all_ood_tests, ood_by_category
from .data import TrainingExample, generate_all, generate_training_examples, generate_counterexamples

_PIPELINE_EXPORTS = {
    "TrainingConfig",
    "build_vocabulary",
    "build_tokenizer",
    "pretrain",
    "train_scores",
    "save_checkpoint",
    "load_checkpoint",
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
    "EconomicScenario",
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
    "ood_by_category",
    "pretrain",
    "save_checkpoint",
    "train_scores",
]
