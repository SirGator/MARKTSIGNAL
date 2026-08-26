"""
EconomyEncoder V1 model, tokenizer, serializer and tensor adapter.

Torch-backed exports are resolved lazily so serialization remains available
when the optional ``ml`` dependencies are not installed.
"""

from .context_serializer import (
    NO_SUMMARY,
    SERIALIZER_CONTRACT_VERSION,
    SUMMARY_MODE_FULL,
    SUMMARY_MODE_NONE,
    SUMMARY_MODES,
    ContextSerializer,
    normalize_summary_mode,
)
from .tokenizer import BPETokenizer

_TORCH_EXPORTS = {
    "ContextTensorEncoder": (".context_encoder", "ContextTensorEncoder"),
    "EconomyEncoder": (".model", "EconomyEncoder"),
    "EconomyTensorInput": (".adapter", "EconomyTensorInput"),
    "TorchEconomyModel": (".adapter", "TorchEconomyModel"),
    "load_economy_model": (".checkpoint", "load_economy_model"),
}


def __getattr__(name: str):
    target = _TORCH_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "BPETokenizer",
    "ContextSerializer",
    "ContextTensorEncoder",
    "EconomyEncoder",
    "EconomyTensorInput",
    "NO_SUMMARY",
    "SERIALIZER_CONTRACT_VERSION",
    "SUMMARY_MODE_FULL",
    "SUMMARY_MODE_NONE",
    "SUMMARY_MODES",
    "TorchEconomyModel",
    "load_economy_model",
    "normalize_summary_mode",
]
