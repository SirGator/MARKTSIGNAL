"""Synthetic company profiles and deterministic structural distributions."""

from .distributions import (
    beta_ratio,
    correlated_ratio,
    sample_beta_ratio,
    sample_correlated_ratio,
    sample_truncated_ratio,
    truncated_ratio,
)
from .generator import (
    COMPANY_GENERATOR_VERSION,
    SyntheticCompany,
    company_context,
    context_for_event,
    generate_companies,
    generate_company,
)

__all__ = [
    "COMPANY_GENERATOR_VERSION",
    "SyntheticCompany",
    "beta_ratio",
    "company_context",
    "context_for_event",
    "correlated_ratio",
    "generate_companies",
    "generate_company",
    "sample_beta_ratio",
    "sample_correlated_ratio",
    "sample_truncated_ratio",
    "truncated_ratio",
]
