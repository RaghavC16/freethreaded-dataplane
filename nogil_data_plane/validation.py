"""Structural checks before submitting ADD to the state-owning actor."""
from __future__ import annotations
from .errors import InvalidObjectError

ABCROWN_BOUNDS_KEYS = frozenset({"lower_bounds", "upper_bounds", "x_Ls", "x_Us",
    "input_split_idx", "lAs", "betas", "split_history", "sub_domain_clip_decisions",
    "decision_info", "unstable_bounds", "alphas", "c"})
ABCROWN_DOMAIN_DATA_KEYS = frozenset({"history", "thresholds", "depths"})

def validate_mapping_add(bounds, domain_data) -> None:
    if not isinstance(bounds, dict) or not isinstance(domain_data, dict):
        raise InvalidObjectError("ADD bounds and domain_data must be dictionaries")

def validate_alpha_beta_crown_add(bounds, domain_data) -> None:
    validate_mapping_add(bounds, domain_data)
    missing_bounds = ABCROWN_BOUNDS_KEYS - bounds.keys()
    missing_data = ABCROWN_DOMAIN_DATA_KEYS - domain_data.keys()
    if missing_bounds or missing_data:
        raise InvalidObjectError(
            f"ADD is missing keys: bounds={sorted(missing_bounds)}, domain_data={sorted(missing_data)}")
