"""Select the exact local feature extractor required by a bundled matcher."""

from features import FEATURE_NAMES, pair_features
from number_features import NUMBER_FEATURE_NAMES, number_pair_features
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features


def extractor_for(model):
    names = list(model.feature_name_)
    if names == FEATURE_NAMES:
        return FEATURE_NAMES, pair_features
    if names == NUMBER_FEATURE_NAMES:
        return NUMBER_FEATURE_NAMES, number_pair_features
    if names == GENERALIZED_FEATURE_NAMES:
        return GENERALIZED_FEATURE_NAMES, generalized_pair_features
    raise ValueError(f"Unsupported model features: {names}")
