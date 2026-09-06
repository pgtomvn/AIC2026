"""Shared, side-effect-free types for spatial query decomposition."""

from __future__ import annotations

from enum import Enum


class SpatialRelationEnum(str, Enum):
    """Canonical spatial relations understood by Advanced Textual KIS."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR_FROM = "far_from"
    INSIDE = "inside"
    SURROUNDING = "surrounding"
    OVERLAPPING = "overlapping"
    BETWEEN = "between"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"


SPATIAL_RELATION_VALUES = frozenset(value.value for value in SpatialRelationEnum)


__all__ = ["SPATIAL_RELATION_VALUES", "SpatialRelationEnum"]
