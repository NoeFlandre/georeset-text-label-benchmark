"""Explicit failures for invalid inputs and non-one-to-one joins."""


class OverlapError(ValueError):
    """Base exception for pipeline input and join failures."""


class DataValidationError(OverlapError):
    """A source row violates its published schema or value contract."""


class CardinalityError(OverlapError):
    """A declared unique key is duplicated or a required match is not unique."""


class ProvenanceError(OverlapError):
    """Source revisions or regional provenance do not agree."""
