from __future__ import annotations


class UnsupportedUpdate(Exception):
    pass


class RebuildRequiredError(RuntimeError):
    """Raised when an in-place update is required but a rebuild is needed.

    Carries the :class:`RebuildReason` that forced the rebuild attempt.
    """

    def __init__(self, reason: object, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or f"rebuild required: {reason}")
