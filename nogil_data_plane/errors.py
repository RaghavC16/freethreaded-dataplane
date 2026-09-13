"""Exceptions raised by the no-GIL domain data plane."""

from __future__ import annotations


class DataPlaneError(RuntimeError):
    """Base class for data-plane failures."""


class ConnectionClosedError(DataPlaneError):
    """The peer closed a connection before a frame was complete."""


class ProtocolError(DataPlaneError):
    """A peer sent an invalid or unsupported protocol message."""

    code = "BAD_FRAME"


class VersionMismatchError(ProtocolError):
    """The peer uses a different job, protocol, or schema version."""

    code = "VERSION_MISMATCH"


class InvalidObjectError(DataPlaneError):
    """A domain packet contains an unsupported or malformed value."""

    code = "INVALID_OBJECT"


class InvalidStateError(DataPlaneError):
    """An operation violates shared-checkout ordering."""

    code = "INVALID_STATE"


class DomainOperationError(DataPlaneError):
    """The wrapped BatchedDomainList operation failed."""

    code = "DOMAIN_OPERATION_FAILED"


class ActorFailedError(DataPlaneError):
    """The domain actor has permanently failed."""

    code = "ACTOR_FAILED"


class DataPlaneCancelledError(DataPlaneError):
    """The caller's control-plane cancellation signal is set."""


class RemoteDataPlaneError(DataPlaneError):
    """A fatal error returned by the remote actor host."""

    def __init__(self, code: str, message: str, *, fatal: bool = True) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.fatal = fatal
