"""Custom exceptions for gguf_sync."""

from __future__ import annotations


class GGUFSyncError(Exception):
    """Base exception for all gguf_sync errors."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(GGUFSyncError):
    """Configuration-related errors."""

    pass


class GGUFError(GGUFSyncError):
    """GGUF parsing errors."""

    pass


class SyncError(GGUFSyncError):
    """File synchronization errors."""

    pass


class BackendError(GGUFSyncError):
    """Backend-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        backend_name: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.backend_name = backend_name


class WatchError(GGUFSyncError):
    """Filesystem watching errors."""

    pass


class ServiceError(GGUFSyncError):
    """Service installation/management errors."""

    pass
