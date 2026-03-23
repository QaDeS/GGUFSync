"""Tests for ServiceInstaller."""

from __future__ import annotations

import pytest

from gguf_sync.core.exceptions import ServiceError
from gguf_sync.core.service import ServiceInstaller


class TestServiceInstallerValidation:
    """Tests for service installer validation."""

    def test_valid_service_name(self) -> None:
        """Test that valid service names work."""
        installer = ServiceInstaller("gguf-sync")
        assert installer.service_name == "gguf-sync"

    def test_valid_service_name_with_hyphen(self) -> None:
        """Test that service names with hyphens work."""
        installer = ServiceInstaller("my-gguf-sync")
        assert installer.service_name == "my-gguf-sync"

    def test_valid_service_name_with_underscore(self) -> None:
        """Test that service names with underscores work."""
        installer = ServiceInstaller("my_gguf_sync")
        assert installer.service_name == "my_gguf_sync"

    def test_valid_service_name_uppercase(self) -> None:
        """Test that service names with uppercase work."""
        installer = ServiceInstaller("GGUFSync")
        assert installer.service_name == "GGUFSync"

    def test_invalid_service_name_with_slash(self) -> None:
        """Test that service names with slashes are rejected."""
        with pytest.raises(ServiceError) as exc_info:
            ServiceInstaller("gguf/sync")
        assert "Invalid service name" in str(exc_info.value.message)

    def test_invalid_service_name_with_dot(self) -> None:
        """Test that service names with dots are rejected."""
        with pytest.raises(ServiceError) as exc_info:
            ServiceInstaller("gguf.sync")
        assert "Invalid service name" in str(exc_info.value.message)

    def test_invalid_service_name_with_space(self) -> None:
        """Test that service names with spaces are rejected."""
        with pytest.raises(ServiceError) as exc_info:
            ServiceInstaller("gguf sync")
        assert "Invalid service name" in str(exc_info.value.message)

    def test_invalid_service_name_empty(self) -> None:
        """Test that empty service names are rejected."""
        with pytest.raises(ServiceError) as exc_info:
            ServiceInstaller("")
        assert "Invalid service name" in str(exc_info.value.message)


class TestServiceInstallerMethods:
    """Tests for service installer methods (platform checks)."""

    def test_executable_path_default(self) -> None:
        """Test that default executable path is set correctly."""
        installer = ServiceInstaller()
        assert installer.executable_path.name.startswith("python")

    def test_system_detection(self) -> None:
        """Test that system is detected."""
        installer = ServiceInstaller()
        assert installer.system in ("Linux", "Darwin", "Windows")
