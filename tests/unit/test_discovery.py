"""Tests for backend auto-discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from link_models.core.discovery import (
    BackendDiscovery,
    DiscoveredBackend,
    create_config_from_discovered,
)


class TestBackendDiscovery:
    """Tests for BackendDiscovery class."""

    @pytest.fixture
    def discovery(self) -> BackendDiscovery:
        return BackendDiscovery()

    def test_discovery_finds_vllm_cache(self, discovery: BackendDiscovery) -> None:
        """Test that discovery can find vLLM cache directory."""
        backends = discovery.discover_all()
        # Should find at least one backend (vllm cache)
        assert len(backends) >= 1
        names = [b.name for b in backends]
        assert "vllm" in names or any("vllm" in n for n in names)

    def test_discovery_returns_list(self, discovery: BackendDiscovery) -> None:
        """Test that discover_all returns a list."""
        result = discovery.discover_all()
        assert isinstance(result, list)

    def test_discovered_backend_has_required_fields(
        self,
        discovery: BackendDiscovery,
    ) -> None:
        """Test that discovered backends have required fields."""
        backends = discovery.discover_all()
        for backend in backends:
            assert hasattr(backend, "name")
            assert hasattr(backend, "backend_type")
            assert hasattr(backend, "install_dir")
            assert isinstance(backend.install_dir, Path)


class TestCreateConfigFromDiscovered:
    """Tests for create_config_from_discovered function."""

    def test_empty_list_returns_empty_config(self) -> None:
        """Test that empty list produces empty config."""
        result = create_config_from_discovered([])
        assert result == {}

    def test_single_backend_config(self) -> None:
        """Test config generation from single backend."""
        discovered = [
            DiscoveredBackend(
                name="ollama",
                backend_type="ollama",
                install_dir=Path("/usr/local/ollama"),
                models_dir=Path("/usr/local/ollama/models"),
            )
        ]
        result = create_config_from_discovered(discovered)

        assert "ollama" in result
        assert result["ollama"]["enabled"] is True

    def test_multiple_backends_config(self) -> None:
        """Test config generation from multiple backends."""
        discovered = [
            DiscoveredBackend(
                name="llama_cpp",
                backend_type="llama_cpp",
                install_dir=Path("/usr/share/llama.cpp"),
                models_dir=Path("/usr/share/llama.cpp"),
            ),
            DiscoveredBackend(
                name="ollama",
                backend_type="ollama",
                install_dir=Path("/usr/local/ollama"),
                models_dir=Path("/usr/local/ollama/models"),
            ),
            DiscoveredBackend(
                name="localai",
                backend_type="localai",
                install_dir=Path("/localai"),
                models_dir=Path("/localai/models"),
                is_running=True,
                port=8080,
            ),
        ]
        result = create_config_from_discovered(discovered)

        assert "llama_cpp" in result
        assert "ollama" in result
        assert "localai" in result
        assert result["localai"]["is_running"] is True
        assert result["localai"]["port"] == 8080

    def test_running_backend_includes_status(self) -> None:
        """Test that running backends include status info."""
        discovered = [
            DiscoveredBackend(
                name="ollama",
                backend_type="ollama",
                install_dir=Path("/test"),
                models_dir=Path("/test/models"),
                is_running=True,
                port=11434,
            )
        ]
        result = create_config_from_discovered(discovered)

        assert result["ollama"]["is_running"] is True
        assert result["ollama"]["port"] == 11434
