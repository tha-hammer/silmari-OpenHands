"""LLM integration using BAML."""

import os
from typing import Any

from agent_harness.config import LLMConfig
from agent_harness.llm.baml_adapter import (
    call_baml_completion,
    call_baml_completion_async,
)
from agent_harness.utils.logging import setup_logger

logger = setup_logger(__name__)


class LLM:
    """LLM class using BAML for all LLM calls.

    This is a BAML-only implementation for the standalone harness.
    No LiteLLM fallback - BAML is the exclusive API layer.
    """

    def __init__(
        self,
        config: LLMConfig,
        service_id: str = "agent",
    ):
        """Initialize LLM instance.

        Args:
            config: LLM configuration
            service_id: Service identifier
        """
        self.config = config
        self.service_id = service_id
        self.use_baml = config.use_baml  # Always True for harness
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables for BAML client from config."""
        if self.config.api_key:
            os.environ["BAML_API_KEY"] = self.config.api_key
        else:
            os.environ["BAML_API_KEY"] = ""  # Empty string for local LLMs

        if self.config.base_url:
            os.environ["BAML_BASE_URL"] = self.config.base_url
        else:
            os.environ.pop("BAML_BASE_URL", None)

        if self.config.model:
            os.environ["BAML_MODEL"] = self.config.model

    def completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous completion call via BAML.

        Args:
            messages: List of message dicts
            tools: Optional list of tool definitions
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dict compatible with LiteLLM ModelResponse format
        """
        return call_baml_completion(
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **kwargs,
        )

    async def async_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Asynchronous completion call via BAML.

        Args:
            messages: List of message dicts
            tools: Optional list of tool definitions
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dict compatible with LiteLLM ModelResponse format
        """
        return await call_baml_completion_async(
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **kwargs,
        )

