"""LLM registry for managing LLM instances."""

from typing import Optional

from agent_harness.config import LLMConfig
from agent_harness.llm.llm import LLM
from agent_harness.utils.logging import setup_logger

logger = setup_logger()


class LLMRegistry:
    """Registry for managing LLM instances per service."""

    def __init__(self, default_llm_config: LLMConfig):
        """Initialize LLM registry.

        Args:
            default_llm_config: Default LLM configuration
        """
        self.default_config = default_llm_config
        self._llms: dict[str, LLM] = {}
        self._default_llm: Optional[LLM] = None

    def get_llm(
        self,
        service_id: str = "agent",
        config: Optional[LLMConfig] = None,
    ) -> LLM:
        """Get or create LLM instance for service.

        Args:
            service_id: Service identifier
            config: Optional LLM configuration (uses default if not provided)

        Returns:
            LLM instance
        """
        if service_id in self._llms:
            return self._llms[service_id]

        llm_config = config or self.default_config
        llm = LLM(config=llm_config, service_id=service_id)
        self._llms[service_id] = llm

        if service_id == "agent" and self._default_llm is None:
            self._default_llm = llm

        return llm

    def get_default_llm(self) -> LLM:
        """Get default LLM instance.

        Returns:
            Default LLM instance
        """
        if self._default_llm is None:
            self._default_llm = self.get_llm("agent")
        return self._default_llm

