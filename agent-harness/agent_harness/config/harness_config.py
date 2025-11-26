"""Configuration models for agent harness."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    use_baml: bool = True  # Always True for harness


@dataclass
class HarnessConfig:
    """Harness configuration."""

    llm: LLMConfig
    agent_name: str = "CodeActAgent"
    tools: list[str] = field(default_factory=lambda: ["cmd", "editor", "think"])
    runtime_type: str = "local"
    workspace_path: str = "./workspace"
    storage_type: str = "local"
    storage_path: str = "~/.agent-harness"
    max_iterations: int = 100
    max_budget: Optional[float] = None
    headless: bool = True

