"""State management for agent execution."""

import base64
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_harness.events.event import Event
from agent_harness.interfaces.storage import StorageInterface
from agent_harness.utils.logging import setup_logger

logger = setup_logger()


@dataclass
class State:
    """Represents the running state of an agent.

    This is a simplified version for the standalone harness.
    """

    session_id: str = ''
    history: list[Event] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    extra_data: dict[str, Any] = field(default_factory=dict)
    last_error: str = ''
    iteration: int = 0
    max_iterations: int = 100

    async def save_to_session(
        self, session_id: str, storage: StorageInterface
    ) -> None:
        """Save state to storage.

        Args:
            session_id: Session identifier
            storage: Storage interface
        """
        try:
            # Serialize state (excluding history which is stored separately)
            state_dict = {
                "session_id": self.session_id,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "extra_data": self.extra_data,
                "last_error": self.last_error,
                "iteration": self.iteration,
                "max_iterations": self.max_iterations,
            }
            pickled = pickle.dumps(state_dict)
            encoded = base64.b64encode(pickled).decode('utf-8')

            await storage.save_file(
                f"sessions/{session_id}/state.pkl", encoded.encode('utf-8')
            )
            logger.debug(f"Saved state to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save state to session: {e}")
            raise

    @staticmethod
    async def restore_from_session(
        session_id: str, storage: StorageInterface
    ) -> "State":
        """Restore state from storage.

        Args:
            session_id: Session identifier
            storage: Storage interface

        Returns:
            Restored State instance
        """
        try:
            encoded = await storage.load_file(f"sessions/{session_id}/state.pkl")
            pickled = base64.b64decode(encoded)
            state_dict = pickle.loads(pickled)

            state = State(
                session_id=state_dict.get("session_id", session_id),
                inputs=state_dict.get("inputs", {}),
                outputs=state_dict.get("outputs", {}),
                extra_data=state_dict.get("extra_data", {}),
                last_error=state_dict.get("last_error", ""),
                iteration=state_dict.get("iteration", 0),
                max_iterations=state_dict.get("max_iterations", 100),
            )
            logger.debug(f"Restored state from session {session_id}")
            return state
        except FileNotFoundError:
            # Return new state if not found
            logger.debug(f"No saved state found for session {session_id}, creating new state")
            return State(session_id=session_id)
        except Exception as e:
            logger.error(f"Failed to restore state from session: {e}")
            raise

