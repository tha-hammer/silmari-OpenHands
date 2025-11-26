import asyncio
from functools import partial
from typing import Any, Callable

from litellm import acompletion as litellm_acompletion

from openhands.core.exceptions import UserCancelledError
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import (
    LLM,
    LLM_RETRY_EXCEPTIONS,
)
from openhands.llm.model_features import get_features
from openhands.utils.shutdown_listener import should_continue

# Import BAML adapter (optional)
try:
    from openhands.llm.baml_adapter import call_baml_completion_async
except ImportError:
    # BAML adapter not available
    call_baml_completion_async = None


class AsyncLLM(LLM):
    """Asynchronous LLM class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._async_completion = partial(
            self._call_acompletion,
            model=self.config.model,
            api_key=self.config.api_key.get_secret_value()
            if self.config.api_key
            else None,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=self.config.drop_params,
            seed=self.config.seed,
        )

        async_completion_unwrapped = self._async_completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        async def async_completion_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the litellm acompletion function that adds logging and cost tracking."""
            messages: list[dict[str, Any]] | dict[str, Any] = []

            # some callers might send the model and messages directly
            # litellm allows positional args, like completion(model, messages, **kwargs)
            # see llm.py for more details
            if len(args) > 1:
                messages = args[1] if len(args) > 1 else args[0]
                kwargs['messages'] = messages

                # remove the first args, they're sent in kwargs
                args = args[2:]
            elif 'messages' in kwargs:
                messages = kwargs['messages']

            # Set reasoning effort for models that support it
            if get_features(self.config.model).supports_reasoning_effort:
                kwargs['reasoning_effort'] = self.config.reasoning_effort

            # ensure we work with a list of messages
            messages_list = messages if isinstance(messages, list) else [messages]

            # Format Message objects to dict format if needed (LiteLLM expects dicts)
            from openhands.core.message import Message
            from typing import cast
            if messages_list and isinstance(messages_list[0], Message):
                messages = self.format_messages_for_llm(
                    cast(list[Message], messages_list)
                )
            else:
                messages = cast(list[dict[str, Any]], messages_list)

            # Update kwargs with converted messages
            kwargs['messages'] = messages

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            self.log_prompt(messages)

            async def check_stopped() -> None:
                while should_continue():
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        return
                    await asyncio.sleep(0.1)

            stop_check_task = asyncio.create_task(check_stopped())

            try:
                # Route to BAML if enabled
                resp: dict[str, Any]
                if self.use_baml and call_baml_completion_async is not None:
                    try:
                        # Prepare kwargs for BAML
                        baml_kwargs = {
                            'temperature': kwargs.get('temperature'),
                            'max_completion_tokens': kwargs.get('max_completion_tokens') or kwargs.get('max_tokens'),
                            'max_tokens': kwargs.get('max_completion_tokens') or kwargs.get('max_tokens'),
                            'top_p': kwargs.get('top_p'),
                            'top_k': kwargs.get('top_k'),
                            'seed': kwargs.get('seed'),
                            'stop': kwargs.get('stop')
                        }

                        # Call BAML completion asynchronously
                        baml_resp = await call_baml_completion_async(
                            messages=messages,
                            tools=kwargs.get('tools'),
                            **baml_kwargs
                        )
                        # Convert ModelResponse to dict format for compatibility
                        resp = baml_resp.model_dump() if hasattr(baml_resp, 'model_dump') else dict(baml_resp)
                        logger.debug('BAML async completion successful')
                    except Exception as e:
                        logger.warning(f'BAML async completion failed, falling back to LiteLLM: {e}')
                        # Fall back to LiteLLM
                        self.use_baml = False
                        resp = await async_completion_unwrapped(*args, **kwargs)
                else:
                    # Use LiteLLM (default or fallback)
                    resp = await async_completion_unwrapped(*args, **kwargs)

                message_back = resp['choices'][0]['message']['content']
                self.log_response(message_back)

                # log costs and tokens used
                self._post_completion(resp)

                # We do not support streaming in this method, thus return resp
                return resp

            except UserCancelledError:
                logger.debug('LLM request cancelled by user.')
                raise
            except Exception as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                await asyncio.sleep(0.1)
                stop_check_task.cancel()
                try:
                    await stop_check_task
                except asyncio.CancelledError:
                    pass

        self._async_completion = async_completion_wrapper

    async def _call_acompletion(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper for the litellm acompletion function."""
        # Used in testing?
        return await litellm_acompletion(*args, **kwargs)

    @property
    def async_completion(self) -> Callable:
        """Decorator for the async litellm acompletion function."""
        return self._async_completion
