---
date: 2025-11-25T20:27:36-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "Agent Architecture: File System Handling, Tool Calls, Looping Strategy, and Context Windows"
tags: [research, codebase, agents, architecture, delegation, file-system, tool-calls, context-windows]
status: complete
last_updated: 2025-11-25
last_updated_by: Auto
---

# Research: Agent Architecture: File System Handling, Tool Calls, Looping Strategy, and Context Windows

**Date**: 2025-11-25T20:27:36-05:00
**Researcher**: Auto
**Git Commit**: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
**Branch**: main
**Repository**: silmari-OpenHands

## Research Question

How do agents in this codebase handle the file system? How are the tool calls structured? What is the looping strategy with the main agent and sub-agents? Do the sub-agents have separate context windows? How would I replicate the agents?

## Summary

This research documents the agent architecture in OpenHands, focusing on four key areas:

1. **File System Handling**: Agents interact with the file system through a Runtime abstraction layer. File operations (read/write/edit) are handled via `FileReadAction`, `FileEditAction`, and `FileWriteAction` events that flow through the EventStream to the Runtime, which executes them in a sandboxed environment with path resolution and security checks.

2. **Tool Call Structure**: Tool calls are structured using LiteLLM's function calling format. The agent's LLM response contains tool calls that are parsed by `response_to_actions()` in `function_calling.py`, converting them into OpenHands Action objects. Tools are defined as `ChatCompletionToolParam` objects with function schemas.

3. **Looping Strategy**: The system uses a parent-delegate pattern where the main agent can spawn sub-agents (delegates) via `AgentDelegateAction`. Delegates share the same EventStream but have separate State objects with their own `start_id` and `end_id` ranges. The parent agent forwards events to active delegates and resumes control when delegates finish.

4. **Context Windows**: Sub-agents (delegates) do NOT have separate context windows. They share the same EventStream and LLM registry, but they have separate State objects that track different event ranges via `start_id`/`end_id`. The conversation memory and condenser operate on the full event history, but each agent's view is filtered by its state's event range.

## Detailed Findings

### 1. File System Handling

#### Architecture Overview

Agents interact with the file system through a layered architecture:

```
Agent → Action (FileReadAction/FileEditAction) → EventStream → Runtime → File System
```

#### File Operations Flow

**1.1 Action Creation**
- Agents create file-related actions through tool calls:
  - `FileReadAction` - for reading files
  - `FileEditAction` - for editing files (supports both LLM-based and ACI-based editing)
  - `FileWriteAction` - for writing files

These actions are created in `openhands/agenthub/codeact_agent/function_calling.py` when the LLM makes tool calls:

```296:300:openhands/agenthub/codeact_agent/function_calling.py
def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
```

**1.2 Path Resolution and Security**

File paths are resolved through `resolve_path()` in `openhands/runtime/utils/files.py`:

```12:51:openhands/runtime/utils/files.py
def resolve_path(
    file_path: str,
    working_directory: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
) -> Path:
    """Resolve a file path to a path on the host filesystem.

    Args:
        file_path: The path to resolve.
        working_directory: The working directory of the agent.
        workspace_mount_path_in_sandbox: The path to the workspace inside the sandbox.
        workspace_base: The base path of the workspace on the host filesystem.

    Returns:
        The resolved path on the host filesystem.
    """
    path_in_sandbox = Path(file_path)

    # Apply working directory
    if not path_in_sandbox.is_absolute():
        path_in_sandbox = Path(working_directory) / path_in_sandbox

    # Sanitize the path with respect to the root of the full sandbox
    # (deny any .. path traversal to parent directories of the sandbox)
    abs_path_in_sandbox = path_in_sandbox.resolve()

    # If the path is outside the workspace, deny it
    if not abs_path_in_sandbox.is_relative_to(workspace_mount_path_in_sandbox):
        raise PermissionError(f'File access not permitted: {file_path}')

    # Get path relative to the root of the workspace inside the sandbox
    path_in_workspace = abs_path_in_sandbox.relative_to(
        Path(workspace_mount_path_in_sandbox)
    )

    # Get path relative to host
    path_in_host_workspace = Path(workspace_base) / path_in_workspace

    return path_in_host_workspace
```

Key security features:
- Paths are resolved relative to the working directory
- Absolute paths are sanitized to prevent directory traversal
- Access is restricted to files within the workspace mount path
- Permission errors are returned as `ErrorObservation` objects

**1.3 Runtime Execution**

The Runtime base class (`openhands/runtime/base.py`) subscribes to the EventStream and handles file operations:

```148:151:openhands/runtime/base.py
        self.event_stream = event_stream
        if event_stream:
            event_stream.subscribe(
                EventStreamSubscriber.RUNTIME, self.on_event, self.sid
            )
```

File read operations are handled asynchronously:

```71:98:openhands/runtime/utils/files.py
async def read_file(
    path: str,
    workdir: str,
    workspace_base: str,
    workspace_mount_path_in_sandbox: str,
    start: int = 0,
    end: int = -1,
) -> Observation:
    try:
        whole_path = resolve_path(
            path, workdir, workspace_base, workspace_mount_path_in_sandbox
        )
    except PermissionError:
        return ErrorObservation(
            f"You're not allowed to access this path: {path}. You can only access paths inside the workspace."
        )

    try:
        with open(whole_path, 'r', encoding='utf-8') as file:  # noqa: ASYNC101
            lines = read_lines(file.readlines(), start, end)
    except FileNotFoundError:
        return ErrorObservation(f'File not found: {path}')
    except UnicodeDecodeError:
        return ErrorObservation(f'File could not be decoded as utf-8: {path}')
    except IsADirectoryError:
        return ErrorObservation(f'Path is a directory: {path}. You can only read files')
    code_view = ''.join(lines)
    return FileReadObservation(path=path, content=code_view)
```

**1.4 File Edit Operations**

File editing supports two modes:
- **LLM-based editing**: Deprecated, uses `LLMBasedFileEditTool`
- **ACI-based editing**: Uses `create_str_replace_editor_tool()` which creates `FileEditAction` with commands like `replace`, `insert`, `delete`

The editor tool is created in `openhands/agenthub/codeact_agent/tools/str_replace_editor.py` and converts tool calls to `FileEditAction` objects.

### 2. Tool Call Structure

#### 2.1 Tool Definition

Tools are defined as `ChatCompletionToolParam` objects (from LiteLLM). Each agent maintains a list of tools:

```50:51:openhands/controller/agent.py
        self.mcp_tools: dict[str, ChatCompletionToolParam] = {}
        self.tools: list = []
```

Tools are registered in the agent's `_get_tools()` method. For example, in `CodeActAgent`:

```108:153:openhands/agenthub/codeact_agent/codeact_agent.py
    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            # For historical reasons, previously OpenAI enforces max function description length of 1k characters
            # https://community.openai.com/t/function-call-description-max-length/529902
            # But it no longer seems to be an issue recently
            # https://community.openai.com/t/was-the-character-limit-for-schema-descriptions-upgraded/1225975
            # Tested on GPT-5 and longer description still works. But we still keep the logic to be safe for older models.
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            if sys.platform == 'win32':
                logger.warning('Windows runtime does not support browsing yet')
            else:
                tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )
        return tools
```

#### 2.2 Tool Call Processing

When the LLM responds with tool calls, they are processed by `response_to_actions()`:

```73:100:openhands/agenthub/codeact_agent/function_calling.py
def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e
```

The function then maps tool call names to specific Action types:

```104:122:openhands/agenthub/codeact_agent/function_calling.py
            if tool_call.function.name == create_cmd_run_tool()['function']['name']:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                # convert is_input to boolean
                is_input = arguments.get('is_input', 'false') == 'true'
                action = CmdRunAction(command=arguments['command'], is_input=is_input)

                # Set hard timeout if provided
                if 'timeout' in arguments:
                    try:
                        action.set_hard_timeout(float(arguments['timeout']))
                    except ValueError as e:
                        raise FunctionCallValidationError(
                            f"Invalid float passed to 'timeout' argument: {arguments['timeout']}"
                        ) from e
                set_security_risk(action, arguments)
```

#### 2.3 Tool Call Metadata

Tool calls include metadata for tracking:

```818:824:openhands/controller/agent_controller.py
        # associate the delegate action with the initiating tool call
        for event in reversed(self.state.history):
            if isinstance(event, AgentDelegateAction):
                delegate_action = event
                obs.tool_call_metadata = delegate_action.tool_call_metadata
                break
```

This metadata links observations back to their originating tool calls.

#### 2.4 System Message with Tools

Tools are included in the system message sent to the LLM:

```59:90:openhands/controller/agent.py
    def get_system_message(self) -> 'SystemMessageAction | None':
        """Returns a SystemMessageAction containing the system message and tools.
        This will be added to the event stream as the first message.

        Returns:
            SystemMessageAction: The system message action with content and tools
            None: If there was an error generating the system message
        """
        # Import here to avoid circular imports
        from openhands.events.action.message import SystemMessageAction

        try:
            if not self.prompt_manager:
                logger.warning(
                    f'[{self.name}] Prompt manager not initialized before getting system message'
                )
                return None

            system_message = self.prompt_manager.get_system_message(
                cli_mode=self.config.cli_mode
            )

            # Get tools if available
            tools = getattr(self, 'tools', None)

            system_message_action = SystemMessageAction(
                content=system_message, tools=tools, agent_class=self.name
            )
            # Set the source attribute
            system_message_action._source = EventSource.AGENT  # type: ignore

            return system_message_action
        except Exception as e:
            logger.warning(f'[{self.name}] Failed to generate system message: {e}')
            return None
```

### 3. Looping Strategy: Main Agent and Sub-Agents

#### 3.1 Agent Controller Architecture

The `AgentController` class manages the agent's execution loop and handles delegation:

```99:112:openhands/controller/agent_controller.py
class AgentController:
    id: str
    agent: Agent
    max_iterations: int
    event_stream: EventStream
    state: State
    confirmation_mode: bool
    agent_to_llm_config: dict[str, LLMConfig]
    agent_configs: dict[str, AgentConfig]
    parent: 'AgentController | None' = None
    delegate: 'AgentController | None' = None
    _pending_action_info: tuple[Action, float] | None = None  # (action, timestamp)
    _closed: bool = False
    _cached_first_user_message: MessageAction | None = None
```

#### 3.2 Delegation Flow

**3.2.1 Creating a Delegate**

When an agent wants to delegate, it creates an `AgentDelegateAction`:

```76:86:openhands/events/action/agent.py
@dataclass
class AgentDelegateAction(Action):
    agent: str
    inputs: dict
    thought: str = ''
    action: str = ActionType.DELEGATE

    @property
    def message(self) -> str:
        return f"I'm asking {self.agent} for help with this task."
```

The controller handles this action and starts a delegate:

```701:760:openhands/controller/agent_controller.py
    async def start_delegate(self, action: AgentDelegateAction) -> None:
        """Start a delegate agent to handle a subtask.

        OpenHands is a multi-agentic system. A `task` is a conversation between
        OpenHands (the whole system) and the user, which might involve one or more inputs
        from the user. It starts with an initial input (typically a task statement) from
        the user, and ends with either an `AgentFinishAction` initiated by the agent, a
        stop initiated by the user, or an error.

        A `subtask` is a conversation between an agent and the user, or another agent. If a `task`
        is conducted by a single agent, then it's also a `subtask`. Otherwise, a `task` consists of
        multiple `subtasks`, each executed by one agent.

        Args:
            action (AgentDelegateAction): The action containing information about the delegate agent to start.
        """
        agent_cls: type[Agent] = Agent.get_cls(action.agent)
        agent_config = self.agent_configs.get(action.agent, self.agent.config)
        # Make sure metrics are shared between parent and child for global accumulation
        delegate_agent = agent_cls(
            config=agent_config, llm_registry=self.agent.llm_registry
        )

        # Take a snapshot of the current metrics before starting the delegate
        state = State(
            session_id=self.id.removesuffix('-delegate'),
            user_id=self.user_id,
            inputs=action.inputs or {},
            iteration_flag=self.state.iteration_flag,
            budget_flag=self.state.budget_flag,
            delegate_level=self.state.delegate_level + 1,
            # global metrics should be shared between parent and child
            metrics=self.state.metrics,
            # start on top of the stream
            start_id=self.event_stream.get_latest_event_id() + 1,
            parent_metrics_snapshot=self.state_tracker.get_metrics_snapshot(),
            parent_iteration=self.state.iteration_flag.current_value,
        )
        self.log(
            'debug',
            f'start delegate, creating agent {delegate_agent.name}',
        )

        # Create the delegate with is_delegate=True so it does NOT subscribe directly
        self.delegate = AgentController(
            sid=self.id + '-delegate',
            file_store=self.file_store,
            user_id=self.user_id,
            agent=delegate_agent,
            event_stream=self.event_stream,
            conversation_stats=self.conversation_stats,
            iteration_delta=self._initial_max_iterations,
            budget_per_task_delta=self._initial_max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            initial_state=state,
            is_delegate=True,
            headless_mode=self.headless_mode,
            security_analyzer=self.security_analyzer,
        )
```

Key points:
- Delegates share the same `EventStream` and `LLMRegistry`
- Delegates have separate `State` objects with their own `start_id` (beginning of their event range)
- Delegates have `is_delegate=True` so they don't subscribe to the event stream directly
- Delegates share global metrics but track local metrics via snapshots

**3.2.2 Event Forwarding**

The parent controller forwards events to active delegates:

```441:473:openhands/controller/agent_controller.py
    def on_event(self, event: Event) -> None:
        """Callback from the event stream. Notifies the controller of incoming events.

        Args:
            event (Event): The incoming event to process.
        """
        # If we have a delegate that is not finished or errored, forward events to it
        if self.delegate is not None:
            delegate_state = self.delegate.get_agent_state()
            if (
                delegate_state
                not in (
                    AgentState.FINISHED,
                    AgentState.ERROR,
                    AgentState.REJECTED,
                )
                or 'RuntimeError: Agent reached maximum iteration.'
                in self.delegate.state.last_error
                or 'RuntimeError:Agent reached maximum budget for conversation'
                in self.delegate.state.last_error
            ):
                # Forward the event to delegate and skip parent processing
                asyncio.get_event_loop().run_until_complete(
                    self.delegate._on_event(event)
                )
                return
            else:
                # delegate is done or errored, so end it
                self.end_delegate()
                return

        # continue parent processing only if there's no active delegate
        asyncio.get_event_loop().run_until_complete(self._on_event(event))
```

**3.2.3 Delegate Completion**

When a delegate finishes, the parent ends it and resumes control:

```762:827:openhands/controller/agent_controller.py
    def end_delegate(self) -> None:
        """Ends the currently active delegate (e.g., if it is finished or errored).

        so that this controller can resume normal operation.
        """
        if self.delegate is None:
            return

        delegate_state = self.delegate.get_agent_state()

        # update iteration that is shared across agents
        self.state.iteration_flag.current_value = (
            self.delegate.state.iteration_flag.current_value
        )

        # Calculate delegate-specific metrics before closing the delegate
        delegate_metrics = self.state.get_local_metrics()
        logger.info(f'Local metrics for delegate: {delegate_metrics}')

        # close the delegate controller before adding new events
        asyncio.get_event_loop().run_until_complete(self.delegate.close())

        if delegate_state in (AgentState.FINISHED, AgentState.REJECTED):
            # retrieve delegate result
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )

            # prepare delegate result observation
            # TODO: replace this with AI-generated summary (#2395)
            # Filter out metrics from the formatted output to avoid clutter
            display_outputs = {
                k: v for k, v in delegate_outputs.items() if k != 'metrics'
            }
            formatted_output = ', '.join(
                f'{key}: {value}' for key, value in display_outputs.items()
            )
            content = (
                f'{self.delegate.agent.name} finishes task with {formatted_output}'
            )
        else:
            # delegate state is ERROR
            # emit AgentDelegateObservation with error content
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )
            content = (
                f'{self.delegate.agent.name} encountered an error during execution.'
            )

        content = f'Delegated agent finished with result:\n\n{content}'

        # emit the delegate result observation
        obs = AgentDelegateObservation(outputs=delegate_outputs, content=content)

        # associate the delegate action with the initiating tool call
        for event in reversed(self.state.history):
            if isinstance(event, AgentDelegateAction):
                delegate_action = event
                obs.tool_call_metadata = delegate_action.tool_call_metadata
                break

        self.event_stream.add_event(obs, EventSource.AGENT)

        # unset delegate so parent can resume normal handling
        self.delegate = None
```

**3.2.4 Step Execution**

The main step loop checks for delegates before executing:

```829:895:openhands/controller/agent_controller.py
    async def _step(self) -> None:
        """Executes a single step of the parent or delegate agent. Detects stuck agents and limits on the number of iterations and the task budget."""
        if self.get_agent_state() != AgentState.RUNNING:
            self.log(
                'debug',
                f'Agent not stepping because state is {self.get_agent_state()} (not RUNNING)',
                extra={'msg_type': 'STEP_BLOCKED_STATE'},
            )
            return

        if self._pending_action:
            action_id = getattr(self._pending_action, 'id', 'unknown')
            action_type = type(self._pending_action).__name__
            self.log(
                'debug',
                f'Agent not stepping because of pending action: {action_type} (id={action_id})',
                extra={'msg_type': 'STEP_BLOCKED_PENDING_ACTION'},
            )
            return

        self.log(
            'debug',
            f'LEVEL {self.state.delegate_level} LOCAL STEP {self.state.get_local_step()} GLOBAL STEP {self.state.iteration_flag.current_value}',
            extra={'msg_type': 'STEP'},
        )

        # Synchronize spend across all llm services with the budget flag
        self.state_tracker.sync_budget_flag_with_metrics()
        if self._is_stuck():
            await self._react_to_exception(
                AgentStuckInLoopError('Agent got stuck in a loop')
            )
            return

        try:
            self.state_tracker.run_control_flags()
        except Exception as e:
            logger.warning('Control flag limits hit')
            await self._react_to_exception(e)
            return

        action: Action = NullAction()

        if self._replay_manager.should_replay():
            # in replay mode, we don't let the agent to proceed
            # instead, we replay the action from the replay trajectory
            action = self._replay_manager.step()
        else:
            try:
                action = self.agent.step(self.state)
                if action is None:
                    raise LLMNoActionError('No action was returned')
                action._source = EventSource.AGENT  # type: ignore [attr-defined]
            except (
                LLMMalformedActionError,
                LLMNoActionError,
                LLMResponseError,
                FunctionCallValidationError,
                FunctionCallNotExistsError,
            ) as e:
                self.event_stream.add_event(
                    ErrorObservation(
                        content=str(e),
                    ),
                    EventSource.AGENT,
                )
                return
```

#### 3.3 State Isolation

Each agent (parent or delegate) maintains its own `State` object with event range tracking:

```48:101:openhands/controller/state/state.py
@dataclass
class State:
    """Represents the running state of an agent in the OpenHands system, saving data of its operation and memory.

    - Multi-agent/delegate state:
      - store the task (conversation between the agent and the user)
      - the subtask (conversation between an agent and the user or another agent)
      - global and local iterations
      - delegate levels for multi-agent interactions
      - almost stuck state

    - Running state of an agent:
      - current agent state (e.g., LOADING, RUNNING, PAUSED)
      - traffic control state for rate limiting
      - confirmation mode
      - the last error encountered

    - Data for saving and restoring the agent:
      - save to and restore from a session
      - serialize with pickle and base64

    - Save / restore data about message history
      - start and end IDs for events in agent's history
      - summaries and delegate summaries

    - Metrics:
      - global metrics for the current task
      - local metrics for the current subtask

    - Extra data:
      - additional task-specific data
    """

    session_id: str = ''
    user_id: str | None = None
    iteration_flag: IterationControlFlag = field(
        default_factory=lambda: IterationControlFlag(
            limit_increase_amount=100, current_value=0, max_value=100
        )
    )
    conversation_stats: ConversationStats | None = None
    budget_flag: BudgetControlFlag | None = None
    confirmation_mode: bool = False
    history: list[Event] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    agent_state: AgentState = AgentState.LOADING
    resume_state: AgentState | None = None

    # root agent has level 0, and every delegate increases the level by one
    delegate_level: int = 0
    # start_id and end_id track the range of events in history
    start_id: int = -1
    end_id: int = -1
```

The `start_id` and `end_id` fields define the event range for each agent's view of history.

### 4. Context Windows and Sub-Agents

#### 4.1 Shared EventStream and LLM Registry

**Sub-agents do NOT have separate context windows.** They share:
- The same `EventStream` (all events go to the same stream)
- The same `LLMRegistry` (metrics are tracked globally)

This is evident in the delegate creation:

```720:722:openhands/controller/agent_controller.py
        delegate_agent = agent_cls(
            config=agent_config, llm_registry=self.agent.llm_registry
        )
```

And the shared EventStream:

```749:750:openhands/controller/agent_controller.py
            event_stream=self.event_stream,
```

#### 4.2 State-Based Event Filtering

While delegates share the EventStream, they have separate `State` objects that filter events by range:

```725:738:openhands/controller/agent_controller.py
        # Take a snapshot of the current metrics before starting the delegate
        state = State(
            session_id=self.id.removesuffix('-delegate'),
            user_id=self.user_id,
            inputs=action.inputs or {},
            iteration_flag=self.state.iteration_flag,
            budget_flag=self.state.budget_flag,
            delegate_level=self.state.delegate_level + 1,
            # global metrics should be shared between parent and child
            metrics=self.state.metrics,
            # start on top of the stream
            start_id=self.event_stream.get_latest_event_id() + 1,
            parent_metrics_snapshot=self.state_tracker.get_metrics_snapshot(),
            parent_iteration=self.state.iteration_flag.current_value,
        )
```

The `start_id` is set to the current event stream's latest ID + 1, meaning the delegate only sees events that occur after its creation.

#### 4.3 History Loading

The `StateTracker` loads history based on the `start_id`:

```115:137:openhands/controller/state/state_tracker.py
        # delegates start with a start_id and initially won't find any events

        # Get the start and end IDs for the history range
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = self.event_stream.get_latest_event_id()

        if start_id > end_id + 1:
            logger.warning(
                f'start_id {start_id} is greater than end_id + 1 ({end_id + 1}). History will be empty.',
            )
            self.state.history = []
            return

        # Load events from the event stream
        events = self.event_stream.search_events(
            start_id=start_id,
            end_id=end_id,
        )

        self.state.history = events
        logger.debug(
            f'Loaded {len(events)} events for agent {id} (start_id={start_id}, end_id={end_id})',
            extra={
                'start_id': start_id,
```

#### 4.4 Conversation Memory and Condensation

The conversation memory and condenser operate on the full event history, but each agent's view is filtered:

```192:202:openhands/agenthub/codeact_agent/codeact_agent.py
        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action
```

The condenser receives the agent's `state`, which contains only the events in that agent's range (from `start_id` to `end_id`).

#### 4.5 Metrics Sharing

Metrics are shared globally via the `LLMRegistry`, but local metrics are tracked via snapshots:

```287:296:openhands/controller/state/state.py
    def get_local_step(self):
        if not self.parent_iteration:
            return self.iteration_flag.current_value

        return self.iteration_flag.current_value - self.parent_iteration

    def get_local_metrics(self):
        if not self.parent_metrics_snapshot:
            return self.metrics
        return self.metrics.diff(self.parent_metrics_snapshot)
```

This allows tracking both global (task-level) and local (subtask-level) metrics.

### 5. How to Replicate the Agents

#### 5.1 Agent Base Class

To create a new agent, inherit from the `Agent` base class:

```25:51:openhands/controller/agent.py
class Agent(ABC):
    DEPRECATED = False
    """
    This abstract base class is an general interface for an agent dedicated to
    executing a specific instruction and allowing human interaction with the
    agent during execution.
    It tracks the execution status and maintains a history of interactions.
    """

    _registry: dict[str, type['Agent']] = {}
    sandbox_plugins: list[PluginRequirement] = []

    config_model: type[AgentConfig] = AgentConfig
    """Class field that specifies the config model to use for the agent. Subclasses may override with a derived config model if needed."""

    def __init__(
        self,
        config: AgentConfig,
        llm_registry: LLMRegistry,
    ):
        self.llm = llm_registry.get_llm_from_agent_config('agent', config)
        self.llm_registry = llm_registry
        self.config = config
        self._complete = False
        self._prompt_manager: 'PromptManager' | None = None
        self.mcp_tools: dict[str, ChatCompletionToolParam] = {}
        self.tools: list = []
```

#### 5.2 Required Methods

Implement the abstract `step()` method:

```104:109:openhands/controller/agent.py
    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """Starts the execution of the assigned instruction. This method should
        be implemented by subclasses to define the specific execution logic.
        """
        pass
```

#### 5.3 Agent Registration

Register the agent in the registry:

```120:133:openhands/controller/agent.py
    @classmethod
    def register(cls, name: str, agent_cls: type['Agent']) -> None:
        """Registers an agent class in the registry.

        Parameters:
        - name (str): The name to register the class under.
        - agent_cls (Type['Agent']): The class to register.

        Raises:
        - AgentAlreadyRegisteredError: If name already registered
        """
        if name in cls._registry:
            raise AgentAlreadyRegisteredError(name)
        cls._registry[name] = agent_cls
```

#### 5.4 Example: CodeActAgent

The `CodeActAgent` provides a complete example:

```49:107:openhands/agenthub/codeact_agent/codeact_agent.py
class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)
```

#### 5.5 Step Implementation Pattern

The `step()` method typically:
1. Processes pending actions
2. Condenses history
3. Builds messages from events
4. Calls LLM with tools
5. Converts response to actions

```161:225:openhands/agenthub/codeact_agent/codeact_agent.py
    def step(self, state: State) -> 'Action':
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - CondensationAction(...) - condense conversation history by forgetting specified events and optionally providing a summary
        - FileReadAction(path, ...) - read file content from specified path
        - FileEditAction(path, ...) - edit file using LLM-based (deprecated) or ACI-based editing
        - AgentThinkAction(thought) - log agent's thought/reasoning process
        - CondensationRequestAction() - request condensation of conversation history
        - BrowseInteractiveAction(browser_actions) - interact with browser using specified actions
        - MCPAction(name, arguments) - interact with MCP server tools
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            'messages': messages,
        }
        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {
            'metadata': state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = self.response_to_actions(response)
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()
```

#### 5.6 Key Components for Replication

To replicate the agent system, you need:

1. **Agent Base Class**: Abstract base with `step()` method
2. **AgentController**: Manages agent execution, state, and delegation
3. **EventStream**: Central event bus for all actions and observations
4. **State Management**: Tracks agent state, history, and metrics
5. **Runtime**: Executes actions in a sandboxed environment
6. **Tool System**: Converts LLM function calls to Actions
7. **Memory System**: Manages conversation history and condensation

## Code References

### Core Agent Files
- `openhands/controller/agent.py:25-184` - Agent base class definition
- `openhands/controller/agent_controller.py:99-1213` - AgentController implementation
- `openhands/agenthub/codeact_agent/codeact_agent.py:49-301` - CodeActAgent example implementation

### File System Handling
- `openhands/runtime/utils/files.py:12-151` - File path resolution and read/write operations
- `openhands/runtime/base.py:90-906` - Runtime base class with file operation handling
- `openhands/events/action/commands.py` - FileReadAction, FileEditAction definitions

### Tool Call Structure
- `openhands/agenthub/codeact_agent/function_calling.py:73-339` - Tool call to Action conversion
- `openhands/agenthub/codeact_agent/codeact_agent.py:108-153` - Tool definition and registration

### Delegation and Looping
- `openhands/controller/agent_controller.py:701-827` - Delegate creation and management
- `openhands/controller/agent_controller.py:441-500` - Event forwarding to delegates
- `openhands/events/action/agent.py:76-86` - AgentDelegateAction definition

### State and Context Management
- `openhands/controller/state/state.py:48-312` - State class with event range tracking
- `openhands/controller/state/state_tracker.py:92-213` - History loading based on start_id/end_id
- `openhands/memory/condenser/condenser.py` - Conversation history condensation

### Event Stream
- `openhands/events/stream.py:43-292` - EventStream implementation with subscribers

## Architecture Documentation

### Event-Driven Architecture

The system uses an event-driven architecture where:
- **Actions** are events that agents want to perform
- **Observations** are events that represent the results of actions
- All events flow through a central `EventStream`
- Subscribers (Runtime, AgentController, Memory) react to events

### Multi-Agent Delegation Pattern

The delegation pattern follows:
1. Parent agent creates `AgentDelegateAction`
2. Controller creates new `AgentController` with `is_delegate=True`
3. Delegate gets separate `State` with `start_id` set to current event ID + 1
4. Parent forwards events to delegate while delegate is active
5. Delegate processes events and adds its own events to the stream
6. When delegate finishes, parent ends it and resumes control
7. Delegate's results are summarized in `AgentDelegateObservation`

### State Isolation Strategy

Each agent maintains:
- **Separate State object**: With its own `start_id`, `end_id`, `delegate_level`
- **Shared EventStream**: All events go to the same stream
- **Shared LLMRegistry**: Metrics are tracked globally
- **Filtered History**: Each agent's history is filtered by its event range

### Context Window Management

- **No separate context windows**: All agents share the same LLM and EventStream
- **Event range filtering**: Each agent's view is limited by `start_id`/`end_id`
- **Condensation**: History is condensed to fit within context limits
- **View caching**: The State object caches filtered views for performance

## Historical Context (from thoughts/)

No relevant historical context found in thoughts/ directory for this topic.

## Related Research

No related research documents found in thoughts/shared/research/.

## Open Questions

1. How does the system handle nested delegation (delegate spawning another delegate)?
2. What happens if a delegate tries to access events before its `start_id`?
3. How are tool call metadata preserved across delegation boundaries?
4. What is the maximum delegation depth supported?
5. How does the condenser handle events from multiple agents in the same stream?

