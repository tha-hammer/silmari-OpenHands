---
date: 2025-11-16T17:26:56-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "BAML LLM Integration Implementation"
tags: [implementation, llm, baml, integration]
status: in_progress
last_updated: 2025-11-16
last_updated_by: Auto
type: implementation
---

# Handoff: BAML LLM Integration Implementation

## Task(s)

Implementing BAML (Boundary AI Markup Language) integration into OpenHands' LLM system as an optional layer. Working from the implementation plan at `thoughts/shared/plans/2025-11-16-baml-llm-integration.md`.

**Completed:**
- ✅ **Phase 1**: BAML Configuration and Client Setup
  - Updated BAML generator to Python/pydantic (`openhands/llm/baml_src/generators.baml`)
  - Added `baml-py` dependency to `pyproject.toml`
  - Added `use_baml` flag to `LLMConfig` (`openhands/core/config/llm_config.py:97-100`)
  - Updated config template (`config.template.toml:214-215`)

- ✅ **Phase 2**: BAML Function Definitions
  - Created BAML types (`openhands/llm/baml_src/types.baml`)
  - Added LiteLLMClient configuration (`openhands/llm/baml_src/clients.baml:5-13`)
  - Created completion function (`openhands/llm/baml_src/completion.baml`)

- ✅ **Phase 3**: BAML Adapter Implementation
  - Created BAML adapter module (`openhands/llm/baml_adapter.py`)
  - Integrated BAML routing into LLM class (`openhands/llm/llm.py:88-98, 339-369`)
  - Fixed BAML syntax errors (removed invalid `import` statements, fixed type issues)

- ✅ **Completion Function Implementation**
  - Implemented text-based message formatting approach
  - Added `_format_messages_to_text()` function for converting structured messages to text
  - Updated completion function to use `FormattedLLMRequest` type

**In Progress:**
- ⏳ **Phase 4**: Function Calling and Advanced Features (not started)

**Planned:**
- ⏳ **Phase 5**: Testing and Documentation

## Critical References

1. **Implementation Plan**: `thoughts/shared/plans/2025-11-16-baml-llm-integration.md` - Complete implementation plan with phases and success criteria
2. **Research Document**: `thoughts/shared/research/2025-11-16-baml-llm-integration.md` - Original research on BAML integration approach
3. **BAML Documentation**: https://docs.boundaryml.com - For understanding BAML syntax and capabilities

## Recent changes

**Configuration:**
- `openhands/llm/baml_src/generators.baml:6-9` - Changed output_type to "python/pydantic", output_dir to "../baml_client", default_client_mode to "sync"
- `pyproject.toml:30` - Added `baml-py = "^0.213.0"` dependency
- `openhands/core/config/llm_config.py:97-100` - Added `use_baml: bool` field with default False
- `config.template.toml:214-215` - Added commented `use_baml` option

**BAML Files:**
- `openhands/llm/baml_src/types.baml` - Created types for messages, tools, and completion request/response
- `openhands/llm/baml_src/clients.baml:5-13` - Added LiteLLMClient configuration
- `openhands/llm/baml_src/completion.baml` - Created completion function with FormattedLLMRequest approach

**Adapter Implementation:**
- `openhands/llm/baml_adapter.py` - New file with conversion functions and BAML completion caller
- `openhands/llm/baml_adapter.py:238-314` - `_format_messages_to_text()` function for text formatting
- `openhands/llm/baml_adapter.py:317-356` - `call_baml_completion()` updated to use text formatting

**LLM Integration:**
- `openhands/llm/llm.py:43-48` - Added optional BAML adapter import
- `openhands/llm/llm.py:88-98` - Added BAML flag storage and environment variable setup
- `openhands/llm/llm.py:339-369` - Added BAML routing logic in completion wrapper
- `openhands/llm/llm.py:450-468` - Added `_call_litellm_completion()` helper method

## Learnings

1. **BAML Syntax**: BAML doesn't use `import` statements - all files in `baml_src/` are automatically available. Types and clients can be referenced directly by name.

2. **BAML Type System**:
   - BAML doesn't support `any` type - must use specific types like `string`, `map<string, string>`, etc.
   - `function` is a reserved keyword in BAML - renamed to `function_def` in Tool class
   - Union types (like `Content = TextContent | ImageContent`) can't be directly accessed in templates - need type guards or formatting

3. **BAML Function Design**: BAML functions are designed for structured extraction from text, not raw LLM completion with message lists. Current implementation uses a text formatting approach:
   - Messages are converted to text format in the adapter
   - BAML extracts structured response from LLM output
   - Response is converted back to ModelResponse format

4. **Current Limitations**:
   - Function calling is simplified (tools listed as text descriptions, not native function calling)
   - Vision support is basic (images shown as `[Image: url]` placeholders)
   - Some structured information may be lost in text conversion
   - These limitations should be addressed in Phase 4

5. **Error Handling**: BAML integration includes automatic fallback to LiteLLM if BAML fails (`openhands/llm/llm.py:361-365`)

## Artifacts

**Implementation Files:**
- `openhands/llm/baml_src/generators.baml` - BAML generator configuration
- `openhands/llm/baml_src/types.baml` - BAML type definitions
- `openhands/llm/baml_src/clients.baml` - BAML client configurations
- `openhands/llm/baml_src/completion.baml` - BAML completion function
- `openhands/llm/baml_adapter.py` - Python adapter for BAML integration
- `openhands/llm/llm.py` - Updated LLM class with BAML routing

**Configuration Files:**
- `pyproject.toml` - Added baml-py dependency
- `openhands/core/config/llm_config.py` - Added use_baml configuration flag
- `config.template.toml` - Added use_baml option

**Documentation:**
- `thoughts/shared/plans/2025-11-16-baml-llm-integration.md` - Implementation plan (checkboxes updated for Phases 1-3)
- `thoughts/shared/research/2025-11-16-baml-llm-integration.md` - Original research document

**Checkpoints:**
- `phase_1_start` - Starting Phase 1
- `phase_3_complete` - Completed Phase 3
- `completion_function_implemented` - Completion function implementation

## Action Items & Next Steps

1. **Generate BAML Client** (Required before testing):
   ```bash
   cd openhands/llm/baml_src && baml generate
   ```
   This will generate the Python client code in `openhands/llm/baml_client/`

2. **Phase 4: Function Calling and Advanced Features** (`thoughts/shared/plans/2025-11-16-baml-llm-integration.md:693-785`):
   - Enhance BAML function for proper function calling support
   - Add async BAML support (`openhands/llm/async_llm.py`)
   - Add streaming BAML support (`openhands/llm/streaming_llm.py`) - if BAML supports it
   - Ensure vision and prompt caching are properly handled

3. **Phase 5: Testing and Documentation** (`thoughts/shared/plans/2025-11-16-baml-llm-integration.md:788-913`):
   - Create unit tests (`tests/unit/llm/test_baml_adapter.py`)
   - Create integration tests (`tests/integration/test_baml_agent.py`)
   - Update documentation

4. **Refine Completion Function** (if needed):
   - Current text-based approach works but has limitations
   - Consider if BAML's native message passing can be used instead
   - May need to investigate BAML's provider SDK capabilities

5. **Verify Success Criteria** (from plan):
   - Run `baml check` to validate BAML files
   - Run `baml generate` to create Python client
   - Test config validation: `python -c "from openhands.core.config import LLMConfig; LLMConfig(use_baml=True)"`
   - Test BAML client import: `from openhands.llm.baml_client.sync_client import b`

## Other Notes

**Key Files to Review:**
- `openhands/llm/llm.py:54-887` - Main LLM class implementation
- `openhands/core/message.py` - Message class structure (important for conversion)
- `openhands/llm/fn_call_converter.py` - Function calling conversion logic (may need BAML integration)

**BAML Client Generation:**
- The BAML client hasn't been generated yet - this is a critical next step
- Client will be generated in `openhands/llm/baml_client/` directory
- After generation, the adapter imports should work: `from openhands.llm.baml_client.sync_client import b`

**Testing Strategy:**
- Start with simple unit tests for message conversion
- Test BAML completion with a simple message
- Verify ModelResponse format matches expected structure
- Test fallback behavior when BAML fails

**Known Issues:**
- The text formatting approach may not preserve all message structure
- Function calling needs proper implementation (currently just text descriptions)
- Vision support is limited (image URLs as placeholders)
- These are documented limitations to address in Phase 4

**BAML Version:**
- Using BAML version `0.213.0` (specified in `generators.baml:13`)
- Ensure `baml-py` package version matches

