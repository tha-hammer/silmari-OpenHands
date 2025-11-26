---
date: 2025-11-16T19:19:03-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "How to add .env file loading to OpenHands configuration system"
tags: [research, codebase, configuration, environment-variables, dotenv]
status: complete
last_updated: 2025-11-16
last_updated_by: Auto
---

# Research: How to add .env file loading to OpenHands configuration system

**Date**: 2025-11-16T19:19:03-05:00
**Researcher**: Auto
**Git Commit**: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
**Branch**: main
**Repository**: silmari-OpenHands

## Research Question

Research the code to determine how to add `.env` file loading. It seems the only method is an `export` to the actual environment.

## Summary

The OpenHands configuration system **already supports `.env` file loading** through the `python-dotenv` library. The `.env` files are loaded into `os.environ` at module import time in three locations, and then the configuration system reads from `os.environ` using the `load_from_env()` function. The current implementation uses the default `load_dotenv()` behavior, which loads `.env` files from the current working directory and merges them into the environment without overriding existing variables.

## Detailed Findings

### Current .env File Loading Implementation

The codebase already uses `python-dotenv` to load `.env` files. The `load_dotenv()` function is called at module import time in three locations:

1. **`openhands/core/config/utils.py:37`** - Called at module level
   ```12:37:openhands/core/config/utils.py
   from dotenv import load_dotenv
   ...
   load_dotenv()
   ```

2. **`openhands/server/shared.py:21`** - Called at module level
   ```4:21:openhands/server/shared.py
   from dotenv import load_dotenv
   ...
   load_dotenv()
   ```

3. **`openhands/agenthub/__init__.py:3`** - Called at module level
   ```1:3:openhands/agenthub/__init__.py
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Configuration Loading Flow

The configuration loading follows this sequence:

1. **Module Import Time**: When any of the above modules are imported, `load_dotenv()` is called, which:
   - Looks for a `.env` file in the current working directory (default behavior)
   - Loads key-value pairs from the `.env` file into `os.environ`
   - Does not override existing environment variables (default `override=False`)

2. **Configuration Initialization**: When `load_openhands_config()` is called:
   ```819:836:openhands/core/config/utils.py
   def load_openhands_config(
       set_logging_levels: bool = True, config_file: str = 'config.toml'
   ) -> OpenHandsConfig:
       """Load the configuration from the specified config file and environment variables.

       Args:
           set_logging_levels: Whether to set the global variables for logging levels.
           config_file: Path to the config file. Defaults to 'config.toml' in the current directory.
       """
       config = OpenHandsConfig()
       load_from_toml(config, config_file)
       load_from_env(config, os.environ)
       finalize_config(config)
       register_custom_agents(config)
       if set_logging_levels:
           logger.DEBUG = config.debug
           logger.DISABLE_COLOR_PRINTING = config.disable_color
       return config
   ```

3. **Environment Variable Reading**: The `load_from_env()` function reads from `os.environ`:
   ```40:136:openhands/core/config/utils.py
   def load_from_env(
       cfg: OpenHandsConfig, env_or_toml_dict: dict | MutableMapping[str, str]
   ) -> None:
       """Sets config attributes from environment variables or TOML dictionary.

       Reads environment-style variables and updates the config attributes accordingly.
       Supports configuration of LLM settings (e.g., LLM_BASE_URL), agent settings
       (e.g., AGENT_MEMORY_ENABLED), sandbox settings (e.g., SANDBOX_TIMEOUT), and more.

       Args:
           cfg: The OpenHandsConfig object to set attributes on.
           env_or_toml_dict: The environment variables or a config.toml dict.
       """
   ```

### How Environment Variables Map to Configuration

The `load_from_env()` function uses a naming convention to map environment variables to configuration fields:

- **Prefix**: Uppercase name of the configuration class followed by an underscore (e.g., `LLM_`, `AGENT_`)
- **Field Names**: All uppercase
- **Full Variable Name**: Prefix + Field Name (e.g., `LLM_API_KEY`, `AGENT_MEMORY_ENABLED`)

The function recursively processes nested configuration models, building environment variable names from the field hierarchy.

### Dependencies

The `python-dotenv` package is already included as a dependency:

```58:58:pyproject.toml
python-dotenv = "*"
```

### Current Limitations

The current implementation has the following characteristics:

1. **Default Behavior**: Uses `load_dotenv()` with default parameters, which:
   - Looks for `.env` in the current working directory only
   - Does not override existing environment variables
   - Does not provide explicit control over which `.env` file to load

2. **Multiple Call Sites**: `load_dotenv()` is called in three different modules, which could lead to:
   - Redundant calls (though `load_dotenv()` is idempotent)
   - Inconsistent behavior if different modules are imported in different orders
   - Difficulty in customizing `.env` file location

3. **No Explicit Path Control**: There's no way to specify a custom path to a `.env` file through configuration

### Entry Points

The configuration is loaded at these entry points:

1. **CLI**: `openhands/cli/main.py:590` - Uses `setup_config_from_args()`
2. **Main Module**: `openhands/core/main.py:289` - Uses `setup_config_from_args()`
3. **Server**: `openhands/server/shared.py:23` - Uses `load_openhands_config()` directly
4. **Storage**: `openhands/storage/data_models/settings.py:130` - Uses `load_openhands_config()` directly

## Code References

- `openhands/core/config/utils.py:12` - Import of `load_dotenv`
- `openhands/core/config/utils.py:37` - Module-level call to `load_dotenv()`
- `openhands/core/config/utils.py:40-136` - `load_from_env()` function implementation
- `openhands/core/config/utils.py:819-836` - `load_openhands_config()` function
- `openhands/core/config/utils.py:839-904` - `setup_config_from_args()` function
- `openhands/server/shared.py:4,21` - Server-side `load_dotenv()` call
- `openhands/agenthub/__init__.py:1,3` - Agent hub `load_dotenv()` call
- `pyproject.toml:58` - `python-dotenv` dependency declaration

## Architecture Documentation

### Configuration Loading Precedence

Based on the code flow, the configuration loading precedence is:

1. **Default values** (defined in Pydantic model fields)
2. **TOML file** (`config.toml`) - loaded via `load_from_toml()`
3. **Environment variables** (including from `.env` files) - loaded via `load_from_env()`
4. **Command-line arguments** - applied via `setup_config_from_args()`

Since `.env` files are loaded into `os.environ` at module import time, and `load_from_env()` reads from `os.environ`, the `.env` file values are effectively part of step 3. However, if an environment variable is already set in the actual environment, it will take precedence over the `.env` file value (due to `override=False` by default).

### Current Pattern

The current pattern follows this flow:

```
Module Import → load_dotenv() → os.environ populated
     ↓
load_openhands_config() called
     ↓
load_from_toml() → config.toml values applied
     ↓
load_from_env(config, os.environ) → environment variables (including .env) applied
     ↓
finalize_config() → post-processing and validation
```

## How to Enhance .env File Loading

Based on the current implementation, here are the areas where `.env` file loading could be enhanced:

### Option 1: Centralize `load_dotenv()` Call

Currently, `load_dotenv()` is called in three different modules. This could be centralized to:
- A single location in `openhands/core/config/utils.py` before `load_openhands_config()` is called
- Or within `load_openhands_config()` itself to ensure it happens at the right time

### Option 2: Add Explicit .env File Path Configuration

The `load_dotenv()` function accepts a `dotenv_path` parameter. This could be:
- Added as a configuration option in `OpenHandsConfig`
- Passed through command-line arguments
- Determined by a standard location (e.g., `~/.openhands/.env` or project root)

### Option 3: Add Override Control

The `load_dotenv()` function accepts an `override` parameter. This could be:
- Made configurable to allow `.env` files to override existing environment variables
- Documented in the configuration documentation

### Option 4: Add Explicit .env File Loading in Config Functions

Instead of relying on module import time, `.env` file loading could be:
- Explicitly called at the start of `load_openhands_config()`
- Made conditional based on configuration
- Logged for debugging purposes

## Related Files

- `openhands/core/config/README.md` - Configuration documentation (does not mention `.env` files)
- `openhands/core/config/openhands_config.py` - Main configuration model
- `openhands/core/config/llm_config.py` - LLM configuration model
- `openhands/core/config/agent_config.py` - Agent configuration model

## Open Questions

1. Should `.env` file loading be centralized to a single location?
2. Should there be explicit configuration for `.env` file path?
3. Should `.env` files override existing environment variables?
4. Should `.env` file loading be logged or made more visible?
5. Should there be support for multiple `.env` files (e.g., `.env.local`, `.env.production`)?

