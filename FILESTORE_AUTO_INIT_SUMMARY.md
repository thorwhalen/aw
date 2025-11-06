# FileStore Auto-Initialization Enhancement

## Problem

The download agent was displaying a warning when trying to update Claude Desktop configuration:

```
⚠️  Warning: Could not automatically update Claude Desktop config: 'mcpServers'
```

This occurred in two scenarios:
1. **Missing file**: The `claude_desktop_config.json` file didn't exist yet
2. **Missing key_path**: The file existed but didn't have the `mcpServers` section

## Solution

Enhanced `FileStore` in `config2py.sync_store` with two new optional parameters:

### New Parameters

1. **`create_file_content`**: Factory callable that returns initial dict for missing files
   - Type: `Optional[Callable[[], dict]]`
   - Default: `None` (raises `FileNotFoundError` for missing files)
   - When provided: Creates the file with returned content

2. **`create_key_path_content`**: Factory callable that returns initial content for missing key_path
   - Type: `Optional[Callable[[], Any]]`
   - Default: `None` (raises `KeyError` for missing key paths)
   - When provided: Creates the key_path section with returned content

### Implementation Details

Modified `FileStore._load_from_file()` to:

1. Check if file exists
   - If missing and `create_file_content` provided → create file with factory content
   - If missing and no factory → raise `FileNotFoundError`

2. Try to access key_path
   - If missing and `create_key_path_content` provided → create section with factory content
   - If missing and no factory → raise `KeyError`

3. Automatically create parent directories when creating files

4. Preserve existing file content when adding new key_path sections

## Usage

### Basic Usage (claude_desktop_config)

```python
from aw.util import claude_desktop_config

# No longer fails even if file or mcpServers section missing
mcp = claude_desktop_config()

# Add server configuration
mcp['download'] = {
    'command': 'python',
    'args': ['/path/to/agent.py']
}
```

### Direct FileStore Usage

```python
from config2py.sync_store import FileStore

# Auto-create file and key_path if missing
store = FileStore(
    'path/to/config.json',
    key_path='servers',
    create_file_content=lambda: {},  # Empty dict for new files
    create_key_path_content=lambda: {}  # Empty dict for new sections
)

store['myserver'] = {'command': 'python', 'args': ['script.py']}
```

## Benefits

1. **No warnings**: Agent setup is now seamless for first-time users
2. **Preserves data**: Existing file content is maintained when adding new sections
3. **Flexible**: Factory pattern allows custom initialization for different use cases
4. **Backward compatible**: Default behavior unchanged (raises errors for missing content)
5. **Directory creation**: Automatically creates parent directories as needed

## Files Modified

### config2py/sync_store.py
- Added `create_file_content` and `create_key_path_content` parameters to `FileStore.__init__`
- Enhanced `_load_from_file()` to handle missing files and key_paths
- Updated docstring with new parameters and examples
- Added doctest demonstrating auto-initialization

### aw/util.py
- Updated `claude_desktop_config()` to use new FileStore parameters
- Passes factory lambdas: `lambda: {}` for both file and key_path initialization
- Updated docstring noting automatic creation behavior

## Testing

All tests pass:

✅ Missing file scenario - file created with proper structure
✅ Missing key_path scenario - section added while preserving existing content  
✅ Existing file/key_path - no changes to current behavior
✅ All doctests passing in sync_store.py
✅ Agent can update config without warnings

## Example Scenarios

### Scenario 1: Fresh Install (No Config File)

```python
# Before: FileNotFoundError
# After: File created automatically
mcp = claude_desktop_config()  # Creates file with {"mcpServers": {}}
```

### Scenario 2: File Exists, Section Missing

```python
# Initial file: {"otherSection": {...}}
mcp = claude_desktop_config()
mcp['server'] = {...}
# Result: {"otherSection": {...}, "mcpServers": {"server": {...}}}
```

### Scenario 3: Everything Exists

```python
# No change in behavior - works as before
mcp = claude_desktop_config()
mcp['newserver'] = {...}
```

## Impact

- **Download agent**: Now works seamlessly on first run
- **User experience**: No confusing warnings or manual config editing needed
- **Reliability**: Handles edge cases gracefully
- **Maintainability**: Factory pattern allows easy customization

## Next Steps

Consider applying this pattern to other config files that may benefit from auto-initialization:
- Other agent configurations
- User settings files
- Plugin/extension registries
