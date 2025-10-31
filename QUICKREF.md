# AW Quick Reference

## Installation

```bash
cd /Users/thorwhalen/Dropbox/py/proj/t/aw
pip install -e .
```

## Quickest Start

```python
from aw import load_for_cosmo

# Load and prepare data for cosmograph in one line
df, meta = load_for_cosmo('data.csv')

# Get suggested visualization parameters
params = meta['preparing']['metadata']['validation_result']['params']
# params = {'points_x_by': 'col1', 'points_y_by': 'col2', ...}

# Use with cosmograph
from cosmograph import cosmo
cosmo(df, **params)
```

## Core Components

### Loading Data

```python
from aw import LoadingAgent, Context

agent = LoadingAgent()
context = Context()
df, metadata = agent.execute('data.csv', context)
```

### Preparing Data

```python
from aw import PreparationAgent
from aw.cosmo import basic_cosmo_validator

agent = PreparationAgent(
    target='cosmo-ready',
    target_validator=basic_cosmo_validator()
)
prepared_df, metadata = agent.execute(df, context)
```

### Full Workflow

```python
from aw import create_cosmo_prep_workflow

workflow = create_cosmo_prep_workflow(max_retries=3)
result_df, metadata = workflow.run('data.csv')
```

## Validation

### Schema Validation

```python
from aw.validation import schema_validator
from pydantic import BaseModel

class DataSchema(BaseModel):
    x: float
    y: float

validator = schema_validator(DataSchema)
success, info = validator(data)
```

### Info-Dict Validation

```python
from aw.validation import info_dict_validator

def compute_info(df):
    return {'null_count': df.isnull().sum().sum()}

def check_info(info):
    return info['null_count'] == 0, "No nulls"

validator = info_dict_validator(compute_info, check_info)
```

### Functional Validation

```python
from aw.validation import functional_validator

def try_visualization(df):
    return cosmo(df, points_x_by='x', points_y_by='y')

validator = functional_validator(try_visualization)
```

## Configuration

### Simple Config

```python
from aw import StepConfig, LoadingAgent

config = StepConfig(
    llm="gpt-4",
    max_retries=5,
    human_in_loop=False
)

agent = LoadingAgent(config)
```

### Callable Config (Dependency Injection)

```python
def my_llm(prompt):
    return "Response to: " + prompt

def my_validator(artifact):
    return True, {'info': 'All good'}

config = StepConfig(
    llm=my_llm,
    validator=my_validator,
    max_retries=3
)
```

### Cascading Config

```python
from aw import GlobalConfig

global_cfg = GlobalConfig(llm="gpt-4", max_retries=3)
step_cfg = global_cfg.override(max_retries=5)
```

## Code Execution

```python
from aw import CodeInterpreterTool

tool = CodeInterpreterTool()
result = tool("x = 5; y = x * 2; print(y)")

print(result.success)  # True
print(result.output)   # "10\n"
```

## Custom Agents

```python
from aw.base import StepConfig
from collections.abc import MutableMapping

class MyAgent:
    def __init__(self, config: StepConfig = None):
        self.config = config or StepConfig()
        self.validator = self.config.resolve_validator()
    
    def execute(self, input_data, context: MutableMapping):
        for attempt in range(self.config.max_retries):
            result = self._process(input_data)
            is_valid, info = self.validator(result)
            if is_valid:
                return result, {'success': True}
        return None, {'success': False}
```

## Interactive Workflows

```python
from aw import InteractiveWorkflow, LoadingAgent

workflow = InteractiveWorkflow()
workflow.add_step('loading', LoadingAgent(), require_approval=True)

def approve(step_name, artifact, metadata):
    print(f"Step {step_name} done. Shape: {artifact.shape}")
    return input("Continue? (y/n): ").lower() == 'y'

workflow.set_approval_callback(approve)
result, meta = workflow.run_interactive('data.csv')
```

## Module Reference

- `aw.base` - Protocols, configs, context
- `aw.validation` - Three validation flavors
- `aw.tools` - Code execution
- `aw.utils` - Helpers, file sampling, LLM facades
- `aw.loading` - LoadingAgent
- `aw.preparing` - PreparationAgent  
- `aw.cosmo` - Cosmograph validators
- `aw.orchestration` - Workflows

## Key Design Patterns

1. **AgenticStep Protocol**: All agents implement `execute(input, context) -> (artifact, metadata)`
2. **ReAct Pattern**: Reason → Act → Observe → Validate → Repeat
3. **Dependency Injection**: Pass functions instead of strings for maximum flexibility
4. **Context as Mapping**: Use MutableMapping for shared state
5. **Generators**: Use lazy evaluation for efficiency
6. **Functional Validation**: Validate by trying the actual purpose

## Common Patterns

### Load and Prepare Generic Data

```python
from aw import load_and_prepare

df, meta = load_and_prepare('data.csv', target='generic')
```

### Load with Custom Validator

```python
from aw import load_and_prepare

def my_validator(df):
    return len(df) > 100, {'rows': len(df)}

df, meta = load_and_prepare(
    'data.csv',
    validator=my_validator,
    max_retries=5
)
```

### Partial Workflow

```python
workflow = create_cosmo_prep_workflow()

# Run only loading
df, meta = workflow.run_partial('data.csv', stop_after='loading')

# Inspect, then continue
prepared_df, meta = workflow.run_partial(df, stop_after='preparing')
```

## Troubleshooting

### Import Errors

```python
# Optional dependencies in try/except blocks
try:
    from aw import create_langchain_executor
except ImportError:
    print("Install: pip install langchain langchain-experimental")
```

### Check Metadata

```python
df, meta = load_for_cosmo('data.csv')

# Check success
if meta.get('success'):
    print("Success!")
else:
    print(f"Failed at: {meta.get('failed_at')}")
    
# Check attempts
for step in meta['steps']:
    print(f"{step['name']}: {step['success']}")
```

### Debug Mode

```python
# Set max_retries=1 to see errors immediately
workflow = create_cosmo_prep_workflow(max_retries=1)
```

## Examples

Run the examples:

```bash
python examples/basic_usage.py
```

This demonstrates:
1. Basic loading
2. Preparation for cosmograph
3. Complete workflows
4. Different validators
5. Custom configurations
6. Code execution
