# aw - Agentic Workflows for Data Preparation

An AI agent package for data preparation with a focus on modular, extensible workflows that follow best practices from modern agentic architectures.

## Features

- **AgenticStep Protocol**: Build modular agents that follow a consistent interface
- **ReAct Pattern**: Reason-Act-Observe loops with automatic retry logic
- **Three Validation Flavors**: Schema-based, info-dict based, and functional validation
- **Code Execution**: Safe defaults with extensibility for robust backends (LangChain, DSPy)
- **Loading Agent**: Automatically loads data from various sources into pandas DataFrames
- **Preparation Agent**: Transforms data to meet target requirements
- **Cosmograph Support**: Specialized validators for cosmograph visualization
- **Orchestration**: Chain multiple steps together with context management
- **Human-in-Loop**: Optional human approval at any step

## Installation

```bash
pip install -e .
```

### Optional Dependencies

```bash
# For LangChain code execution
pip install langchain langchain-experimental

# For cosmograph validation
pip install cosmograph

# For OpenAI LLM
pip install openai

# For oa package (preferred LLM interface)
pip install oa
```

## Quick Start

### Simple Example: Load and Prepare for Cosmograph

```python
from aw import load_for_cosmo

# Load and prepare data in one call
df, metadata = load_for_cosmo('data.csv')

# Use with cosmograph
from cosmograph import cosmo
params = metadata['preparing']['metadata']['validation_result']['params']
cosmo(df, **params)
```

### Step-by-Step Example

```python
from aw import LoadingAgent, PreparationAgent, Context
from aw.cosmo import basic_cosmo_validator

# Create context
context = Context()

# Step 1: Load data
loading_agent = LoadingAgent()
df, loading_meta = loading_agent.execute('data.csv', context)

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Prepare data
preparing_agent = PreparationAgent(
    target='cosmo-ready',
    target_validator=basic_cosmo_validator()
)
prepared_df, prep_meta = preparing_agent.execute(df, context)

print(f"Prepared data: {prep_meta['validation_result']}")
```

### Workflow Example

```python
from aw import create_cosmo_prep_workflow

# Create a complete workflow
workflow = create_cosmo_prep_workflow(max_retries=5)

# Run it
result_df, metadata = workflow.run('data.csv')

# Check results
if metadata['success']:
    print("Success!")
    print(f"Steps executed: {[s['name'] for s in metadata['steps']]}")
else:
    print(f"Failed at: {metadata.get('failed_at')}")
```

## Architecture

### Core Concepts

#### AgenticStep Protocol

All agents implement the `AgenticStep` protocol:

```python
def execute(
    self, 
    input_data: Any, 
    context: MutableMapping[str, Any]
) -> tuple[ArtifactType, dict[str, Any]]:
    """Execute the agentic step following ReAct pattern"""
    ...
```

#### ReAct Pattern

Each agent follows the Reason-Act-Observe loop:

1. **Reason (Thought)**: Analyze input and context to decide action
2. **Act (Action)**: Generate and execute code or use tools
3. **Observe**: Capture results and validate
4. **Repeat or Finish**: Based on validation, retry or proceed

#### Three Validation Flavors

1. **Schema-based**: Use Pydantic models or JSON schemas
2. **Info-dict based**: Compute info → check info → pass/fail
3. **Functional**: Try to use the artifact for its purpose

```python
from aw.validation import (
    schema_validator,
    info_dict_validator, 
    functional_validator
)

# Schema validation
from pydantic import BaseModel
class DataSchema(BaseModel):
    x: float
    y: float
validate = schema_validator(DataSchema)

# Info-dict validation
def compute_info(df):
    return {'null_count': df.isnull().sum().sum()}

def check_info(info):
    return info['null_count'] == 0, "No nulls allowed"

validate = info_dict_validator(compute_info, check_info)

# Functional validation (try the purpose)
def try_visualize(df):
    return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')

validate = functional_validator(try_visualize)
```

### Configuration System

Supports both simple objects and callables with cascading defaults:

```python
from aw import GlobalConfig, StepConfig

# Global defaults
global_config = GlobalConfig(
    llm="gpt-4",
    max_retries=3
)

# Step-specific override
step_config = global_config.override(
    llm="gpt-3.5-turbo",
    max_retries=5
)

# Or pass callables
def my_llm(prompt):
    return "Response to: " + prompt

step_config = StepConfig(
    llm=my_llm,  # Function instead of string
    validator=lambda x: (True, {}),
    max_retries=3
)
```

### Tools

#### CodeInterpreterTool

Safe code execution with extensibility:

```python
from aw import CodeInterpreterTool

# Default (safe exec)
tool = CodeInterpreterTool()
result = tool("x = 5; y = x * 2; print(y)")
print(result.output)  # "10"

# With LangChain backend
from aw import create_langchain_executor
tool = CodeInterpreterTool(executor=create_langchain_executor())
```

#### FileSamplerTool

Sample and analyze files before loading:

```python
from aw import FileSamplerTool

sampler = FileSamplerTool()
info = sampler('data.csv')
print(info['extension'])  # '.csv'
print(info['sample_text'])  # First 1024 bytes
```

### Custom Agents

Build your own agents following the pattern:

```python
from aw.base import AgenticStep, StepConfig
from collections.abc import MutableMapping
from typing import Any

class MyCustomAgent:
    """Custom agent following AgenticStep protocol"""
    
    def __init__(self, config: StepConfig = None):
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()
        self.validator = self.config.resolve_validator()
    
    def execute(
        self,
        input_data: Any,
        context: MutableMapping[str, Any]
    ) -> tuple[Any, dict]:
        attempts = []
        
        for attempt in range(self.config.max_retries):
            # 1. Reason: Analyze situation
            thought = self._analyze(input_data, attempts)
            
            # 2. Act: Perform action
            result = self._act(thought)
            
            # 3. Observe: Validate
            is_valid, info = self.validator(result)
            attempts.append({'attempt': attempt, 'info': info})
            
            if is_valid:
                context['my_agent'] = {'result': result, 'info': info}
                return result, {'success': True, 'attempts': attempts}
        
        return None, {'success': False, 'attempts': attempts}
```

## Advanced Usage

### Interactive Workflows with Human-in-Loop

```python
from aw import InteractiveWorkflow, LoadingAgent, PreparationAgent

workflow = InteractiveWorkflow()
workflow.add_step('loading', LoadingAgent(), require_approval=True)
workflow.add_step('preparing', PreparationAgent(), require_approval=False)

# Set custom approval callback
def approve(step_name, artifact, metadata):
    print(f"\nStep '{step_name}' completed")
    print(f"Artifact shape: {artifact.shape}")
    return input("Continue? (y/n): ").lower() == 'y'

workflow.set_approval_callback(approve)

# Run interactively
result, metadata = workflow.run_interactive('data.csv')
```

### Custom Validators

Create domain-specific validators:

```python
from aw.validation import info_dict_validator

def compute_ml_info(df):
    return {
        'num_features': len(df.columns) - 1,
        'num_samples': len(df),
        'class_balance': df['target'].value_counts().to_dict()
    }

def check_ml_requirements(info):
    if info['num_features'] < 5:
        return False, "Need at least 5 features"
    if info['num_samples'] < 100:
        return False, "Need at least 100 samples"
    return True, "ML ready"

ml_validator = info_dict_validator(compute_ml_info, check_ml_requirements)
```

### Partial Workflow Execution

```python
from aw import create_data_prep_workflow

workflow = create_data_prep_workflow()

# Run only the loading step
df, metadata = workflow.run_partial('data.csv', stop_after='loading')

# Inspect intermediate result
print(f"Loaded {df.shape}")

# Continue with preparing
final_df, final_meta = workflow.run_partial(df, stop_after='preparing')
```

## Design Principles

The package follows these principles:

1. **Functional over OOP**: Favor functions and generators
2. **Modularity**: Small, focused functions with clear purposes
3. **Open-Closed**: Easy to extend without modifying core
4. **Dependency Injection**: Abstract interfaces with injectable implementations
5. **Mapping Interfaces**: Use `Mapping`/`MutableMapping` from `collections.abc`
6. **Generators over Lists**: Use lazy evaluation when possible
7. **Minimal Doctests**: Simple examples in docstrings

## Project Structure

```
aw/
├── __init__.py         # Main exports
├── base.py            # Core protocols and configs
├── validation.py      # Three validation flavors
├── tools.py           # Code execution and tools
├── utils.py           # Helpers and facades
├── loading.py         # LoadingAgent
├── preparing.py       # PreparationAgent
├── cosmo.py          # Cosmograph validators
└── orchestration.py   # Workflow management
```

## Contributing

Contributions welcome! Key areas:

- Additional validators for other visualization libraries
- More robust LLM reasoning in code generation
- Integration with more backends (DSPy, CrewAI)
- State management with lakeFS/DVC
- Additional agent types (sourcing, finding, etc.)

## License

See LICENSE file.

## Credits

Built following best practices from:
- ReAct pattern (Reasoning and Acting)
- LangChain's agent architectures
- CrewAI's multi-agent systems
- Data Version Control (DVC) principles
