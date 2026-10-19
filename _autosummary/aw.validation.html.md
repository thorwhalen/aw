# aw.validation

Validation system supporting three flavors: schema, info-dict, and functional.

This module provides validators that can be used to check if artifacts
meet requirements. Validators follow the pattern:

```default
validator(artifact) -> (success: bool, info: dict)
```

### Functions

| [`all_validators`](#aw.validation.all_validators)(\*validators)                  | Combine multiple validators - all must pass.                       |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`any_validator`](#aw.validation.any_validator)(\*validators)                   | Combine multiple validators - at least one must pass.              |
| [`functional_validator`](#aw.validation.functional_validator)(try_function[, ...])     | Create a validator that tries to use the artifact for its purpose. |
| [`has_attributes`](#aw.validation.has_attributes)(\*\*required_attrs)            | Validator that checks artifact has required attributes.            |
| [`info_dict_validator`](#aw.validation.info_dict_validator)(compute_info, check_info) | Create a validator that computes info then checks it.              |
| [`is_not_empty`](#aw.validation.is_not_empty)()                                | Validator that checks artifact is not empty.                       |
| [`is_type`](#aw.validation.is_type)(expected_type)                        | Validator that checks artifact type.                               |
| [`schema_validator`](#aw.validation.schema_validator)(schema)                      | Create a validator from a schema (Pydantic, JSON Schema, etc.).    |

### aw.validation.all_validators(\*validators)

Combine multiple validators - all must pass.

* **Parameters:**
  **\*validators** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Validator functions to combine
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator that passes only if all validators pass

### Example

```pycon
>>> validate = all_validators(
...     is_dataframe,
...     has_required_columns(['x', 'y']),
...     has_no_nulls
... )
```

### aw.validation.any_validator(\*validators)

Combine multiple validators - at least one must pass.

* **Parameters:**
  **\*validators** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Validator functions to combine
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator that passes if any validator passes

### aw.validation.functional_validator(try_function, success_check=None)

Create a validator that tries to use the artifact for its purpose.

This validator actually attempts to use the artifact in the way it’s
intended to be used (e.g., try to visualize the data, try to train a model).

* **Parameters:**
  * **try_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Function that tries to use the artifact
  * **success_check** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Optional function to check if result is acceptable
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> def try_cosmo(df):
...     return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')
>>> validate = functional_validator(try_cosmo)
>>> validate(df)
```

### aw.validation.has_attributes(\*\*required_attrs)

Validator that checks artifact has required attributes.

### Example

```pycon
>>> validate = has_attributes(shape=lambda s: len(s) == 2, columns=lambda c: len(c) > 0)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.validation.info_dict_validator(compute_info, check_info)

Create a validator that computes info then checks it.

This is a two-stage validator:

1. Compute information about the artifact (e.g., df.shape, df.dtypes)
2. Check if that information meets requirements

* **Parameters:**
  * **compute_info** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Function to extract info from artifact
  * **check_info** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Function to check if info is acceptable
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> def compute_df_info(df):
...     return {'shape': df.shape, 'null_count': df.isnull().sum().sum()}
>>> def check_df_info(info):
...     if info['null_count'] > 10:
...         return False, "Too many nulls"
...     return True, "OK"
>>> validate = info_dict_validator(compute_df_info, check_df_info)
```

### aw.validation.is_not_empty()

Validator that checks artifact is not empty.

Works for sequences, mappings, DataFrames, etc.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.validation.is_type(expected_type)

Validator that checks artifact type.

### Example

```pycon
>>> import pandas as pd
>>> validate = is_type(pd.DataFrame)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.validation.schema_validator(schema)

Create a validator from a schema (Pydantic, JSON Schema, etc.).

* **Parameters:**
  **schema** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A schema object with validation capability
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> from pydantic import BaseModel
>>> class DataSchema(BaseModel):
...     x: float
...     y: float
>>> validate = schema_validator(DataSchema)
>>> validate({'x': 1.0, 'y': 2.0})
(True, {'validated': ...})
```
