# aw.cosmo

Cosmograph-specific validator tool.

Provides functional validation for cosmograph visualization requirements.

### Functions

| [`basic_cosmo_validator`](#aw.cosmo.basic_cosmo_validator)()                       | Basic validator that checks structural requirements only.   |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`create_cosmo_validator`](#aw.cosmo.create_cosmo_validator)([cosmo_function, ...]) | Create a validator for cosmograph requirements.             |
| [`infer_cosmo_params`](#aw.cosmo.infer_cosmo_params)(df)                        | Infer suitable cosmograph parameters from DataFrame.        |
| [`strict_cosmo_validator`](#aw.cosmo.strict_cosmo_validator)()                      | Strict validator that actually calls cosmograph.            |
| [`try_cosmo_visualization`](#aw.cosmo.try_cosmo_visualization)(df[, cosmo_function]) | Try to create a cosmograph visualization.                   |

### aw.cosmo.basic_cosmo_validator()

Basic validator that checks structural requirements only.

Does not require cosmograph to be installed.

### Example

```pycon
>>> validator = basic_cosmo_validator()
>>> success, info = validator(df)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### aw.cosmo.create_cosmo_validator(cosmo_function=None, required_columns=2, allow_generated_params=True)

Create a validator for cosmograph requirements.

This validator attempts to actually call cosmograph.cosmo() with the
DataFrame to see if it’s ready for visualization.

* **Parameters:**
  * **cosmo_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – The cosmograph.cosmo function (imported if None)
  * **required_columns** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum number of numeric columns needed
  * **allow_generated_params** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to auto-infer x/y columns
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  Validator function following (artifact) -> (success, info) protocol

### Example

```pycon
>>> validator = create_cosmo_validator()
>>> success, info = validator(df)
>>> if success:
...     print(f"Can visualize with: {info['params']}")
```

### aw.cosmo.infer_cosmo_params(df)

Infer suitable cosmograph parameters from DataFrame.

* **Parameters:**
  **df** – pandas DataFrame
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict of suggested parameters for cosmograph.cosmo()

### Example

```pycon
>>> params = infer_cosmo_params(df)
>>> from cosmograph import cosmo
>>> cosmo(df, **params)
```

### aw.cosmo.strict_cosmo_validator()

Strict validator that actually calls cosmograph.

Requires cosmograph to be installed.

### Example

```pycon
>>> validator = strict_cosmo_validator()
>>> success, info = validator(df)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### aw.cosmo.try_cosmo_visualization(df, cosmo_function=None, \*\*cosmo_kwargs)

Try to create a cosmograph visualization.

This is a functional validator that actually attempts the visualization.

* **Parameters:**
  * **df** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – DataFrame to visualize
  * **cosmo_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – The cosmograph.cosmo function
  * **\*\*cosmo_kwargs** – Additional arguments for cosmo
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (success, info)

### Example

```pycon
>>> from cosmograph import cosmo
>>> success, info = try_cosmo_visualization(df, cosmo)
```
