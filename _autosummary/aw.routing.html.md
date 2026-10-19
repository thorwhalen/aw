# aw.routing

Configurable routing and decision chains for agentic workflows.

This module provides tools for building extensible, inspectable routing chains
that follow the open-closed principle. Users can:

- Use default configurations out of the box
- Inspect all routing rules and mappings
- Customize individual components
- Extend with new routing strategies

The key pattern is: provide sensible defaults, make them visible, keep them mutable.

### Functions

| [`detect_explicit`](#aw.routing.detect_explicit)(ctx)                           | Use explicit extension if provided.       |
|-------------------------------------------------------------------------------------------------|-------------------------------------------|
| [`detect_from_url`](#aw.routing.detect_from_url)(ctx)                           | Extract extension from URL path.          |
| [`make_content_type_detector`](#aw.routing.make_content_type_detector)([content_type_map]) | Factory for content-type based detection. |
| [`make_magic_bytes_detector`](#aw.routing.make_magic_bytes_detector)([magic_bytes_map])   | Factory for magic bytes based detection.  |

### Classes

| [`ConditionalRouter`](#aw.routing.ConditionalRouter)([strategies, short_circuit])   | Router with short-circuit conditions.                                                |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`ConfidentRetrievalStrategy`](#aw.routing.ConfidentRetrievalStrategy)(search[, floor, gap]) | A routing strategy backed by a ranked search: answer only when the ranking is clear. |
| [`ExtensionContext`](#aw.routing.ExtensionContext)(url[, content, ...])            | Context for extension detection.                                                     |
| [`ExtensionRouter`](#aw.routing.ExtensionRouter)([priority_extensions, ...])      | Route extension detection through configurable strategies.                           |
| [`MappingRouter`](#aw.routing.MappingRouter)([mapping, default, key_transform]) | Router based on a lookup mapping.                                                    |
| [`PriorityRouter`](#aw.routing.PriorityRouter)([strategies])                     | Route through strategies in priority order, short-circuiting on first match.         |
| [`RoutingStrategy`](#aw.routing.RoutingStrategy)(\*args, \*\*kwargs)              | Protocol for routing strategies.                                                     |

### *class* aw.routing.ConditionalRouter(strategies=list(), short_circuit=None)

Bases: [`PriorityRouter`](#aw.routing.PriorityRouter)

Router with short-circuit conditions.

Only accepts results that match the short-circuit condition,
otherwise continues to next strategy.

```pycon
>>> def try_url(x): return x.get('url_ext')
>>> def try_type(x): return x.get('content_type_ext')
>>>
>>> # Only accept .pdf or .md, otherwise keep trying
>>> router = ConditionalRouter(
...     [try_url, try_type],
...     short_circuit=lambda r: r in {'.pdf', '.md'}
... )
>>>
>>> # .pdf matches short-circuit, return immediately
>>> router({'url_ext': '.pdf', 'content_type_ext': '.html'})
'.pdf'
>>>
>>> # .jpg doesn't match, try next strategy, .md matches
>>> router({'url_ext': '.jpg', 'content_type_ext': '.md'})
'.md'
```

### *class* aw.routing.ConfidentRetrievalStrategy(search, floor=0.0, gap=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A routing strategy backed by a ranked search: answer only when the ranking is clear.

`search(context)` returns `(candidate, score)` pairs, best first (any retrieval
engine: `ir.discover`, a BM25 index, a dict lookup). The strategy returns the top
candidate when its score reaches `floor` *and* beats the runner-up by at least
`gap`; otherwise it returns `None` so a [`PriorityRouter`](#aw.routing.PriorityRouter) can fall through
to the next, more expensive, strategy (an LLM adjudicator, a human). That is the
“cheap-first, escalate on ambiguity” shape: no model in the hot path for the clear
cases, abstention rather than a guess for the close ones.

```pycon
>>> ranked = {'open an issue': [('ge-write-issue', 0.9), ('ge', 0.4)],
...           'do the thing': [('a', 0.5), ('b', 0.48)],
...           'nothing': []}
>>> route = ConfidentRetrievalStrategy(search=ranked.get, floor=0.3, gap=0.1)
>>> route('open an issue')
'ge-write-issue'
>>> route('do the thing') is None   # too close to call: escalate
True
>>> route('nothing') is None
True
>>> router = PriorityRouter([route, lambda q: 'ask-the-llm'])
>>> router('do the thing')
'ask-the-llm'
```

`explain` tells a caller why a query fell through, for calibration logs:

```pycon
>>> route.explain('do the thing')
'ambiguous: a=0.5 over b=0.48 is below gap 0.1'
```

#### explain(context)

Why the strategy answered or abstained, in one line.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* aw.routing.ExtensionContext(url, content=b'', content_type='', explicit_extension=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Context for extension detection.

```pycon
>>> ctx = ExtensionContext(
...     url='https://example.com/file.pdf',
...     content=b'%PDF-1.5',
...     content_type='text/html'
... )
>>> ctx.url
'https://example.com/file.pdf'
```

### *class* aw.routing.ExtensionRouter(priority_extensions=<factory>, content_type_map=<factory>, magic_bytes_map=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Route extension detection through configurable strategies.

This router implements the default logic for detecting file extensions,
but every component is visible and customizable.

### Examples

```pycon
>>> # Use with defaults
>>> router = ExtensionRouter()
>>> ctx = ExtensionContext('https://example.com/file.pdf', b'%PDF-1.5')
>>> router(ctx)
'.pdf'
>>>
>>> # URL without extension, falls back to magic bytes
>>> ctx = ExtensionContext('https://example.com/download', b'%PDF-1.5')
>>> router(ctx)
'.pdf'
>>>
>>> # Priority extensions short-circuit
>>> ctx = ExtensionContext('https://example.com/file.pdf', b'<html>')
>>> router(ctx)  # .pdf from URL wins even though content is HTML
'.pdf'
>>>
>>> # Customize content type mapping
>>> router.content_type_map['.txt'] = 'text/x-log'
>>>
>>> # Add custom detector
>>> def my_detector(ctx): return '.custom' if 'special' in ctx.url else None
>>> router = router.with_prepended_strategy(my_detector)
```

#### with_appended_strategy(strategy)

Create new router with added strategy at lowest priority.

* **Return type:**
  [`ExtensionRouter`](#aw.routing.ExtensionRouter)

#### with_prepended_strategy(strategy)

Create new router with added strategy at highest priority.

* **Return type:**
  [`ExtensionRouter`](#aw.routing.ExtensionRouter)

### *class* aw.routing.MappingRouter(mapping=dict(), default=None, key_transform=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Router based on a lookup mapping.

Provides a clean interface for switch-case style routing with:

- Visible, mutable mapping
- Optional default value or factory
- Easy extension via dict interface

```pycon
>>> # Content-Type to extension mapping
>>> ct_map = {
...     'application/pdf': '.pdf',
...     'text/html': '.html',
...     'application/json': '.json',
... }
>>> router = MappingRouter(ct_map)
>>>
>>> router('application/pdf')
'.pdf'
>>> router('text/plain') is None  # No default
True
>>>
>>> # With default
>>> router_with_default = MappingRouter(ct_map, default='.bin')
>>> router_with_default('text/plain')
'.bin'
```

#### copy()

Create a copy with same configuration.

* **Return type:**
  [`MappingRouter`](#aw.routing.MappingRouter)

#### update(\*args, \*\*kwargs)

Update mapping.

### *class* aw.routing.PriorityRouter(strategies=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Route through strategies in priority order, short-circuiting on first match.

This is a simpler, more specialized version of i2.routing_forest that’s
optimized for the common case of “try these in order, stop at first success”.

```pycon
>>> def try_a(x): return 'A' if x > 10 else None
>>> def try_b(x): return 'B' if x > 5 else None
>>> def try_c(x): return 'C'
>>>
>>> router = PriorityRouter([try_a, try_b, try_c])
>>> router(15)
'A'
>>> router(8)
'B'
>>> router(3)
'C'
```

#### append(strategy)

Add a strategy at the end (lowest priority).

* **Return type:**
  [`PriorityRouter`](#aw.routing.PriorityRouter)

#### insert(index, strategy)

Insert a strategy at a specific position.

* **Return type:**
  [`PriorityRouter`](#aw.routing.PriorityRouter)

#### prepend(strategy)

Add a strategy at the beginning (highest priority).

* **Return type:**
  [`PriorityRouter`](#aw.routing.PriorityRouter)

### *class* aw.routing.RoutingStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Protocol for routing strategies.

A routing strategy takes a context and returns a result or None.

### aw.routing.detect_explicit(ctx)

Use explicit extension if provided.

```pycon
>>> ctx = ExtensionContext('', explicit_extension='pdf')
>>> detect_explicit(ctx)
'.pdf'
>>> ctx = ExtensionContext('', explicit_extension='.json')
>>> detect_explicit(ctx)
'.json'
```

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### aw.routing.detect_from_url(ctx)

Extract extension from URL path.

```pycon
>>> ctx = ExtensionContext('https://example.com/file.pdf')
>>> detect_from_url(ctx)
'.pdf'
>>> ctx = ExtensionContext('https://example.com/file')
>>> detect_from_url(ctx) is None
True
```

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### aw.routing.make_content_type_detector(content_type_map=None)

Factory for content-type based detection.

Returns a detector function with the given mapping.
Users can create custom detectors with different mappings.

```pycon
>>> # Use default mapping
>>> detector = make_content_type_detector()
>>> ctx = ExtensionContext('', content_type='application/pdf')
>>> detector(ctx)
'.pdf'
>>>
>>> # Custom mapping
>>> custom_map = {'text/plain': '.log'}
>>> custom_detector = make_content_type_detector(custom_map)
>>> ctx = ExtensionContext('', content_type='text/plain')
>>> custom_detector(ctx)
'.log'
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`ExtensionContext`](#aw.routing.ExtensionContext)], [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### aw.routing.make_magic_bytes_detector(magic_bytes_map=None)

Factory for magic bytes based detection.

```pycon
>>> detector = make_magic_bytes_detector()
>>> ctx = ExtensionContext('', content=b'%PDF-1.5...')
>>> detector(ctx)
'.pdf'
>>> ctx = ExtensionContext('', content=b'<html>...')
>>> detector(ctx)
'.html'
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`ExtensionContext`](#aw.routing.ExtensionContext)], [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
