# aw.dispatch

Cheap-first dispatch to a skill: calibrated retrieval first, an LLM adjudicator only on ambiguity.

The chain decided in opsward ADR 0011 (`misc/docs/decisions/0011-cheap-first-dispatch-no-router-agent.md`
in thorwhalen/opsward): a route in the instructions and the host’s own skill listing are
the first two layers and belong to the host; this module is layers three and four, as code:

```default
chain = skills_dispatch_chain(adjudicate=Adjudicator(ask=my_llm))
chain("open an issue for the client about the broken export")   # -> 'ge-write-issue' or None
chain.trace("…")   # which layer answered, the ranked candidates, the reason
```

- [`ir_search()`](#aw.dispatch.ir_search) adapts `ir.discover` (lexical by default: the only mode fast enough
  for a hot path) to the `search(query) -> [(candidate, score)]` shape, with copies of one
  skill (`name@base`) collapsed to the best one.
- [`aw.routing.ConfidentRetrievalStrategy`](aw.routing.html.md#aw.routing.ConfidentRetrievalStrategy) answers only on a clear ranking and
  abstains until a boundary (floor, gap or relative gap) is configured.
- [`Adjudicator`](#aw.dispatch.Adjudicator) is the escalation: an LLM shown the raw request and the top-k
  candidates with their descriptions, answering with one of them or none. Never the whole
  catalog, never a summary of the request.
- [`DispatchChain`](#aw.dispatch.DispatchChain) wires them and keeps the ranked list between the two, which a
  plain [`aw.routing.PriorityRouter`](aw.routing.html.md#aw.routing.PriorityRouter) cannot do.

`ir` is an optional dependency (`pip install aw[ir]`); everything here is testable with
a fake search and a fake `ask`.

### Functions

| [`ir_search`](#aw.dispatch.ir_search)([corpus, mode, k, discover])            | A `search(query) -> [(name, score)]` over an `ir` corpus, copies collapsed.          |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`skill_name`](#aw.dispatch.skill_name)(candidate)                             | The skill a corpus record stands for: `'branch-hygiene@priv'` -> `'branch-hygiene'`. |
| [`skills_dispatch_chain`](#aw.dispatch.skills_dispatch_chain)(\*[, corpus, mode, k, ...]) | The ADR 0011 chain over the `ir` skills corpus, ready to call.                       |

### Classes

| [`Adjudicator`](#aw.dispatch.Adjudicator)([ask, describe, k, none_token])       | The escalation layer: an LLM picks one of the top-k candidates for the raw request, or none.   |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`DispatchChain`](#aw.dispatch.DispatchChain)(search[, floor, gap, rel_gap, ...]) | Retrieval, then confident selection, then adjudication; with a trace of which layer answered.  |

### *class* aw.dispatch.Adjudicator(ask=<function \_default_ask>, describe=None, k=5, none_token='NONE')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The escalation layer: an LLM picks one of the top-k candidates for the raw request, or none.

`ask(prompt) -> str` is the model call (default: `oa.chat`); `describe(name)`
returns a candidate’s description so the model sees cards, not bare names. The model
sees the user’s words verbatim: a summary is a lossy channel (opsward ADR 0011 §3).

```pycon
>>> adj = Adjudicator(ask=lambda p: 'ge-write-issue', describe={'ge-write-issue': 'Write an issue.'}.get)
>>> adj('open an issue for the client', [('ge-write-issue', 9.0), ('ge', 8.9)])
'ge-write-issue'
>>> Adjudicator(ask=lambda p: 'NONE')('x', [('a', 1.0)]) is None
True
>>> Adjudicator(ask=lambda p: 'something-else')('x', [('a', 1.0)]) is None   # not a candidate: no guess
True
```

### *class* aw.dispatch.DispatchChain(search, floor=0.0, gap=0.0, rel_gap=None, adjudicate=None, k_adjudicate=5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Retrieval, then confident selection, then adjudication; with a trace of which layer answered.

```pycon
>>> table = {'clear': [('a', 10.0), ('b', 5.0)], 'close': [('a', 10.0), ('b', 9.8)], 'none': []}
>>> chain = DispatchChain(search=table.get, rel_gap=0.9, adjudicate=Adjudicator(ask=lambda p: 'b'))
>>> chain('clear'), chain('close'), chain('none')
('a', 'b', None)
>>> chain.trace('close')['layer'], chain.trace('clear')['layer'], chain.trace('none')['layer']
('adjudicate', 'retrieval', 'none')
>>> DispatchChain(search=table.get, rel_gap=0.9)('close') is None   # no adjudicator: abstain
True
```

#### trace(request)

`{'layer', 'answer', 'ranked', 'reason'}` for *request*; the model is called only on ambiguity.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aw.dispatch.ir_search(corpus='skills', , mode='lexical', k=10, discover=None)

A `search(query) -> [(name, score)]` over an `ir` corpus, copies collapsed.

`discover` defaults to `ir.discover` (imported lazily, so `aw` itself does not
depend on `ir`); pass any callable with the same result shape to test without it.
Scores are the engine’s (BM25-like for lexical), so calibrate `floor` per corpus.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]

### aw.dispatch.skill_name(candidate)

The skill a corpus record stands for: `'branch-hygiene@priv'` -> `'branch-hygiene'`.

```pycon
>>> skill_name('ge-write-issue@g_ai'), skill_name('deslop')
('ge-write-issue', 'deslop')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### aw.dispatch.skills_dispatch_chain(, corpus='skills', mode='lexical', k=10, floor=0.0, rel_gap=0.75, adjudicate=None, discover=None)

The ADR 0011 chain over the `ir` skills corpus, ready to call.

Defaults: lexical mode; a relative gap of 0.75 (the runner-up must score at most 75% of
the top), the point on the 14-prompt probe set of opsward ADR 0011 where the chain
gives no confident wrong answer (9 confident, 5 escalated; at 0.9 it answers 12 right
and 2 wrong); no floor until `ir calibrate-min-score` has produced one for the
corpus; no adjudicator (pass an [`Adjudicator`](#aw.dispatch.Adjudicator) to escalate instead of abstaining).

* **Return type:**
  [`DispatchChain`](#aw.dispatch.DispatchChain)
