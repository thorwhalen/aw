

I'd like to build an AI agent for data preparation.

It should have several aspects of the problem. These should be able to be done individually, or in a chain (in the ordered listed below). Each step of the chain should be able to be done automatically (with all the classics like validation, max num of tries, etc.) or with human in the loop (systematically, or via some decision the AI agent would make):

finding data

sourcing the data (downloading)

loading the data

(optionally) getting stats on the data

preparing the data

I'd like each step to use the same pattern, with a bunch of parametrizations possible (but not required (we'll use hardcoded as well as smart defaults)). There should be parameters to specify validation, max retries, tools (if possible via a specific reusable schema for the interface) etc.

Whenever possible, I'd like to be able to specify parameters via functions as well as via simple objects such as strings, numbers, etc. For example, a text-to-text chat model could be specified via a string, but could also be specified via a function that is to be considered as the text-to-text chat function (assuming all the agent needs is a means to "ask AI" by composing a question, and getting text back).

There should be an easy, boilerplate-minimal, convention over configuration way of specifying all parameters we want to specify, with classic configuration patterns such as being able to set some defaults at any level of the nested dependencies (e.g. a global chat, or embedding model, to be taken if no other is specified, but with the ability of specifying a different one in some sub-component or agent).

That's the general context I would like you to think of when designing the architecture.

But I'd also like to work on a concrete example (which will be specified via configuration as much as possible, but hard coded when ever it's simpler).

The concrete use case is as follows:

Only the loading the data and preparing the data steps.

In loading data, we have an agent whose job it is to take some sort of source specification (e.g. URI (URL, local folder or file path, etc.), etc.) and resolve it into a pandas dataframe. (Remember that this target "loaded data" type/format must not be hardcoded if possible to avoid, since we might want our data in a different format at some point. Though we might want to create a module, that dependency-injects these aspects). This agent will have do what a human does when faced with this situation. It has a look at some aspects of the source specification (e.g. the file extension), and possibly a sample of the first n bytes of the contents (possibly sampling over different files too), try out one way of loading it (which here corresponds to transforming it into a pandas dataframe), if it fails, it captures failure information (e.g. traceback) and analyzes it to think of the next steps (parametrization of the loader, try a different loader, etc.)

After successfully getting a dataframe, further validation may also happen. This can be generalized as "run some computation(s) on this dataframe, producing some info dict for example, and then another function over this info dict that might result in some action (warning the user, asking the user, pausing the normal flow of things, etc.)". Think of some good design patterns for such kind of validation.

This step (like any step) produces some data artifacts that can then be used by other agents. In this case, is, for sure, a dataframe, but could also provide analysis information (e.g. produced during validation) that can then be used by downstream processes. (This can be seen as a memory management aspect, or as a "data flow" (e.g. DAGs) aspect).

The next step is preparation. Preparation's target is to transform the dataframe into a something (possibly still a dataframe, or something else, possibly even multiple outputs) that is expected. This expectation is of course operationalized by the validation function, though that may not necessarily (and often is not) the way the user will express it. The validation function spits out some information as well as, critically, a boolean that says if the expectations are met, or not (which signals, in the retry loop, whether we should retry or can now go to the next step (in our current case, finish the process and output the result)). Note that sometimes the validation can be expressed via a schema (pydantic model, json schema, etc.), but schemas can also be too limited for general cases. The most general case is always the function (that's why I suggested all parameters should have functional forms (as well as "simple objects/primitives" forms, to make the UX easier)). One functional form we can have for validation is to "try out the purpose of the data prep" -- that is, actually apply the downstream processes or proxies thereof. More on that later.

So the preparation step will need access to a python environment, critically, with pandas installed. It will need to be able to write some code, run it to try to prepare transform the data in a way that moves it closer to being validated. Here we really want to look into the existing best practices and design patterns regarding "inplace versus copy". Data preparation is not a straight line: We try things, perhaps over several transformation steps, then retract, try something else, etc. If we transform inplace (mutating the dataframe), we have better usage of memory, but then how do we "rewind" to a previous state? What are the design patterns for that? We could transform inplace, but remembering all the steps of the transformation in a list, then to rewind, we redo all those steps up to the point we want, from loading the data. We could also keep a buffer of copies of the dataframe (plus other artifacts) to be able to rewind at least a few steps? We could only transform over copies of data, keeping (caching) all the copies, releasing cached copies according to memory-aware rules etc. If we can't find some clear robust patterns and preferably tools for this, I prefer to go for the simplest (recompute from load OR only work on copies (and pray nothing blows up)).

More on functional validation. We'll take the example of the goal of the data preparation as being "preparing it for visualization". Take this example seriously, since this is the actual concrete example I need to implement with my data prep AI agent system.

Even more concretely: We'll use a specific visualization target: The cosmograph.cosmo(data, ...) function. This function needs data (a dataframe) plus, at the very minimum, we need to specify a points_x_by and points_y_by which tells cosmo what columns to use for the coordinates of the points it will plot. The values for these columns must all be numerical (no missing values). So we already know one condition: There must be at least two distinct columns with numerical values. To make the visualization even more interesting, we might also want to have point_size_by (a column to determine the size -- values must be numerical) and point_color_by (to determine the color -- values could be numerical or categorical (but if categorical, shouldn't have too many unique values)).

Whatever it is, one way we could validate our data prep is to have yet another Agent (AI or not) that will take the data, find some possible data-to-viz mappings and literally try them out by calling the cosmo function and seeing if it goes boom (and optionally, do some validation on the output).

I pasted here a bit of research so you have some extra information on design patterns etc.

Don't write any code yet. I first want to talk architecture and design.

First give me some concise takeaways about your thoughts on this, knowing what my preferences are etc.





---





🎯 Architecture & Design Takeaways

Core Design Philosophy

1. Unified AgenticStep Protocol



Every step (Finding, Sourcing, Loading, Preparing) implements the same interface

Each has: execute(input, context) → (artifact, metadata)

Internal loop: Thought → Action (code gen) → Observe (run) → Validate → repeat or finish

This gives you the ReAct pattern naturally, with your retry/validation baked in

2. Configuration as Callable-or-Simple-Object



Every parameter (llm_model, validation, tools) accepts both:Simple form: "gpt-4" or max_retries=3

Functional form: lambda prompt: my_llm(prompt) or custom_validator(df)

Use dataclasses with Union types for the config objects

Cascade: GlobalConfig → StepConfig → AgentConfig via simple dict merging or dataclass inheritance

3. Validation as Three Flavors (Pick by Use Case)



Schema-based: PydanticModel (for structure)

Info-dict based: df → compute_info() → check_info() → bool + reason

Functional/Try: df → actually_try_cosmo() → bool + error_or_params

For your concrete example: Use functional validation (try cosmo, capture exception)



State Management: The Pragmatic Path

4. Start with "Copy-on-Write" + Smart Caching



For your concrete case (cosmograph prep), DataFrames are likely small enough to copy

Pattern: Each transformation attempt works on a shallow copy of the last successful state

Cache last N states (e.g., 3) as a simple list: state_history = [df_0, df_1, df_2]

On failure: pop the stack, retry with the previous state

On success: commit to history, continue

If you need industrial strength later: Layer in lakeFS (the research nailed this—it's Git for data, perfect for your use case)

5. Memory/Artifacts as Nested Mapping



context = {

'loading': {'df': df, 'info': {...}, 'trace': [...]},

'preparing': {'df': df, 'cosmo_params': {...}, 'trace': [...]}

}

Each step reads from and writes to this context

Aligns with your Mapping/MutableMapping preference

Makes "chain of steps" trivial: prepare_agent.execute(context['loading']['df'], context)

Concrete Implementation Strategy

6. Loading Agent



Input: source_uri

Core loop: infer_format() → gen_loader_code() → execute() → [fail? → observe_error() → retry]

Validation: basic (is_dataframe + not_empty) OR custom info-dict validator

Output: df + {shape, dtypes, null_counts, sample_rows}

7. Preparing Agent



Input: df + target_requirements (e.g., "cosmo-ready")

Core loop: analyze_current_state() → gen_transform_code() → execute() → validate_with_cosmo()

Validation: CosmoValidatorAgent as a tool (literally tries cosmo call)

Output: prepared_df + {successful_viz_params, applied_transformations}

8. CosmoValidatorAgent (as Internal Tool)



Not a full agent—just a validator function wrapped as a tool

Takes: df → infers possible column mappings → tries cosmo(df, x=..., y=...)

Returns: {'success': bool, 'params': {...} or 'error': traceback}

This becomes the "Observation" in the Preparing Agent's loop

Architecture Wins for Your Preferences

✅ Functional-first: Steps are functions, validators are functions, config is callable



✅ Mapping interface: Context/artifacts as nested dicts



✅ Minimal boilerplate: @dataclass for configs, Protocol for step interface



✅ Iterables over lists: Trace/history as generators where possible



✅ Inner functions: Validators/helpers nested in agent implementations



✅ Open-closed: New steps/validators plug in without modifying core





## Critical Questions Before We Code

Orchestration framework? Should we use CrewAI/LangChain/DSPy or roll our own minimal ReAct loop? (I lean toward: roll our own for max control + your preferences)

Code execution sandbox? Real CodeInterpreterTool (external service) or just exec() with safety? (I lean toward: exec in limited namespace for simplicity, layer in sandbox later)

LLM interface? Should I design for Anthropic API specifically, or abstract to work with any? (I lean toward: abstract with Claude as default)

What's your take on these? Then we architect the actual module structure.





My answer to all three of the questions is: Always favor abstractions. Especially abstractions aligned with standard lib python (collections.abc etc.). I prefer to have abstractions that I can then inject dependencies into (dependency injection). That said:

(1) I should always have defaults for these abstractions (preferably with simple pure python)

(2) I would like to have easy access to more robust full-featured "backends" that use langchain, dspy... (In fact, perhaps we can have specific modules that depend on these, that provide us their ability to use our stretches with them immediately.



As far as code execution, yes, I would like some safe exec default, but definitely have the more robust tools at my fingertips.



And as far as LLMs go; definitely abstract facade for what the agents need, but use openAI default, not claude. In fact, use the oa package (check it out and see if you have the facades you need there (for example, the chat function and the prompt_function and the prompt_json_function may be useful -- see the oa.py.md package about oa).

The other packages I attached are other of my packages you might find useful (they're pypi installable).