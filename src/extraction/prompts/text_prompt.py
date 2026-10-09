"""
extraction prompt for text chunks (Chunk.source_type == "text").

Covers markdown/docs prose AND config/data files (.yaml, .json, .sh, ...)
that default to "text" - worded generically enough to degrade
reasonably for both, not just narrative prose.
"""

TEXT_PROMPT_TEMPLATE = """\
You are extracting a knowledge graph from a chunk of text. It may be \
documentation prose, or a structured file such as configuration or data.

Identify:
- entities: named concepts, components, or things discussed in this \
chunk. Give each a name (the actual name as it appears in the text, e.g. \
"LoRAConfig", "enable_lora" - never the literal words "Function", "Class", \
"Module", "Concept", or "Entity", which are only the allowed values for \
node_type, not names) and a node_type - one of: \
Function, Class, Module, Concept, Entity.
- relationships: how those entities relate to each other, as \
(subject, relation, target) triples. Valid relations:
  - RELATES_TO: the text explicitly links two named things - one uses, \
configures, requires, enables or belongs to the other. subject is the one \
that acts (uses, configures, ...).
  - DEFINED_IN: a component is defined or configured in a named module or \
file
  - CALLS, IMPORTS, INHERITS_FROM: only use these if the text explicitly \
describes that code-level relationship (e.g. "X calls Y internally") - \
most text chunks won't have any of these

Rules:
- Do not use REFERENCES for text; use RELATES_TO for every explicit link.
- Both ends of a relationship must be named in the same section (a \
heading and the text under it).
- Code snippets inside the text follow the usual code meaning: "from X \
import Y" gives Y DEFINED_IN X, and a call inside a function gives CALLS.

Example. Given this text:

    LoRA adapters are configured via LoRAConfig, which enable_lora() uses \
to set up adapter weights.

Correct extraction:
  entities: [{{"name": "LoRAConfig", "node_type": "Class"}}, \
{{"name": "enable_lora", "node_type": "Function"}}]
  relationships: [{{"subject": "enable_lora", "relation": "RELATES_TO", "target": "LoRAConfig"}}]

Only extract entities and relationships actually present in the text chunk \
below - never reuse names from the example above, that example is not part \
of this chunk. Do not invent things that aren't there, and do not extract \
the file or chunk itself as an entity.

subject and target must always be short identifier names (a function, \
class, or module name) - never a full sentence or line of text.

File: {file_path}

Text:
{text}
"""


def build_prompt(file_path: str, text: str) -> str:
    return TEXT_PROMPT_TEMPLATE.format(file_path=file_path, text=text)
