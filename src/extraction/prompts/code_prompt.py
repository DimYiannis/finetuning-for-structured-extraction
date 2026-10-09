"""
extraction prompt for code chunks (Chunk.source_type == "code").

Outlines' FSM guarantees the *shape* of the output matches
ExtractionResult - it says nothing about whether the model picks the
semantically right node/relation type for what it actually reads. This
prompt's job is steering that semantic choice; schema.py enforces it
can never be an invalid one.
"""

CODE_PROMPT_TEMPLATE = """\
You are extracting a knowledge graph from a chunk of source code.

Identify:
- entities: functions, classes, modules, or other named things defined or \
used in this chunk. Give each a name (the actual identifier as it appears \
in the code, e.g. "LoRARequest", "encode" - never the literal words \
"Function", "Class", "Module", "Concept", or "Entity", which are only the \
allowed values for node_type, not names) and a node_type - one of: \
Function, Class, Module, Concept, Entity.
- relationships: how those entities relate to each other, as \
(subject, relation, target) triples. Valid relations:
  - CALLS: a function calls another function or creates an instance of a \
class. subject is the function whose body contains the call.
  - INHERITS_FROM: a class inherits from a base class
  - DEFINED_IN: a function or class is defined in a module. For \
"from X import Y", Y is DEFINED_IN X.
  - REFERENCES: a function or class names another class or function \
without calling it (a type annotation, an isinstance check, passing it as \
a value). subject is the function or class where the name appears.
  - IMPORTS: a module imports another module
  - RELATES_TO: an explicit link not covered by the above

Rules:
- Never extract the file, the chunk, or the current module itself as an \
entity. So "import X" gives the entity X but no relationship.
- For CALLS and REFERENCES, skip Python builtins (len, abs, hash, open, \
print, isinstance, super, str, int, dict, ...), typing names (Optional, \
Union, Any, ...) and logging calls (logger.warning, ...).
- Skip calls made outside any function (at module level or directly in a \
class body), and skip applying a decorator.
- If a function both calls and references a name, use CALLS only.
- A method called on an object (self.x(), request.validate()) is named by \
the method alone: "validate". A call through a module or class keeps the \
dotted name as written: "os.path.join".
- Every subject and target must also be listed in entities.

Example. Given this code:

    from vllm.lora.request import LoRARequest

    class LoRAResolver:
        def resolve(self, request: LoRARequest):
            return request.validate()

Correct extraction:
  entities: [{{"name": "LoRARequest", "node_type": "Class"}}, \
{{"name": "vllm.lora.request", "node_type": "Module"}}, \
{{"name": "LoRAResolver", "node_type": "Class"}}, \
{{"name": "resolve", "node_type": "Function"}}, \
{{"name": "validate", "node_type": "Function"}}]
  relationships: [{{"subject": "LoRARequest", "relation": "DEFINED_IN", "target": "vllm.lora.request"}}, \
{{"subject": "resolve", "relation": "REFERENCES", "target": "LoRARequest"}}, \
{{"subject": "resolve", "relation": "CALLS", "target": "validate"}}]

Only extract entities and relationships actually present in the code chunk \
below - never reuse names from the example above, that example is not part \
of this chunk. Do not invent things that aren't there.

subject and target must always be short identifier names (a function, \
class, or module name) - never a full line of code, an import statement, \
or a sentence.

File: {file_path}

Code:
{text}
"""


def build_prompt(file_path: str, text: str) -> str:
    return CODE_PROMPT_TEMPLATE.format(file_path=file_path, text=text)
