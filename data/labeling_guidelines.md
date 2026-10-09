# Labelling guidelines — v2

The rules every gold record in `data/eval.jsonl` (and the training split) follows. Gold must be consistent, because precision and recall are only meaningful if the same situation is always labelled the same way. When a case isn't covered here, decide, add the rule, and re-check earlier records it affects.

Only **relationships** are scored (`src/evaluate.py` compares `(subject, relation, target)` triples). Entities are still labelled, but mistakes there don't move the metrics.

## Source of truth

This document is the full rule set. The extraction prompts (`src/extraction/prompts/`) state the same rules in short form, and their worked examples are labelled by these rules, so the model is asked for exactly what the gold contains. **Change both together:** a rule added here without a matching prompt line measures convention-following, not extraction.

## General

1. **Only what the chunk says.** Label relationships visible in the chunk text. No outside knowledge of vLLM, except reading a name's module off an import that appears in the chunk.
2. **Never the file or the chunk itself.** The current file/module is not an entity, so it can't be a subject or target. (Prompt: "do not extract the file or chunk itself as an entity".)
3. **Names as written, identifiers only.** Use the identifier exactly as it appears; it must match `^[A-Za-z_][A-Za-z0-9_.]*$`. Names that don't (`--enable-lora`, `"LoRA adapters"`) are skipped, never rewritten.
4. **Dotted vs short names.**
   - Qualified by a module or class name → keep the dotted form: `os.path.join`, `json.load`, `msgspec.Struct`.
   - Called on an instance (`self`, a parameter, a local variable) → method name only: `request.validate()` → `validate`.
5. **Both ends are entities.** Every subject and target also appears in `entities`.
6. **Each fact once.** No duplicate triples (the scorer de-duplicates anyway).
7. **Empty is valid.** A chunk with nothing extractable gets `{"entities": [], "relationships": []}` — license headers, bare constants, `__init__.py` re-exports with nothing to say.
8. **Direction reads left to right:** `subject RELATION target` = "subject calls / inherits from / is defined in / references target".

## Node types

| Type | Use for |
| --- | --- |
| `Function` | functions, methods, decorators, callables |
| `Class` | classes, including base classes and exception types |
| `Module` | Python modules/packages named by an import or in prose (`vllm.lora.request`, `json`) |
| `Concept` | single-identifier ideas in prose (`LoRA`, `PagedAttention`) |
| `Entity` | anything else named: environment variables, config keys, fields, typing special forms (`Optional`) |

## Code relations

| Relation | Label when | Subject | Target | Don't label |
| --- | --- | --- | --- | --- |
| `CALLS` | a function body calls something, including instantiating a class | innermost enclosing function/method | callee (rule 4) | builtins (`len`, `abs`, `hash`, `open`, `print`, `isinstance`, `super`, ...); logging calls (`logger.warning`); calls outside any function (module level, class body) |
| `INHERITS_FROM` | a class lists a base class | the class | each base class as written | keyword arguments in the class header (`omit_defaults=True`), metaclass keywords |
| `DEFINED_IN` | `from X import Y` | each imported name `Y` | module `X` | `import X` (gives the entity `X`, no relation — the importing module is the file, rule 2); methods inside a class. Imports are always labelled, including typing and stdlib names (`Optional DEFINED_IN typing`). |
| `REFERENCES` | a function/class names another class/function without calling it: type annotations, `isinstance` targets, passing it as a value, class-body assignments (`__metaclass__ = X`) | innermost enclosing function/class | the named class/function | builtins and typing constructs (`str`, `dict`, `Optional`, `Union`, `Any`, `Set`); anything also called in the same function (`CALLS` wins) |
| `IMPORTS` | the chunk explicitly states one module imports another | importer | imported | ordinary import statements (rule 2 rules out the file as importer) |
| `RELATES_TO` | an explicit link no other relation covers | — | — | use as a last resort in code |

Applying a decorator is not labelled.

## Text relations (`.md`, `.rst`, `.txt`, config files)

- Relationships need both ends named in the **same section** (a header and its body).
- `RELATES_TO`: any explicit link between two named things — one uses, configures, requires, enables or belongs to the other. Both directions read the same; put the one that acts (uses, configures) as subject.
- `DEFINED_IN`: the text says a component is defined or configured in a named module or file.
- `CALLS`, `IMPORTS`, `INHERITS_FROM`: only when the text states that code relationship outright ("X calls Y internally").
- **No `REFERENCES` in prose.** Use `RELATES_TO` for every link that isn't one of the above.
- Code snippets inside documentation follow the code rules.

## Worked examples

From `data/sample.jsonl`.

**Imports** (`vllm/lora/resolver.py [0:358]`)

```python
from abc import ABC, abstractmethod
from vllm.lora.request import LoRARequest
logger = init_logger(__name__)
```

`ABC DEFINED_IN abc`, `abstractmethod DEFINED_IN abc`, `LoRARequest DEFINED_IN vllm.lora.request`. No `IMPORTS` (the importer is the file itself), no `CALLS` for `init_logger` (module level).

**Class with annotations** (`vllm/lora/resolver.py [358:1385]`)

```python
class LoRAResolver(ABC):
    @abstractmethod
    async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
```

`LoRAResolver INHERITS_FROM ABC`, `resolve_lora REFERENCES LoRARequest`. Not `Optional`/`str` (typing, builtins), not the decorator, not `resolve_lora DEFINED_IN LoRAResolver` (methods inside a class).

**Calls** (`vllm/plugins/lora_resolvers/filesystem_resolver.py [293:1438]`)

```python
lora_path = os.path.join(self.lora_cache_dir, lora_name)
...
lora_request = LoRARequest(lora_name=lora_name, lora_int_id=abs(hash(lora_name)), lora_path=lora_path)
```

`resolve_lora CALLS os.path.join`, `resolve_lora CALLS LoRARequest` (instantiation). Not `abs`/`hash` (builtins). `LoRARequest` is also in the return annotation, but it's called, so only `CALLS`.

## Changelog

- **v1** — initial rules.
- **v2** — prose uses `RELATES_TO` for every explicit link, never `REFERENCES`. Prompts rewritten to state these rules and their examples relabelled to match (import → `DEFINED_IN`, annotation → `REFERENCES`); `REFERENCES` now has a defined subject (the enclosing function/class). Imports are labelled even for typing/stdlib names.
