"""Tool registry shared by the W15 assistant and the W16 agent.

Each tool = OpenAI-style JSON schema + async implementation + argument validation.
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .config import Settings, get_settings
from .rag.ingest import load_catalog


class ToolError(Exception):
    pass


class ToolUnavailable(ToolError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Awaitable[Any]]

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def validate(self, args: dict) -> list[str]:
        """Minimal JSON-schema validation (required, types, enums). Returns list of problems."""
        errs = []
        props = self.parameters.get("properties", {})
        for r in self.parameters.get("required", []):
            if r not in args:
                errs.append(f"missing required argument '{r}'")
        types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        for k, v in args.items():
            if k not in props:
                errs.append(f"unknown argument '{k}'")
                continue
            t = props[k].get("type")
            if t and v is not None and not isinstance(v, types[t]):
                errs.append(f"argument '{k}' should be {t}")
            if "enum" in props[k] and v is not None and v not in props[k]["enum"]:
                errs.append(f"argument '{k}' must be one of {props[k]['enum']}")
        return errs


class ToolBox:
    """Holds the tool implementations; supports fault injection for W16 failure tests."""

    def __init__(self, store, router=None, s: Settings | None = None, fail_inject: str | None = None):
        self.s = s or get_settings()
        self.store = store
        self.router = router
        self.catalog = load_catalog(self.s)
        self.fail_inject = fail_inject if fail_inject is not None else self.s.fail_inject
        self.tools: dict[str, Tool] = {}
        self._register()

    # ------------------------------------------------------------ implementations
    async def search_papers(self, query: str, k: int = 5, paper_id: str | None = None, **_) -> dict:
        if self.fail_inject == "unavailable":
            raise ToolUnavailable("search backend unavailable (vector DB connection refused)")
        if self.fail_inject == "timeout":
            await asyncio.sleep(self.s.tool_timeout_s + 5)
        if paper_id and paper_id not in self.catalog:
            raise ToolError(f"unknown paper_id '{paper_id}'. Valid: {sorted(self.catalog)}")
        hits = await asyncio.to_thread(self.store.search, query, k, paper_id)
        results = [h.to_dict(self.s.chunk_char_cap) for h in hits]
        if self.fail_inject == "malformed":
            # garbled text + stripped metadata: simulates a corrupted index / broken parser
            rnd = random.Random(query)
            results = [{"chunk_id": None, "paper_id": "", "page": -1, "score": "NaN",
                        "text": "".join(rnd.choice("#%&@~^") for _ in range(120))} for _ in results]
        return {"query": query, "paper_id": paper_id, "n": len(results), "results": results}

    async def get_chunk(self, chunk_id: str, **_) -> dict:
        if self.fail_inject == "unavailable":
            raise ToolUnavailable("search backend unavailable (vector DB connection refused)")
        h = await asyncio.to_thread(self.store.get, chunk_id)
        if h is None:
            raise ToolError(f"no chunk with id '{chunk_id}'")
        return h.to_dict()

    async def list_papers(self, **_) -> dict:
        return {"papers": [{"paper_id": k, "title": v["title"]} for k, v in self.catalog.items()]}

    async def route_support_intent(self, message: str, **_) -> dict:
        if self.router is None:
            raise ToolUnavailable("intent router model not loaded (run export_onnx first)")
        return await self.router.route(message)

    # ------------------------------------------------------------ registry
    def _register(self):
        pid = {"type": "string", "description": "Optional: restrict to one paper id (see list_papers)",
               "enum": list(self.catalog)}
        self.add(Tool("search_papers",
                      "Semantic + keyword search over the research-paper corpus. Returns chunks with chunk_id, paper_id, page.",
                      {"type": "object", "properties": {
                          "query": {"type": "string", "description": "What to look for"},
                          "k": {"type": "integer", "description": f"Number of chunks (max {self.s.max_top_k})"},
                          "paper_id": pid},
                       "required": ["query"]}, self.search_papers))
        self.add(Tool("get_chunk", "Fetch the full text of one chunk by chunk_id (e.g. 'linevd:p3:c1').",
                      {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"]},
                      self.get_chunk))
        self.add(Tool("list_papers", "List all papers in the corpus with their ids and titles.",
                      {"type": "object", "properties": {}}, self.list_papers))
        self.add(Tool("route_support_intent",
                      "Classify an e-commerce customer-support message into one of 11 support agents "
                      "(the W14 fine-tuned router, served with ONNX Runtime).",
                      {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                      self.route_support_intent))

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if names is None or n in names]

    async def call(self, name: str, args: dict) -> tuple[bool, Any]:
        """Execute with validation + timeout. Returns (ok, result_or_error_string). Never raises."""
        tool = self.tools.get(name)
        if tool is None:
            return False, f"ERROR: unknown tool '{name}'. Available: {list(self.tools)}"
        problems = tool.validate(args)
        if problems:
            return False, "ERROR: invalid arguments: " + "; ".join(problems)
        try:
            return True, await asyncio.wait_for(tool.fn(**args), timeout=self.s.tool_timeout_s)
        except asyncio.TimeoutError:
            return False, f"ERROR: tool '{name}' timed out after {self.s.tool_timeout_s}s"
        except ToolUnavailable as e:
            return False, f"ERROR: tool unavailable: {e}"
        except ToolError as e:
            return False, f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001
            return False, f"ERROR: {type(e).__name__}: {e}"


def dumps(obj: Any) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
