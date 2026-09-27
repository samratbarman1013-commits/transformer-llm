"""Tool registry for the Transformer agent.

Design: every tool is a function + a JSON-schema-style descriptor. The loop
(layered above) decides when to call what. At prototype scale (1M params) the
model itself cannot emit reliable tool calls, so `route()` maps user intent to
tools with deterministic rules — the tool interface, registry and execution
plumbing are production-shaped, only the router is a placeholder that gets
replaced by the model's function-calling once it is instruction-tuned at
larger scale.
"""

from __future__ import annotations

import ast
import operator
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolResult:
    ok: bool
    output: str
    tool: str = ""


@dataclass
class Tool:
    name: str
    description: str
    args: dict
    fn: Callable[[str], ToolResult]
    examples: list[str] = field(default_factory=list)


# --- calculator: safe AST arithmetic (no eval, no names, no imports) ------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


def calc(expr: str) -> ToolResult:
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
        return ToolResult(True, str(value), "calculator")
    except Exception as e:
        return ToolResult(False, f"could not evaluate: {e}", "calculator")


# --- python sandbox: subprocess, timeout, no shell -----------------------

def run_python(code: str) -> ToolResult:
    try:
        p = subprocess.run(
            ["python3", "-c", code], capture_output=True, text=True,
            timeout=10, cwd="/tmp",
        )
        out = (p.stdout or p.stderr or "(no output)").strip()[:2000]
        return ToolResult(p.returncode == 0, out, "python")
    except subprocess.TimeoutExpired:
        return ToolResult(False, "timed out after 10s", "python")


# --- time -----------------------------------------------------------------

def now(_: str = "") -> ToolResult:
    return ToolResult(True, time.strftime("%Y-%m-%d %H:%M:%S %Z"), "clock")


# --- web search: pluggable provider --------------------------------------

class SearchProvider:
    """Implement `search()` and register on the agent. Prototype ships with a
    local no-op so the app runs offline; wire a real provider (DuckDuckGo,
    Brave, Tavily...) in server/app.py via SEARCH_PROVIDER."""

    def search(self, query: str) -> list[dict]:
        return [{"title": "Search not configured",
                 "snippet": f"Set a search provider to look up: {query}",
                 "url": ""}]


def web_search(query: str, provider: SearchProvider | None = None) -> ToolResult:
    if provider is None:
        provider = SearchProvider()
    results = provider.search(query)[:5]
    text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)
    return ToolResult(True, text or "no results", "search")


# --- registry --------------------------------------------------------------

DEFAULT_TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate arithmetic expressions, e.g. 12*(3+4)/2",
        args={"expr": "string"},
        fn=calc,
        examples=["2+2", "sqrt? no - arithmetic only", "(17*31)-6"],
    ),
    "python": Tool(
        name="python",
        description="Run a short Python snippet in a sandboxed subprocess",
        args={"code": "string"},
        fn=run_python,
        examples=["print(sum(range(100)))"],
    ),
    "clock": Tool(
        name="clock", description="Current date and time",
        args={}, fn=lambda _: now(), examples=["what time is it"],
    ),
    "search": Tool(
        name="search", description="Web search via a pluggable provider",
        args={"query": "string"}, fn=web_search,
        examples=["search latest ISRO news"],
    ),
}


def route(message: str) -> list[str]:
    """Deterministic intent router (placeholder for learned function-calling)."""
    m = message.lower()
    calls: list[str] = []
    if re.search(r"\b(calc|calculate|what(?:'s| is)\s+[\d(])|\d+\s*[-+*/^]\s*\d+", m):
        expr = re.search(r"[-+*/().\d\s^%]{3,}", message)
        if expr:
            calls.append(f"calculator:{expr.group().strip()}")
    if m.startswith(("run ", "python ")) or "```python" in message:
        code = message.split("```python")[-1].split("```")[0] if "```python" in message \
            else message.split(" ", 1)[-1]
        calls.append(f"python:{code}")
    if any(k in m for k in ("search", "look up", "news about", "find online")):
        q = message.split(":", 1)[-1] if ":" in message else message
        calls.append(f"search:{q}")
    if "time" in m and "what" in m:
        calls.append("clock:")
    return calls
