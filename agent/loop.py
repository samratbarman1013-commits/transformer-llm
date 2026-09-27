"""Agent loop: observe → (route → tool → observe)* → respond.

This is the 'loop working' engine. The 1M prototype uses the deterministic
router in agent.tools; the loop itself — bounded iterations, tool results
folded back into context, stop conditions — is what carries over unchanged
when the model grows to 500M params and starts emitting its own tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import tools as T


@dataclass
class Turn:
    user: str
    tool_calls: list[dict] = field(default_factory=list)
    final: str = ""


class Agent:
    def __init__(self, model=None, tok=None, max_tool_iters: int = 4,
                 search_provider: T.SearchProvider | None = None):
        self.model, self.tok = model, tok
        self.max_tool_iters = max_tool_iters
        self.search_provider = search_provider

    def _run_tool(self, call: str) -> dict:
        name, _, arg = call.partition(":")
        tool = T.DEFAULT_TOOLS.get(name)
        if tool is None:
            return {"tool": name, "ok": False, "output": f"unknown tool '{name}'"}
        if name == "search" and self.search_provider:
            res = tool.fn(arg, provider=self.search_provider)
        else:
            res = tool.fn(arg)
        return {"tool": name, "ok": res.ok, "output": res.output[:1500]}

    def respond(self, user: str, history: str = "") -> Turn:
        turn = Turn(user=user)
        context = user

        for _ in range(self.max_tool_iters):
            calls = T.route(context)
            if not calls:
                break
            for c in calls:
                result = self._run_tool(c)
                turn.tool_calls.append(result)
                context = f"{user}\n[tool {result['tool']} says] {result['output']}"
            break  # prototype: one routing pass per turn

        # did a tool fully answer? (calculator hits, successful python)
        answered = any(r["ok"] and r["tool"] in ("calculator", "python", "clock")
                       for r in turn.tool_calls)
        if answered:
            pieces = [f"[{r['tool']}] {r['output']}" for r in turn.tool_calls if r["ok"]]
            turn.final = "\n".join(pieces)
            return turn

        # otherwise (or on top), the LLM generates the reply
        if self.model is not None:
            from transformer.chat import reply
            llm = reply(self.model, self.tok, f"{history}\nUser: {user}".strip())
            if turn.tool_calls:
                notes = "\n".join(f"[{r['tool']}] {r['output']}"
                                  for r in turn.tool_calls if not r["ok"])
                turn.final = (notes + "\n\n" + llm).strip()
            else:
                turn.final = llm
        else:
            turn.final = "No language model loaded."
        return turn
