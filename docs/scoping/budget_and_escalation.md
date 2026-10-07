# Claude budget policy (FW13) and one escalation primitive (FW35)

Scoped 2026-10-07.

## FW13 — when may normal stages use Claude?

### Today

`APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY` (default true, `llm.py`) keeps the
`claude_cli` tier out of every score/tailor/cover/judge fallback chain, so
the whole Pro-plan window is saved for auto-apply. It came from the
2026-08-20 incident where a cover run cascaded into `claude_cli` and burned
the window apply needed. The user's concern: an all-or-nothing reservation
wastes the window when no apply run is happening.

Facts that shape the design:
- Interactive Claude (these sessions) and the apply agent share the same
  Pro pool (decisions #166/#185/#195). The interactive use is invisible to
  the pipeline.
- `claude_status` already tracks a durable apply-side exhaustion signal and
  reads Claude's own usage cache (`read_cached_usage_state`,
  `binding_window`).

### Proposed policy (three modes, one env var)

`APPLYPILOT_CLAUDE_FOR_STAGES = never | idle | always` (default `idle`;
the old flag maps `true -> never`, `false -> always`).

`idle` lets normal stages fall back to `claude_cli` only when **all** of:
1. no `applypilot apply` process holds the scheduler/apply lock (a lock
   file the apply orchestrator writes at start and removes at exit);
2. the cached usage state shows the current window below a threshold
   (default 50% used), via `read_cached_usage_state`;
3. the stage has used fewer than N claude_cli calls this window
   (default 20), counted in a small JSON next to the exhaustion state.

Any `claude_status.record_apply_exhaustion` immediately suspends stage use
for the rest of the window.

**Done when:** a cover run during Gemini exhaustion with no apply running
falls back to `claude_cli` and stops at the cap; the same run with an
apply process running never touches it. All cloud-testable with fakes.

### Decision (2026-10-07, #234)

Keep the current policy: Claude stays reserved for apply. The modes below are kept for reference if testing or distillation resumes.

### Original decision needed

- Default `idle` with a 20-call / 50%-of-window cap, or keep `never` as
  the default and make `idle` opt-in?

## FW35 — one escalation primitive

### Today's hand-rolled cascades

| Where | Chain | Trigger to move on |
|---|---|---|
| `llm.LLMClient` | Gemini → OpenAI → Anthropic → local (→ claude_cli if unreserved) | 429/503/quota, marks provider exhausted |
| `scoring/deterministic_fallback` | qwen3:1.7b → qwen3:8b | ambiguous title regex |
| `scorer._try_quota_cooldown_fallback` | cloud → local hybrid | quota-cooldown error text |
| `apply/launcher.run_job` | sonnet → haiku (downgrade) | fresh `successful_paths` memo |
| `local_tailor` editor | one fixed local model | none (no escalation) |

They differ in **direction** (escalate for quality vs. downgrade for cost)
and in **trigger** (an error vs. a property of the input), but each is
"an ordered list of options plus a rule for when to try the next one."

### Proposed shape

```python
@dataclass
class Rung:
    name: str
    call: Callable[[Request], Result]
    cost: float          # relative, for ordering and reporting
    skip_if: Callable[[Request], bool] = lambda r: False

class Ladder:
    def __init__(self, rungs, *, accept: Callable[[Result], bool], on_error: Callable[[Exception], bool]): ...
    def run(self, req) -> Result  # tries rungs in order; records which rung answered
```

Migrate in this order, each its own PR with existing tests unchanged:
1. Local tailor editor: 1.7b → 8b when the 1.7b output fails the safety
   checks (the FW29 cross-pollination idea; real quality win).
2. Deterministic scorer escalation (pure refactor).
3. Leave `LLMClient` alone until 1–2 prove the shape; it carries
   persistence and exhaustion bookkeeping the others don't need.

**Value check:** this is worth doing only together with item 1 (a real
behavior gain). As a pure refactor it is Q4.
