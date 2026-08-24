# Week 1 — Cache measurement gate

Every downstream number is unfalsifiable until cache behaviour is observably
controllable. These probes ran against `qwen/qwen3.7-flash` pinned to `alibaba`,
76 real MCP tools (~12.4K prompt tokens). Total spend: under $0.01.

## Q1 — Where can tool schemas live and still cache?

**The single biggest architectural question in the project, and the docs are silent on it.**
OpenRouter documents `cache_control` only for `text` blocks in `system`/`user`.

| Schemas carried in | `cache_write_tokens` reported | Hit rate (4 trials) | Works immediately |
|---|---|---|---|
| native `tools` field | **never** — 0 on every trial | 0/4 at 0s, 3/4 after 4s | no |
| `system` text block | 11,434 | **4/4 both arms** | yes |

The `tools` field receives only **best-effort implicit** caching: `cache_control` is
not honoured, no write is ever billed or reported, and reads appear only after a
delay and only sometimes. System text receives real explicit caching.

**Consequence: Layer B (hot set) must render as text inside the cached system block,
not via the native `tools` field.** The `tools` field then carries only a small
dispatcher so native tool-call output formatting is preserved.

## Q2 — Does a pinned provider hold one cache across a session?

Only if the layout from Q1 is used. Same model, same pin, same corpus:

| Layout | Cold misses over 11 turns | Steady-state hit rate |
|---|---|---|
| native `tools` field | 3 / 11 | ~88% |
| cached `system` text | **0 / 11** | **98.5%** |

The sporadic zeroes were not provider drift — they were the `tools` field's
best-effort caching failing. With the correct layout the cache is stable.

## Q3 — Minimum cacheable prefix

| tokens | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| cached | no | no | **yes** | yes | yes | yes |

Floor is 1024 for Qwen, confirming the registry's previously unverified guess.
Below it, caching silently no-ops with no error.

## Q4 — Silent cache killer, reproduced on purpose

Identical tool set, identical semantics, order reversed:

| | cached tokens |
|---|---|
| byte-identical repeat | 11,136 |
| after reordering tools | **0** |

A pure byte-level change with no semantic content destroys the entire prefix.
This is why serialization is pinned in `hotset/layout/serialize.py`.

## Methodological note

The first pass of Q1 ran single-shot and reported MISS, contradicting Q4 in the
same run. Repeats showed the `tools` path is non-deterministic. **Every probe
here runs multiple trials; single-shot cache measurements are not trustworthy.**

---

# Week 2 — Layout measurement

## Q5 — Does the dispatcher actually route?

No, and that is better than the design intended. Given one tool named `call_tool`
in the native field and 76 tools described only in cached system text, the model
emits a native `tool_call` for `get_current_time` — **a name that appears nowhere
in the tools field** — with correct arguments.

The tools field acts as a **format primer**, not a router: its presence switches the
model into structured-call mode, and it then names whatever tool the *text* described.
So all 76 tools get native `tool_calls` while only one tool's bytes sit in the
uncached upstream field. `parse_call()` accepts both the direct and wrapped shapes.

## Q6 — Is the ephemeral tail really a pure suffix?

Yes, against a live provider. Four turns, tail injected only on turn 2:

| turn | tail | cached | uncached | tool called |
|---|---|---|---|---|
| 0 | no | 0 (write 3107) | 22 | `get_current_time` |
| 1 | no | 3107 | 78 | `convert_time` |
| 2 | **yes** | 3107 | 354 | `read_text_file` |
| 3 | no | 3107 | 212 | — |

`read_text_file` had only a Layer A index line, no schema, until the tail supplied
one; the model called it correctly. Dropping the tail on turn 3 left the prefix
byte-identical, so tail-loading costs nothing in cache terms.

## Q7 — Split breakpoints, and why the admission threshold is provider-dependent

Layer A is frozen; Layer B changes on every admission. Giving each its own
breakpoint should let admission re-write B alone. Whether it does is a **provider
capability**, not a property of the layout:

| | admission → cached | admission → write |
|---|---|---|
| Qwen, split | 0 | 6,616 |
| Haiku, no split | 0 | 7,420 |
| Haiku, **split** | **6,652** | **767** |

Alibaba silently ignores the second breakpoint. Anthropic honours it, and Layer A
survives admission.

Marginal cost of one admission on Haiku over a warm turn, against one tail-load
at $1.00/MTok:

| layout | marginal admission | break-even `n*` |
|---|---|---|
| single breakpoint | $0.008554 | **54.8 uses** |
| split breakpoints | $0.000902 | **5.8 uses** |

**A 9.5× shift in the admission threshold from a structural choice carrying no
semantic content.** `n*` therefore depends not only on a model's `u/c` ratio but on
how many cache segments its provider honours — recorded as
`ModelSpec.cache_breakpoints`.
