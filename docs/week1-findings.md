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
