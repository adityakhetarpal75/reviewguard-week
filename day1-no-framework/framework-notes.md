# ReviewGuard — Framework Notes

## Day 1 — No framework (Anthropic SDK + tool use)

What surprised me:
- The agent loop is just a while-loop with tool dispatch. All "intelligence" lives in Claude.
- Tool descriptions ARE the prompt — they directly drive how reliably tools get called.

What was painful:
- Filename mismatch (sample_reviews.py vs sample_data.py) cost me time.
- Windows hides file extensions by default — bit me with .py.txt confusion.

What I'd use this approach for:
- Learning what frameworks abstract away.
- Cases where I want full control over every API call and message in context.

What I'd avoid it for:
- Anything beyond ~5 tools or stateful workflows — the loop gets messy.

Cost / time:
- API cost: ~$0.05 for 5 reviews on Opus
- Time: ~3 hours including setup
- Lines of code: ~250 across 3 files

---

## Day 2 — RAG + real data

What surprised me:
- HIL fired 3/5 times once data was real. Mock data hid this entirely.
- Agent disagreed with eval labels twice (E0000, E0001) — and arguably was right.
- E0004 caught a machine-generated review through coherence analysis, not keywords.
- Pure semantic similarity returned both fake AND genuine for spammy queries.
  Confirmed my paper's thesis: text alone is insufficient.

What was painful:
- Setup detours: Voyage rate limits → OpenAI billing → finally local embeddings.
- Partial-file edits ("replace the top of the file") broke things; full file safer.

What I'd use this approach for:
- Building real RAG when I need to understand retrieval, not just use it.
- Stakes high enough to warrant HIL escalation between conflicting signals.

What I'd avoid it for:
- Production at scale — local embeddings have model-load latency, no batch service.

Cost / time:
- API cost so far: ~$0.10 total (Anthropic only, local embeddings free)
- Time: ~6 hours across Day 1 + Day 2
- New code: retriever.py (~120 lines), prepare_data.py (~80 lines)

## Day 3 — LangGraph

What surprised me:
- Parallel nodes cut latency 73% (13.6s → 3.7s) with zero logic changes
- HIL rate dropped from 26.5% to 10% — less deliberation = more hard calls
- Typed state makes every node's output inspectable — huge for debugging

What was painful:
- langchain-anthropic doesn't auto-load .env like raw anthropic SDK does
- load_dotenv() must be called explicitly before ChatAnthropic is instantiated

What I'd use LangGraph for:
- Any agent where latency matters and tool calls are independent
- Workflows where you need checkpointing or human-in-the-loop mid-graph
- Complex state machines with conditional routing

What I'd avoid it for:
- Simple single-tool agents — the abstraction tax isn't worth it
- Teams not already in the LangChain ecosystem — steep learning curve

Latency: p50=3.7s (Day2: 13.6s) — 73% improvement from parallelism
Accuracy: binary F1=0.741 (Day2: 0.821) — tradeoff for speed
HIL rate: 10.0% (Day2: 26.5%)
Lines of code: ~150 across 4 files (Day2: ~350 across 6 files)

## Day 4 — CrewAI

What surprised me:
- Built a 3-agent system faster than any previous day
- 9 parse errors — CrewAI output is less structured than LangGraph
- Built-in execution traces out of the box — no setup needed
- Role/goal/backstory shapes agent behavior differently than system prompts

What was painful:
- Can't pass langchain ChatAnthropic directly — must use model name string
- Output format unreliable — moderator sometimes returned prose not JSON
- No easy way to enforce structured output like Pydantic AI will

What I'd use CrewAI for:
- Rapid prototyping of multi-agent collaboration
- When you want to try "what if I had 3 specialists" in an afternoon
- Internal tools where parse errors don't matter much

What I'd avoid it for:
- Production systems where structured output is required
- Latency-sensitive applications (27s per review)
- Any case where you need fine-grained control over what each agent sees

Latency: p50=27.1s (Day3: 3.7s) — slowest yet, 3 LLM calls per review
Accuracy: binary F1=0.750 (Day2: 0.821)
Parse errors: 9/50 — biggest reliability problem so far
HIL rate: 19.5%