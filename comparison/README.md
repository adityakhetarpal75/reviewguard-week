\# ReviewGuard — Building the Same Agent Four Ways



\## What is ReviewGuard



ReviewGuard is a fake review detection system. It takes a product 

review and a reviewer ID as input and returns one of three verdicts: 

fake, genuine, or suspicious (flagged for human review).



To make the decision it uses two signals:

1\. Similar reviews retrieved from a corpus of 200 labeled Amazon reviews

2\. The reviewer's behavioral profile — account age, review volume, 

&#x20;  average rating, 5-star percentage



Text alone is insufficient for fake review detection. This was the 

central finding of my earlier research on the Yelp and Amazon datasets. 

Behavioral signals are essential. ReviewGuard was built to test that 

claim in an agentic setting.



\---



\## The experiment



I built the same agent four times using four different frameworks. 

Same problem. Same data. Same 50 eval reviews. Same metrics. Only 

the framework changed. This isolates the framework as the variable 

and makes the comparison honest.



\*\*Data:\*\* Amazon fake reviews dataset (40,432 reviews). Sampled 200 

for the retrieval corpus and 50 for evaluation. Labels: fake (CG) 

and genuine (OR). Synthesized reviewer behavioral profiles that 

probabilistically correlate with the label.



\*\*Retriever:\*\* built from scratch using sentence-transformers 

(all-MiniLM-L6-v2), NumPy cosine similarity, and top-k selection. 

No vector database. The entire retrieval system is \~120 lines of 

Python. Same retriever used across all four implementations.



\*\*Eval metrics:\*\* precision, recall, F1 per label. Binary F1 

treating suspicious as caught. Latency p50 and p95. Parse error 

rate. HIL escalation rate.



\---



\## Results



| Framework | Latency p50 | Binary F1 | Parse Errors | HIL Rate |

|---|---|---|---|---|

| Raw SDK (Day 1+2) | 13.6s | 0.821 | 1 | 26.5% |

| LangGraph (Day 3) | 3.7s | 0.741 | 0 | 10.0% |

| CrewAI (Day 4) | 27.1s | 0.750 | 9 | 19.5% |

| Pydantic AI (Day 5) | pending | pending | 0 guaranteed | pending |



\---



\## Day 1 and Day 2 — Raw Anthropic SDK



\### What it does

A hand-rolled while loop. Claude decides which tools to call and 

in what order. The loop dispatches tool calls, sends results back, 

and continues until Claude returns a final verdict. Claude is called 

3-4 times per review.



Day 1 used mock data. Day 2 replaced the mock retriever with real 

vector search and replaced mock reviewer profiles with real 

synthesized profiles from the Amazon dataset. The loop did not 

change — only the tool internals changed.



\### Key insight

The agent loop is just a while loop with tool dispatch. Every 

framework this week is a more structured version of this loop. 

Building it from scratch first made every subsequent framework 

immediately understandable.



Semantic similarity alone cannot distinguish fake from genuine in 

the Amazon CG dataset. A query for "Best product ever amazing buy 

now" returned both fake and genuine reviews. Behavioral signals 

carried most of the discriminative weight. This confirms the 

central finding of the underlying research.



\### When to use

Simple agents where you want full control and minimal dependencies. 

Learning and prototyping. Any task where the loop logic is 

straightforward enough that a framework adds complexity without 

benefit.



\### When not to use

Production systems where latency matters. 13.6s per review is too 

slow for real-time moderation. Systems where you need parallel 

execution or guaranteed structured output.



\---



\## Day 3 — LangGraph



\### What it does

The same agent rebuilt as a directed graph. Four nodes: retrieve, 

history, decide, escalate. Retrieve and history both connect from 

START — LangGraph runs them in parallel automatically. Decide waits 

for both to finish. Escalate runs conditionally if confidence is 

below 0.7.



Claude is called exactly once per review, inside the decide node. 

The parallel execution of retrieve and history is what cut latency 

from 13.6s to 3.7s.



\### Key insight

Parallelizing two independent steps with four lines of graph 

configuration — no threading code, no async/await — cut latency 

73%. In a system processing 10,000 reviews per day that difference 

is 34 hours of processing time saved daily.



The accuracy drop from 0.821 to 0.741 is a direct consequence of 

Claude getting one shot instead of multiple turns. The multi-turn 

reasoning of Day 2 built up evidence gradually. The single-shot 

approach of Day 3 assembles all evidence upfront and decides once. 

One-shot is faster but slightly less thorough.



\### When to use

Production systems where latency is critical. Workflows with 

independent parallel steps. Complex conditional routing between 

steps. Long-running agents that need checkpointing. Any system 

where you need explicit control over the agent's structure.



\### When not to use

Simple single-step agents where the graph adds overhead without 

benefit. Rapid prototyping where you need something running in 

minutes not hours.



\---



\## Day 4 — CrewAI



\### What it does

Three specialist agents: Retriever Agent, Behavioral Analyst, 

Moderator. Each has a role, goal, and backstory. The Retriever 

Agent finds similar reviews. The Analyst checks behavioral signals. 

The Moderator reads both reports and makes the final decision. 

Each agent makes its own Claude API call — three calls per review.



\### Key insight

CrewAI is the fastest framework to build with. Describing three 

job roles took less code than any other implementation. But the 

cost was significant: 27.1s latency (three sequential Claude calls) 

and 9 parse errors out of 50 reviews because agents respond in 

natural language and structured JSON extraction is unreliable.



The 9 parse errors are the most important finding of Day 4. They 

reveal CrewAI's fundamental limitation for production use: when 

agents communicate in natural language, getting reliably structured 

output back requires additional work that CrewAI does not provide 

out of the box.



\### When to use

Rapid prototyping of multi-agent collaboration. Exploring whether 

specialist agents improve quality before committing to a full 

implementation. Demos and stakeholder presentations. Any situation 

where you need a working multi-agent system in an afternoon.



\### When not to use

Production systems. 27.1s latency and 9/50 parse errors are both 

unacceptable in a system processing real traffic. Rebuild in 

LangGraph with Pydantic AI output once you have validated the 

approach.



\---



\## Day 5 — Pydantic AI



\### What it does

A single agent with a defined output schema. The output must be a 

ReviewVerdict object with four fields: label (one of three exact 

strings), confidence (float between 0 and 1), reasoning (string), 

flagged\_for\_human (boolean). Pydantic AI validates Claude's response 

against this schema and retries automatically if it does not match.



Claude is called once per review. All JSON parsing code is 

eliminated. The output is a typed Python object, not a string or 

dict.



\### Key insight

Every previous day had a JSON parsing block that could fail. Day 4 

proved this failure is real and frequent under certain conditions. 

Pydantic AI eliminates the failure mode entirely by making the 

framework responsible for output validation rather than the 

application code.



The elimination of parsing code also makes the codebase 

significantly cleaner. The entire "extract verdict from Claude's 

response" logic, which was 15-20 lines in every other day, becomes 

two lines: call the agent, access result.output.



\### When to use

Any production agent where output feeds a database or downstream 

system. Systems where a single parse error breaks the pipeline. 

Any situation where you need type-safe guaranteed output. Combine 

with LangGraph for the best production architecture.



\### When not to use

Agents that return free-form text like summaries or drafts. Simple 

scripts where output format does not matter.



\---



\## Framework recommendation matrix



| Need | Best choice |

|---|---|

| Learning how agents work | Raw SDK |

| Speed critical, parallel steps | LangGraph |

| Multiple specialists, quick prototype | CrewAI |

| Guaranteed structured output | Pydantic AI |

| Production fake review system | LangGraph + Pydantic AI |



\---



\## What I would build for production ReviewGuard



LangGraph for the agent structure with Pydantic AI output from 

the decide node.



LangGraph gives parallel execution of retrieve and history nodes 

(3.7s latency), conditional routing to escalation, typed state 

inspectable at every step, and checkpointing for reliability.



Pydantic AI gives a guaranteed ReviewVerdict object from the decide 

node with zero parse errors, type-safe fields, and no parsing code.



Together: the speed of Day 3 and the reliability of Day 5.



\---



\## What I built vs what a production system needs



This implementation is a proof of concept. A production system 

would additionally need:



\- A real vector database (Chroma, Qdrant, or Pinecone) instead 

&#x20; of NumPy files for scale beyond 10,000 reviews

\- Real reviewer behavioral data from platform logs instead of 

&#x20; synthesized profiles

\- A human review queue that the HIL escalations actually write to

\- Observability with LangSmith or Langfuse to monitor agent 

&#x20; decisions in production

\- A continuous eval pipeline that runs the 50-review eval set 

&#x20; automatically on every code change



\---



\## Repository structure



\- day1-no-framework/ — raw SDK, mock data, hand-rolled loop

\- day3-langgraph/ — LangGraph, parallel nodes, 3.7s latency

\- day4-crewai/ — CrewAI, three specialist agents

\- day5-pydantic/ — Pydantic AI, guaranteed structured output

