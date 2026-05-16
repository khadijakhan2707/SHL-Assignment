# Approach Document â€” SHL Conversational Assessment Recommender

**Candidate:** [Your Name] | **Role:** AI Intern, SHL Labs | **Model:** gemini-2.5-flash (Google Gemini API)

---

## 1. Problem Decomposition

The task has four interlocking sub-problems: (a) catalog ingestion and representation, (b) conversational agent design covering clarify/recommend/refine/compare behaviors, (c) a stateless FastAPI service that handles non-deterministic multi-turn dialogue, and (d) a principled evaluation framework. Each is addressed below.

---

## 2. Catalog Representation: Full Catalog in System Prompt

**Decision:** Inject the entire SHL product catalog as structured text directly into the Gemini system prompt on every request â€” no vector database, no retrieval step.

**Why not RAG?** The catalog fits comfortably in Gemini's large context window for this assignment-sized catalog. RAG would add retrieval latency (critical given the 30-second timeout), risk missed items from imperfect embedding similarity, and create a two-step failure mode. Full catalog injection gives the model complete visibility for every query, maximizing Recall@10.

**Catalog format:** Each product is serialized as a pipe-delimited line: `PRODUCT|name|type:code|levels:...|duration:...|langs:...|remote:...|adaptive:...|url:...|desc:...`. This format is dense, token-efficient, and easy for the model to pattern-match against.

**Caching:** The catalog JSON is downloaded once and committed to the repo as `catalog_cache.json`, so the server avoids a remote fetch on every cold start (important for Render.com's free tier).

---

## 3. Agent Design

The system prompt encodes six behavioral rules with explicit priority ordering:

| Rule | Trigger | Action |
|---|---|---|
| SCOPE GUARD | Off-topic / injection | Refuse and redirect |
| CLARIFY | Vague query (no role/use-case) | Ask ONE targeted question |
| RECOMMEND | Role + â‰¥1 signal present | Return 1â€“10 catalog items |
| REFINE | User adds/removes constraints | Update shortlist in-place |
| COMPARE | Explicit comparison request | Answer from catalog only |
| END | User confirms completion | Set `end_of_conversation: true` |

**Key design choices:**
- Rules are numbered and ordered to prevent ambiguity (e.g. scope guard fires before clarify).
- Role-specific heuristics are embedded in the prompt: technical roles get K+A+P; leadership gets OPQ32r+report products; volume screening gets simulation+personality; graduate cohorts get Verify G+ + Graduate Scenarios.
- The model is instructed to be honest about catalog gaps (e.g. no Rust-specific test exists) rather than hallucinating a substitute.

**Output format:** The model outputs JSON only â€” no markdown, no fences. This keeps parsing trivial and reduces format-compliance failures. The schema is repeated and annotated inside the prompt to minimize deviations.

---

## 4. Hallucination Guard (Post-Processing)

After every LLM call, a validation layer checks each recommended item:
1. **Exact URL match** against the catalog URL set â†’ keep, normalize name.
2. **Exact name match** (case-insensitive) â†’ correct URL from catalog.
3. **Partial name match** â†’ pick the longest overlapping catalog name.
4. **No match** â†’ drop silently (logged as a warning).

This means a hallucinated product never reaches the evaluator, regardless of how the model responds. URLs are always from the real catalog.

---

## 5. Evaluation Framework

The service exposes a `POST /evaluate` endpoint implementing Recall@K and three behavior metrics:

**Recall@10:** For each trace, the fraction of expected assessments that appear in the agent's final shortlist. Averaged across all traces: `Mean Recall@10 = (1/N) Î£ Recall@10_i`.

**Behavior metrics (each 0â€“1):**
- **Schema compliance:** All required fields (`reply`, `recommendations`, `end_of_conversation`) present and correctly typed.
- **URL validity:** Every returned URL exists in the catalog.
- **Turn limit compliance:** Conversation completed within 8 turns.

**Behavior probes (manual test suite):** `test_agent.py` runs 16 probe categories derived from the 10 sample conversations:
- Vague queries produce clarification, not recommendations
- Clear queries produce recommendations without unnecessary clarification
- Off-topic and prompt-injection attempts return empty recommendations
- Refinement correctly adds/removes items without restarting
- `end_of_conversation` is false until explicit user confirmation
- All returned URLs pass the SHL catalog pattern check

---

## 6. What Didn't Work and How I Iterated

**Vector search (FAISS):** Prototyped early. Embedding similarity occasionally retrieved products that matched the query surface form but were wrong for the role (e.g. returning a sales simulation when the query mentioned "communication skills"). Dropped in favor of full-catalog injection, which let the model reason holistically across all products.

**Streaming responses:** Considered for latency. Added complexity to JSON assembly without meaningful UX gain (the evaluator is automated). Reverted to single synchronous call.

**Prompt routing by type:** Tried pre-filtering the catalog to the relevant type (K for technical queries, P for leadership). Hurt recall for queries spanning multiple types (e.g. a senior IC role needing K + A + P together). Full catalog injection resolved this.

**Improvement measurement:** After each prompt iteration, I replayed all 10 sample conversations and counted: (a) turns to first recommendation, (b) number of recs matching the expected shortlist, (c) any hallucinated URLs. Each prompt version was compared against the previous on these three metrics before merging.

---

## 7. Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Gemini gemini-2.5-flash (Google Gemini API) | Strong instruction-following, large context window, free tier available |
| Framework | FastAPI + Pydantic | Fast, schema-validated, industry standard for ML APIs |
| HTTP client | httpx (async) | Native async for FastAPI; timeout control |
| Deployment | Render.com (free tier) | Free, GitHub-connected, no credit card for basic web services |
| Catalog retrieval | Full prompt injection | Maximizes recall, eliminates retrieval latency |

**AI tools used:** ChatGPT/Codex was used to help structure boilerplate and iterate on the system prompt. All design decisions and code logic were authored and reviewed by me.




