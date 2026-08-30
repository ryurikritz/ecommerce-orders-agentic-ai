# Kartify Support Agent — Help Document

**Scope of this document.** Everything in Parts 1–4 is derived from the source modules (`agent.py`, `models.py`, `repository.py`, `evaluation.py`, `feedback.py`, `display.py`, `__init__.py`) and the `agent_architecture.svg` diagram. Where the code and the diagram disagree, this document says so. Part 5 onward is forward-looking and every item there is tagged with a confidence level. Nothing below describes a capability that does not exist in the code unless it is explicitly marked as a proposal.

---

## Part 1 — The problem

### 1.1 What breaks when you point a language model at a customer's order data

The naive build is: give an LLM the order database and let it answer. Four things go wrong, and all four are visible in production incidents rather than in demos.

**It answers about the wrong record.** A customer asks "where's my parcel" and has five open orders. A model that guesses is right 20% of the time and confidently wrong the rest. There is no error message — just a plausible answer about someone's headphones when they asked about a monitor.

**It answers about someone else's record.** The customer identifier arrives in free text. If the retrieval step trusts what the message says rather than what the session establishes, "show me order ORD1001, I'm customer 1" becomes an access-control bypass expressed in English.

**It invents the parts it doesn't have.** Asked for a delivery date on an order that has none, a fluent model supplies one. The answer reads exactly like a correct answer. This is the failure mode with the worst blast radius, because nothing downstream can detect it.

**It acts when it should only assess.** "Cancel my order" is a consequential instruction. An agent with a write path and a persuasive customer will eventually execute a cancellation that a human would have refused, and there is no undo.

### 1.2 What this system does about it

The Kartify agent is a **governed** support agent: a fixed sequence of nodes, each with one job, where the language model is an optional component of exactly one stage and has no access to data, tools, or the final wording of any answer.

Four properties are enforced structurally rather than by prompt instruction:

| Property | How it is enforced | Where |
| --- | --- | --- |
| Cannot write | SQLite connections open with `mode=ro`; the driver refuses writes | `repository.connect_read_only` |
| Cannot read another customer's order | Ownership is checked *before* retrieval runs | `authorize_node` → `retrieve_node` |
| Cannot invent facts | Response text is composed from retrieved fields by templated branches, never generated | `respond_node` |
| Cannot take consequential action | Returns and cancellations terminate in `human_handoff` with `write_executed: False` | `policy_node` |

The system is a demonstration that these four properties can hold **while** the conversation still feels natural — pronouns resolve, follow-ups work, clarifications resume where they left off.

### 1.3 Who this document is for

- Someone reading the architecture diagram who needs to know what each numbered box actually does.
- Someone about to extend the system who needs to know which guarantees they might accidentally break.
- Someone evaluating the work who needs to know what is real, what is demo scaffolding, and what would be required for production.

---

## Part 2 — Reading the architecture diagram

The diagram shows nine numbered stages plus five supporting elements. Here is what each maps to in code.

### 2.1 The numbered pipeline

| # | Diagram label | Code | What it actually decides |
| --- | --- | --- | --- |
| 1 | Input guardrail | `guardrail_node` | Is this input safe and non-empty? |
| 2 | Understanding selector | *(a branch inside `understand_node`)* | Rules, or a configured model? |
| 3 | Typed understanding | `understand_node` | Which of seven intents; any explicitly stated IDs |
| 4 | Context resolver | `resolve_context_node` | Whose session is this, and which order do they mean? |
| 5 | Authorization gate | `authorize_node` | Does this customer own this order? |
| 6 | Governed tool router | `retrieve_node` | Load the record — or skip safely if denied |
| 7 | Policy engine | `policy_node` | Return window, cancellation boundary, product match |
| 8 | Grounded response | `respond_node` | Compose the answer from retrieved fields |
| 9 | Response critic | `evaluate_node` | Score the turn on four independent signals |

### 2.2 Three places the diagram and the code differ

These are worth knowing before anyone presents the diagram as documentation.

**The diagram shows nine stages; the graph has eight nodes.** Stage 2, the "Understanding selector," is not a separate node. It is a conditional inside `understand_node` (`if mode != DETERMINISTIC_MODE and not (bare_order_selection and pending_intent)`). Drawing it as its own box is defensible for explanation, but a reader tracing the code will not find a `selector` node.

**"Orders service" and "Product evidence" are not two services.** The diagram shows the tool router fanning out to both, and both feeding the policy engine. In code there is one function, `get_order`, which returns a single aggregate with product evidence joined in. There is no separate products call and no fan-in.

**One arrow is doing two jobs.** The `context → respond` edge is labelled "missing slot." There is a second, distinct reason that edge is taken: `direct_response`, set for `general_help` and `end_conversation`, which need no record at all. Two different journeys, one arrow.

### 2.3 Supporting elements

- **Session memory** (`SupportSession`) holds four bounded slots: customer, pending task, active order, active product. Not a transcript — four values. This bounds both prompt size on the LLM path and how much any single turn can expose.
- **Optional structured LLM** is bound to a strict JSON schema and receives only those four context slots plus the current message. It returns an intent and, at most, IDs the customer stated explicitly.
- **Customer feedback** (rating, resolved flag, comment) is collected separately and persisted to SQLite.
- **Quality analytics** combines the automated critic score with the customer rating. See §5.2 — this element is partially implemented.

---

## Part 3 — Workflow logic

### 3.1 The turn lifecycle

Every turn runs the same eight nodes in the same order. There are exactly two places the pipeline can short-circuit, and both route to `respond`.

```
START
  │
  ▼
guardrail ──[blocked / empty]──────────────────┐
  │ safe                                        │
  ▼                                             │
understand   (selector: rules or model)         │
  │                                             │
  ▼                                             │
context ─────[missing slot OR no data needed]──┤
  │ context ready                               │
  ▼                                             │
authorize                                       │
  │ (unconditional edge — denial handled inside)│
  ▼                                             │
tools        (safe skip if unauthorized)        │
  │                                             │
  ▼                                             │
policy                                          │
  │                                             │
  ▼                                             │
respond ◄───────────────────────────────────────┘
  │
  ▼
evaluate
  │
  ▼
END
```

**The unconditional `authorize → tools` edge is deliberate.** A denied turn still executes the tools node, which performs no read and writes explicit empties. The trace therefore shows the same node sequence whether or not access was granted, so a denial is auditable rather than invisible.

### 3.2 The five decisions that shape a turn

**Decision 1 — Is the input safe?** A substring denylist covering SQL manipulation and prompt-injection phrasings. Empty input is treated as a clarification, not a block; the distinct `error_code` keeps the analytics honest. *(See §5.4 — this control is weaker than it appears.)*

**Decision 2 — Rules or model?** Determined by the selected mode and by whether credentials exist. The model is skipped entirely when the turn is a bare order ID answering a pending clarification, because the answer is already known. Any failure — no key, rate limit, timeout, malformed response — falls back to the deterministic classifier. The fallback is recorded in state, so "the rules decided" and "the model failed and the rules decided" are distinguishable in the analytics.

**Decision 3 — Which order?** A five-step ladder, strongest evidence first:

1. An order ID stated in this message
2. The words "latest" / "most recent" / "newest"
3. The order carried over from the previous turn
4. A status question with no reference → the newest order
5. The customer has exactly one order

If none apply, the agent presents up to four candidates and parks the intent. The chosen route is recorded in `order_reference_source`, which is the field that answers a customer disputing which order was discussed.

> **Asymmetry worth understanding.** Step 4 applies to `order_status` only. Returns and cancellations deliberately fall through to clarification rather than defaulting, because guessing the target of a consequential action is not acceptable. The `Ambiguous return` benchmark scenario exists specifically to pin this in place. *(See §5.4 — the trace message on the clarification path overstates this.)*

**Decision 4 — Does this customer own this order?** A single `SELECT 1` — existence only, no columns. The result is identical whether the order belongs to someone else or does not exist, so a denial reveals nothing about what exists.

**Decision 5 — What does policy permit?**

- *Return:* order age at the frozen policy date vs. the product's return window, excluding `Cancelled` and `Returned` statuses. When no single product resolves, the shortest window across all items is used — conservative by construction.
- *Cancellation:* eligible only while status is `Processing`.
- Both produce a **request for human approval**, never an action.

### 3.3 How conversation memory works

Memory is updated in `SupportSession.ask` after the graph returns, not inside a node. Three rules govern it:

- **Order carries forward** until a new one is resolved.
- **Product resets when the order changes.** Without this, "is it covered?" could answer about an item from a previous order.
- **Pending intent is cleared** once an order resolves or the turn reaches a terminal outcome, so a stale pending task cannot hijack a later bare order ID.

The clarification-continuation loop is the most important behaviour in the system and the hardest to get right:

```
Turn 1  "which products are in my order?"   → no order resolvable
                                            → park intent=product_help
                                            → offer 4 candidates
Turn 2  "ORD1009"                           → bare ID, pending intent exists
                                            → skip the model entirely
                                            → resume product_help on ORD1009
Turn 3  "what warranty does it have?"       → "it" resolves to the matched product
Turn 4  "where is it now?"                  → intent changes, order does NOT reset
```

---

## Part 4 — The final answer flow

### 4.1 Six terminal outcomes

Every turn ends in exactly one. Only one of the six means "we answered from data."

| Outcome | Reached when | Response character |
| --- | --- | --- |
| `resolved` | Evidence retrieved and rendered | The answer |
| `clarification` | A required slot was missing | A question, with candidates |
| `denied` | Ownership check failed | Refusal that reveals nothing |
| `blocked` | Guardrail matched | Boundary stated, no detail |
| `human_handoff` | Consequential intent assessed | Assessment + explicit no-change statement |
| `ended` | Customer closed | Thanks + feedback prompt |

### 4.2 How each answer is composed

**No branch of `respond_node` generates prose.** Every response is either a fixed string or an f-string interpolating fields that a prior node retrieved. This is why hallucination is structurally impossible rather than merely unlikely — even in LLM mode, the model influenced only the intent label, never a word of the output.

The three-way delivery-date handling in the status branch is the clearest example:

- Date on record → state it
- No date, order is `Cancelled` or `Returned` → explain why there is none
- No date, order is active → *"A delivery date has not yet been assigned; I will not invent one."*

The third case is where a generative agent fabricates. Here it is a branch.

**Consequential answers are hedged by construction.** A return says *"appears eligible to request"* and shows its working — age, window, scope — because the agent cannot inspect item condition. A cancellation always ends by stating that no order change has been made, on both the eligible and ineligible paths.

### 4.3 What happens after the answer

1. **Critic** scores four signals: access control held, the answer was grounded, policy was evaluated where required, and the trace contains every node the turn should have visited. Unweighted mean, so the score decomposes cleanly.
2. **Memory** updates per §3.3.
3. **Trace** returns to the channel alongside the answer — one row per node with decision, detail, timing, and a field-level record of what data was touched.
4. **Customer rating** is collected independently at conversation close.

**The two scores are deliberately independent.** An automated score of 100% on a conversation the customer rated 2/5 is the single most informative signal the system can produce, and it is only visible because neither measure can influence the other.

---

## Part 5 — What the diagram does not capture

Everything in this part is grounded in the code as it stands. Items are ordered by how much they matter for taking this to a final version.

### 5.1 There is no downstream from `human_handoff` — [Certain]

The diagram shows "handoff" as one of four outputs of stage 8. Nothing consumes it. `human_handoff` is a string in the outcome field; no ticket is created, no queue is written to, no specialist is notified, and the "cancellation request prepared for human approval" is not prepared anywhere — it exists only in the sentence shown to the customer.

This is the largest gap between what the architecture claims and what the code does. A customer told a specialist will confirm their return has been told something the system cannot deliver on.

### 5.2 Automated quality signals are computed but never stored — [Certain]

The diagram shows `critic → analytics` labelled "automated signals," and analytics as a persistent store. `FeedbackStore` persists the **customer** side (rating, resolved, comment, turns, duration, intent path). Nothing persists the **automated** side. The `quality` dict lives for the duration of the turn and is discarded.

Consequence: the comparison described in §4.3 — automated score against customer rating — cannot currently be performed on real traffic. It is the system's most valuable metric and there is no table to compute it from.

Related: `intents` is written to the feedback table as a comma-joined string and never split on read. The column is write-only.

### 5.3 Traces are ephemeral — [Certain]

The trace is the system's central explainability claim. It is returned to the caller and rendered in the notebook, then discarded. There is no trace store, no correlation ID index, and no way to answer "show me the reasoning for the conversation this customer is complaining about" after the session ends.

### 5.4 Two safety controls are weaker than they present — [Certain]

**The guardrail is a substring denylist.** `"ignore previous instructions"` is caught; `"disregard prior instructions"` is not. It is a reasonable demo control and a poor security boundary. The real protection is architectural — no write path, ownership checked before retrieval — and that is the stronger claim to make.

**The clarification trace overstates the guarantee.** `resolve_context_node` emits *"the agent will not guess"* on the clarification path, while the `helpful_latest_default` branch in the same function silently picks the newest order for status questions. Both behaviours are defensible; the trace message describing them is not accurate for both.

### 5.5 `get_order` is the one unscoped read — [Certain]

`list_customer_orders` and `customer_owns_order` filter by customer in SQL and cannot leak another customer's data regardless of caller. `get_order` takes an order ID alone and returns the full aggregate — including `customer_id`, `customer_name`, `shipping_address`, and `payment_method`.

Its safety is an **orchestration** guarantee (the caller checks ownership first), not a data-layer one. The diagram shows authorization gating the tools; the tool itself is ungated. A reordered edge or a new caller loses the guarantee with no second line of defence.

### 5.6 Resource lifecycle — [Certain]

`with connect_read_only(...) as connection:` manages the *transaction*, not the connection. sqlite3's context manager never closes. Eleven call sites across `repository.py` and `feedback.py` leave handles open. On a read-only connection the transaction management does nothing useful, so the `with` provides no benefit while implying cleanup that is not happening.

### 5.7 The evaluation never exercises the LLM path — [Certain]

`run_benchmark` constructs `SupportSession(customer_id)` with no `mode`, so every scenario runs deterministic. The component most likely to regress — model classification — is never tested. Reported `intent_accuracy` measures a regex cascade against labels written for that cascade.

Related measurement issues: `p95_latency_ms` over ~14 turns interpolates near the maximum and is not meaningfully a percentile; the `access_control` column blends two different definitions (self-reported on scenario rows, structurally verified on probe rows).

### 5.8 Nothing survives a restart — [Certain]

`SupportSession` is in-memory. `FeedbackStore` defaults to a fixed filename in the OS temp directory, which is both ephemeral **and shared** — on a multi-visitor deployment every session reads and writes the same table, so one visitor's ratings move the quality band another visitor sees.

### 5.9 Smaller items — [Certain]

- `POLICY_AS_OF` is a hardcoded date. Reproducible for the demo; silently wrong against live data.
- `get_customer` selects `email`; only `name` is used. `get_order` selects address and payment method; neither is rendered. Both enter agent state as unused PII.
- `general_help` returns a fixed capability string. There is no knowledge base, so any question outside the six intents gets the same sentence.
- `confusion_matrix` is defined in `evaluation.py` but not re-exported from the package.
- `import kartify` compiles the LangGraph and loads pandas eagerly; a failure in any subsystem breaks the whole package surface.
- There is no authentication. Identity comes from a dropdown.
- There is no output-side validation on the LLM path — mitigated by the fact that the model never writes response text.

---

## Part 6 — Route to a final version

Five phases, ordered so each depends only on the ones before it. Confidence tags reflect how certain the *approach* is, not whether you will choose to do it.

### Phase 1 — Make the existing guarantees structural (days)

Nothing new; close the gap between what the system claims and what it enforces.

| Change | Why | Confidence |
| --- | --- | --- |
| Wrap `_connect()` in `contextlib.closing` at all 11 sites | Removes the handle leak; the `with` then means what it appears to mean | [Certain] |
| Add optional `customer_id` to `get_order` with an `AND o.customer_id = ?` predicate | Makes scoping structural, matching the other reads; defence in depth instead of defence in ordering | [Certain] |
| Narrow the SELECT lists to columns actually rendered | Removes unused PII from agent state and the trace panel | [Certain] |
| Add `Field(pattern=r"^ORD\d{4}$")` to `Classification.order_id` | A fabricated ID becomes a validation error rather than a denied lookup | [Likely] |
| Add a minimum-n floor to the quality band in `metrics()` | Currently one self-submitted 5-star rating produces "Strong" | [Certain] |
| Promote the `database_summary` table tuple to a named module constant | Makes the injection allowlist self-evident to a reviewer | [Likely] |
| Correct the "will not guess" trace message, or gate `helpful_latest_default` | The trace must be accurate to be evidence | [Certain] |

**Outcome:** every claim in Part 1's guarantee table survives a hostile read of the code.

### Phase 2 — Close the analytics loop (1–2 weeks)

The system produces its most valuable signal and throws it away.

1. **Persist automated quality per turn.** A `turn_quality` table keyed on `(conversation_id, turn_number)` holding the four signals, the score, the intent, the outcome, the understanding provider, and the fallback flag. Written from `SupportSession.ask`, which already has all of it in `self.history`.
2. **Persist traces.** Same key, one row per node. This is what makes "show me the reasoning for this complaint" answerable after the fact. [Likely] JSON-per-turn is sufficient at demo scale; a row-per-node table is better for querying node-level latency.
3. **Join the two scorecards.** With both persisted, the divergence metric becomes a query: conversations where the automated score was high and the customer rating was low. That is the analysis worth putting in a results chapter.
4. **Make `intents` readable.** Split on read, or normalise to a child table. Per-intent satisfaction is the most interesting cut available in this data.

**Outcome:** the "Quality analytics" box in the diagram becomes real, and the automated-vs-human comparison becomes a chart rather than an aspiration.

### Phase 3 — Make the evaluation prove something (1–2 weeks)

1. **Parameterise `run_benchmark(mode=...)`** and run the suite across deterministic, Groq, and OpenAI. Report the delta. [Certain] — this is the single highest-value change in the whole roadmap, because it converts a regression test into an actual evaluation and produces the comparison table any reviewer will ask for.
2. **Expand the suite** enough that the rates are not moved several points by one turn. Prioritise paraphrases the rules were not written against — that is where deterministic and LLM modes will actually diverge.
3. **Add an adversarial set:** injection phrasings the denylist misses, cross-customer attempts in varied wording, consequential requests with ambiguous targets, requests for data that does not exist.
4. **Separate the safety metric** from the labelled-accuracy metric, and use one consistent definition of `access_control`.
5. **Report `max` as `max`,** or gather enough turns for a percentile to mean something.
6. **Add a coupling test:** assert every `Intent` member appears in the LLM system prompt. The vocabulary is currently duplicated in three uncoupled places and a new member fails silently.

**Outcome:** you can state what the LLM path costs and buys, with numbers.

### Phase 4 — Complete the handoff (2–4 weeks)

The gap in §5.1 is where "governed agent" stops being true.

1. **A handoff artifact.** When `policy_node` sets `handoff: True`, persist a request record: conversation ID, customer, order, intent, the policy assessment with all inputs, the trace, and a status of `pending`. This is the thing the response sentence currently promises.
2. **A specialist view** listing pending requests with the agent's assessment and reasoning visible. [Guessing] on the interface — a Streamlit page is adequate for demonstration; a real deployment routes into whatever ticketing system already exists.
3. **An approval path.** A specialist approves or rejects. Only *then* does a write occur, executed by the specialist's credentials rather than the agent's. The agent's read-only connection stays read-only — that property should never be relaxed.
4. **Close the loop to the customer.** Notify on resolution, and record whether the specialist agreed with the agent's assessment. That agreement rate is a genuine measure of policy-engine quality and it does not exist today.

**Outcome:** `human_handoff` becomes a state in a workflow rather than a word in a sentence.

### Phase 5 — Production shape (scope depends on target)

Grouped rather than sequenced; pick by deployment context.

**Identity and data**
- Replace the demo picker with real session authentication. The `claimed_customer_id` / `customer_id` split already in `models.py` is exactly the right shape for this — the impersonation check works unchanged once identity arrives from a verified token. [Certain]
- Move off SQLite to whatever the order system actually is. `repository.py` is the only module that would change; that isolation is why it exists. [Certain]
- Replace `POLICY_AS_OF` with the real clock, keeping the override for reproducible tests. [Certain]

**Observability**
- Export traces to OpenTelemetry or LangSmith rather than a local table. [Likely]
- Redact PII from stored traces. The `data_used` field already records *which* fields were touched, which is most of the work. [Likely]
- Alert on fallback rate: a rising `understanding_fallback` means the model path is degrading silently.

**Safety**
- Replace the substring denylist with layered controls: input classification, output validation, and rate limiting per customer. [Likely]
- Keep the architectural controls as primary. They are what actually holds.

**Capability**
- Retrieval over policy documents for `general_help`, so out-of-scope questions get an answer instead of a capability list. This is the natural place for RAG in this architecture — and note it belongs *outside* the governed order path, reading documents rather than customer records. [Likely]
- Channel adapters (email, voice) reusing the same graph. The graph is already channel-agnostic; only the presentation layer differs. [Likely]

---

## Part 7 — Verification checklist

Before presenting or submitting, confirm each of these yourself rather than taking this document's word for it.

**Correctness**
- [ ] `pytest` passes; note which paths have no coverage at all
- [ ] `mypy` or `pyright` runs clean over the package — the `total=False` TypedDict means the type checker is your only structural guarantee, and it is currently unverified
- [ ] `run_benchmark()` executes end to end and every row's `passed` column is explicable

**Claims you will be asked to defend**
- [ ] "Read-only" — show `mode=ro` and demonstrate a refused write
- [ ] "Authorization before retrieval" — show the node order and the safe-skip branch
- [ ] "No hallucination" — show that every `respond_node` branch is a template
- [ ] "Trace is complete" — show a trace for a denied turn and for a blocked turn
- [ ] "The LLM cannot escalate" — show that it returns only an intent label and stated IDs

**Known weaknesses to state before you are asked**
- [ ] `get_order` scoping is enforced by the caller (§5.5)
- [ ] The guardrail is a denylist (§5.4)
- [ ] The benchmark is deterministic-mode only (§5.7)
- [ ] `human_handoff` has no downstream (§5.1)

**Environment**
- [ ] The Mermaid architecture cell renders as a diagram in the exact environment you will present from — it degrades to raw text in Colab and some VS Code configurations
- [ ] `ARCHITECTURE_MERMAID` matches what `build_graph()` compiles; it is hand-maintained and can drift

---

## Appendix A — Node reference

| Node | Reads | Writes | Can short-circuit |
| --- | --- | --- | --- |
| `guardrail` | `query` | `blocked`, `error`, `error_code` | Yes → `respond` |
| `understand` | `query`, memory slots | `intent`, `order_id`, `claimed_customer_id`, `understanding_*` | No |
| `context` | identity, memory, owned orders | `customer_id`, `order_id`, `order_reference_source`, `candidate_orders`, `pending_intent` | Yes → `respond` |
| `authorize` | `customer_id`, `order_id` | `authorized` | No |
| `tools` | `order_id`, `customer_id` | `order`, `orders` | No (safe-skips instead) |
| `policy` | `order`, `query`, `active_product_name` | `policy`, `matched_product`, `handoff`, `write_executed` | No |
| `respond` | everything above | `response`, `outcome` | Terminal |
| `evaluate` | `trace` and all flags | `quality` | Terminal |

## Appendix B — Intent and outcome vocabularies

**Intents** (`order_status`, `order_list`, `product_help`, `return_help`, `cancel_request`, `end_conversation`, `general_help`) are defined once in `models.py` but consumed in three uncoupled places: the branch order of `_deterministic_intent`, the LLM system prompt text, and the branch order of `respond_node`. Adding a member requires touching all three, and omitting one fails silently.

**Outcomes** (`resolved`, `clarification`, `denied`, `blocked`, `human_handoff`, `ended`) are the grouping axis for all quality analytics. Only `resolved` means "answered from evidence"; the other five are legitimate, non-failure endings, and a system that reported them as failures would be optimising toward exactly the behaviour this architecture exists to prevent.
