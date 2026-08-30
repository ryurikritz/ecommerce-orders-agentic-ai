"""Context-aware LangGraph support agent with evaluation and session memory."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# This module defines a governed customer-support agent for a demo e-commerce
# domain ("Kartify"). It is built as a LangGraph state machine with a fixed,
# auditable node sequence:
#
#   guardrail -> understand -> context -> authorize -> tools -> policy
#             -> respond -> evaluate
#
# Design principles visible in the code below:
#   1. Fail closed. Every optional capability (LLM classification, provider
#      credentials) degrades to a deterministic path rather than erroring out.
#   2. Read-only by default. No node performs a write against the order store;
#      cancellations and returns are prepared as *requests* for human approval.
#   3. Authorization before retrieval. `authorize_node` runs before
#      `retrieve_node`, so an unauthorized turn never loads customer records
#      into state at all (object-level access control, not response filtering).
#   4. Explainability. Every node appends a TraceEvent recording the decision,
#      a human-readable detail, elapsed time, and which data fields it touched.
#   5. Reproducibility. Policy maths is anchored to a fixed date constant, not
#      to wall-clock "today", so demo results do not drift.
# =============================================================================

from __future__ import annotations

# --- Standard library -------------------------------------------------------
import os          # environment-driven configuration (API keys, model names)
import re          # all intent/slot extraction is regex-based, no NLP deps
import time        # perf_counter used for per-node latency measurement
import uuid        # conversation identifiers for the session façade
from datetime import date, datetime
from typing import Any

# --- Third-party ------------------------------------------------------------
from langgraph.graph import END, START, StateGraph  # the orchestration engine
from rapidfuzz import fuzz                          # fuzzy product-name matching

# --- Local ------------------------------------------------------------------
# `models` supplies the typed state contract; `repository` is the only module
# permitted to touch the data store, which keeps data access auditable.
from .models import AgentState, Classification, Intent, TraceEvent
from .repository import (
    customer_owns_order,
    get_customer,
    get_order,
    list_customer_orders,
)


# =============================================================================
# SECTION 1 — CONFIGURATION CONSTANTS
# =============================================================================

# Frozen "as of" date for all policy arithmetic (return windows, order age).
# Overridable via KARTIFY_POLICY_DATE so a demo or test run can pin a scenario.
# Deliberately NOT date.today(): a live clock would make return-eligibility
# assertions in the test suite start failing as the fixture data ages.
POLICY_AS_OF = date.fromisoformat(os.getenv("KARTIFY_POLICY_DATE", "2025-10-31"))

# Understanding-mode labels. These strings are shown in the UI selector AND used
# as routing keys in `_provider_configuration`, so they must stay in sync with
# whatever the Streamlit/notebook front end renders.
DETERMINISTIC_MODE = "Deterministic demo"          # rules only, zero network calls
GROQ_MODE = "Free LLM assisted: GPT OSS 20B"       # GroqCloud OpenAI-compatible endpoint
OPENAI_MODE = "OpenAI assisted"                    # OpenAI proper
LEGACY_LLM_MODE = "LLM-assisted"                   # older label, retained for back-compat

# Bumped whenever the persisted session shape changes, so stored sessions from
# an earlier build can be detected and discarded rather than silently misread.
SESSION_SCHEMA_VERSION = 3


def available_understanding_modes() -> list[str]:
    """Return only modes whose credentials are configured, without exposing secrets."""
    # The deterministic mode is always offered — it has no dependencies.
    modes = [DETERMINISTIC_MODE]
    # Presence checks only: the key value itself never leaves this function,
    # so the UI can render available options without handling secrets.
    if os.getenv("GROQ_API_KEY"):
        modes.append(GROQ_MODE)
    if os.getenv("OPENAI_API_KEY"):
        modes.append(OPENAI_MODE)
    return modes

# Documentation artefact only — rendered in the notebook/app to show the graph
# topology. It is hand-maintained, so it can drift from `build_graph()`; treat
# `build_graph()` as the source of truth if the two ever disagree.
ARCHITECTURE_MERMAID = """flowchart TD
    UI[Streamlit / Notebook] --> SESSION[Demo identity + conversation session]
    SESSION --> G[Input guardrail]
    G -->|safe| S[Understanding selector]
    G -->|blocked| R[Grounded response]
    S --> U[Rules or structured LLM classification]
    L[Optional language model: Groq GPT-OSS or OpenAI] --> U
    U --> C[Context resolver]
    C -->|needs one slot| R
    C --> A[Object-level authorization]
    A -->|denied| T[Tool router / safe skip]
    A -->|allowed| T
    T --> O[(Orders tool)]
    T --> P[(Products evidence)]
    T --> Y[Policy engine]
    O --> Y
    P --> Y
    Y --> R
    R --> E[Response critic + quality signals]
    E --> M[Session memory]
    E --> UI
    UI --> F[Customer rating + resolution feedback]
    F --> Q[(Quality analytics)]
"""


# =============================================================================
# SECTION 2 — TRACE AND SLOT-EXTRACTION HELPERS
# =============================================================================


def _event(
    node: str,
    decision: str,
    detail: str,
    *,
    started: float | None = None,
    data_used: list[str] | None = None,
) -> TraceEvent:
    # Single constructor for audit events so every node emits an identical shape.
    # `started` is a perf_counter() reading taken at node entry; passing it lets
    # this helper compute the elapsed time rather than each node duplicating it.
    # `data_used` is the explainability payload: which governed fields the node
    # actually read, which is what an auditor reviews after the fact.
    return {
        "node": node,
        "decision": decision,
        "detail": detail,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2)
        if started
        else 0.0,
        "data_used": data_used or [],
    }


def _extract_customer_id(query: str) -> int | None:
    # Matches "customer 42", "customer id: 42", "cid#42". Only fires on an
    # explicit keyword so a bare number in the text (a price, a quantity) is
    # never mistaken for an identity claim.
    match = re.search(r"(?:customer|customer\s*id|cid)\s*[:#-]?\s*(\d+)", query, re.I)
    return int(match.group(1)) if match else None


def _extract_order_id(query: str) -> str | None:
    # Canonical order format is ORD followed by exactly four digits. Separators
    # and spacing are tolerated on input ("ORD-1001", "ord #1001") but the
    # return value is always normalised to the canonical "ORD1001" form.
    match = re.search(r"\bORD\s*[-#]?\s*(\d{4})\b", query, re.I)
    return f"ORD{match.group(1)}" if match else None


# =============================================================================
# SECTION 3 — DETERMINISTIC INTENT CLASSIFIER
# =============================================================================


def _deterministic_intent(query: str, previous_intent: Intent | None) -> Intent:
    # Rule-based classifier used both as the default mode and as the fallback
    # whenever the LLM path is unavailable or fails.
    #
    # ORDER OF CHECKS IS SIGNIFICANT. The branches are arranged from most
    # consequential to least, because phrases overlap: "cancel my return"
    # contains both cancel and return vocabulary. Cancellation is evaluated
    # first so the more consequential intent wins ties.
    text = re.sub(r"\s+", " ", query.lower()).strip()

    # --- Closing signals ---------------------------------------------------
    if any(term in text for term in ("bye", "goodbye", "end conversation", "that's all", "thats all")):
        return "end_conversation"

    # --- Cancellation (highest consequence, checked first) -----------------
    # Three families of evidence: the explicit verb; "stop" within ~35 chars of
    # an order noun; and idiomatic pre-shipment phrasings. The bounded {0,35}
    # window is what keeps "stop" from matching across an unrelated clause.
    if (
        re.search(r"\b(cancel|cancellation)\b", text)
        or re.search(r"\bstop\b.{0,35}\b(order|purchase)\b", text)
        or re.search(r"\b(order|purchase)\b.{0,35}\b(stop|cancelled|canceled)\b", text)
        or any(
            term in text
            for term in (
                "do not ship",
                "don't ship",
                "dont ship",
                "before it ships",
                "does not go ahead",
                "doesn't go ahead",
                "doesnt go ahead",
            )
        )
    ):
        return "cancel_request"

    # --- Returns / replacements -------------------------------------------
    # Includes condition words (damaged, defective) because customers usually
    # describe the fault rather than naming the process they want.
    if (
        any(
            term in text
            for term in (
                "return",
                "replace",
                "damaged",
                "broken",
                "defective",
                "changed my mind",
                "change my mind",
            )
        )
        or re.search(r"\b(send|take|ship)\b.{0,35}\b(it|that|this|one|item|product)?\s*back\b", text)
    ):
        return "return_help"

    # --- Order list --------------------------------------------------------
    # Checked before order_status: "my orders" is plural and needs no slot,
    # whereas order_status resolves a single order reference.
    if any(term in text for term in ("all orders", "my orders", "order history", "list orders")):
        return "order_list"

    # --- Product / warranty questions --------------------------------------
    # Deliberately ahead of order_status because "what came in the parcel" is
    # about contents, not tracking, even though it mentions the parcel.
    if (
        any(
            term in text
            for term in (
                "product",
                "item",
                "warranty",
                "guarantee",
                "covered",
                "coverage",
                "what did i buy",
                "what is in",
                "what's in",
                "whats in",
                "what came in",
                "came in that parcel",
                "came in the parcel",
                "parcel contents",
                "order contents",
                "what i bought",
                "what i purchased",
            )
        )
        or re.search(r"\b(what|which)\b.{0,35}\b(buy|bought|purchase|purchased|receive|received)\b", text)
    ):
        return "product_help"

    # --- Tracking / delivery ------------------------------------------------
    if any(
        term in text
        for term in (
            "status",
            "arrive",
            "delivery",
            "track",
            "where is",
            "when will",
            "shipped",
            "shipping",
            "processing",
            "delivered",
            "get here",
            "reach me",
            "reach us",
            "parcel now",
            "order now",
            "whereabouts",
        )
    ):
        return "order_status"

    # --- Pronoun continuation ----------------------------------------------
    # Short follow-ups carrying only a pronoun ("what about it?") inherit the
    # previous intent. The <=8 word cap stops a long, genuinely new question
    # from being absorbed into the prior topic just because it says "it".
    if len(text.split()) <= 8 and any(
        term in text for term in ("it", "that", "this order", "that one", "this one")
    ):
        return previous_intent or "order_status"

    # --- Default -----------------------------------------------------------
    # Nothing matched: answer with a capability statement rather than guessing.
    return "general_help"


def _attributive_duration(value: str) -> str:
    """Convert '3 years' into the natural modifier '3-year' for customer text."""
    # Cosmetic only. Stored policy values are phrases like "2 years"; used as an
    # adjective they must read "a 2-year warranty". Anything that does not match
    # the simple "<number> <word>" shape is passed through untouched.
    match = re.fullmatch(r"\s*(\d+)\s+([a-zA-Z]+)\s*", value)
    if not match:
        return value.strip()
    count, unit = match.groups()
    # rstrip("s") de-pluralises the unit: "years" -> "year".
    return f"{count}-{unit.rstrip('s')}"


# =============================================================================
# SECTION 4 — OPTIONAL LLM UNDERSTANDING LAYER
# =============================================================================


def _provider_configuration(mode: str) -> dict[str, str] | None:
    """Resolve a provider configuration without returning it to application state."""
    # Returns credentials for local use inside `_llm_classification` only. The
    # dict deliberately never reaches AgentState, so the API key cannot leak
    # into a trace, a session history entry, or a UI render.
    #
    # LEGACY_LLM_MODE resolution: prefer Groq when only Groq is configured,
    # otherwise fall through to the OpenAI branch below.
    if mode == GROQ_MODE or (
        mode == LEGACY_LLM_MODE
        and os.getenv("GROQ_API_KEY")
        and not os.getenv("OPENAI_API_KEY")
    ):
        # Re-checked because the legacy branch above can be entered on the
        # mode label alone; returning None here triggers the deterministic path.
        if not os.getenv("GROQ_API_KEY"):
            return None
        return {
            "provider": "GroqCloud",
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "api_key": os.environ["GROQ_API_KEY"],
            # Groq exposes an OpenAI-compatible surface, which is why one
            # ChatOpenAI client can serve both providers.
            "base_url": "https://api.groq.com/openai/v1",
        }
    if mode in {OPENAI_MODE, LEGACY_LLM_MODE}:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {
            "provider": "OpenAI",
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": "https://api.openai.com/v1",
        }
    # Deterministic mode, or an unrecognised label: no provider.
    return None


def _safe_llm_failure(error: Exception) -> str:
    """Map provider failures to non-sensitive operational categories."""
    # Only the exception *class name* is inspected — never str(error), which can
    # carry request bodies, prompt fragments, or key prefixes. The caller stores
    # the returned category in state, so it must be safe to display.
    error_name = type(error).__name__.lower()
    if "rate" in error_name:
        return "rate_limit"
    if "auth" in error_name or "permission" in error_name:
        return "authentication"
    if "timeout" in error_name:
        return "timeout"
    if "connection" in error_name:
        return "connection"
    # Catch-all so an unexpected exception type still yields a bounded label.
    return "provider_error"


def _llm_classification(
    query: str,
    *,
    mode: str,
    previous_intent: Intent | None,
    active_order_id: str | None,
    active_product_name: str | None,
    pending_intent: Intent | None,
) -> tuple[Classification | None, dict[str, Any]]:
    """Use a structured model for understanding and fail closed to deterministic routing."""
    # Returns (classification_or_None, understanding_metadata). A None result is
    # not an error — it instructs the caller to use the deterministic classifier.
    # The metadata dict is what surfaces in the trace and quality dashboard.
    configured = _provider_configuration(mode)
    if not configured:
        # An LLM mode was requested but no key exists: record that a fallback
        # occurred so the dashboard can distinguish "chose rules" from
        # "wanted the model and could not reach it".
        return None, {
            "provider": "deterministic",
            "model": None,
            "fallback": mode != DETERMINISTIC_MODE,
            "failure": "credentials_unavailable" if mode != DETERMINISTIC_MODE else None,
        }
    try:
        # Imported lazily so the deterministic path carries no LangChain
        # dependency at import time.
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=configured["model"],
            api_key=configured["api_key"],
            base_url=configured["base_url"],
            temperature=0,     # classification must be reproducible
            timeout=12,        # bounded so a hung provider cannot stall a turn
            max_retries=1,     # one retry; beyond that, fall back to rules
        # strict json_schema binding forces the response into the Classification
        # model, which removes free-text parsing from the critical path.
        ).with_structured_output(Classification, method="json_schema", strict=True)
        raw_result = model.invoke(
            [
                (
                    # System prompt. Two jobs: pin down the intent taxonomy with
                    # worked examples for the ambiguous cases, and — crucially —
                    # forbid the model from copying an identifier out of the
                    # supplied context. Slot values must come from the current
                    # user message only; context is for reference resolution.
                    "system",
                    "Classify one e-commerce order-support turn into exactly one allowed intent. "
                    "Use order_status for tracking, delivery, arrival, or parcel whereabouts; "
                    "order_list for a list or history of orders; product_help for order contents, "
                    "purchased items, warranty, guarantee, or whether a product is covered; "
                    "return_help for returns, replacements, defects, changed-mind requests, or "
                    "sending an item back; cancel_request for stopping or cancelling an order; "
                    "end_conversation for closing; and general_help only when the request is truly "
                    "outside those categories. Examples: 'what came in that parcel?' is "
                    "product_help; 'how long is the monitor covered?' is product_help; 'could I "
                    "send that one back?' is return_help; 'when should the parcel get here?' is "
                    "order_status; 'stop this order before it ships' is cancel_request. Use the "
                    "bounded conversation context to resolve references such as it, that one, or "
                    "the parcel. Extract an order_id or customer_id only when the current user "
                    "message states it explicitly; never copy or invent an identifier from context.",
                ),
                (
                    # Human turn: the query plus a deliberately *bounded* context
                    # summary. Only four slots are shared, never raw history, so
                    # prompt size and data exposure both stay predictable.
                    "human",
                    "Current request: " + query + "\n"
                    f"Previous intent: {previous_intent or 'none'}\n"
                    f"Active order reference: {active_order_id or 'none'}\n"
                    f"Active product reference: {active_product_name or 'none'}\n"
                    f"Pending clarification intent: {pending_intent or 'none'}",
                ),
            ]
        )
        # Defensive normalisation: depending on the LangChain version the
        # structured output arrives either as the pydantic object or as a dict.
        result = (
            raw_result
            if isinstance(raw_result, Classification)
            else Classification.model_validate(raw_result)
        )
        return result, {
            "provider": configured["provider"],
            "model": configured["model"],
            "fallback": False,
            "failure": None,
        }
    except Exception as error:
        # Broad catch is intentional: a classification failure must degrade the
        # turn to rules, never propagate and break the conversation.
        return None, {
            "provider": configured["provider"],
            "model": configured["model"],
            "fallback": True,
            "failure": _safe_llm_failure(error),
        }


# =============================================================================
# SECTION 5 — GRAPH NODES
# -----------------------------------------------------------------------------
# Every node takes the full AgentState and returns a PARTIAL state dict that
# LangGraph merges in. `trace` is an accumulating channel, so each node returns
# only the event(s) it generated rather than the whole list.
# =============================================================================


def guardrail_node(state: AgentState) -> AgentState:
    # Node 1. Input safety gate — the only node that can short-circuit the graph
    # straight to `respond`.
    started = time.perf_counter()
    query = (state.get("query") or "").strip()

    # Two threat families: SQL-ish data manipulation, and prompt-injection
    # attempts aimed at the understanding layer. This is a substring denylist,
    # which is a cheap demo control, not a security boundary — the real
    # protection is that the agent has no write path and no raw SQL surface.
    blocked_patterns = (
        "drop table",
        "delete from",
        "insert into",
        "update orders",
        "bypass authorization",
        "ignore previous instructions",
        "reveal system prompt",
    )

    # Empty input: not "blocked" (nothing hostile happened), just unanswerable.
    # The distinct error_code keeps the quality analytics honest.
    if not query:
        return {
            "blocked": False,
            "error": "Please enter a question.",
            "error_code": "empty_input",
            "outcome": "clarification",
            "trace": [_event("guardrail", "reject", "Rejected an empty request.", started=started)],
        }

    # `next(...)` captures WHICH pattern matched so the trace names it. The
    # matched pattern goes only into the internal trace, not the user reply.
    matched = next((pattern for pattern in blocked_patterns if pattern in query.lower()), None)
    if matched:
        return {
            "blocked": True,
            # User-facing text states the boundary without confirming what was
            # detected, so probing the denylist yields no signal.
            "error": "I can help with order information, but I cannot execute or bypass protected operations.",
            "error_code": "unsafe_instruction",
            "outcome": "blocked",
            "trace": [
                _event(
                    "guardrail",
                    "block",
                    f"Blocked unsafe instruction pattern: {matched!r}.",
                    started=started,
                )
            ],
        }

    # Clean input. `error` is explicitly reset to None so a stale error from an
    # earlier turn cannot survive into this one.
    return {
        "blocked": False,
        "error": None,
        "trace": [_event("guardrail", "allow", "Input accepted.", started=started)],
    }


def understand_node(state: AgentState) -> AgentState:
    # Node 2. Decide intent and extract any explicitly stated slots.
    started = time.perf_counter()
    query = state["query"]

    # Regex extraction runs first and unconditionally: it is the trusted source
    # for slots, and the LLM result is only allowed to fill gaps it leaves.
    explicit_order = _extract_order_id(query)
    claimed_customer = _extract_customer_id(query)

    # Detects a reply that is *nothing but* an order id — i.e. the customer
    # answering the "which order?" clarification. Such a turn carries no intent
    # of its own and must inherit the pending one.
    bare_order_selection = bool(
        explicit_order
        and re.fullmatch(
            r"\s*(?:order\s*)?ORD\s*[-#]?\s*\d{4}\s*[.!]?\s*",
            query,
            re.I,
        )
    )
    pending_intent = state.get("pending_intent")
    mode = state.get("mode") or DETERMINISTIC_MODE

    # Skip the model call entirely for a bare order selection with a pending
    # intent: the answer is already known, so a network round trip would add
    # latency and risk without adding information.
    if mode != DETERMINISTIC_MODE and not (bare_order_selection and pending_intent):
        llm_result, understanding = _llm_classification(
            query,
            mode=mode,
            previous_intent=state.get("previous_intent"),
            active_order_id=state.get("active_order_id"),
            active_product_name=state.get("active_product_name"),
            pending_intent=pending_intent,
        )
    else:
        llm_result = None
        understanding = {
            # Label distinguishes "resolved from memory" from "rules decided",
            # which matters when reviewing provider mix in the dashboard.
            "provider": "conversation memory"
            if bare_order_selection and pending_intent
            else "deterministic",
            "model": None,
            "fallback": False,
            "failure": None,
        }

    # --- Resolution precedence: memory > LLM > rules -----------------------
    if bare_order_selection and pending_intent:
        intent = pending_intent
        method = "pending-clarification continuation"
    elif llm_result:
        intent = llm_result.intent
        # `or explicit_order` keeps the regex value when the model returns none.
        # Regex-extracted slots therefore act as the floor, never overwritten
        # away by a model that failed to echo them.
        explicit_order = llm_result.order_id or explicit_order
        claimed_customer = llm_result.customer_id or claimed_customer
        method = "optional structured LLM"
    else:
        intent = _deterministic_intent(query, state.get("previous_intent"))
        # Distinguishes "rules by design" from "rules because the model failed".
        method = (
            "deterministic fallback"
            if understanding["fallback"]
            else "deterministic domain classifier"
        )

    if explicit_order:
        # Normalise to canonical form (strip dashes, hashes, spaces; upper-case)
        # so downstream ownership lookups compare like with like.
        explicit_order = re.sub(r"[^A-Z0-9]", "", explicit_order.upper())
        # Salvage rule: naming a specific order but classifying as general_help
        # is almost certainly a missed status question. Treat the order id as
        # the stronger signal.
        if intent == "general_help":
            intent = "order_status"
            method += " + explicit-order fallback"

    return {
        "intent": intent,
        # The four understanding_* fields are the observability payload for the
        # classification layer: who decided, with what model, and whether the
        # preferred path degraded.
        "understanding_provider": understanding["provider"],
        "understanding_model": understanding["model"],
        "understanding_fallback": understanding["fallback"],
        "understanding_failure": understanding["failure"],
        "order_id": explicit_order,
        # Named "claimed" rather than "customer_id": an id asserted in free text
        # is an unverified claim until the context node reconciles it.
        "claimed_customer_id": claimed_customer,
        "trace": [
            _event(
                "understand",
                intent,
                f"Intent={intent}; explicit_order={explicit_order}; method={method}; "
                f"provider={understanding['provider']}; model={understanding['model'] or 'none'}; "
                f"fallback={understanding['fallback']}; "
                f"failure={understanding['failure'] or 'none'}.",
                started=started,
                data_used=["query", "previous_intent", "pending_intent"],
            )
        ],
    }


def resolve_context_node(state: AgentState) -> AgentState:
    # Node 3. Establish identity, then resolve which order the turn is about.
    # This node never fetches order *contents* — it only settles references.
    started = time.perf_counter()
    intent = state["intent"]

    # Session identity takes precedence over any id asserted in the message.
    # The `or` order is the security-relevant part of this line.
    customer_id = state.get("customer_id") or state.get("claimed_customer_id")
    customer = get_customer(customer_id) if customer_id else None

    # No valid identity: stop here and ask for a session. Nothing customer-
    # specific has been read at this point.
    if not customer:
        return {
            "customer_id": customer_id,
            "needs_clarification": True,
            "candidate_orders": [],
            "candidate_order_ids": [],
            "trace": [
                _event(
                    "context",
                    "need_identity",
                    "No valid session identity is available.",
                    started=started,
                )
            ],
        }

    identity_source = state.get("identity_source") or "query_demo"

    # Impersonation guard. In an authenticated session, a message naming a
    # DIFFERENT customer id is refused outright rather than silently ignored —
    # silent ignoring would leave the customer believing they queried the other
    # account. `not in (None, customer_id)` allows the no-claim and self-claim
    # cases through.
    if state.get("authenticated") and state.get("claimed_customer_id") not in (None, customer_id):
        return {
            "customer_id": customer_id,
            "customer_name": customer["name"],
            "needs_clarification": True,
            "error": "The customer identifier in the message does not match the active session.",
            "error_code": "identity_mismatch",
            "candidate_orders": [],
            "candidate_order_ids": [],
            "trace": [
                _event(
                    "context",
                    "identity_mismatch",
                    "Ignored an identifier that conflicted with the active session.",
                    started=started,
                    data_used=["session_customer_id"],
                )
            ],
        }

    # Shared base for the success paths below. `direct_response` tells the
    # router to skip authorize/tools/policy for intents that touch no records.
    base: AgentState = {
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "identity_source": identity_source,
        "needs_clarification": False,
        "candidate_orders": [],
        "candidate_order_ids": [],
        "direct_response": intent in {"general_help", "end_conversation"},
    }

    # These three intents need no order reference: two are conversational,
    # and order_list is scoped by customer rather than by order.
    if intent in {"general_help", "end_conversation", "order_list"}:
        base["trace"] = [
            _event(
                "context",
                "identity_ready",
                f"Using {identity_source}; no order reference is required for {intent}.",
                started=started,
                data_used=["customer_id"],
            )
        ]
        return base

    # --- Order reference resolution ----------------------------------------
    order_id = state.get("order_id")
    # `source` is recorded so the trace explains *why* this order was chosen —
    # essential when a customer disputes which order the agent answered about.
    source = "explicit" if order_id else None
    query = state["query"].lower()
    # Ownership-scoped listing; the repository returns newest-first, which every
    # "latest" rule below depends on.
    owned_orders = list_customer_orders(customer_id)

    # Resolution ladder, strongest evidence first:
    #   1. explicit id in the message (already set above)
    #   2. the customer said "latest"/"most recent"
    #   3. an order carried over from the previous turn
    #   4. a status question with no reference — assume the newest order
    #   5. the customer has exactly one order, so there is nothing to disambiguate
    if not order_id and any(term in query for term in ("latest", "most recent", "newest")):
        order_id = owned_orders[0]["order_id"] if owned_orders else None
        source = "latest_order_rule"
    elif not order_id and state.get("active_order_id"):
        order_id = state["active_order_id"]
        source = "conversation_memory"
    elif not order_id and intent == "order_status" and owned_orders:
        # Note: this default applies to order_status only. Returns and
        # cancellations are excluded on purpose — guessing the target of a
        # consequential action is not acceptable, so those fall through to the
        # clarification branch below.
        order_id = owned_orders[0]["order_id"]
        source = "helpful_latest_default"
    elif not order_id and len(owned_orders) == 1:
        order_id = owned_orders[0]["order_id"]
        source = "single_owned_order"

    # Ambiguous: present up to four candidates and stash the pending intent so
    # the customer's one-word answer next turn resumes the original request.
    if not order_id:
        return {
            **base,
            "candidate_orders": owned_orders[:4],
            "candidate_order_ids": [row["order_id"] for row in owned_orders[:4]],
            "pending_intent": intent,
            "pending_candidate_order_ids": [
                row["order_id"] for row in owned_orders[:4]
            ],
            "needs_clarification": True,
            "trace": [
                _event(
                    "context",
                    "clarify_order",
                    f"{len(owned_orders)} candidate orders exist; the agent will not guess.",
                    started=started,
                    data_used=["scoped_order_index"],
                )
            ],
        }

    # Resolved. `active_order_id` seeds conversation memory; the pending-*
    # fields are cleared because the clarification (if any) is now answered.
    return {
        **base,
        "order_id": order_id,
        "active_order_id": order_id,
        "pending_intent": None,
        "pending_candidate_order_ids": [],
        "order_reference_source": source,
        "trace": [
            _event(
                "context",
                "order_resolved",
                f"Resolved {order_id} via {source}.",
                started=started,
                data_used=["active_order_id", "scoped_order_index"],
            )
        ],
    }


def authorize_node(state: AgentState) -> AgentState:
    # Node 4. Object-level access control. Runs BEFORE retrieval so that a
    # denied turn never loads the record — this is the difference between
    # genuine authorization and merely redacting a response after the fact.
    started = time.perf_counter()
    customer_id, order_id = state.get("customer_id"), state.get("order_id")

    if state["intent"] == "order_list":
        # A list is inherently scoped to the caller by the repository query, so
        # a valid identity is the only requirement.
        allowed = customer_id is not None
        detail = "Authorized a customer-scoped order list."
    elif customer_id is not None and order_id:
        # The actual ownership check, delegated to the repository so the rule
        # lives next to the data.
        allowed = customer_owns_order(customer_id, order_id)
        detail = "Ownership verified." if allowed else "Ownership check failed."
    else:
        # Fail closed: missing context denies rather than defaults to allow.
        allowed = False
        detail = "Required identity or order context is missing."

    return {
        "authorized": allowed,
        "trace": [
            _event(
                "authorize",
                "allow" if allowed else "deny",
                detail,
                started=started,
                data_used=["customer_id", "order_id"],
            )
        ],
    }


def retrieve_node(state: AgentState) -> AgentState:
    # Node 5. The only node that pulls order data into state.
    started = time.perf_counter()

    # Safe skip. The graph edge from authorize is unconditional, so the denial
    # is enforced here: the node still runs but performs no read and writes
    # explicit empties, guaranteeing no record can leak into state or trace.
    if not state.get("authorized"):
        return {
            "order": None,
            "orders": [],
            "trace": [
                _event(
                    "tools",
                    "safe_skip",
                    "No customer record entered state because authorization failed.",
                    started=started,
                )
            ],
        }

    if state["intent"] == "order_list":
        orders = list_customer_orders(state["customer_id"])
        return {
            "orders": orders,
            "trace": [
                _event(
                    "tools",
                    "list_customer_orders",
                    f"Returned {len(orders)} scoped order summaries.",
                    started=started,
                    # Field-level disclosure record: summaries only, no line items.
                    data_used=["orders.order_id", "status", "order_date", "total_amount"],
                )
            ],
        }

    # Single-order path. `orders` is reset to [] so a list from a previous turn
    # cannot linger and be rendered alongside this answer.
    order = get_order(state["order_id"])
    return {
        "order": order,
        "orders": [],
        "trace": [
            _event(
                "tools",
                "get_order",
                "Loaded one governed order aggregate." if order else "No matching order exists.",
                started=started,
                data_used=["order", "order_items", "products"] if order else [],
            )
        ],
    }


def _match_product(query: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Fuzzy-matches a product mentioned in free text against the order's line
    # items. Tolerates typos, partial names, and casing without a catalogue
    # lookup or embedding model.
    if not items:
        return None
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        name = item["name"]
        name_tokens = re.findall(r"[a-z0-9]+", name.lower())
        # Two complementary scorers:
        #  - token_score: best single-word match, which catches "monitor" inside
        #    "UltraWide Curved Monitor" where a whole-phrase score would be low.
        #  - phrase_score: WRatio over the full strings, which catches multi-word
        #    references and length mismatches.
        token_score = max(
            (fuzz.ratio(query_token, name_token) for query_token in query_tokens for name_token in name_tokens),
            default=0.0,
        )
        phrase_score = fuzz.WRatio(query, name)
        scored.append((max(token_score, phrase_score), item))
    score, matched_item = max(scored, key=lambda pair: pair[0])
    # High threshold (85) chosen so ambiguity resolves to "no match" rather than
    # a confident wrong product. Callers treat None as a signal to fall back to
    # remembered context or to list everything.
    if score < 85:
        return None
    return matched_item


def policy_node(state: AgentState) -> AgentState:
    # Node 6. Deterministic business-rule evaluation. Produces a decision
    # *recommendation* plus its inputs; it never mutates an order.
    started = time.perf_counter()
    intent, order = state["intent"], state.get("order")

    # --- Product context resolution (no policy maths) ----------------------
    if order and intent == "product_help":
        # Three-tier resolution, most specific first:
        #   1. fuzzy match on this turn's wording
        #   2. the product remembered from the previous turn ("is it covered?")
        #   3. a single-item order, where there is nothing else it could mean
        matched = _match_product(state["query"], order["items"])
        if not matched and state.get("active_product_name"):
            matched = next(
                (
                    item
                    for item in order["items"]
                    if item["name"].lower() == state["active_product_name"].lower()
                ),
                None,
            )
        if not matched and len(order["items"]) == 1:
            matched = order["items"][0]
        return {
            "matched_product": matched,
            # Refreshes conversation memory so later pronouns resolve.
            "active_product_name": matched["name"] if matched else None,
            "policy": {},
            "handoff": False,
            "write_executed": False,
            "trace": [
                _event(
                    "policy",
                    "product_context",
                    (
                        f"Resolved product context to {matched['name']}."
                        if matched
                        else "No single product reference was required."
                    ),
                    started=started,
                    data_used=["query", "active_product_name", "order.items"],
                )
            ],
        }

    # Non-consequential intents (status, list) need no policy evaluation.
    if not order or intent not in {"return_help", "cancel_request"}:
        return {
            "policy": {},
            "handoff": False,
            "write_executed": False,
            "trace": [
                _event(
                    "policy",
                    "not_required",
                    "No consequential policy decision is required.",
                    started=started,
                )
            ],
        }

    # Age is measured against the frozen POLICY_AS_OF, not today. max(0, ...)
    # guards against fixture data dated after the pinned policy date.
    order_day = datetime.strptime(order["order_date"], "%Y-%m-%d").date()
    age_days = max(0, (POLICY_AS_OF - order_day).days)

    # --- Return eligibility -------------------------------------------------
    if intent == "return_help":
        # Same three-tier product resolution as product_help above.
        matched = _match_product(state["query"], order["items"])
        if not matched and state.get("active_product_name"):
            matched = next(
                (
                    item
                    for item in order["items"]
                    if item["name"].lower() == state["active_product_name"].lower()
                ),
                None,
            )
        if not matched and len(order["items"]) == 1:
            matched = order["items"][0]
        # Unresolved product: evaluate every item and take the SHORTEST window.
        # Conservative by design — better to under-promise than to assert an
        # eligibility the customer cannot actually exercise.
        considered = [matched] if matched else order["items"]
        windows = [int(re.search(r"\d+", item["return_policy"]).group()) for item in considered]
        window_days = min(windows)
        # Two independent conditions: terminal statuses are excluded outright,
        # and the order must sit inside the window.
        eligible = order["status"] not in {"Cancelled", "Returned"} and age_days <= window_days
        product_name = matched["name"] if matched else "all items (conservative window)"
        # The policy dict is the audit record: every input to the decision is
        # captured alongside the outcome so the result can be re-derived later.
        policy = {
            "policy_date": POLICY_AS_OF.isoformat(),
            "age_days": age_days,
            "window_days": window_days,
            # "to_request", not "eligible": the agent assesses, a human decides.
            "eligible_to_request": eligible,
            "product_scope": product_name,
            "requires_human_confirmation": True,
        }
        return {
            "matched_product": matched,
            # Preserves prior memory when this turn resolved nothing new.
            "active_product_name": matched["name"] if matched else state.get("active_product_name"),
            "policy": policy,
            "handoff": True,          # returns always route to a specialist
            "write_executed": False,  # asserted explicitly for the audit trail
            "trace": [
                _event(
                    "policy",
                    "return_assessment",
                    f"Age={age_days}; window={window_days}; eligible_to_request={eligible}.",
                    started=started,
                    data_used=["order_date", "status", "product.return_policy"],
                )
            ],
        }

    # --- Cancellation eligibility ------------------------------------------
    # Only "Processing" orders can be cancelled: anything shipped or beyond has
    # left the fulfilment window. Note this is a status test, not an age test.
    eligible = order["status"] == "Processing"
    policy = {
        "eligible_to_request": eligible,
        "current_status": order["status"],
        "requires_human_approval": True,
        # Recorded inside the policy payload as well as at state level, so the
        # "no write occurred" fact survives wherever the policy dict is logged.
        "write_executed": False,
    }
    return {
        "policy": policy,
        "handoff": True,
        "write_executed": False,
        "trace": [
            _event(
                "policy",
                "cancellation_assessment",
                f"Status={order['status']}; eligible_to_request={eligible}; write_executed=False.",
                started=started,
                data_used=["status", "cancellation_policy"],
            )
        ],
    }


def respond_node(state: AgentState) -> AgentState:
    # Node 7. Renders customer-facing text. Every branch composes from fields
    # already resolved upstream — no branch invents facts or re-queries data.
    #
    # Branch order matters: errors, then clarifications, then conversational
    # intents, then the authorization denial, then the grounded answers.
    started = time.perf_counter()

    # --- Controlled error surface ------------------------------------------
    if state.get("error"):
        # Only an unsafe-instruction error is reported as "blocked"; everything
        # else (empty input, identity mismatch) is a recoverable clarification.
        outcome = "blocked" if state.get("error_code") == "unsafe_instruction" else "clarification"
        return {
            "response": state["error"],
            "outcome": outcome,
            "trace": [_event("respond", outcome, "Returned a controlled response.", started=started)],
        }

    # --- Clarification ------------------------------------------------------
    if state.get("needs_clarification"):
        candidates = state.get("candidate_orders", [])
        if candidates:
            # Status and date are included so the customer can identify the
            # right order without a second round trip.
            choices = "\n".join(
                f"• {row['order_id']} — {row['status']} — ordered {row['order_date']}"
                for row in candidates
            )
            text = "Which order should I use? Choose one of your recent orders:\n" + choices
        else:
            # No candidates means identity itself is unresolved.
            text = "Please start or select a verified demo customer session before requesting order data."
        return {
            "response": text,
            "outcome": "clarification",
            "trace": [_event("respond", "clarify", "Asked for only the missing context.", started=started)],
        }

    # --- Conversational intents (no data access) ---------------------------
    if state["intent"] == "end_conversation":
        # Closing prompt doubles as the feedback solicitation that feeds the
        # quality dashboard.
        return {
            "response": "Thanks for contacting Kartify. Please rate the conversation so the quality dashboard can learn from the outcome.",
            "outcome": "ended",
            "trace": [_event("respond", "close", "Closed the conversation and requested feedback.", started=started)],
        }
    if state["intent"] == "general_help":
        # Capability statement rather than an apology: it tells the customer
        # exactly which reformulations will succeed.
        return {
            "response": "I can check your latest order, delivery status, purchased products, return eligibility, or prepare a cancellation request for human approval.",
            "outcome": "resolved",
            "trace": [_event("respond", "capabilities", "Explained the governed support scope.", started=started)],
        }

    # --- Authorization denial ----------------------------------------------
    if not state.get("authorized"):
        # Privacy-preserving wording: it does not reveal whether the order
        # exists, only that this session cannot reach it. The second sentence
        # states the non-retrieval fact explicitly.
        return {
            "response": "I cannot access that order from the active customer session. No order record was retrieved.",
            "outcome": "denied",
            "trace": [_event("respond", "deny", "Returned a privacy-preserving denial.", started=started)],
        }

    # --- Grounded: order list ----------------------------------------------
    if state["intent"] == "order_list":
        rows = state.get("orders", [])
        text = "Here are your orders:\n" + "\n".join(
            f"• {row['order_id']}: {row['status']} — ${row['total_amount']:,.2f}"
            for row in rows
        )
        return {
            "response": text,
            "outcome": "resolved",
            "trace": [_event("respond", "grounded_list", "Formatted scoped tool evidence.", started=started)],
        }

    order = state.get("order")
    # Authorized but no record: an evidence gap, distinct from a denial.
    if not order:
        return {
            "response": "I could not find that order.",
            "outcome": "clarification",
            "trace": [_event("respond", "not_found", "Reported an evidence gap.", started=started)],
        }

    # --- Grounded: product help --------------------------------------------
    if state["intent"] == "product_help":
        matched = state.get("matched_product")
        # Narrow answer only when a specific product resolved AND the question
        # is about coverage terms; otherwise list the contents.
        if matched and any(
            term in state["query"].lower()
            for term in ("warranty", "guarantee", "return window", "how long")
        ):
            text = (
                f"The {matched['name']} in {order['order_id']} has a "
                f"{_attributive_duration(matched['warranty_period'])} warranty and a "
                f"{_attributive_duration(matched['return_policy'])} "
                "return window."
            )
        else:
            lines = [
                f"• {item['name']} × {item['quantity']} — warranty {item['warranty_period']}; return window {item['return_policy']}"
                for item in order["items"]
            ]
            text = f"{order['order_id']} contains:\n" + "\n".join(lines)
        outcome = "resolved"

    # --- Grounded: return assessment ---------------------------------------
    elif state["intent"] == "return_help":
        policy = state["policy"]
        # Hedged phrasing ("appears eligible to request") is deliberate: the
        # agent cannot inspect item condition, so it must not state a verdict.
        decision = (
            "appears eligible to request a return"
            if policy["eligible_to_request"]
            else "does not appear eligible for a return"
        )
        # The response shows its working — age, window, and scope — so the
        # customer can see the basis of the assessment.
        text = (
            f"{order['order_id']} {decision} for {policy['product_scope']}. "
            f"At the reproducible policy date it was {policy['age_days']} days old; "
            f"the applicable window is {policy['window_days']} days. A support specialist must confirm condition and exceptions."
        )
        outcome = "human_handoff"

    # --- Grounded: cancellation request ------------------------------------
    elif state["intent"] == "cancel_request":
        policy = state["policy"]
        # Both branches end by stating that no order change was made — the
        # customer should never be left believing the cancellation is done.
        if policy["eligible_to_request"]:
            text = (
                f"{order['order_id']} is Processing, so I can prepare a cancellation request for human approval. "
                "I have not changed the order."
            )
        else:
            text = (
                f"{order['order_id']} is {order['status']}, so an automated cancellation request is not eligible. "
                "I can hand this to a support specialist; no order change has been made."
            )
        outcome = "human_handoff"

    # --- Grounded: order status (default branch) ---------------------------
    else:
        # Three-way delivery-date handling. The final branch is the anti-
        # hallucination case: with no date on record the agent says so rather
        # than estimating one.
        if order["delivery_date"]:
            delivery = f"The recorded delivery date is {order['delivery_date']}."
        elif order["status"] in {"Cancelled", "Returned"}:
            delivery = f"There is no active delivery date because the order is {order['status']}."
        else:
            delivery = "A delivery date has not yet been assigned; I will not invent one."
        text = f"{order['order_id']} is {order['status']}. {delivery} Total: ${order['total_amount']:,.2f}."
        outcome = "resolved"

    return {
        "response": text,
        "outcome": outcome,
        "trace": [
            _event(
                "respond",
                "grounded_response",
                "Generated the answer from governed fields and policy output.",
                started=started,
                data_used=["order", "policy"] if state.get("policy") else ["order"],
            )
        ],
    }


def evaluate_node(state: AgentState) -> AgentState:
    # Node 8. Automated self-assessment. Deliberately kept separate from the
    # customer rating so the two signals can be compared rather than conflated —
    # a high automated score with a low customer rating is the interesting case.
    started = time.perf_counter()
    nodes = {event["node"] for event in state.get("trace", [])}

    # Signal 1 — access control: an unauthorized turn must not have an order
    # in state. This asserts the safe-skip in retrieve_node actually held.
    access_control = not (state.get("authorized") is False and state.get("order"))

    # Signal 2 — grounding: the answer rests on retrieved evidence, OR it is one
    # of the outcome types that legitimately has no evidence behind it.
    grounded = bool(
        state.get("order")
        or state.get("orders")
        or state.get("outcome") in {"clarification", "denied", "blocked", "ended"}
        or state.get("intent") == "general_help"
    )

    # Signal 3 — policy: consequential intents must carry a policy payload.
    policy_checked = state.get("intent") not in {"return_help", "cancel_request"} or bool(
        state.get("policy")
    )

    # Signal 4 — trace completeness. The expected node set is built to match the
    # path this turn should have taken, so short-circuited turns are not
    # penalised for skipping nodes they were never meant to reach.
    expected = {"guardrail", "respond"}
    if not state.get("error"):
        expected.add("understand")
    if not state.get("error") and state.get("intent") not in {"general_help", "end_conversation"}:
        expected.add("context")
    if not state.get("needs_clarification") and state.get("intent") not in {"general_help", "end_conversation"} and not state.get("error"):
        expected.update({"authorize", "tools", "policy"})
    # Subset, not equality: extra nodes are acceptable, missing ones are not.
    trace_complete = expected.issubset(nodes)

    # Unweighted mean of the four booleans, so each signal counts equally and
    # the score decomposes cleanly (0.75 = exactly one failed check).
    signals = [access_control, grounded, policy_checked, trace_complete]
    score = round(sum(signals) / len(signals), 3)
    quality = {
        "access_control": access_control,
        "grounded": grounded,
        "policy_checked": policy_checked,
        "trace_complete": trace_complete,
        "automated_quality_score": score,
        # Placeholder filled in later by the UI feedback path.
        "customer_rating": None,
    }
    return {
        "quality": quality,
        "trace": [
            _event(
                "evaluate",
                "pass" if all(signals) else "review",
                f"Automated quality score={score:.0%}; customer rating remains independent.",
                started=started,
                data_used=["trace", "authorization", "grounding", "policy"],
            )
        ],
    }


# =============================================================================
# SECTION 6 — ROUTING AND GRAPH ASSEMBLY
# =============================================================================


def route_after_guardrail(state: AgentState) -> str:
    # Blocked or empty input skips understanding entirely and goes straight to
    # the controlled response.
    return "respond" if state.get("error") else "understand"


def route_after_context(state: AgentState) -> str:
    # Two ways to bypass the data path: a missing slot (clarification), or an
    # intent that needs no records at all (`direct_response`).
    return "respond" if state.get("needs_clarification") or state.get("direct_response") else "authorize"


def build_graph():
    """Compile the executable state graph used by notebook, tests, and app."""
    # Topology is fixed and declared here — the agent cannot choose its own
    # control flow, which is what makes the path auditable. Note that
    # authorize -> tools -> policy -> respond are UNCONDITIONAL edges: the
    # denial is enforced inside the nodes (safe skip), not by rerouting, so the
    # trace shows the same node sequence whether or not access was granted.
    builder = StateGraph(AgentState)

    # --- Nodes --------------------------------------------------------------
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("understand", understand_node)
    builder.add_node("context", resolve_context_node)
    builder.add_node("authorize", authorize_node)
    builder.add_node("tools", retrieve_node)
    builder.add_node("policy", policy_node)
    builder.add_node("respond", respond_node)
    builder.add_node("evaluate", evaluate_node)

    # --- Edges --------------------------------------------------------------
    # Safety gate is the mandatory entry point.
    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail, {"understand": "understand", "respond": "respond"}
    )
    builder.add_edge("understand", "context")
    builder.add_conditional_edges(
        "context", route_after_context, {"authorize": "authorize", "respond": "respond"}
    )
    builder.add_edge("authorize", "tools")
    builder.add_edge("tools", "policy")
    builder.add_edge("policy", "respond")
    # Evaluation is unconditional and terminal: every turn is scored, including
    # blocked and denied ones.
    builder.add_edge("respond", "evaluate")
    builder.add_edge("evaluate", END)
    return builder.compile()


# Module-level singleton. Compiled once at import so per-turn latency measures
# execution rather than graph construction.
GRAPH = build_graph()


# =============================================================================
# SECTION 7 — SESSION FAÇADE AND PUBLIC ENTRY POINTS
# =============================================================================


class SupportSession:
    """Conversation façade that keeps identity and active-order context across turns."""

    # The graph itself is stateless per invocation. This class owns the
    # cross-turn memory and injects it into each call, which keeps the graph
    # pure and testable while still supporting pronouns and follow-ups.

    def __init__(self, customer_id: int, mode: str = DETERMINISTIC_MODE):
        # Identity is validated once, at construction, so every later turn can
        # be treated as authenticated.
        customer = get_customer(customer_id)
        if not customer:
            raise ValueError(f"Unknown demo customer: {customer_id}")
        self.conversation_id = str(uuid.uuid4())   # correlation id for analytics
        self.schema_version = SESSION_SCHEMA_VERSION
        self.customer_id = customer_id
        self.customer_name = customer["name"]
        self.mode = mode
        # --- Conversation memory slots (bounded on purpose) ----------------
        self.active_order_id: str | None = None        # last order discussed
        self.active_product_name: str | None = None    # last product discussed
        self.previous_intent: Intent | None = None     # for pronoun continuation
        self.pending_intent: Intent | None = None      # intent awaiting an order pick
        self.pending_candidate_order_ids: list[str] = []
        self.turns = 0
        self.started_at = time.perf_counter()          # for session duration
        self.history: list[dict[str, Any]] = []        # per-turn analytics rows

    def ask(self, query: str) -> AgentState:
        # One conversational turn: inject memory, invoke the graph, then fold
        # the result back into memory.
        started = time.perf_counter()
        result = GRAPH.invoke(
            {
                "query": query,
                "mode": self.mode,
                # Session-established identity, which is what lets the context
                # node reject a conflicting id claimed in the message text.
                "authenticated": True,
                "identity_source": "demo_session",
                "customer_id": self.customer_id,
                "customer_name": self.customer_name,
                "active_order_id": self.active_order_id,
                "active_product_name": self.active_product_name,
                "previous_intent": self.previous_intent,
                "pending_intent": self.pending_intent,
                "pending_candidate_order_ids": self.pending_candidate_order_ids,
                "trace": [],   # fresh trace per turn
            }
        )
        self.turns += 1

        # --- Memory update: order -------------------------------------------
        previous_active_order = self.active_order_id
        # `or self.active_order_id` retains the prior value when a turn resolved
        # no order (a clarification, for instance).
        self.active_order_id = result.get("active_order_id") or self.active_order_id
        # Switching orders invalidates the remembered product — otherwise "is it
        # covered?" could answer about an item from the previous order.
        if self.active_order_id != previous_active_order:
            self.active_product_name = None

        # --- Memory update: product -----------------------------------------
        # A freshly matched product wins; otherwise carry forward whatever the
        # policy node preserved.
        if result.get("matched_product"):
            self.active_product_name = result["matched_product"]["name"]
        elif result.get("active_product_name"):
            self.active_product_name = result["active_product_name"]

        self.previous_intent = result.get("intent") or self.previous_intent

        # --- Memory update: pending clarification ---------------------------
        # Set when the agent asked which order; cleared once an order was
        # resolved or the turn reached a terminal outcome. Without the clear,
        # a stale pending intent would hijack a later bare order id.
        if result.get("needs_clarification") and result.get("candidate_order_ids"):
            self.pending_intent = result.get("intent")
            self.pending_candidate_order_ids = list(result["candidate_order_ids"])
        elif result.get("order_id") or result.get("outcome") in {
            "resolved",
            "denied",
            "human_handoff",
            "ended",
        }:
            self.pending_intent = None
            self.pending_candidate_order_ids = []

        # End-to-end latency, including graph overhead — distinct from the
        # per-node durations already recorded in the trace.
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        result.setdefault("quality", {})["latency_ms"] = elapsed_ms
        result["conversation_id"] = self.conversation_id
        result["turn_number"] = self.turns

        # Flat analytics row per turn: enough to reconstruct behaviour and
        # provider mix without retaining the full state object.
        self.history.append(
            {
                "turn": self.turns,
                "query": query,
                "response": result["response"],
                "intent": result.get("intent"),
                "active_order_id": self.active_order_id,
                "active_product_name": self.active_product_name,
                "pending_intent": self.pending_intent,
                "understanding_provider": result.get("understanding_provider"),
                "understanding_model": result.get("understanding_model"),
                "understanding_fallback": result.get("understanding_fallback", False),
                "outcome": result.get("outcome"),
                "quality_score": result.get("quality", {}).get("automated_quality_score"),
                "latency_ms": elapsed_ms,
            }
        )
        return result

    def context(self) -> dict[str, Any]:
        # Inspectable snapshot of session memory — used by the UI debug panel
        # and by tests asserting that context carried across turns correctly.
        return {
            "conversation_id": self.conversation_id,
            "schema_version": self.schema_version,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "active_order_id": self.active_order_id,
            "active_product_name": self.active_product_name,
            "previous_intent": self.previous_intent,
            "pending_intent": self.pending_intent,
            "pending_candidate_order_ids": self.pending_candidate_order_ids,
            "turns": self.turns,
        }

    def feedback_payload(
        self, rating: int, resolved: bool, comment: str = ""
    ) -> dict[str, Any]:
        # Builds the customer-rating record for the quality store, joining the
        # subjective rating to objective session metrics (turn count, duration,
        # intent path) so the two can be correlated.
        return {
            "conversation_id": self.conversation_id,
            "customer_id": self.customer_id,
            "rating": rating,
            "resolved": resolved,
            "comment": comment,
            # max(1, ...) avoids a divide-by-zero if feedback arrives before any
            # turn has been taken.
            "turns": max(1, self.turns),
            "duration_seconds": round(time.perf_counter() - self.started_at, 2),
            "intents": [row["intent"] for row in self.history if row.get("intent")],
        }


def ask(query: str, mode: str = DETERMINISTIC_MODE) -> AgentState:
    """Backward-compatible one-turn helper; prefer SupportSession for conversation memory."""
    # Single-shot entry point retained for notebook cells and older tests.
    # If the query names a valid demo customer, a throwaway authenticated
    # session is created so the turn behaves like a real one.
    customer_id = _extract_customer_id(query)
    if customer_id and get_customer(customer_id):
        return SupportSession(customer_id, mode=mode).ask(query)
    # Otherwise invoke unauthenticated: the context node will stop at the
    # identity check and ask for a session rather than exposing any data.
    return GRAPH.invoke(
        {
            "query": query,
            "mode": mode,
            "authenticated": False,
            "identity_source": "query_demo",
            "trace": [],
        }
    )
