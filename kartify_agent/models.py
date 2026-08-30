"""Typed contracts shared by the graph, tools, evaluation, and UI."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# The shared vocabulary for the whole system. Every other module imports from
# here and none of them define their own copies, so a change to an intent name
# or a state key is a one-line change with compiler-visible consequences.
#
# IMPORTANT DISTINCTION — this module contains two kinds of "contract" and they
# behave completely differently at runtime:
#
#   TypedDict  (TraceEvent, AgentState)
#       Static hints only. Erased at runtime. A wrong key, a missing key, or a
#       value of the wrong type raises nothing and is caught only by a type
#       checker being run over the codebase. These are documentation with
#       tooling support, not guarantees.
#
#   BaseModel  (Classification, FeedbackRecord)
#       Pydantic. Validated at runtime, every construction. A bad rating or a
#       malformed LLM response raises immediately.
#
# The two are used where each is appropriate: AgentState is internal and passes
# through LangGraph's own merge machinery, where per-node validation would cost
# more than it catches; Classification and FeedbackRecord sit on trust
# boundaries (a language model, and a user-facing form) where validation is the
# whole point.
#
# Structure below:
#   Section 1 — closed vocabularies (Intent, Outcome)
#   Section 2 — TypedDict state contracts (TraceEvent, AgentState)
#   Section 3 — validated pydantic models (Classification, FeedbackRecord)
#
# NOTE ON THIS FILE: only comments and standard blank-line spacing have been
# added. No declaration, annotation, default, or constraint has been altered.
# =============================================================================

from __future__ import annotations

# `operator.add` is imported for exactly one purpose: it is the reducer
# function attached to the `trace` channel at the bottom of AgentState.
import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# SECTION 1 — CLOSED VOCABULARIES
# -----------------------------------------------------------------------------
# Literal types rather than str or Enum. Literal gives exhaustiveness checking
# in a type checker — add a member here and every match/if-chain that fails to
# handle it becomes a reportable error — while still serialising as a plain
# string into JSON, SQLite, and the LLM's structured-output schema with no
# conversion step.
# =============================================================================

# The complete intent taxonomy. This tuple is duplicated in three places by
# necessity and they must stay aligned: the branch order of
# `_deterministic_intent`, the instruction text of the LLM system prompt, and
# the branch order of `respond_node`. Adding a member here without touching
# those three is the most likely way to introduce a silent gap.
#
# The members are ordered roughly by specificity, with the catch-all last.
Intent = Literal[
    "order_status",       # tracking, delivery dates, parcel whereabouts
    "order_list",         # order history; customer-scoped, needs no order id
    "product_help",       # order contents, warranty, coverage terms
    "return_help",        # returns, replacements, defects — consequential
    "cancel_request",     # stopping an order before fulfilment — consequential
    "end_conversation",   # closing signal; triggers the feedback prompt
    "general_help",       # catch-all; answers with a capability statement
]

# How a turn terminated. This is the axis the quality analytics group by, which
# is why "resolved" is narrow: only a turn that actually answered from evidence
# earns it. The other five are all legitimate, non-failure endings.
Outcome = Literal[
    "resolved",           # answered from retrieved evidence
    "clarification",      # a required slot was missing; the agent asked
    "denied",             # authorization failed; no record was retrieved
    "blocked",            # stopped at the input guardrail
    "human_handoff",      # assessed, then routed to a specialist — never a write
    "ended",              # customer closed the conversation
]


# =============================================================================
# SECTION 2 — TYPEDDICT STATE CONTRACTS
# =============================================================================


class TraceEvent(TypedDict, total=False):
    # One audit record per node execution, built exclusively by `_event()` in
    # the agent module. total=False makes every key optional, which matters
    # because these are constructed by one helper and consumed by loosely
    # coupled readers (the UI trace panel, the evaluate node) that should not
    # break if a field is absent.
    node: str              # which graph node emitted this
    decision: str          # the branch taken — "allow", "deny", "clarify_order"
    detail: str            # human-readable explanation, safe to display
    duration_ms: float     # per-node timing, computed by the _event helper
    data_used: list[str]   # field-level disclosure record for audit


class AgentState(TypedDict, total=False):
    """State contract for one turn; session context is injected before invocation."""

    # THE total=False DECISION, stated plainly: every key below is optional.
    # That is required by the architecture — LangGraph nodes return PARTIAL
    # dicts that get merged, so no node ever holds a complete state object and
    # marking anything required would make every node return type invalid.
    #
    # The cost is real and worth knowing: a type checker cannot tell you that
    # `state["query"]` might be absent, and a mistyped key name is a runtime
    # KeyError rather than a caught error. The reads in agent.py compensate by
    # using `.get()` with defaults almost everywhere; the few direct `state[...]`
    # accesses are the places that assume an upstream node has already run.
    #
    # The keys are grouped below by lifecycle stage — which node writes them.

    # --- Injected by the caller before invocation --------------------------
    query: str                       # the raw customer message
    mode: str                        # understanding mode label (see agent.py)
    authenticated: bool              # True only for a real SupportSession
    identity_source: str             # "demo_session" or "query_demo", for audit

    # --- Identity ----------------------------------------------------------
    customer_id: int | None          # the trusted, session-established identity
    customer_name: str | None
    # Kept deliberately separate from customer_id: an id asserted in message
    # text is an unverified CLAIM until the context node reconciles it against
    # the session. Merging these two fields would erase that distinction and
    # with it the impersonation check.
    claimed_customer_id: int | None

    # --- Conversation memory, injected by SupportSession -------------------
    # Bounded on purpose — four slots, not a transcript. This caps both prompt
    # size on the LLM path and how much context a single turn can leak.
    active_order_id: str | None
    active_product_name: str | None
    previous_intent: Intent | None
    pending_intent: Intent | None            # intent parked awaiting an order pick
    pending_candidate_order_ids: list[str]   # what was offered in that question

    # --- Written by understand_node ----------------------------------------
    intent: Intent
    # The four understanding_* fields are the observability payload for the
    # classification layer. They exist so the dashboard can answer "who decided
    # this, and did the preferred path degrade?" without re-running the turn.
    understanding_provider: str              # "GroqCloud", "deterministic", etc.
    understanding_model: str | None          # None on the rules path
    understanding_fallback: bool             # True when an LLM path degraded
    understanding_failure: str | None        # bounded category, never raw error text

    # --- Written by resolve_context_node -----------------------------------
    order_id: str | None
    # Records WHY this order was chosen ("explicit", "conversation_memory",
    # "helpful_latest_default"). Not decorative: it is the field that answers a
    # customer disputing which order the agent talked about.
    order_reference_source: str | None
    candidate_orders: list[dict[str, Any]]   # full rows, for rendering the choice
    candidate_order_ids: list[str]           # ids only, for session memory
    needs_clarification: bool                # routes to respond, skipping data access
    direct_response: bool                    # intent needs no record at all

    # --- Written by authorize_node and retrieve_node -----------------------
    authorized: bool
    order: dict[str, Any] | None             # single-order path
    orders: list[dict[str, Any]]             # order_list path
    matched_product: dict[str, Any] | None   # resolved by fuzzy match in policy

    # --- Written by policy_node --------------------------------------------
    policy: dict[str, Any]                   # decision plus all of its inputs
    handoff: bool                            # requires a human specialist

    # --- Safety and error flags --------------------------------------------
    blocked: bool
    # Asserted explicitly on every consequential path so the audit trail
    # carries a positive record that no mutation occurred, rather than relying
    # on the absence of evidence.
    write_executed: bool
    error: str | None                        # customer-safe message
    error_code: str | None                   # machine-readable, for analytics

    # --- Written by respond_node and evaluate_node -------------------------
    response: str
    outcome: Outcome
    quality: dict[str, Any]

    # --- The one accumulating channel --------------------------------------
    # `Annotated[..., operator.add]` tells LangGraph to REDUCE rather than
    # replace: each node's returned trace list is concatenated onto the
    # existing one. Every other key above is last-write-wins, which is why a
    # node returning a partial dict silently clears nothing it omits but
    # overwrites everything it includes.
    #
    # Without this reducer each node would erase the previous node's audit
    # record and only the final event would survive — the whole explainability
    # story depends on this single annotation.
    trace: Annotated[list[TraceEvent], operator.add]


# =============================================================================
# SECTION 3 — VALIDATED MODELS (RUNTIME-ENFORCED)
# =============================================================================


class Classification(BaseModel):
    """Structured model output used only when optional LLM mode is enabled."""

    # The trust boundary between a language model and the rest of the system.
    # Bound via `.with_structured_output(..., method="json_schema", strict=True)`,
    # so the provider enforces this schema server-side and pydantic validates
    # again on arrival. Free-text parsing never enters the critical path.

    # Constrained to the closed vocabulary, so an invented intent name fails
    # validation and the caller falls back to the deterministic classifier.
    intent: Intent

    # Both slot fields are REQUIRED but NULLABLE — note there is no `= None`
    # default, only a Field with a description. Strict json_schema requires
    # every property to be present, so the model must explicitly emit null
    # rather than omitting the key; that removes the ambiguity between "no
    # order was mentioned" and "the model forgot to answer".
    #
    # The `description` text is not documentation — it is shipped to the model
    # as part of the generated JSON schema, so it functions as instruction. It
    # deliberately repeats the system prompt's rule about explicit statement.
    #
    # Worth knowing: there is no format constraint here. A value like "ORD9999"
    # for a non-existent order passes validation and reaches `understand_node`,
    # which normalises it and hands it to the ownership check. The authorization
    # node is what actually stops a fabricated id, not this model.
    order_id: str | None = Field(
        description="Order ID stated explicitly in the current message, otherwise null."
    )
    customer_id: int | None = Field(
        description="Customer ID stated explicitly in the current message, otherwise null."
    )


class FeedbackRecord(BaseModel):
    """Conversation-level customer feedback and operational context."""

    # The trust boundary for user-submitted feedback. Validated here before it
    # reaches FeedbackStore.save, which is why that method can pass values
    # straight into SQL parameters without further checking.
    #
    # Note the constraints are duplicated at the storage layer where they
    # matter most (rating has a CHECK in the DDL) — defence in depth, so a
    # direct SQL write cannot poison the averages either.

    # Correlates a rating back to a specific conversation. UNIQUE in the
    # database, which is what makes the store's UPSERT idempotent against a
    # Streamlit rerun replaying the same submission.
    conversation_id: str
    customer_id: int

    # Bounded 1-5 inclusive. Enforced here AND in the SQLite CHECK constraint.
    rating: int = Field(ge=1, le=5)

    # The second, independent satisfaction axis. Kept separate from rating
    # because a pleasant conversation that solved nothing is a different
    # product outcome from a terse one that worked — the quality bands in
    # feedback.py require both to be good before reporting "Strong".
    resolved: bool

    # Defaulted rather than optional, so downstream string handling never has
    # to test for None. The 500-char cap bounds free text at the model layer;
    # note the TEXT column itself carries no length limit.
    comment: str = Field(default="", max_length=500)

    # --- Operational context, carried alongside the subjective score --------
    # These are what make the rating analysable: a 5-star result after nine
    # turns is not the same outcome as a 5-star result after two.
    turns: int = Field(ge=1)              # at least one turn must have occurred
    duration_seconds: float = Field(ge=0)  # non-negative; comes from perf_counter

    # The sequence of intents the conversation passed through. Typed as
    # list[str] rather than list[Intent], so an unrecognised value would pass
    # validation here — a looser constraint than the Classification model
    # above applies to the same vocabulary.
    #
    # default_factory (not `= []`) avoids the shared-mutable-default trap:
    # every instance gets its own list.
    intents: list[str] = Field(default_factory=list)
