"""Public API for the advanced Kartify agentic-commerce teaching package."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# The package façade. It re-exports the small set of names a notebook, a test,
# or the Streamlit app should reach for, so consumers write
#
#     from kartify import SupportSession, run_benchmark
#
# rather than binding themselves to the internal module layout. That indirection
# is the point: `agent.py` can be split or renamed without breaking a single
# import in the notebook, as long as the names below keep resolving.
#
# WHAT IS AND IS NOT EXPOSED — the rule is implicit, so stating it here:
#   exported     agent (session, graph, modes), evaluation, feedback storage
#   NOT exported models (type contracts), repository (data tools), display
#
# The `display` omission is deliberate and correct: it imports IPython, which
# would become a hard dependency of the Streamlit app and of every test run for
# the sake of three notebook-only helpers.
#
# The `models` and `repository` omissions are less clearly reasoned. A caller
# using FeedbackStore.save() needs `FeedbackRecord` to construct a record, and
# the app's data-audit panel needs `repository.database_summary` — both must
# reach past this façade into the submodules, which is the coupling the façade
# exists to prevent. Worth deciding explicitly whether those are internal or
# public rather than leaving it to whichever import someone wrote first.
#
# IMPORT COST — the significant operational note. These three imports are
# EAGER, and they cascade:
#   .agent      -> langgraph, rapidfuzz, pydantic, and it executes
#                  `GRAPH = build_graph()` at module scope, so the state graph
#                  is COMPILED during `import kartify`
#   .evaluation -> pandas, plus .agent again
#   .feedback   -> pandas, sqlite3
#
# Consequence: `import kartify` costs a graph compilation and a pandas load
# even when the caller only wants FeedbackStore. And because the imports are
# unguarded, a failure anywhere — a langgraph version mismatch, a missing
# rapidfuzz — takes down the entire package surface, including the parts that
# do not depend on the broken subsystem. Deferring to lazy module-level
# __getattr__ would decouple them, at the cost of some clarity.
#
# NOTE ON THIS FILE: only comments and standard blank-line spacing have been
# added. No import, name, or list entry has been altered.
# =============================================================================

# --- Agent subsystem ---------------------------------------------------------
# The core runtime: the conversation façade, the graph builder, the mode
# constants, and the one-shot helper.
from .agent import (
    # Mermaid source for the architecture diagram. Exported because the
    # notebook and app both render it. Hand-maintained, so it can drift from
    # what build_graph() actually compiles.
    ARCHITECTURE_MERMAID,
    # The three understanding-mode labels. Exported because they are the
    # argument callers pass to SupportSession(mode=...) and because the UI
    # renders them as selector options — the string values are part of the
    # public contract, not internal enum members.
    #
    # Note LEGACY_LLM_MODE is deliberately absent: it exists in agent.py for
    # backward compatibility with older saved configurations and should not be
    # offered as a new choice.
    DETERMINISTIC_MODE,
    GROQ_MODE,
    OPENAI_MODE,
    # Session-shape version. Exported so a caller can detect and discard a
    # session persisted by an earlier build rather than silently misreading it.
    SESSION_SCHEMA_VERSION,
    # The primary entry point. Owns conversation memory across turns; this is
    # what the notebook, the app, and the benchmark all drive.
    SupportSession,
    # One-shot helper retained for backward compatibility. No memory between
    # calls — the docstring in agent.py steers callers to SupportSession.
    ask,
    # Credential-aware mode list, so the UI can render only what is actually
    # configured without ever handling a key value.
    available_understanding_modes,
    # Exposed so a test or notebook can compile a fresh, isolated graph rather
    # than sharing the module-level GRAPH singleton.
    #
    # Note GRAPH itself is NOT exported: sharing one compiled instance is the
    # intended runtime behaviour, but handing it out as a public name would
    # invite callers to invoke it directly and bypass SupportSession — losing
    # conversation memory and the per-turn analytics with it.
    build_graph,
)

# --- Evaluation subsystem ----------------------------------------------------
# Note `confusion_matrix` is defined in evaluation.py but NOT re-exported here,
# unlike its two siblings. A notebook doing `from kartify import *` gets the
# benchmark and the summary but must import the confusion matrix from the
# submodule — an inconsistency in the public surface rather than a considered
# exclusion, since all three are notebook-facing analysis functions.
from .evaluation import benchmark_summary, run_benchmark

# --- Feedback subsystem ------------------------------------------------------
# The class is exported; FeedbackRecord, its input contract, is not (see the
# models note in the header).
from .feedback import FeedbackStore


# `__all__` defines what `from kartify import *` binds, and — more usefully —
# documents the intended public surface for anyone reading the package cold.
# It also suppresses linter warnings about the re-exports above being unused.
#
# Every entry here must have a matching import above; a name listed but not
# imported raises AttributeError on a star-import, and one imported but not
# listed is still reachable by direct import but is excluded from star-imports.
#
# Ordering is plain ASCII sort, which interleaves kinds: uppercase constants
# sort before lowercase functions, so `FeedbackStore` and `SupportSession` land
# among the mode constants rather than beside each other. Alphabetical is
# defensible for grep-ability; grouping by kind would read better.
__all__ = [
    # --- constants (agent) ---
    "ARCHITECTURE_MERMAID",
    "DETERMINISTIC_MODE",
    # --- class (feedback) ---
    "FeedbackStore",
    # --- constants (agent), continued ---
    "GROQ_MODE",
    "OPENAI_MODE",
    "SESSION_SCHEMA_VERSION",
    # --- class (agent) ---
    "SupportSession",
    # --- functions (agent) ---
    "ask",
    "available_understanding_modes",
    # --- function (evaluation) ---
    "benchmark_summary",
    # --- function (agent) ---
    "build_graph",
    # --- function (evaluation) ---
    "run_benchmark",
]
