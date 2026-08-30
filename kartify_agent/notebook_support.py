"""Compact display helpers so notebook cells focus on experiments and interpretation."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# Presentation layer for the notebook. Its only job is to keep rendering code
# out of the teaching cells, so a reader sees the experiment and its
# interpretation rather than twenty lines of DataFrame formatting.
#
# The design rule this module follows: it renders, it does not compute. No
# function here derives a metric, filters a result, or reshapes an outcome —
# everything displayed comes straight from the agent's own state. That matters
# because a display helper that quietly transformed data would put a layer of
# unaudited logic between the graph and the reader, which is exactly what the
# rest of the system is built to avoid.
#
# Two rendering styles are used, deliberately and inconsistently:
#   display(Markdown(...)) / display(DataFrame) — rich output, for artefacts
#   print(...)                                  — plain stdout, for transcripts
# See the note in run_conversation for why the mixture is a trade-off, not an
# oversight, and where it can bite.
#
# NOTE ON THIS FILE: only comments and standard blank-line spacing have been
# added. No statement, argument, or expression has been altered.
# =============================================================================

from __future__ import annotations

from typing import Any

import pandas as pd

# IPython is a NOTEBOOK-ONLY dependency. Importing this module inside the
# Streamlit app or a plain test run will pull IPython in as a hard requirement
# even though nothing here would be called — which is the reason no other
# module in the package imports from this one.
from IPython.display import Markdown, display

# Imports the diagram constant and the session class from the agent module.
# ARCHITECTURE_MERMAID is hand-maintained rather than generated from
# build_graph(), so it can drift from the compiled topology; treat the graph as
# the source of truth if the two ever disagree.
from .agent import ARCHITECTURE_MERMAID, SupportSession


def show_architecture() -> None:
    # Renders the architecture diagram by wrapping the Mermaid source in a
    # fenced code block tagged `mermaid`.
    #
    # RENDERING IS ENVIRONMENT-DEPENDENT. Whether this appears as a diagram or
    # as literal text depends entirely on the front end:
    #   - JupyterLab 4.x and GitHub's .ipynb viewer render Mermaid natively.
    #   - Classic Notebook, Google Colab, and some VS Code configurations do
    #     not, and will show the raw flowchart source in a grey code block.
    # Worth checking in the exact environment used for submission or demo,
    # because a diagram that degrades to twenty lines of arrow syntax is a
    # visible defect in the one cell most likely to be screenshotted.
    display(Markdown(f"```mermaid\n{ARCHITECTURE_MERMAID}\n```"))


def show_turn(result: dict[str, Any]) -> None:
    # Renders one turn as three stacked artefacts, ordered from what the
    # customer sees to what an auditor needs:
    #
    #   1. the response      — what was actually said
    #   2. the trace         — how the graph arrived at it
    #   3. the quality block — how the agent scored its own work
    #
    # That ordering is the pedagogical point of the function: the answer alone
    # is never the deliverable, and putting the trace immediately beneath it
    # makes the reasoning path impossible to skip past.

    # Direct subscript, not .get(). Justified because respond_node sets
    # `response` on every terminal path including blocked and denied turns —
    # but it does mean a malformed state raises KeyError here rather than
    # rendering a partial view.
    display(Markdown(f"### Agent response\n{result['response']}"))

    # The trace as a table: one row per node, with decision, detail, timing,
    # and the data_used disclosure record. `.get(..., [])` degrades to an empty
    # frame rather than raising if a caller passes a hand-built dict.
    #
    # Note this renders the INTERNAL detail strings verbatim — including, on a
    # blocked turn, the specific unsafe pattern that matched. Appropriate for a
    # developer notebook; it would be a disclosure issue on any customer-facing
    # surface.
    display(pd.DataFrame(result.get("trace", [])))

    # Quality signals transposed into a single-column frame. Series -> to_frame
    # gives a vertical key/value table, which reads better than a one-row wide
    # DataFrame when there are six-odd heterogeneous fields (booleans, a score,
    # a latency). Mixed types collapse the column to object dtype; that is
    # cosmetic here since nothing downstream computes on it.
    display(pd.Series(result.get("quality", {}), name="value").to_frame())


def run_conversation(session: SupportSession, prompts: list[str]) -> pd.DataFrame:
    # Drives a scripted multi-turn conversation through a live session and
    # returns the analytics table. Takes an existing session rather than
    # constructing one, which is what allows a notebook to inspect
    # `session.context()` afterwards and to continue the same conversation in a
    # later cell.

    for prompt in prompts:
        # `session.ask` carries conversation memory across iterations, so
        # prompt N can legitimately depend on prompt N-1 ("Cancel it").
        result = session.ask(prompt)

        # `print` rather than `display(Markdown(...))` — a deliberate choice
        # for the transcript, since a plain USER/AGENT block reads as a
        # conversation and copies cleanly into a report.
        #
        # The trade-off: print writes to the stdout stream while the rest of
        # this module writes to the display channel, and Jupyter does not
        # guarantee ordering between the two. Mixing show_turn and
        # run_conversation output in one cell can therefore interleave oddly.
        print(f"\nUSER: {prompt}\nAGENT: {result['response']}")

    # Returns the session's CUMULATIVE history, not just the turns from this
    # call. Calling this function twice against the same session returns the
    # first batch again inside the second result — correct if the intent is "an
    # analytics table for the whole conversation so far", surprising if read as
    # "the turns I just ran". Worth knowing before slicing the frame.
    #
    # Each row carries intent, resolved order, understanding provider, outcome,
    # quality score, and latency — enough to reconstruct behaviour without
    # retaining the full state objects.
    return pd.DataFrame(session.history)
