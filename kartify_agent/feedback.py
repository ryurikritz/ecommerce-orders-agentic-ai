"""Conversation feedback storage and customer-experience quality metrics."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# The customer-side half of the evaluation story. Where `evaluation.py` scores
# the agent against labelled expectations, this module captures what the
# customer actually thought — a rating, a resolved flag, and the shape of the
# conversation that produced them.
#
# Keeping the two independent is the point. An automated quality score of 100%
# on a conversation the customer rated 2/5 is the single most informative
# signal the system can produce, and it is only visible because neither measure
# is allowed to influence the other.
#
# Layering:
#   FeedbackStore.__init__ / _connect / _create_schema  -> persistence setup
#   FeedbackStore.save                                  -> validated write path
#   FeedbackStore.dataframe                             -> raw read for analysis
#   FeedbackStore.metrics                               -> aggregated scorecard
#
# SQLite is chosen over a dataframe or JSON file for three reasons: the UNIQUE
# constraint on conversation_id gives idempotent writes for free, the CHECK
# constraint enforces the rating scale at the storage layer, and `pd.read_sql_query`
# makes the analysis path a one-liner.
#
# LIMITS worth stating before the numbers are presented:
#   - The default path is the OS temp directory, which on a shared deployment
#     means every visitor writes to the same database file (see __init__).
#   - `metrics()` applies its quality bands at any sample size above zero, so a
#     single rating is enough to produce a confident-looking band.
# =============================================================================

from __future__ import annotations

import sqlite3   # stdlib persistence; no server, no ORM, no migration tooling
import tempfile  # supplies the ephemeral default location
from pathlib import Path
from typing import Any

import pandas as pd

# Pydantic model supplying the validation contract for a feedback submission.
# Storage never constructs its own validation rules — they live in one place.
from .models import FeedbackRecord


class FeedbackStore:
    """Persist demo feedback in SQLite; Community Cloud storage is ephemeral."""

    # The docstring's warning is load-bearing: on Streamlit Community Cloud the
    # container filesystem is wiped between restarts, so this store is a demo
    # artefact. Anything intended to survive needs a mounted volume or a hosted
    # database, and the `path` argument exists to make that swap a one-liner.

    def __init__(self, path: str | Path | None = None):
        # Default location is the OS temp directory. Two consequences worth
        # being explicit about:
        #   1. Ephemeral — cleared on restart, by design for a demo.
        #   2. SHARED — the filename is fixed, so on a multi-user deployment
        #      every visitor reads and writes the same file. Feedback from all
        #      sessions pools into one table, which is why the metrics below
        #      describe "the deployment", not "this user".
        self.path = Path(path or Path(tempfile.gettempdir()) / "kartify_feedback.db")
        # Makes an injected custom path work even if its directory is absent.
        # exist_ok keeps repeated construction idempotent.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Schema creation on every construction, not on first write, so the
        # table is guaranteed present before any caller touches it.
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        # One place to configure connections, so every caller gets identical
        # behaviour.
        connection = sqlite3.connect(self.path)
        # Row factory gives dict-like access by column name rather than by
        # positional index, so adding a column cannot silently break a reader.
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        # `with connection:` wraps the statement in a transaction that commits
        # on success and rolls back on exception. Note it does NOT close the
        # connection — sqlite3's context manager manages the transaction only.
        with self._connect() as connection:
            connection.execute(
                # IF NOT EXISTS makes this safe to call on every construction.
                # There is no migration path here: a change to this DDL will
                # not alter an existing table, so a schema change on a live
                # file needs the file deleted or migrated by hand.
                """
                CREATE TABLE IF NOT EXISTS conversation_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- UNIQUE is what makes the upsert in save() possible: one
                    -- row per conversation, so a resubmitted rating corrects
                    -- rather than duplicates.
                    conversation_id TEXT NOT NULL UNIQUE,
                    customer_id INTEGER NOT NULL,
                    -- Scale enforced at the storage layer as well as in the
                    -- pydantic model: defence in depth, so a direct SQL write
                    -- cannot poison the average.
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    -- SQLite has no boolean type; stored as 0/1 and cast back
                    -- in dataframe(). No CHECK here, unlike rating.
                    resolved INTEGER NOT NULL,
                    -- Defaulted rather than nullable, so downstream string
                    -- handling never has to test for None.
                    comment TEXT NOT NULL DEFAULT '',
                    -- Conversation shape, carried alongside the rating so the
                    -- subjective score can be correlated with effort: a 5-star
                    -- rating after nine turns is not the same product outcome
                    -- as a 5-star rating after two.
                    turns INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    -- Denormalised: the intent path is joined into one
                    -- comma-separated string rather than given a child table.
                    -- Cheap to write, but it comes back out of dataframe() as
                    -- a string, so any per-intent analysis needs a split first.
                    intents TEXT NOT NULL,
                    -- Set once by SQLite on insert. Deliberately NOT touched by
                    -- the upsert below, so it records first submission time
                    -- even when a rating is later corrected.
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(
        self,
        record: FeedbackRecord | dict[str, Any] | None = None,
        **fields: Any,
    ) -> FeedbackRecord:
        """Validate and upsert feedback from a record, mapping, or keyword fields."""
        # Three accepted call shapes — a model instance, a mapping, or loose
        # keywords — because the callers differ: the agent's
        # `feedback_payload()` returns a dict, a UI form supplies keywords, and
        # tests construct the model directly.

        # Ambiguity is rejected rather than silently resolved: if both forms
        # are supplied there is no defensible precedence rule.
        if record is not None and fields:
            raise ValueError("Pass either a record or keyword fields, not both.")
        payload = fields if record is None else record
        # Single validation gate. Everything below this line can assume a
        # well-formed record, which is why the SQL parameters are untyped.
        validated = payload if isinstance(payload, FeedbackRecord) else FeedbackRecord(**payload)
        with self._connect() as connection:
            connection.execute(
                # UPSERT keyed on the UNIQUE conversation_id. This makes the
                # write idempotent, which matters because a Streamlit rerun can
                # replay the same submission — without it, one customer's
                # opinion would be counted several times in the averages.
                #
                # Last write wins: a corrected rating overwrites the previous
                # one and no history is retained. Acceptable for a scorecard,
                # but it means rating-change behaviour is not analysable.
                """
                INSERT INTO conversation_feedback (
                    conversation_id, customer_id, rating, resolved, comment,
                    turns, duration_seconds, intents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    rating=excluded.rating,
                    resolved=excluded.resolved,
                    comment=excluded.comment,
                    turns=excluded.turns,
                    duration_seconds=excluded.duration_seconds,
                    intents=excluded.intents
                """,
                # Parameterised throughout — no string interpolation anywhere in
                # this module, so the comment field cannot carry SQL into the
                # statement.
                (
                    validated.conversation_id,
                    validated.customer_id,
                    validated.rating,
                    # Explicit bool -> int for the SQLite column.
                    int(validated.resolved),
                    validated.comment,
                    validated.turns,
                    validated.duration_seconds,
                    # Flattening the intent path for storage. The inverse
                    # (.split(",")) is not implemented on the read side.
                    ",".join(validated.intents),
                ),
            )
        # Returns the validated model, not the raw input, so the caller sees
        # exactly what was persisted including any coercion pydantic applied.
        return validated

    def dataframe(self) -> pd.DataFrame:
        # Raw read for exploratory analysis and notebook display. Deliberately
        # unaggregated — `metrics()` is built on top of this rather than
        # issuing its own SQL, so there is one read path to reason about.
        with self._connect() as connection:
            frame = pd.read_sql_query(
                # Chronological order so the frame can be read as a timeline
                # or used directly for a trend plot.
                "SELECT * FROM conversation_feedback ORDER BY created_at", connection
            )
        # Restores the boolean type that SQLite could not store. Guarded on
        # non-empty because .astype on an empty frame's column would operate on
        # a column that read_sql_query may not have typed as expected.
        if not frame.empty:
            frame["resolved"] = frame["resolved"].astype(bool)
        return frame

    def metrics(self) -> dict[str, float | int | str]:
        # Aggregated customer-experience scorecard, the counterpart to
        # `benchmark_summary()` in the evaluation module.
        frame = self.dataframe()

        # Cold-start branch. Returns the same key set as the populated branch
        # so any UI can render without conditional logic.
        #
        # Note the zeros are sentinels, not measurements: 0.0 sits outside the
        # valid 1-5 rating scale, so charting this row alongside real data
        # would plot "no feedback" as worse than the worst possible rating.
        # The quality_band string is the field that carries the real meaning.
        if frame.empty:
            return {
                "responses": 0,
                "average_rating": 0.0,
                "resolution_rate": 0.0,
                "five_star_share": 0.0,
                "average_turns": 0.0,
                "quality_band": "Insufficient feedback",
            }

        average_rating = float(frame["rating"].mean())
        # Mean over the boolean column gives the proportion resolved.
        resolution_rate = float(frame["resolved"].mean())

        # --- Quality banding ----------------------------------------------
        # Two conjunctive conditions per band: satisfaction AND resolution.
        # The conjunction is the deliberate part — a high rating with poor
        # resolution means customers are being handled pleasantly but not
        # helped, and that should not read as "Strong".
        #
        # Thresholds are hardcoded rather than configurable, and the ladder is
        # evaluated at ANY sample size above zero. With one response, a single
        # 5-star resolved rating produces "Strong". The `responses` count is
        # returned alongside the band for exactly this reason — the band is
        # only interpretable next to its denominator.
        if average_rating >= 4.5 and resolution_rate >= 0.80:
            band = "Strong"
        elif average_rating >= 3.5 and resolution_rate >= 0.60:
            band = "Monitor"
        else:
            # Catch-all: also where a high-rating/low-resolution split lands.
            band = "Needs improvement"

        return {
            # Sample size first: every figure below is meaningless without it.
            "responses": int(len(frame)),
            "average_rating": round(average_rating, 2),
            "resolution_rate": round(resolution_rate, 3),
            # Top-box share. Tracked separately from the mean because the mean
            # hides distribution: 3,3,5,5 and 4,4,4,4 both average 4.0, and the
            # first is a polarised experience worth investigating.
            "five_star_share": round(float((frame["rating"] == 5).mean()), 3),
            # Effort proxy. Rising average turns alongside a stable rating
            # suggests the agent is resolving issues, but slowly.
            "average_turns": round(float(frame["turns"].mean()), 2),
            "quality_band": band,
        }
