"""Allowlisted, read-only domain tools over the teaching SQLite database."""

# =============================================================================
# MODULE OVERVIEW
# -----------------------------------------------------------------------------
# The ONLY module permitted to touch the order database. Every other module —
# the graph nodes, the evaluation harness, the app — reaches data through the
# six functions below and never opens a connection of its own. That single
# constraint is what makes data access auditable: the complete set of queries
# this system can issue is visible in one file, on one screen.
#
# Three defences operate here:
#
#   1. READ-ONLY AT THE DRIVER. Connections open with `mode=ro`, so a write is
#      refused by SQLite itself rather than by convention. This is the reason
#      `write_executed: False` can be asserted honestly upstream — there is no
#      code path in the process capable of mutating an order.
#
#   2. ALLOWLISTED QUERIES. No query builder, no ORM, no caller-supplied SQL.
#      Six fixed statements, all parameterised, with one documented exception
#      in `database_summary` (see the note there).
#
#   3. SCOPED READS — PARTIALLY. `list_customer_orders` and
#      `customer_owns_order` filter by customer_id in SQL, so they cannot
#      return another customer's data regardless of caller behaviour.
#      `get_order` does NOT: it takes an order_id alone and returns the full
#      aggregate. Its scoping lives in the CALLER (`authorize_node` runs
#      `customer_owns_order` before `retrieve_node` runs `get_order`), which
#      means the security property is an orchestration guarantee, not a data
#      layer one. See the extended note on `get_order`.
#
# NOTE ON THIS FILE: only comments and standard blank-line spacing have been
# added. No query text, parameter, or expression has been altered.
# =============================================================================

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# Resolved at import time, relative to this file: up one level out of the
# package, then into data/orders.db. Path-relative rather than cwd-relative, so
# the notebook, the test runner, and the Streamlit app all find the same file
# regardless of where they were launched from.
#
# The `parents[1]` index is coupled to the package layout — moving this module
# one directory deeper silently redirects the path, and the failure surfaces as
# "unable to open database file" rather than as an import error.
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "orders.db"


def connect_read_only(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open SQLite in read-only mode and return rows as dictionaries."""
    # `mode=ro` via the URI form (uri=True) is the load-bearing line of this
    # module. It is enforced by the SQLite driver, not by discipline: an
    # INSERT, UPDATE, or DROP on this connection raises OperationalError.
    #
    # A second, useful side effect: `mode=ro` will NOT create a missing file.
    # Default sqlite3.connect silently creates an empty database, which would
    # turn a wrong path into an empty result set — a far worse failure than the
    # loud "unable to open database file" this produces instead.
    #
    # as_posix() normalises Windows backslashes, which the URI form rejects.
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    # Row factory gives dict-like access by column name, which is what lets
    # every function below convert results with a plain dict(row) and keeps
    # callers insulated from column ordering.
    connection.row_factory = sqlite3.Row
    return connection


def list_customers(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return safe demo identities; real authentication belongs outside chat."""
    # Powers the demo identity picker in the UI. The docstring's second clause
    # is the important disclaimer: selecting yourself from a dropdown is not
    # authentication, and in a real deployment customer_id would arrive from a
    # verified session token rather than from this list.
    #
    # Column selection is minimal by design — id and name only, no email —
    # because this result is rendered to whoever opens the app.
    #
    # `with connection:` manages the TRANSACTION (commit/rollback). It does not
    # close the connection; sqlite3's context manager never does. Every
    # function in this module follows the same pattern, so each call leaves an
    # open handle behind.
    with connect_read_only(db_path) as connection:
        rows = connection.execute(
            # Deterministic ordering so the picker does not reshuffle between
            # reruns and a screenshot stays reproducible.
            "SELECT customer_id, name FROM customers ORDER BY customer_id"
        ).fetchall()
    # sqlite3.Row objects are converted to plain dicts at the boundary, so no
    # driver-specific type escapes this module into agent state.
    return [dict(row) for row in rows]


def get_customer(customer_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Resolve a customer profile for session context."""
    # Called twice per session lifecycle: once in SupportSession.__init__ to
    # validate the identity, and once per turn in resolve_context_node. A None
    # return is the signal that no valid identity exists, which routes the turn
    # to a clarification before any order data is touched.
    with connect_read_only(db_path) as connection:
        row = connection.execute(
            # Parameterised — the id is bound, never interpolated.
            #
            # Note `email` is selected here but nothing downstream uses it:
            # resolve_context_node reads only `name`. It therefore enters
            # session context as unused PII. Narrowing this SELECT to the two
            # columns actually consumed would be a free data-minimisation win.
            "SELECT customer_id, name, email FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    # None rather than an exception: absence is an expected outcome the caller
    # branches on, not an error condition.
    return dict(row) if row else None


def list_customer_orders(
    customer_id: int, db_path: Path = DB_PATH
) -> list[dict[str, Any]]:
    """Return orders belonging to one customer, newest first."""
    # SCOPED BY CONSTRUCTION. The WHERE clause makes it structurally impossible
    # for this function to return another customer's orders, whatever the
    # caller does. Contrast with get_order below.
    sql = """
        SELECT order_id, order_date, status, delivery_date, total_amount
        FROM orders
        WHERE customer_id = ?
        -- Summary columns only: no address, no payment method, no line items.
        -- This result is used for the order_list response and for the
        -- clarification candidate list, neither of which needs detail.
        ORDER BY order_date DESC, order_id DESC
    """
    # The ORDER BY is depended on elsewhere: resolve_context_node's
    # "latest_order_rule", "helpful_latest_default", and the Context memory
    # benchmark scenario all assume owned_orders[0] is the newest order.
    # order_id DESC is the tie-breaker that keeps that deterministic when two
    # orders share a date — without it, same-day orders would resolve
    # arbitrarily and the benchmark labels would flap.
    with connect_read_only(db_path) as connection:
        rows = connection.execute(sql, (customer_id,)).fetchall()
    return [dict(row) for row in rows]


def customer_owns_order(
    customer_id: int, order_id: str, db_path: Path = DB_PATH
) -> bool:
    """Object-level authorization: the session customer must own the order."""
    # The system's actual access-control primitive, called by authorize_node
    # BEFORE retrieve_node runs. Ordering is the whole point: a denied turn
    # never loads a record, so there is nothing to redact.
    with connect_read_only(db_path) as connection:
        row = connection.execute(
            # `SELECT 1` deliberately returns no data — only the existence of a
            # matching row. An authorization probe that returned columns could
            # leak whatever it selected through a logging path or an exception.
            #
            # Both predicates are bound parameters, so a crafted order_id
            # string cannot alter the clause.
            "SELECT 1 FROM orders WHERE customer_id = ? AND order_id = ?",
            (customer_id, order_id),
        ).fetchone()
    # Boolean only. Note the result is identical whether the order belongs to
    # someone else or does not exist at all — the caller cannot distinguish the
    # two, so a denial reveals nothing about which orders exist.
    return row is not None


def get_order(order_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Return one governed order aggregate containing its product evidence."""
    # THE UNSCOPED READ. This function takes an order_id and nothing else, and
    # returns the full aggregate including customer_id, customer_name,
    # shipping_address, and payment_method. It performs no ownership check.
    #
    # Safety therefore depends entirely on the CALLER having run
    # customer_owns_order first — which retrieve_node does, guarded by its
    # `if not state.get("authorized")` safe-skip. The property holds today, but
    # it is an orchestration guarantee rather than a data-layer one: a new
    # caller, a reordered graph edge, or a future direct import gets full
    # order PII for any id with no second line of defence.
    #
    # Adding an optional customer_id parameter and a matching predicate would
    # make the scoping structural, matching list_customer_orders above.

    order_sql = """
        SELECT o.order_id, o.order_date, o.status, o.delivery_date,
               o.total_amount, o.shipping_address, o.payment_method,
               o.customer_id, c.name AS customer_name
        FROM orders AS o
        -- Inner join, so an order with a dangling customer_id returns nothing
        -- rather than a partial record. Referential integrity enforced at read
        -- time as well as by the schema.
        JOIN customers AS c ON c.customer_id = o.customer_id
        WHERE o.order_id = ?
    """
    # Over-fetch worth noting: shipping_address and payment_method are selected
    # here but no branch of respond_node renders either one. They enter
    # AgentState["order"] regardless, so they are present anywhere state is
    # inspected or displayed — including the notebook trace panel.

    item_sql = """
        SELECT oi.product_id, p.name, p.category, p.description,
               -- These two columns are the policy engine's inputs: the return
               -- window and warranty text parsed by policy_node. They are
               -- read from the product record rather than hardcoded, so a
               -- catalogue change flows through without a code change.
               p.return_policy, p.warranty_period,
               -- price_at_purchase, not p.price: the historical price is the
               -- correct figure for a return or refund conversation, and it
               -- does not move when the catalogue is repriced.
               oi.quantity, oi.price_at_purchase
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        WHERE oi.order_id = ?
        -- Stable alphabetical ordering, so the product_help contents listing
        -- and the _match_product fallback both behave reproducibly.
        ORDER BY p.name
    """

    with connect_read_only(db_path) as connection:
        order = connection.execute(order_sql, (order_id,)).fetchone()
        # Early return before the second query: no order means no items worth
        # fetching, and the caller gets a clean None to branch on.
        if order is None:
            return None
        items = connection.execute(item_sql, (order_id,)).fetchall()

    # Two statements on one connection, in autocommit — they are not a single
    # atomic snapshot. Immaterial here because the database is read-only with
    # no concurrent writer, but it would matter against a live store.

    # Assembled as a nested aggregate rather than a flat join result, so a
    # multi-item order is one object with an items list instead of duplicated
    # order columns across N rows. policy_node and respond_node both iterate
    # `order["items"]` and depend on this shape.
    result = dict(order)
    result["items"] = [dict(item) for item in items]
    return result


def database_summary(db_path: Path = DB_PATH) -> dict[str, int]:
    """Return counts used by the app and notebook data audit."""
    # Feeds the data-audit cell and the app sidebar: a quick assertion that the
    # fixture database is populated as expected before any results are read.
    with connect_read_only(db_path) as connection:
        return {
            # THE ONE PLACE IN THIS MODULE THAT BUILDS SQL BY STRING
            # FORMATTING. Table and column IDENTIFIERS cannot be supplied as
            # bound parameters in SQLite — only values can — so interpolation
            # is unavoidable for a query that varies by table name.
            #
            # It is safe here because the loop iterates a hardcoded literal
            # tuple on the very next line: no caller input reaches the f-string,
            # and the allowlist is visible in the same expression. Promoting
            # that tuple to a module-level constant would make the allowlist
            # explicit to anyone scanning for injection risk, which is the
            # first thing a reviewer will grep for.
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            # The allowlist itself. Any future addition must be a literal here;
            # accepting a table name as an argument would turn this into a
            # genuine injection surface.
            for table in ("customers", "orders", "order_items", "products")
        }
        # Note the dict comprehension returns from INSIDE the `with` block, so
        # every count is read on the same connection before it is released.
