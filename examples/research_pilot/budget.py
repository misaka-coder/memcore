"""Local, fail-closed spending reservations shared by every paid pilot stage.

Commit a reservation before sending any request. A crash, transport failure, or
unverifiable usage leaves its entire amount pending and prevents another request.
Amounts computed from usage are tariff estimates, not provider billing records.
This module performs no network operations and never exports the SQLite path.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

MAX_LIMIT_CNY = Decimal("50")
_MAX_TOKENS = 2**63 - 1
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_STAGE_ID = re.compile(r"[0-9a-f]{32}\Z")
_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


class BudgetError(RuntimeError):
    """A fixed error code safe to record without request data or local paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _decimal(value: str | Decimal, code: str) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise BudgetError(code)
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise BudgetError(code) from None
    if not result.is_finite() or result < 0:
        raise BudgetError(code)
    return result


def _amount(value: Decimal) -> str:
    return format(value, "f")


def _tokens(value: Any, code: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > _MAX_TOKENS:
        raise BudgetError(code)
    return value


def _label(value: str, code: str) -> str:
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise BudgetError(code)
    return value


@dataclass(frozen=True)
class Tariff:
    """CNY per million tokens; callers must explicitly select a reservation rate."""

    cached_input_cny_per_million: Decimal
    uncached_input_cny_per_million: Decimal
    output_cny_per_million: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _decimal(getattr(self, name), "invalid_tariff")
            # Keep exact arithmetic bounded and exclude accidental free rates.
            if value <= 0 or value > 1_000_000 or value.as_tuple().exponent < -12:
                raise BudgetError("invalid_tariff")
            object.__setattr__(self, name, value)

    @classmethod
    def peak(cls) -> Tariff:
        """The pilot's explicit conservative rates, independent of dispatch time."""
        return cls(Decimal("0.10"), Decimal("3"), Decimal("9"))

    def cost(self, *, cached_input: int, uncached_input: int, output: int) -> Decimal:
        for value in (cached_input, uncached_input, output):
            _tokens(value, "invalid_token_count")
        with localcontext() as context:
            context.prec = 80
            return (
                cached_input * self.cached_input_cny_per_million
                + uncached_input * self.uncached_input_cny_per_million
                + output * self.output_cny_per_million
            ) / Decimal("1000000")

    def reserve_cost(self, *, input_token_upper_bound: int, max_output_tokens: int) -> Decimal:
        _tokens(input_token_upper_bound, "invalid_input_token_upper_bound", positive=True)
        _tokens(max_output_tokens, "invalid_max_output_tokens", positive=True)
        # No expected cache discount may reduce the pre-dispatch reservation.
        if self.cached_input_cny_per_million > self.uncached_input_cny_per_million:
            return self.cost(cached_input=input_token_upper_bound, uncached_input=0, output=max_output_tokens)
        return self.cost(cached_input=0, uncached_input=input_token_upper_bound, output=max_output_tokens)


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    request_id: str
    stage: str
    input_token_upper_bound: int
    max_output_tokens: int
    reserved_cny: str


class BudgetLedger:
    """One persistent budget, immutable limit, and at most one pending request.

    Use the same local path for preflight, pilot, and later paid stages. Opening
    an existing ledger permits inspection; assert_ready/reserve refuse to proceed
    if a previous process left any reservation unresolved. There is deliberately
    no automatic release, retry, or uncertain-charge reconciliation method.
    Freeze snapshot()["stage_id"] before paid work, then always pass it as
    expected_stage_id so a mistaken path cannot create a fresh budget.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        limit_cny: str | Decimal = "50",
        expected_stage_id: str | None = None,
    ):
        limit = _decimal(limit_cny, "invalid_budget_limit")
        if limit <= 0 or limit > MAX_LIMIT_CNY or limit.as_tuple().exponent < -12:
            raise BudgetError("invalid_budget_limit")
        if str(path) in {"", ":memory:"}:
            raise BudgetError("persistent_ledger_path_required")
        if expected_stage_id is not None and (
            not isinstance(expected_stage_id, str) or _STAGE_ID.fullmatch(expected_stage_id) is None
        ):
            raise BudgetError("invalid_expected_stage_id")
        self._lock = threading.RLock()
        self._closed = False
        # mode=rw enforces existence in SQLite itself, with no check/create race.
        database = str(path) if expected_stage_id is None else Path(path).resolve().as_uri() + "?mode=rw"
        try:
            self._connection = sqlite3.connect(
                database,
                uri=expected_stage_id is not None,
                isolation_level=None,
                timeout=10,
                check_same_thread=False,
            )
        except sqlite3.OperationalError:
            if expected_stage_id is not None:
                raise BudgetError("existing_budget_ledger_required") from None
            raise
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA synchronous=FULL")
            with self._transaction() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS pilot_budget_metadata ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                    "schema_version INTEGER NOT NULL, limit_cny TEXT NOT NULL, stage_id TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS pilot_budget_reservations ("
                    "reservation_id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, stage TEXT NOT NULL, "
                    "input_token_upper_bound INTEGER NOT NULL, max_output_tokens INTEGER NOT NULL, "
                    "reserved_cny TEXT NOT NULL, cached_rate TEXT NOT NULL, uncached_rate TEXT NOT NULL, "
                    "output_rate TEXT NOT NULL, status TEXT NOT NULL "
                    "CHECK(status IN ('reserved', 'uncertain', 'settled')), "
                    "usage_cost_at_reserved_rates_cny TEXT, reason TEXT, "
                    "prompt_tokens INTEGER, completion_tokens INTEGER, cache_hit_tokens INTEGER, "
                    "cache_miss_tokens INTEGER)"
                )
                connection.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS pilot_budget_single_pending "
                    "ON pilot_budget_reservations ((1)) WHERE status IN ('reserved', 'uncertain')"
                )
                row = connection.execute("SELECT * FROM pilot_budget_metadata WHERE singleton = 1").fetchone()
                if row is None:
                    if expected_stage_id is not None:
                        raise BudgetError("budget_stage_id_missing")
                    stage_id = uuid.uuid4().hex
                    connection.execute(
                        "INSERT INTO pilot_budget_metadata VALUES (1, 2, ?, ?)", (_amount(limit), stage_id)
                    )
                elif row["schema_version"] != 2 or "stage_id" not in row.keys():
                    raise BudgetError("unsupported_budget_schema")
                elif Decimal(row["limit_cny"]) != limit:
                    raise BudgetError("budget_limit_mismatch")
                else:
                    stage_id = row["stage_id"]
                    if not isinstance(stage_id, str) or _STAGE_ID.fullmatch(stage_id) is None:
                        raise BudgetError("invalid_budget_stage_id")
                    if expected_stage_id is not None and stage_id != expected_stage_id:
                        raise BudgetError("budget_stage_id_mismatch")
            self._limit = limit
            self._stage_id = stage_id
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise BudgetError("budget_ledger_closed")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> BudgetLedger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _pending(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute("SELECT 1 FROM pilot_budget_reservations WHERE status != 'settled' LIMIT 1").fetchone()
            is not None
        )

    @staticmethod
    def _totals(connection: sqlite3.Connection) -> tuple[Decimal, Decimal, int, int]:
        rows = connection.execute(
            "SELECT status, reserved_cny, usage_cost_at_reserved_rates_cny FROM pilot_budget_reservations"
        ).fetchall()
        used, reserved = Decimal("0"), Decimal("0")
        pending_count, settled_count = 0, 0
        with localcontext() as context:
            context.prec = 80
            for row in rows:
                if row["status"] == "settled":
                    used += Decimal(row["usage_cost_at_reserved_rates_cny"])
                    settled_count += 1
                else:
                    reserved += Decimal(row["reserved_cny"])
                    pending_count += 1
        return used, reserved, pending_count, settled_count

    def assert_ready(self) -> None:
        with self._transaction() as connection:
            if self._pending(connection):
                raise BudgetError("pending_reservation_blocks_dispatch")

    def snapshot(self) -> dict[str, str | int]:
        with self._transaction() as connection, localcontext() as context:
            context.prec = 80
            used, reserved, pending_count, settled_count = self._totals(connection)
            return {
                "stage_id": self._stage_id,
                "limit_cny": _amount(self._limit),
                "usage_cost_at_reserved_rates_cny": _amount(used),
                "reserved_cny": _amount(reserved),
                "committed_cny": _amount(used + reserved),
                "remaining_cny": _amount(self._limit - used - reserved),
                "pending_count": pending_count,
                "settled_requests": settled_count,
            }

    def reserve(
        self,
        *,
        request_id: str,
        stage: str,
        input_token_upper_bound: int,
        max_output_tokens: int,
        tariff: Tariff,
    ) -> Reservation:
        """Atomically persist the full upper bound before a caller sends HTTP."""
        _label(request_id, "invalid_request_id")
        _label(stage, "invalid_stage")
        if not isinstance(tariff, Tariff):
            raise BudgetError("explicit_tariff_required")
        amount = tariff.reserve_cost(
            input_token_upper_bound=input_token_upper_bound, max_output_tokens=max_output_tokens
        )
        with self._transaction() as connection, localcontext() as context:
            context.prec = 80
            if self._pending(connection):
                raise BudgetError("pending_reservation_blocks_dispatch")
            if connection.execute(
                "SELECT 1 FROM pilot_budget_reservations WHERE request_id = ?", (request_id,)
            ).fetchone():
                raise BudgetError("request_id_already_used")
            used, reserved, _, _ = self._totals(connection)
            if used + reserved + amount > self._limit:
                raise BudgetError("budget_limit_exceeded")
            result = Reservation(
                reservation_id=uuid.uuid4().hex,
                request_id=request_id,
                stage=stage,
                input_token_upper_bound=input_token_upper_bound,
                max_output_tokens=max_output_tokens,
                reserved_cny=_amount(amount),
            )
            connection.execute(
                "INSERT INTO pilot_budget_reservations ("
                "reservation_id, request_id, stage, input_token_upper_bound, max_output_tokens, "
                "reserved_cny, cached_rate, uncached_rate, output_rate, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved')",
                (
                    result.reservation_id,
                    request_id,
                    stage,
                    input_token_upper_bound,
                    max_output_tokens,
                    result.reserved_cny,
                    _amount(tariff.cached_input_cny_per_million),
                    _amount(tariff.uncached_input_cny_per_million),
                    _amount(tariff.output_cny_per_million),
                ),
            )
        return result

    @staticmethod
    def _open_reservation(connection: sqlite3.Connection, reservation_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM pilot_budget_reservations WHERE reservation_id = ?", (reservation_id,)
        ).fetchone()
        if row is None:
            raise BudgetError("unknown_reservation")
        if row["status"] == "settled":
            raise BudgetError("reservation_already_settled")
        return row

    def mark_uncertain(self, reservation_id: str, *, reason: str = "provider_failure") -> None:
        """Retain the reservation even if the caller did not receive a response."""
        if reason not in {"provider_failure", "timeout", "interrupted"}:
            raise BudgetError("invalid_uncertainty_reason")
        with self._transaction() as connection:
            row = self._open_reservation(connection, reservation_id)
            if row["status"] == "reserved":
                connection.execute(
                    "UPDATE pilot_budget_reservations SET status = 'uncertain', reason = ? WHERE reservation_id = ?",
                    (reason, reservation_id),
                )

    @staticmethod
    def _validated_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
        if not isinstance(usage, Mapping):
            raise BudgetError("provider_usage_missing")
        if any(name not in usage for name in _USAGE_FIELDS):
            raise BudgetError("provider_usage_missing")
        result = {name: _tokens(usage[name], "invalid_provider_usage") for name in _USAGE_FIELDS}
        if result["prompt_cache_hit_tokens"] + result["prompt_cache_miss_tokens"] != result["prompt_tokens"]:
            raise BudgetError("invalid_provider_usage")
        total = result["prompt_tokens"] + result["completion_tokens"]
        if "total_tokens" in usage and _tokens(usage["total_tokens"], "invalid_provider_usage") != total:
            raise BudgetError("invalid_provider_usage")
        result["total_tokens"] = total
        return result

    def settle(self, reservation_id: str, usage: Mapping[str, Any] | None) -> dict[str, Any]:
        """Settle only a successful response with coherent usage within all bounds.

        Validation failure is durably marked uncertain before raising. Once an
        attempt is uncertain, no subsequent call can automatically release it.
        """
        error: BudgetError | None = None
        result: dict[str, Any] = {}
        with self._transaction() as connection, localcontext() as context:
            context.prec = 80
            row = self._open_reservation(connection, reservation_id)
            if row["status"] == "uncertain":
                raise BudgetError("reservation_uncertain")
            try:
                verified = self._validated_usage(usage)
                tariff = Tariff(Decimal(row["cached_rate"]), Decimal(row["uncached_rate"]), Decimal(row["output_rate"]))
                cost = tariff.cost(
                    cached_input=verified["prompt_cache_hit_tokens"],
                    uncached_input=verified["prompt_cache_miss_tokens"],
                    output=verified["completion_tokens"],
                )
                reserved = Decimal(row["reserved_cny"])
                if cost > reserved:
                    raise BudgetError("usage_cost_exceeds_reservation")
                if (
                    verified["prompt_tokens"] > row["input_token_upper_bound"]
                    or verified["completion_tokens"] > row["max_output_tokens"]
                ):
                    raise BudgetError("usage_tokens_exceed_reservation")
            except BudgetError as exc:
                error = exc
                connection.execute(
                    "UPDATE pilot_budget_reservations SET status = 'uncertain', reason = ? WHERE reservation_id = ?",
                    (exc.code, reservation_id),
                )
            else:
                connection.execute(
                    "UPDATE pilot_budget_reservations SET status = 'settled', usage_cost_at_reserved_rates_cny = ?, "
                    "prompt_tokens = ?, completion_tokens = ?, cache_hit_tokens = ?, cache_miss_tokens = ? "
                    "WHERE reservation_id = ?",
                    (
                        _amount(cost),
                        verified["prompt_tokens"],
                        verified["completion_tokens"],
                        verified["prompt_cache_hit_tokens"],
                        verified["prompt_cache_miss_tokens"],
                        reservation_id,
                    ),
                )
                result = {
                    "reservation_id": reservation_id,
                    "usage_cost_at_reserved_rates_cny": _amount(cost),
                    "reserved_cny": row["reserved_cny"],
                    "released_cny": _amount(reserved - cost),
                    "usage": verified,
                }
        if error is not None:
            raise error
        return result
