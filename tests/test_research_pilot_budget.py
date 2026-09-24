"""Budget admission tests: no retries or optimistic release after uncertainty."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path

from examples.research_pilot.budget import BudgetError, BudgetLedger, Tariff


def usage(prompt: int = 100, completion: int = 10, cached: int = 0) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": cached,
        "prompt_cache_miss_tokens": prompt - cached,
        "total_tokens": prompt + completion,
    }


class PilotBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "budget.sqlite3"
        self.ledgers: list[BudgetLedger] = []

    def tearDown(self) -> None:
        for ledger in self.ledgers:
            ledger.close()
        self.temp.cleanup()

    def ledger(self, *, limit: str = "50", expected_stage_id: str | None = None) -> BudgetLedger:
        ledger = BudgetLedger(self.path, limit_cny=limit, expected_stage_id=expected_stage_id)
        self.ledgers.append(ledger)
        return ledger

    @staticmethod
    def reserve(ledger: BudgetLedger, request: str = "request_1", **overrides):
        arguments = {
            "request_id": request,
            "stage": "preflight",
            "input_token_upper_bound": 100,
            "max_output_tokens": 10,
            "tariff": Tariff.peak(),
            **overrides,
        }
        return ledger.reserve(**arguments)

    def test_explicit_peak_tariff_and_full_context_reservation_are_decimal_exact(self) -> None:
        tariff = Tariff.peak()
        self.assertEqual(tariff.cached_input_cny_per_million, Decimal("0.10"))
        self.assertEqual(tariff.uncached_input_cny_per_million, Decimal("3"))
        self.assertEqual(tariff.output_cny_per_million, Decimal("9"))
        with localcontext() as context:
            context.prec = 3  # The application's decimal context cannot round admission down.
            ledger = self.ledger()
            reservation = self.reserve(ledger, input_token_upper_bound=1048576, max_output_tokens=2048)
            self.assertEqual(reservation.reserved_cny, "3.16416")
            self.assertEqual(ledger.snapshot()["remaining_cny"], "46.83584")
        with self.assertRaises(TypeError):
            ledger.reserve(request_id="missing_rate", stage="pilot", input_token_upper_bound=100, max_output_tokens=10)

    def test_exact_limit_is_admitted_but_one_extra_output_token_is_rejected(self) -> None:
        ledger = self.ledger(limit="0.00039")
        with self.assertRaisesRegex(BudgetError, "budget_limit_exceeded"):
            self.reserve(ledger, max_output_tokens=11)
        self.assertEqual(ledger.snapshot()["pending_count"], 0)
        reservation = self.reserve(ledger)
        self.assertEqual(Decimal(reservation.reserved_cny), Decimal("0.00039"))
        self.assertEqual(Decimal(ledger.snapshot()["remaining_cny"]), Decimal("0"))
        ledger.settle(reservation.reservation_id, usage())
        with self.assertRaisesRegex(BudgetError, "budget_limit_exceeded"):
            self.reserve(ledger, "next", input_token_upper_bound=1, max_output_tokens=1)

    def test_valid_usage_releases_only_unused_reservation_and_never_exports_path(self) -> None:
        ledger = self.ledger()
        reservation = self.reserve(ledger, input_token_upper_bound=1000, max_output_tokens=100)
        settlement = ledger.settle(reservation.reservation_id, usage(prompt=200, completion=20, cached=150))
        self.assertEqual(Decimal(settlement["usage_cost_at_reserved_rates_cny"]), Decimal("0.000345"))
        self.assertEqual(Decimal(settlement["released_cny"]), Decimal("0.003555"))
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["settled_requests"], 1)
        self.assertEqual(Decimal(snapshot["reserved_cny"]), 0)
        self.assertEqual(Decimal(snapshot["committed_cny"]), Decimal("0.000345"))
        self.assertNotIn(str(self.directory), json.dumps([snapshot, settlement]))
        self.assertTrue(all(isinstance(v, str) for k, v in snapshot.items() if k.endswith("_cny")))
        ledger.assert_ready()
        with self.assertRaisesRegex(BudgetError, "request_id_already_used"):
            self.reserve(ledger)
        with self.assertRaisesRegex(BudgetError, "reservation_already_settled"):
            ledger.settle(reservation.reservation_id, usage())

    def test_every_stage_and_reopened_instance_share_prior_cost_and_fixed_limit(self) -> None:
        first = self.ledger()
        reservation = self.reserve(first, input_token_upper_bound=1_000_000, max_output_tokens=1)
        first.settle(reservation.reservation_id, usage(prompt=1_000_000, completion=0))
        first.close()
        second = self.ledger()
        self.assertEqual(Decimal(second.snapshot()["usage_cost_at_reserved_rates_cny"]), Decimal("3"))
        with self.assertRaisesRegex(BudgetError, "budget_limit_exceeded"):
            self.reserve(
                second, "pilot_1", stage="pilot", input_token_upper_bound=1_000_000, max_output_tokens=5_000_000
            )
        self.reserve(second, "pilot_1", stage="pilot", input_token_upper_bound=666_666, max_output_tokens=5_000_000)
        self.assertEqual(Decimal(second.snapshot()["committed_cny"]), Decimal("49.999998"))
        with self.assertRaisesRegex(BudgetError, "budget_limit_mismatch"):
            self.ledger(limit="49")

    def test_single_inflight_is_atomic_across_independent_connections(self) -> None:
        first, second = self.ledger(), self.ledger()
        barrier = threading.Barrier(2)

        def attempt(ledger: BudgetLedger, request: str) -> str:
            barrier.wait(timeout=5)
            try:
                self.reserve(ledger, request)
                return "reserved"
            except BudgetError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(attempt, first, "first"), executor.submit(attempt, second, "second")]
            results = [future.result(timeout=15) for future in futures]
        self.assertCountEqual(results, ["reserved", "pending_reservation_blocks_dispatch"])
        self.assertEqual(first.snapshot()["pending_count"], 1)
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_failure_and_timeout_retain_reservation_after_close(self) -> None:
        for reason in ("provider_failure", "timeout", "interrupted"):
            with self.subTest(reason=reason):
                self.path = self.directory / f"{reason}.sqlite3"
                ledger = self.ledger()
                reservation = self.reserve(ledger)
                ledger.mark_uncertain(reservation.reservation_id, reason=reason)
                before = ledger.snapshot()
                with self.assertRaisesRegex(BudgetError, "reservation_uncertain"):
                    ledger.settle(reservation.reservation_id, usage())
                ledger.close()
                reopened = self.ledger()
                self.assertEqual(reopened.snapshot(), before)
                with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
                    reopened.assert_ready()
                with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
                    self.reserve(reopened, "next")

    def test_missing_or_invalid_usage_is_persistently_uncertain(self) -> None:
        cases = [
            None,
            {},
            {"prompt_tokens": 100, "completion_tokens": 10},
            {**usage(), "prompt_tokens": True},
            {**usage(), "completion_tokens": -1},
            {**usage(), "prompt_cache_hit_tokens": 1},
            {**usage(), "prompt_cache_miss_tokens": "100"},
            {**usage(), "total_tokens": 109},
            {**usage(), "total_tokens": 110.0},
        ]
        for index, invalid in enumerate(cases):
            with self.subTest(usage=invalid):
                self.path = self.directory / f"invalid_{index}.sqlite3"
                ledger = self.ledger()
                reservation = self.reserve(ledger)
                with self.assertRaisesRegex(BudgetError, "provider_usage_missing|invalid_provider_usage"):
                    ledger.settle(reservation.reservation_id, invalid)
                snapshot = ledger.snapshot()
                self.assertEqual(snapshot["pending_count"], 1)
                self.assertEqual(snapshot["settled_requests"], 0)
                self.assertEqual(Decimal(snapshot["reserved_cny"]), Decimal(reservation.reserved_cny))
                ledger.close()
                with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
                    self.ledger().assert_ready()

    def test_cost_or_either_token_bound_violation_keeps_entire_reservation(self) -> None:
        cases = [
            (usage(prompt=101, completion=10), "usage_cost_exceeds_reservation"),
            # Even when cache discounts leave cost below reservation, the upper bound is mandatory.
            (usage(prompt=101, completion=0, cached=101), "usage_tokens_exceed_reservation"),
            (usage(prompt=0, completion=11), "usage_tokens_exceed_reservation"),
        ]
        for index, (invalid, code) in enumerate(cases):
            with self.subTest(code=code, usage=invalid):
                self.path = self.directory / f"bounds_{index}.sqlite3"
                ledger = self.ledger()
                reservation = self.reserve(ledger)
                with self.assertRaisesRegex(BudgetError, code):
                    ledger.settle(reservation.reservation_id, invalid)
                self.assertEqual(Decimal(ledger.snapshot()["reserved_cny"]), Decimal(reservation.reserved_cny))
                with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
                    self.reserve(ledger, "next")

    def test_process_exit_after_reserve_cannot_reset_budget_on_restart(self) -> None:
        script = (
            "import os, sys\n"
            "from examples.research_pilot.budget import BudgetLedger, Tariff\n"
            "ledger = BudgetLedger(sys.argv[1])\n"
            "ledger.reserve(request_id='crashed', stage='preflight', input_token_upper_bound=100, "
            "max_output_tokens=10, tariff=Tariff.peak())\n"
            "os._exit(0)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        ledger = self.ledger()
        self.assertEqual(ledger.snapshot()["pending_count"], 1)
        self.assertEqual(Decimal(ledger.snapshot()["reserved_cny"]), Decimal("0.00039"))
        with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
            self.reserve(ledger, "retry")

    def test_stage_identity_is_random_persistent_and_reopens_same_costs(self) -> None:
        ledger = self.ledger()
        stage_id = ledger.snapshot()["stage_id"]
        self.assertRegex(stage_id, r"\A[0-9a-f]{32}\Z")
        reservation = self.reserve(ledger)
        ledger.settle(reservation.reservation_id, usage())
        before = ledger.snapshot()
        ledger.close()
        self.assertEqual(self.ledger(expected_stage_id=stage_id).snapshot(), before)

    def test_expected_identity_on_wrong_path_cannot_create_fresh_budget(self) -> None:
        stage_id = self.ledger().snapshot()["stage_id"]
        wrong_path = self.directory / "typo.sqlite3"
        with self.assertRaisesRegex(BudgetError, "existing_budget_ledger_required"):
            BudgetLedger(wrong_path, expected_stage_id=stage_id)
        self.assertFalse(wrong_path.exists())
        # An existing empty file is not an initialized budget either.
        wrong_path.touch()
        before = wrong_path.read_bytes()
        with self.assertRaisesRegex(BudgetError, "budget_stage_id_missing"):
            BudgetLedger(wrong_path, expected_stage_id=stage_id)
        self.assertEqual(wrong_path.read_bytes(), before)

    def test_other_ledger_identity_is_rejected_without_changing_either_budget(self) -> None:
        first = self.ledger()
        first_snapshot = first.snapshot()
        self.path = self.directory / "other.sqlite3"
        second = self.ledger()
        reservation = self.reserve(second)
        second_snapshot = second.snapshot()
        self.assertNotEqual(first_snapshot["stage_id"], second_snapshot["stage_id"])
        with self.assertRaisesRegex(BudgetError, "budget_stage_id_mismatch"):
            self.ledger(expected_stage_id=first_snapshot["stage_id"])
        self.assertEqual(first.snapshot(), first_snapshot)
        self.assertEqual(second.snapshot(), second_snapshot)
        same = self.ledger(expected_stage_id=second_snapshot["stage_id"])
        with self.assertRaisesRegex(BudgetError, "pending_reservation_blocks_dispatch"):
            same.assert_ready()
        same.settle(reservation.reservation_id, usage())
        self.assertEqual(second.snapshot()["stage_id"], second_snapshot["stage_id"])

    def test_invalid_limits_and_token_bounds_never_create_reservations(self) -> None:
        for limit in ("50.000001", "0", "-1", "NaN", "Infinity", 50.0):
            with self.subTest(limit=limit), self.assertRaisesRegex(BudgetError, "invalid_budget_limit"):
                self.ledger(limit=limit)
        with self.assertRaisesRegex(BudgetError, "persistent_ledger_path_required"):
            BudgetLedger(":memory:")
        with self.assertRaisesRegex(BudgetError, "invalid_expected_stage_id"):
            self.ledger(expected_stage_id="wrong")
        ledger = self.ledger()
        for overrides in (
            {"input_token_upper_bound": 0},
            {"input_token_upper_bound": True},
            {"input_token_upper_bound": 100.0},
            {"max_output_tokens": -1},
            {"max_output_tokens": 0},
            {"tariff": None},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(BudgetError):
                self.reserve(ledger, **overrides)
        self.assertEqual(ledger.snapshot()["pending_count"], 0)


if __name__ == "__main__":
    unittest.main()
