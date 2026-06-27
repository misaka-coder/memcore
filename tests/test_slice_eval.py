"""评测台回归:跑默认数据集,断言验收阈值(全过 + 零泄漏 + 满 hit@k)。"""

from __future__ import annotations

import unittest

from memcore.eval import default_dataset, run_eval


class EvalHarness(unittest.TestCase):
    def test_default_dataset_meets_acceptance(self) -> None:
        seed, cases = default_dataset()
        report = run_eval(seed=seed, cases=cases)
        self.assertTrue(report.ok, msg="\n" + report.format())
        self.assertEqual(report.leak_count, 0)  # 隔离零泄漏红线
        self.assertEqual(report.hit_rate, 1.0)  # 期望片段全部命中

    def test_report_formats(self) -> None:
        seed, cases = default_dataset()
        report = run_eval(seed=seed, cases=cases)
        text = report.format()
        self.assertIn("verdict: PASS", text)


if __name__ == "__main__":
    unittest.main()
