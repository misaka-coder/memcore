"""Artifact correctness and an optional full, process-restarted host gate."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.research_pilot.akane_host import _safe_metadata
from examples.research_pilot.akane_long_artifacts import PlanWorkspace
from examples.research_pilot.akane_long_run import _accepted, load_long_pack
from examples.research_pilot.akane_run import _contains_sensitive
from examples.research_pilot.runner import digest


class LongExperimentUnitTests(unittest.TestCase):
    def test_nested_timeline_labels_do_not_become_drive_paths(self):
        labels = "content:\n正常正文\nstate:\n状态\ndata:\n资料\ninput:\n参数\noutput:\n结果"
        for _ in range(4):
            self.assertFalse(_contains_sensitive(labels, ""))
            self.assertFalse(_contains_sensitive("stored result:\n" + labels, ""))
            labels = json.dumps({"body": labels}, ensure_ascii=False)
        for path in (
            "C:\\private\\record.txt",
            "C:/private/record.txt",
            "位置C:\\private",
            "file:///private",
            "\\\\server\\share",
        ):
            for _ in range(3):
                self.assertTrue(_contains_sensitive({"body": path}, ""))
                self.assertTrue(_contains_sensitive("stored result:\n" + path, ""))
                path = json.dumps({"body": path})
        self.assertTrue(_contains_sensitive("prefix example-secret-value suffix", "example-secret-value"))

    def test_saved_plan_must_satisfy_constraints_and_old_revision_is_immutable(self):
        rules = {
            "option": "分批迁移",
            "count": 6,
            "per_unit": 120,
            "budget": 900,
            "owner": "许衡",
            "open_items": ["回滚演练"],
        }
        good = {key: copy.deepcopy(value) for key, value in rules.items() if key != "budget"}
        good["total"] = 720
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plans.json"
            workspace = PlanWorkspace(path, {"v1": rules, "v2": rules})
            self.assertEqual(workspace.save("draft_v1", good)["status"], "saved")
            self.assertEqual(workspace.check("draft_v1")["status"], "passed")
            wrong = {**good, "total": 721, "open_items": []}
            self.assertEqual(workspace.save("draft_v1", wrong)["status"], "conflict")
            workspace.save("draft_v2", wrong)
            checked = workspace.check("draft_v2")
            self.assertEqual(checked["status"], "failed")
            self.assertEqual(set(checked["failed_checks"]), {"total_arithmetic", "unresolved_items"})
            workspace.save("draft_v2", good)
            self.assertEqual(workspace.check("draft_v2")["status"], "passed")
            before = digest(workspace.snapshot())
            restored = PlanWorkspace(path, {"v1": rules, "v2": rules})
            self.assertEqual(digest(restored.snapshot()), before)
            self.assertEqual(restored.check("draft_v1")["status"], "passed")
            self.assertEqual(digest(restored.snapshot()), before)

    def test_summary_snapshot_metadata_uses_returned_node_without_inventing_raw_children(self):
        opened = {
            "status": "ok",
            "node_type": "episodic",
            "view": "content",
            "memory_id": "summary-1",
            "result": {
                "card": {"kind": "memory.episode_summary", "source_ids": ["raw-child"]},
                "diary_summary": "阶段记忆",
            },
        }
        records = _safe_metadata({"results": [opened]})
        self.assertEqual([row["source_id"] for row in records], ["summary-1"])
        self.assertEqual(records[0]["kind"], "memory.episode_summary")
        self.assertEqual(records[0]["content_hash"], digest(opened["result"]))

    def test_acceptance_requires_a_completed_public_final_and_exact_speech(self):
        completion = {
            "completed": True,
            "final_entry": {"kind": "message.assistant"},
            "submitted_speech": "完成了",
            "final_projection": None,
        }
        self.assertEqual(len(_accepted([completion], "完成了")), 1)
        self.assertEqual(_accepted([{**completion, "completed": False}], "完成了"), [])
        self.assertEqual(_accepted([completion], "没有保存"), [])
        self.assertEqual(_accepted([completion], ""), [])

    def test_pack_keeps_fixed_denominators_and_model_inputs_separate_from_gold(self):
        pack = Path(__file__).resolve().parents[1] / "docs/research/pilot_v4"
        inputs = load_long_pack(pack)
        self.assertEqual(len(inputs["manifest"]["runs"]), 8)
        self.assertEqual(sum(len(case["steps"]) for case in inputs["scenarios"]["scenarios"]), 48)
        self.assertNotIn("answer_key", inputs)
        self.assertEqual(inputs["manifest"]["raw_token_trigger"], 4000)

    @unittest.skipUnless(os.environ.get("AKANE_TEST_SOURCE_ROOT"), "optional actual Akane checkout not configured")
    def test_actual_engine_compaction_tools_and_two_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "gate"
            completed = subprocess.run(
                [
                    os.environ.get("AKANE_TEST_PYTHON", sys.executable),
                    "-B",
                    "-m",
                    "examples.research_pilot.akane_long",
                    "--offline-test",
                    "--akane-root",
                    os.environ["AKANE_TEST_SOURCE_ROOT"],
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                timeout=900,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            gate = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "passed")
            self.assertTrue(gate["actual_engine_compaction_and_process_restart"])
            self.assertEqual(gate["real_network_calls"], 0)


if __name__ == "__main__":
    unittest.main()
