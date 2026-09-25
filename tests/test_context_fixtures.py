from __future__ import annotations

import json
from pathlib import Path
import unittest

from memcore import official_context_adapters


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "context_surface_v1"


class ContextFixtureTests(unittest.TestCase):
    def test_fixed_clock_fixtures_have_one_current_message_and_no_private_time_format(self) -> None:
        expected = {"host_openai_chat.json", "host_anthropic_messages.json", "harness_deepseek.json"}
        self.assertEqual({path.name for path in FIXTURE_DIR.glob("*.json")}, expected)
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            with self.subTest(fixture=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["fixture_version"], "context_surface_v1")
                self.assertEqual(data["wire_includes_current_once"], True)
                messages = [
                    *data.get("history_messages", []),
                    data["current_message"],
                    *data.get("active_turn_messages", []),
                ]
                current_text = json.dumps(data["current_message"], ensure_ascii=False)
                self.assertEqual(
                    sum(current_text == json.dumps(message, ensure_ascii=False) for message in messages),
                    1,
                )
                serialized = json.dumps(data, ensure_ascii=False)
                self.assertNotIn("[message_time:]", serialized)
                self.assertNotIn("context.inject", serialized)

    def test_fixtures_are_accepted_by_their_official_normalizers(self) -> None:
        adapters = official_context_adapters()
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            with self.subTest(fixture=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                adapter = adapters[data["provider_profile"]]
                events = adapter.normalize(
                    [
                        *data.get("history_messages", []),
                        data["current_message"],
                        *data.get("active_turn_messages", []),
                    ]
                )
                self.assertTrue(any(event.kind == "tool_call" for event in events))
                self.assertTrue(any(event.kind == "tool_result" for event in events))
                self.assertIn("call-current", {event.call_id for event in events})
                self.assertTrue(any("event." in str(event.content) for event in events))


if __name__ == "__main__":
    unittest.main()
