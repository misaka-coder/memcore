from __future__ import annotations

import unittest

from memcore import bind_request_projection_messages


def _message(turn_id: str, source_id: str, index: int, text: str) -> dict:
    return {
        "payload": {"role": "user", "content": text},
        "source_ids": [source_id],
        "turn_id": turn_id,
        "projection_index": index,
        "projection_status": "complete",
        "projection_version": 3,
    }


class RequestBindingContractTests(unittest.TestCase):
    def test_preserves_order_and_separates_standalone_source_turns(self) -> None:
        result = bind_request_projection_messages(
            [
                _message("turn-active", "user-1", 10, "question"),
                _message("legacy.material", "material-1", 11, "attachment"),
                _message("turn-active", "tool-1", 12, "tool result"),
            ],
            active_turn_id="turn-active",
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            [item.payload["content"] for item in result.messages], ["question", "attachment", "tool result"]
        )
        self.assertEqual(
            [(group.turn_id, group.relation) for group in result.groups],
            [
                ("turn-active", "active"),
                ("legacy.material", "standalone"),
            ],
        )
        self.assertEqual(result.active_group.request_indexes, (0, 2))
        self.assertEqual(result.groups[1].request_indexes, (1,))

    def test_missing_source_turn_is_ambiguous_not_assigned_to_active(self) -> None:
        result = bind_request_projection_messages(
            [_message("", "material-1", 4, "attachment")],
            active_turn_id="turn-active",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "request_binding_source_turn_ambiguous")

    def test_requires_an_active_group(self) -> None:
        result = bind_request_projection_messages(
            [_message("legacy.material", "material-1", 4, "attachment")],
            active_turn_id="turn-active",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "request_binding_active_turn_missing")

    def test_rejects_missing_sources_and_invalid_projection_metadata(self) -> None:
        missing_source = _message("turn-active", "source", 1, "x")
        missing_source["source_ids"] = []
        invalid_metadata = _message("turn-active", "source", -1, "x")

        self.assertEqual(
            bind_request_projection_messages([missing_source], active_turn_id="turn-active").reason,
            "request_binding_source_ids_required",
        )
        self.assertEqual(
            bind_request_projection_messages([invalid_metadata], active_turn_id="turn-active").reason,
            "request_binding_projection_metadata_invalid",
        )


if __name__ == "__main__":
    unittest.main()
