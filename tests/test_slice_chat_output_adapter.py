"""Chat Output Adapter:标准输出契约、普通文本解析、句末分段。"""

from __future__ import annotations

import unittest

from memcore import StreamingSpeechParser, build_chat_output_contract_prompt, parse_chat_output, segment_speech


class SpeechSegmenter(unittest.TestCase):
    def test_chinese_punctuation_cluster_stays_together(self) -> None:
        self.assertEqual(segment_speech("哈啊？！真的吗。"), ["哈啊？！", "真的吗。"])

    def test_english_punctuation_cluster_stays_together(self) -> None:
        self.assertEqual(segment_speech("Really?! I see."), ["Really?!", "I see."])

    def test_decimal_is_not_split(self) -> None:
        self.assertEqual(segment_speech("收益率是 3.5%。"), ["收益率是 3.5%。"])

    def test_abbreviation_is_not_split(self) -> None:
        self.assertEqual(segment_speech("U.S. market is open. OK."), ["U.S. market is open.", "OK."])

    def test_domain_is_not_split(self) -> None:
        self.assertEqual(segment_speech("example.com is down. Fixed."), ["example.com is down.", "Fixed."])

    def test_newline_is_strong_boundary(self) -> None:
        self.assertEqual(segment_speech("第一句\n第二句。"), ["第一句", "第二句。"])


class ChatOutputParser(unittest.TestCase):
    def test_memcore_json_extracts_speech_and_coerces_metadata(self) -> None:
        result = parse_chat_output(
            """
            {
              "speech": "哈啊？！真的吗。",
              "emotion": "happy",
              "memory_metadata": {
                "keywords": ["可乐", "饮料", "可乐", "a", "b"],
                "subject_scopes": ["user", "bad"],
                "categories": ["preference", "bad"],
                "mood_tags": ["warm"],
                "importance": 2,
                "confidence": -1
              },
              "debug": "ignored by memory core"
            }
            """,
            mode="memcore_json",
            enable_flavor=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.speech, "哈啊？！真的吗。")
        self.assertEqual(result.segments, ["哈啊？！", "真的吗。"])
        self.assertEqual(result.presentation, {"emotion": "happy"})
        self.assertEqual(result.extra, {"debug": "ignored by memory core"})
        self.assertEqual(result.memory_metadata["keywords"], ["可乐", "饮料", "a", "b"])
        self.assertEqual(result.memory_metadata["subject_scopes"], ["user"])
        self.assertEqual(result.memory_metadata["categories"], ["preference"])
        self.assertEqual(result.memory_metadata["mood_tags"], ["warm"])
        self.assertEqual(result.memory_metadata["importance"], 1.0)
        self.assertEqual(result.memory_metadata["confidence"], 0.0)

    def test_flavor_off_strips_mood_tags(self) -> None:
        result = parse_chat_output(
            {"speech": "好。", "memory_metadata": {"mood_tags": ["warm"]}},
            mode="memcore_json",
            enable_flavor=False,
        )
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.memory_metadata["mood_tags"], [])

    def test_memcore_json_requires_speech(self) -> None:
        result = parse_chat_output({"memory_metadata": {"keywords": ["可乐"]}}, mode="memcore_json")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "invalid_contract")
        self.assertEqual(result.reason, "speech_required")

    def test_plain_text_auto_mode(self) -> None:
        result = parse_chat_output("普通回复。第二句。", mode="auto")
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "plain_text")
        self.assertEqual(result.speech, "普通回复。第二句。")
        self.assertEqual(result.segments, ["普通回复。", "第二句。"])
        self.assertEqual(result.memory_metadata, {})

    def test_broken_json_is_not_stored_as_speech(self) -> None:
        result = parse_chat_output('{"speech": "坏掉"', mode="auto")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "output_unparsed")
        self.assertEqual(result.speech, "")
        self.assertEqual(result.reason, "invalid_json")

    def test_json_array_is_not_object(self) -> None:
        result = parse_chat_output('["not", "object"]', mode="auto")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "output_unparsed")
        self.assertEqual(result.reason, "json_not_object")

    def test_custom_json_without_speech_is_unparsed(self) -> None:
        result = parse_chat_output({"message": "not our contract"}, mode="custom_json")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "output_unparsed")
        self.assertEqual(result.reason, "speech_required")


class ChatOutputPrompt(unittest.TestCase):
    def test_prompt_mentions_contract_and_not_speech_segments(self) -> None:
        prompt = build_chat_output_contract_prompt(
            categories=("preference", "finance_profile"),
            enable_flavor=False,
            enable_sentence_segments=True,
        )
        self.assertIn("字段固定为 speech, memory_metadata", prompt)
        self.assertIn("finance_profile", prompt)
        self.assertIn("mood_tags 必须输出为空数组", prompt)
        self.assertIn("工具调用阶段不适用本 JSON 契约", prompt)
        self.assertIn("不是摆设", prompt)
        self.assertIn("可以继续补查", prompt)
        self.assertIn("结构化信息通道", prompt)
        self.assertIn("legacy 文本 followup", prompt)
        self.assertIn("最终回复", prompt)
        self.assertIn("每个完整句子", prompt)
        self.assertIn("read_timeline", prompt)
        self.assertIn("retrieve", prompt)
        self.assertIn("load_material", prompt)
        self.assertIn("上周二", prompt)
        self.assertIn("群聊", prompt)
        self.assertIn("稳定ID", prompt)
        self.assertIn("不要猜名字", prompt)
        self.assertIn("不要为了显得记得而编造", prompt)
        self.assertIn("confidence", prompt)
        self.assertIn("未来正常聊天", prompt)
        self.assertIn("不要机械补太宽泛的上位词", prompt)
        self.assertIn("饮料/偏好", prompt)
        self.assertIn("不要写整句或短句", prompt)
        self.assertNotIn("speech_segments", prompt)


class StreamingOutputParser(unittest.TestCase):
    def test_plain_stream_delays_punctuation_cluster_segment(self) -> None:
        stream = StreamingSpeechParser(mode="plain")
        events = stream.feed("哈啊？")
        self.assertEqual([e["text"] for e in events if e["type"] == "speech_chunk"], ["哈啊？"])
        self.assertEqual([e for e in events if e["type"] == "speech_segment"], [])

        events += stream.feed("！真的吗。")
        events += stream.finish()

        self.assertEqual(
            [e["text"] for e in events if e["type"] == "speech_segment"],
            ["哈啊？！", "真的吗。"],
        )
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["payload"]["status"], "plain_text")
        self.assertEqual(final["payload"]["speech"], "哈啊？！真的吗。")

    def test_memcore_json_stream_extracts_speech_and_final_metadata(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = []
        events += stream.feed('{"memory_metadata":{"keywords":["英伟达"]},"speech":"哈')
        events += stream.feed('啊？！他说：\\"好吧。\\""}')
        events += stream.finish()

        self.assertEqual(
            [e["text"] for e in events if e["type"] == "speech_chunk"],
            ["哈", '啊？！他说："好吧。"'],
        )
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "speech_segment"],
            ["哈啊？！", '他说："好吧。"'],
        )
        metadata = [e for e in events if e["type"] == "metadata_ready"][0]
        self.assertEqual(metadata["memory_metadata"]["keywords"], ["英伟达"])
        self.assertEqual(events[-1]["payload"]["speech"], '哈啊？！他说："好吧。"')

    def test_stream_ignores_nested_speech_key(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = stream.feed('{"memory_metadata":{"speech":"bad"},"speech":"good。"}')
        events += stream.finish()

        self.assertEqual([e["text"] for e in events if e["type"] == "speech_chunk"], ["good。"])
        self.assertEqual(events[-1]["payload"]["speech"], "good。")

    def test_stream_missing_speech_finishes_with_invalid_contract(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = stream.feed('{"message":"not speech"}')
        events += stream.finish()

        self.assertEqual([e for e in events if e["type"] == "speech_chunk"], [])
        self.assertEqual(events[-1]["payload"]["status"], "invalid_contract")
        self.assertEqual(events[-1]["payload"]["speech"], "")


if __name__ == "__main__":
    unittest.main()
