"""Chat Output Adapter:标准输出契约、普通文本解析、句末分段。"""

from __future__ import annotations

import unittest
import json

from memcore import StreamingSpeechParser, build_chat_output_contract_prompt, parse_chat_output, segment_speech


class SpeechSegmenter(unittest.TestCase):
    def test_pairs_survive_soft_limit_newlines_and_every_stream_boundary(self) -> None:
        pairs = list(zip("([{（【《「『“‘〈〔〖〘〚［｛｟«‹<", ")] }）】》」』”’〉〕〗〙〛］｝｠»›>".replace(" ", "")))
        pairs += [('"', '"'), ("'", "'"), ("`", "`"), ("```", "```")]
        for opener, closer in pairs:
            speech = f"{opener}Long English, with words!\nAnd more words?{closer}结束。下一句。"
            expected = [speech.split("下一句")[0].replace("\n", " "), "下一句。"]
            self.assertEqual(segment_speech(speech, max_chars=12), expected, opener)
            wire = json.dumps({"speech": speech}, ensure_ascii=False)
            for boundary in range(len(wire) + 1):
                parser = StreamingSpeechParser(mode="memcore_json", max_segment_chars=12)
                events = parser.feed(wire[:boundary]) + parser.feed(wire[boundary:]) + parser.finish()
                self.assertEqual(
                    [e["text"] for e in events if e["type"] == "speech_segment"], expected, (opener, boundary)
                )

    def test_nested_quotes_apostrophes_and_unclosed_source(self) -> None:
        cases = [
            ("她说：\"看《书里[还有'nested!']》吧。\"结束。", ["她说：\"看《书里[还有'nested!']》吧。\"结束。"]),
            ("Don't worry. John's here.", ["Don't worry.", "John's here."]),
            ("‘Don’t worry! It’s fine.’结束。", ["‘Don’t worry! It’s fine.’结束。"]),
            ('完成。"Next! Still quoted?"结束。', ["完成。", '"Next! Still quoted?"结束。']),
            ("2 < 3。下一句。", ["2 < 3。", "下一句。"]),
            ("[原文本来未闭合，继续输出。", ["[原文本来未闭合，继续输出。"]),
            ("很长的词语" * 50, ["很长的词语" * 50]),
        ]
        for speech, expected in cases:
            self.assertEqual(segment_speech(speech), expected)
            parser = StreamingSpeechParser(mode="plain")
            events = [event for char in speech for event in parser.feed(char)] + parser.finish()
            self.assertEqual([e["text"] for e in events if e["type"] == "speech_segment"], expected)

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

    def test_numbered_list_marker_stays_with_item(self) -> None:
        self.assertEqual(
            segment_speech("1. 先检查文件。2. 再执行转换。"),
            ["1. 先检查文件。", "2. 再执行转换。"],
        )

    def test_title_punctuation_is_not_an_outer_boundary(self) -> None:
        self.assertEqual(
            segment_speech("《孤独摇滚！》很好看。下一句。"),
            ["《孤独摇滚！》很好看。", "下一句。"],
        )

    def test_nested_quote_and_bracket_punctuation_stays_inside(self) -> None:
        self.assertEqual(
            segment_speech("她说：“我在看《孤独摇滚！》。”然后笑了。"),
            ["她说：“我在看《孤独摇滚！》。”然后笑了。"],
        )


class ChatOutputParser(unittest.TestCase):
    def test_memcore_json_extracts_speech_and_coerces_metadata(self) -> None:
        result = parse_chat_output(
            """
            {
              "speech": "哈啊？！真的吗。",
              "emotion": "happy",
              "memory_metadata": {
                "turn_intent": "memory_query",
                "memory_facets": ["preference", "bad"],
                "about_roles": ["user", "bad"],
                "entity_anchors": ["可乐", "饮料", "可乐"],
                "topic_terms": ["喜欢"],
                "retrieval_priority": "high",
                "mood_tags": ["warm"]
              },
              "debug": "ignored by memory core"
            }
            """,
            mode="memcore_json",
            enable_flavor=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.metadata_status, "accepted")
        self.assertTrue(result.metadata_present)
        self.assertEqual(result.speech, "哈啊？！真的吗。")
        self.assertEqual(result.segments, ["哈啊？！", "真的吗。"])
        self.assertEqual(result.presentation, {"emotion": "happy"})
        self.assertEqual(result.extra, {"debug": "ignored by memory core"})
        self.assertEqual(result.memory_metadata["turn_intent"], "memory_query")
        self.assertEqual(result.memory_metadata["memory_facets"], ["preference"])
        self.assertEqual(result.memory_metadata["about_roles"], ["user"])
        self.assertEqual(result.memory_metadata["entity_anchors"], ["可乐", "饮料"])
        self.assertEqual(result.memory_metadata["topic_terms"], ["喜欢"])
        self.assertEqual(result.memory_metadata["retrieval_priority"], "high")
        self.assertEqual(result.memory_metadata["mood_tags"], ["warm"])

    def test_flavor_off_strips_mood_tags(self) -> None:
        result = parse_chat_output(
            {"speech": "好。", "memory_metadata": {"mood_tags": ["warm"]}},
            mode="memcore_json",
            enable_flavor=False,
        )
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.memory_metadata["mood_tags"], [])

    def test_memcore_json_requires_speech(self) -> None:
        result = parse_chat_output({"memory_metadata": {"entity_anchors": ["可乐"]}}, mode="memcore_json")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "invalid_contract")
        self.assertEqual(result.reason, "speech_required")

    def test_missing_metadata_keeps_speech_without_claiming_model_acceptance(self) -> None:
        result = parse_chat_output({"speech": "照常回复。"}, mode="memcore_json")

        self.assertTrue(result.ok)
        self.assertEqual(result.speech, "照常回复。")
        self.assertEqual(result.metadata_status, "missing")
        self.assertFalse(result.metadata_present)

    def test_wrong_metadata_type_keeps_speech_and_reports_invalid_annotation(self) -> None:
        result = parse_chat_output({"speech": "仍然回复。", "memory_metadata": []}, mode="memcore_json")

        self.assertTrue(result.ok)
        self.assertEqual(result.speech, "仍然回复。")
        self.assertEqual(result.metadata_status, "invalid")
        self.assertTrue(result.metadata_present)

    def test_plain_text_auto_mode(self) -> None:
        result = parse_chat_output("普通回复。第二句。", mode="auto")
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "plain_text")
        self.assertEqual(result.speech, "普通回复。第二句。")
        self.assertEqual(result.segments, ["普通回复。", "第二句。"])
        self.assertEqual(result.memory_metadata, {})
        self.assertEqual(result.metadata_status, "plain")
        self.assertFalse(result.metadata_present)

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
            enable_flavor=False,
            enable_sentence_segments=True,
        )
        self.assertIn("字段固定为 speech, memory_metadata", prompt)
        self.assertIn("memory_facets", prompt)
        self.assertIn("about_roles", prompt)
        self.assertIn("entity_anchors", prompt)
        self.assertIn("mood_tags：输出空数组", prompt)
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
        self.assertIn("宿主指定的本轮记忆标注目标", prompt)
        self.assertIn('include_explicit=true, kind_patterns=["material.*"]', prompt)
        self.assertNotIn("confidence", prompt)
        self.assertIn("准确名称或别名", prompt)
        self.assertIn("优先保留具体、便于检索的名词和主题短语", prompt)
        self.assertIn("两组词去重", prompt)
        self.assertNotIn("speech_segments", prompt)


class StreamingOutputParser(unittest.TestCase):
    def test_closed_speech_emits_final_sentence_before_metadata(self) -> None:
        for chunks in (('{"speech":"第一句？！"',), ('{"speech":"第一句？', "！", '"')):
            with self.subTest(chunks=chunks):
                stream = StreamingSpeechParser(mode="memcore_json")
                events = []
                for chunk in chunks:
                    events += stream.feed(chunk)
                self.assertEqual([e["text"] for e in events if e["type"] == "speech_segment"], ["第一句？！"])
                self.assertFalse(any(e["type"] in {"metadata_ready", "final"} for e in events))
                events += stream.feed(',"memory_metadata":{"topic_terms":["测试"]}}')
                events += stream.finish()
                self.assertEqual([e["text"] for e in events if e["type"] == "speech_segment"], ["第一句？！"])
                self.assertEqual(events[-1]["payload"]["memory_metadata"]["topic_terms"], ["测试"])

    def test_closed_unpunctuated_tail_stays_pending_until_finish(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = stream.feed('{"speech":"完整句。尚未说完"')
        self.assertEqual([e["text"] for e in events if e["type"] == "speech_segment"], ["完整句。"])
        events += stream.feed(',"memory_metadata":')
        self.assertFalse(any(e["type"] == "metadata_ready" for e in events))
        final_events = stream.finish()
        self.assertEqual([e["text"] for e in final_events if e["type"] == "speech_segment"], ["尚未说完"])
        self.assertEqual(final_events[-1]["payload"]["status"], "output_unparsed")
        self.assertFalse(any(e["type"] == "metadata_ready" for e in final_events))

    def test_surrogate_pair_is_intact_at_every_chunk_boundary(self) -> None:
        raw = '{"speech":"\\uD83D\\uDE0A你好。","memory_metadata":{}}'
        for split in range(1, len(raw)):
            with self.subTest(split=split):
                stream = StreamingSpeechParser(mode="memcore_json")
                events = stream.feed(raw[:split]) + stream.feed(raw[split:]) + stream.finish()
                chunks = [e["text"] for e in events if e["type"] == "speech_chunk"]
                for chunk in chunks:
                    chunk.encode("utf-8")
                self.assertEqual("".join(chunks), "😊你好。")
                self.assertEqual([e["text"] for e in events if e["type"] == "speech_segment"], ["😊你好。"])
                self.assertEqual(events[-1]["payload"]["speech"], "😊你好。")

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

    def test_plain_stream_keeps_title_punctuation_across_chunks(self) -> None:
        stream = StreamingSpeechParser(mode="plain")
        events = stream.feed("我在看《孤独摇滚！")
        self.assertEqual([event for event in events if event["type"] == "speech_segment"], [])

        events += stream.feed("》这部作品很好看。下")
        events += stream.feed("一句。")
        events += stream.finish()

        self.assertEqual(
            [event["text"] for event in events if event["type"] == "speech_segment"],
            ["我在看《孤独摇滚！》这部作品很好看。", "下一句。"],
        )

    def test_plain_stream_keeps_numbered_marker_across_chunks(self) -> None:
        stream = StreamingSpeechParser(mode="plain")
        events = stream.feed("1.")
        events += stream.feed(" 先检查文件。")
        events += stream.finish()

        self.assertEqual(
            [event["text"] for event in events if event["type"] == "speech_segment"],
            ["1. 先检查文件。"],
        )

    def test_plain_stream_keeps_numbered_marker_across_character_deltas(self) -> None:
        stream = StreamingSpeechParser(mode="plain")
        events = []
        for character in "1. 先检查文件。2. 再执行转换。":
            events += stream.feed(character)
        events += stream.finish()

        self.assertEqual(
            [event["text"] for event in events if event["type"] == "speech_segment"],
            ["1. 先检查文件。", "2. 再执行转换。"],
        )

    def test_memcore_json_stream_extracts_speech_and_final_metadata(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = []
        events += stream.feed('{"memory_metadata":{"entity_anchors":["英伟达"]},"speech":"哈')
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
        self.assertEqual(metadata["memory_metadata"]["entity_anchors"], ["英伟达"])
        self.assertEqual(metadata["metadata_status"], "accepted")
        self.assertTrue(metadata["metadata_present"])
        self.assertEqual(events[-1]["payload"]["speech"], '哈啊？！他说："好吧。"')
        self.assertEqual(events[-1]["payload"]["metadata_status"], "accepted")
        self.assertTrue(events[-1]["payload"]["metadata_present"])

    def test_stream_missing_metadata_keeps_speech_and_reports_missing(self) -> None:
        stream = StreamingSpeechParser(mode="memcore_json")
        events = stream.feed('{"speech":"正常播放。"}')
        events += stream.finish()

        final = events[-1]["payload"]
        self.assertEqual(final["status"], "parsed")
        self.assertEqual(final["speech"], "正常播放。")
        self.assertEqual(final["metadata_status"], "missing")
        self.assertFalse(final["metadata_present"])
        metadata = [event for event in events if event["type"] == "metadata_ready"][0]
        self.assertEqual(metadata["metadata_status"], "missing")
        self.assertFalse(metadata["metadata_present"])

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
