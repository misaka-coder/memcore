"""温度开关动态贯穿三层写入提示词:开了三层都要 mood,关了三层一个字不提。"""

from __future__ import annotations

import unittest

from memcore.prompts import build_reinforcement_prompts, build_semantic_prompts, build_summary_prompts
from memcore.schema import MOOD_TAGS


def _systems(enable_flavor: bool) -> list[str]:
    return [
        build_summary_prompts(transcript="x", batch_size=1, enable_flavor=enable_flavor)[0],
        build_semantic_prompts(source_text="x", enable_flavor=enable_flavor)[0],
        build_reinforcement_prompts(existing_text="a", incoming_text="b", enable_flavor=enable_flavor)[0],
    ]


class FlavorShapesAllWritePrompts(unittest.TestCase):
    def test_flavor_on_injects_mood_into_all_three(self) -> None:
        for system in _systems(True):
            self.assertIn("mood_tags", system)
            self.assertIn("情感温度(已启用)", system)
            self.assertIn(MOOD_TAGS[0], system)  # 枚举确实列出来了

    def test_flavor_off_mentions_no_mood_anywhere(self) -> None:
        for system in _systems(False):
            self.assertNotIn("mood_tags", system)
            self.assertNotIn("情感温度", system)

    def test_welded_contract_intact_regardless_of_flavor(self) -> None:
        # 温度只是中间插入的一段;契约 + 时间锚点骨架不受影响。
        for enable in (True, False):
            for system in _systems(enable):
                self.assertIn("只输出一个合法 JSON 对象", system)
                self.assertIn("时间锚点规则", system)


if __name__ == "__main__":
    unittest.main()
