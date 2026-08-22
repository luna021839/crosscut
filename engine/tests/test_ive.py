from __future__ import annotations

import unittest

from crosscut.ive import IVEMember


class IVEMemberTests(unittest.TestCase):
    def test_korean_and_english_aliases(self) -> None:
        self.assertEqual(IVEMember.parse("원영"), IVEMember.JANGWONYOUNG)
        self.assertEqual(IVEMember.parse("An Yujin"), IVEMember.ANYUJIN)

    def test_unknown_member_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown IVE member"):
            IVEMember.parse("unknown")


if __name__ == "__main__":
    unittest.main()
