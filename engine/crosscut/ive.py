"""IVE-specific product vocabulary and request validation."""

from __future__ import annotations

from enum import StrEnum


class IVEMember(StrEnum):
    ANYUJIN = "안유진"
    GAEUL = "가을"
    REI = "레이"
    JANGWONYOUNG = "장원영"
    LIZ = "리즈"
    LEESEO = "이서"

    @classmethod
    def parse(cls, value: str) -> "IVEMember":
        normalized = value.strip().casefold().replace(" ", "")
        aliases = {
            "안유진": cls.ANYUJIN,
            "유진": cls.ANYUJIN,
            "anyujin": cls.ANYUJIN,
            "yujin": cls.ANYUJIN,
            "가을": cls.GAEUL,
            "gaeul": cls.GAEUL,
            "레이": cls.REI,
            "rei": cls.REI,
            "장원영": cls.JANGWONYOUNG,
            "원영": cls.JANGWONYOUNG,
            "jangwonyoung": cls.JANGWONYOUNG,
            "wonyoung": cls.JANGWONYOUNG,
            "리즈": cls.LIZ,
            "liz": cls.LIZ,
            "이서": cls.LEESEO,
            "leeseo": cls.LEESEO,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown IVE member {value!r}; expected one of: {allowed}") from exc


class EditMode(StrEnum):
    GROUP = "group"
    MEMBER = "member"


class AspectRatio(StrEnum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
