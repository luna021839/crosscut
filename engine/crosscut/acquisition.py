"""Deterministic, cookie-free YouTube source acquisition for IVE edits.

The acquisition boundary intentionally stays small: public YouTube metadata is
queried through the installed ``yt-dlp`` executable, candidates are filtered
before any media is downloaded, and selected files are materialized in a
visible per-project cache.  This module never supplies cookies, bypasses DRM,
uploads media, or executes a shell.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from .events import EventReporter
from .group_source_quality import (
    GROUP_RELATIVE_QUALITY_GATE_VERSION,
    GROUP_RELATIVE_QUALITY_TARGET,
    GroupSourceQualityAssessment,
    RankedGroupSource,
    analyze_group_source_quality,
    select_group_quality_pool,
)
from .ive import AspectRatio, EditMode, IVEMember, SourceMode
from .media import MediaInfo, MediaProbeError, probe_media
from .request import (
    AnalysisRequest,
    DEFAULT_AUTOMATIC_SOURCE_COUNT,
    MemberAnchorRequest,
)
from .ytdlp_runtime import discover_ytdlp_frontend


class AcquisitionError(RuntimeError):
    """A clear user-facing source discovery or download failure."""


class _MemberAcquisitionIdentityReauthenticationRequired(AcquisitionError):
    """Cached broadcast metadata needs a fresh authoritative lookup."""


_ACQUISITION_IDENTITY_AUTHORITY_SEAL = object()
_FRESH_METADATA_IDENTITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class _VerifiedFreshMetadataIdentity:
    """Process-local proof for one live metadata observation.

    Serialized acquisition manifests deliberately cannot recreate this value.
    It is issued only while consuming a direct query response or a playlist
    response that passed the pinned-owner network validation in this process.
    """

    video_id: str
    title: str
    channel_id: str | None
    broadcast_program: str
    broadcast_date: str
    broadcast_date_basis: str
    origin: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _FRESH_METADATA_IDENTITY_SEAL:
            raise TypeError("fresh metadata identity must be issued by acquisition")
        if self.origin not in {
            "query_search",
            "validated_network_catalog",
        }:
            raise ValueError("fresh metadata identity origin is invalid")
        if self.broadcast_date_basis not in {
            "title",
            "upload_date",
            "timestamp",
        }:
            raise ValueError("fresh metadata broadcast date basis is invalid")


@dataclass(frozen=True, slots=True)
class VerifiedAcquisitionSourceIdentity:
    source_index: int
    source_path: str
    video_id: str
    broadcast_date: str
    broadcast_program: str
    source_size_bytes: int
    source_sha256: str

    @property
    def broadcast_slot(self) -> str:
        return f"{self.broadcast_date}:{self.broadcast_program}"


@dataclass(frozen=True, slots=True)
class VerifiedMemberAcquisitionIdentityAuthority:
    group: str
    song: str
    target_member: str
    manifest_path: str
    manifest_size_bytes: int
    manifest_sha256: str
    sources: tuple[VerifiedAcquisitionSourceIdentity, ...]
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _ACQUISITION_IDENTITY_AUTHORITY_SEAL:
            raise TypeError(
                "member acquisition identity authority must be issued by acquisition"
            )


def is_verified_member_acquisition_identity_authority(
    value: object,
) -> bool:
    return bool(
        isinstance(value, VerifiedMemberAcquisitionIdentityAuthority)
        and value._seal is _ACQUISITION_IDENTITY_AUTHORITY_SEAL
    )


@dataclass(frozen=True, slots=True)
class AcquisitionCandidate:
    video_id: str
    title: str
    channel: str
    channel_id: str | None
    duration_seconds: float
    webpage_url: str
    queries: tuple[str, ...]
    selection_tier: int
    trust_score: int
    broadcast_date: str | None = None
    broadcast_program: str | None = None
    broadcast_date_basis: str | None = None
    official_channel_signal: str = "none"
    rejection_reasons: tuple[str, ...] = ()
    fallback_reason: str | None = None
    discovery_sources: tuple[str, ...] = ("query_search",)
    _fresh_metadata_identities: tuple[_VerifiedFreshMetadataIdentity, ...] = ()
    # Exact current metadata observations are deliberately process-local.
    # Cache-derived candidates leave this empty, so a serialized manifest
    # alone can never mint a download-suppression decision.
    _metadata_observation_sha256: tuple[str, ...] = ()

    @property
    def query(self) -> str:
        return self.queries[0]

    @property
    def selectable(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class AcquisitionArtifact:
    source_paths: tuple[Path, ...]
    member_anchors: tuple[MemberAnchorRequest, ...]
    manifest_path: Path
    manifest: dict[str, Any]
    member_identity_authority: (
        VerifiedMemberAcquisitionIdentityAuthority | None
    ) = None
    member_identity_authority_error: str | None = None


class _MediaContractError(AcquisitionError):
    """Downloaded media violated a request-scoped, fail-closed contract."""

    def __init__(
        self,
        message: str,
        *,
        hard_rejection_reason: str,
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.hard_rejection_reason = hard_rejection_reason
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class _ValidatedPersistentMediaContractRejection:
    evidence: dict[str, Any]
    quarantine_path: Path
    quarantine_file_identity: tuple[int, int, int, int, int]


class _PersistentMediaContractRejectionSkip(AcquisitionError):
    """Internal control flow for a fully revalidated persisted rejection."""

    def __init__(self, evidence: dict[str, Any]) -> None:
        self.evidence = evidence
        reason = evidence["hard_rejection_reason"]
        video_id = evidence["video_id"]
        super().__init__(
            f"{video_id}: unchanged persistent media-contract rejection ({reason})"
        )


class _CatalogCompletenessError(AcquisitionError):
    """A pinned playlist response is partial or regresses a valid cache."""


RunProcess = Callable[..., subprocess.CompletedProcess[str]]
GroupSourceQualityAnalyzer = Callable[..., GroupSourceQualityAssessment]

_VIDEO_ID = re.compile(r"^[0-9A-Za-z_-]{6,64}$")
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
_MINIMUM_DURATION_SECONDS = 95.0
_MAXIMUM_DURATION_SECONDS = 480.0
_SELECTION_POLICY_VERSION = 5
_LEGACY_SELECTION_POLICY_VERSION = 4
_LATEST_SONG_ACQUISITION_POLICY_VERSION = 2
_LEGACY_LATEST_SONG_ACQUISITION_POLICY_VERSION = 1
_PROMOTION_CLUSTER_MAXIMUM_GAP_DAYS = 28
_CURRENT_HARD_TITLE_AUDIT_POLICY = (
    "selected_title_encore_winner_afterstage_and_music_bank_global_special_v3"
)
_MEMBER_PORTRAIT_ASPECT_CONTRACT_VERSION = 2
_AUTOMATIC_MEDIA_QUALITY_CONTRACT_VERSION = 1
_CONTENT_INTEGRITY_CONTRACT_VERSION = 1
_PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION = 1
_PERSISTENT_MEDIA_CONTRACT_REJECTION_REASONS = frozenset(
    {
        "automatic_member_portrait_first_video_stream_missing",
        "automatic_member_portrait_dimensions_unavailable",
        "automatic_member_portrait_aspect_ratio_mismatch",
        "automatic_member_portrait_crop_headroom_below_production",
        "automatic_group_landscape_first_video_stream_missing",
        "automatic_group_landscape_dimensions_unavailable",
        "automatic_group_landscape_aspect_ratio_mismatch",
        "automatic_video_resolution_below_minimum",
        "automatic_video_frame_rate_unavailable",
        "automatic_video_frame_rate_below_minimum",
    }
)
_AUTOMATIC_MEMBER_PORTRAIT_RATIO = 9.0 / 16.0
_AUTOMATIC_MEMBER_PORTRAIT_RATIO_TOLERANCE = 0.01
_AUTOMATIC_GROUP_LANDSCAPE_RATIO = 16.0 / 9.0
_AUTOMATIC_GROUP_LANDSCAPE_RATIO_TOLERANCE = 0.03
_AUTOMATIC_MINIMUM_SHORT_EDGE = 720
_AUTOMATIC_MINIMUM_LONG_EDGE = 1280
_AUTOMATIC_MINIMUM_FRAME_RATE = 23.0
_PROFESSIONAL_DOWNLOAD_PROFILE = "professional_crop_headroom_4k_first_v1"
_FULL_HD_DOWNLOAD_PROFILE = "full_hd_avc_compatibility_v1"
_DOWNLOAD_PROFILES = frozenset(
    {_PROFESSIONAL_DOWNLOAD_PROFILE, _FULL_HD_DOWNLOAD_PROFILE}
)
_PROFESSIONAL_MAXIMUM_LONG_EDGE = 3840
_PROFESSIONAL_MAXIMUM_LOSSLESS_ZOOM = 2.0
_GROUP_PRODUCTION_WIDTH = 1920
_GROUP_PRODUCTION_HEIGHT = 1080
_MEMBER_PRODUCTION_WIDTH = 1080
_MEMBER_PRODUCTION_HEIGHT = 1920
_MEMBER_MINIMUM_LOSSLESS_CROP_HEADROOM_ZOOM = 1.0
_IVE_STAGE_PLAYLIST_ID = "PLPhtNKiHTFyob6KUBxHdAdHlliwZTMBFc"
_IVE_STAGE_PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=" + _IVE_STAGE_PLAYLIST_ID
)
_IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID = "UC-Fnix71vRP64WXeo0ikd0Q"
_IVE_STAGE_PLAYLIST_OWNER_NAME = "IVE"
_CATALOG_CACHE_SCHEMA_VERSION = 1
_CATALOG_CACHE_POLICY_VERSION = 1
_CATALOG_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
_CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT = 1500
_CATALOG_CACHE_INDEX_FILENAME = "official-ive-stage-playlist-index-v1.json"
_CATALOG_CACHE_OBJECT_MAXIMUM_BYTES = 32 * 1024 * 1024
_CATALOG_OBJECT_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _download_format_for_profile(profile: str) -> str:
    if profile == _PROFESSIONAL_DOWNLOAD_PROFILE:
        # 4K broadcast uploads commonly use AV1/VP9 rather than H.264.  Prefer
        # the highest source pixels first, retain M4A audio when available,
        # and keep the proven Full-HD AVC path as a decoder-safe fallback.
        return (
            "bv*[width<=3840][height<=3840]+ba[ext=m4a]/"
            "bv*[width<=3840][height<=3840]+ba/"
            "bv*[vcodec^=avc1][width<=1920][height<=1920]+ba[ext=m4a]/"
            "bv*[ext=mp4][width<=1920][height<=1920]+ba[ext=m4a]/"
            "b[ext=mp4][width<=1920][height<=1920]/"
            "best[width<=1920][height<=1920]"
        )
    if profile == _FULL_HD_DOWNLOAD_PROFILE:
        return (
            "bv*[vcodec^=avc1][width<=1920][height<=1920]+ba[ext=m4a]/"
            "bv*[ext=mp4][width<=1920][height<=1920]+ba[ext=m4a]/"
            "b[ext=mp4][width<=1920][height<=1920]/"
            "best[width<=1920][height<=1920]"
        )
    raise ValueError(f"unsupported yt-dlp download profile: {profile}")

_TRUSTED_CHANNEL_MARKERS = (
    "mnetkpop",
    "m2",
    "kbs kpop",
    "kbs world tv",
    "mbckpop",
    "all the k-pop",
    "sbs kpop",
    "sbs inkigayo",
    "the k-pop",
    "arirang k-pop",
    "kocowa tv",
)
_TRUSTED_CHANNEL_IDS = {
    "UCbD8EppRX3ZwJSou-TVo90A",
    "UCeLPm9yH_a_QH8n6445G-Ow",
    "UCe52oeb7Xv_KaJsEzcKXJJg",
    "UCS_hnpJLQTvBkqALgapi_4g",
    "UCoRXPcv8XK5fAplLbk9PTww",
    "UCTQVIXvcHrR9jYoJ6qaBAow",
    "UCM3jwNRfl5-W8VzgT6DsaEQ",
    "UC5BMQOsAB8hKUyHu9KI6yig",
    # ALL THE K-POP / Show Champion
    "UCPde4guD9yFBRzkxk2PatoA",
    # ARIRANG K-POP / Simply K-Pop
    "UC_6zPf4uuteHFFctnQtEaXQ",
}
_MUSIC_SHOW_PROGRAM_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("m_countdown", ("m countdown", "mcountdown", "엠카운트다운")),
    ("music_bank", ("music bank", "musicbank", "뮤직뱅크", "뮤직 뱅크")),
    ("music_core", ("music core", "musiccore", "음악중심", "음악 중심")),
    ("inkigayo", ("inkigayo", "인기가요")),
    ("show_champion", ("show champion", "쇼챔피언", "쇼 챔피언")),
    ("the_show", ("the show", "더쇼", "더 쇼")),
    ("simply_kpop", ("simply k-pop", "simply kpop")),
)
_FANCAM_MARKERS = ("fancam", "facecam", "fan cam", "직캠", "입덕직캠")
_MEMBER_CLOSEUP_MARKERS = (
    "facecam",
    "face cam",
    "face-cam",
    "페이스캠",
    "얼빡",
    "close-up",
    "close up",
    "closeup",
    "클로즈업",
    "클로즈 업",
)
_FALLBACK_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intro_plus", ("intro +", "intro+", "인트로 +", "인트로+")),
    ("concert_or_festival", ("concert", "festival", "콘서트", "페스티벌", "대학축제")),
    ("dance_practice", ("dance practice", "practice video", "안무 영상", "댄스 프랙티스")),
    ("stage_mix", ("stage mix", "stagemix", "무대모음", "무대 모음", "무대교차편집")),
    (
        "music_video",
        (
            "official mv",
            "official music video",
            "music video",
            " m/v",
            " mv ",
            "mv ",
            "[mv]",
            "(mv)",
            "뮤직비디오",
        ),
    ),
)

_WINNER_AFTERSTAGE_UNRELEASED_MARKERS = (
    "미방분",
    "미방송분",
    "unreleased",
)
_WINNER_AFTERSTAGE_WIN_MARKERS = (
    re.compile(r"(?<![0-9가-힣])1\s*위(?![0-9가-힣])"),
    re.compile(r"(?<![0-9a-z])no\.?\s*1(?![0-9a-z])"),
    re.compile(r"(?<![0-9a-z])1st\s+place(?![0-9a-z])"),
    re.compile(r"(?<![0-9a-z])first\s+place(?![0-9a-z])"),
    re.compile(r"(?<![0-9a-z])winner(?![0-9a-z])"),
)
_WINNER_FANCAM_MARKER = re.compile(
    r"(?<![0-9가-힣])1\s*위"
    r"[^0-9a-z가-힣]{0,12}"
    r"(?:(?:4k|8k|uhd|1080p|2160p|60fps)[^0-9a-z가-힣]{0,12})?"
    r"직\s*캠"
)
_MUSIC_BANK_GLOBAL_SPECIAL_MARKER = re.compile(
    r"(?<![0-9a-z가-힣])(?:music\s*bank|뮤직\s*뱅크)"
    r"[\s|/·:._-]+(?:in|인)[\s|/·:._-]+"
    r"(?!(?:4k|8k|uhd|full\s*hd|1080p|2160p|60fps)(?![0-9a-z]))"
    r"(?=[0-9a-z가-힣])"
)

_MEMBER_ALIASES: dict[IVEMember, tuple[str, ...]] = {
    IVEMember.ANYUJIN: ("안유진", "유진", "an yujin", "anyujin", "yujin"),
    IVEMember.GAEUL: ("가을", "gaeul"),
    IVEMember.REI: ("레이", "rei"),
    IVEMember.JANGWONYOUNG: ("장원영", "원영", "jang wonyoung", "jangwonyoung", "wonyoung"),
    IVEMember.LIZ: ("리즈", "liz"),
    IVEMember.LEESEO: ("이서", "leeseo"),
}

_SONG_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("eleven", "일레븐"),
    ("love dive", "lovedive", "러브 다이브", "러브다이브"),
    ("after like", "afterlike", "애프터 라이크", "애프터라이크"),
    ("kitsch", "키치"),
    ("i am", "iam", "아이 엠", "아이엠"),
    ("either way", "eitherway", "이더 웨이", "이더웨이"),
    (
        "off the record",
        "offtherecord",
        "오프 더 레코드",
        "오프더레코드",
    ),
    ("baddie", "배디"),
    ("heya", "해야"),
    ("accendio", "아센디오"),
    (
        "rebel heart",
        "rebelheart",
        "레블 하트",
        "레블하트",
        "레벨 하트",
        "레벨하트",
    ),
    ("attitude", "애티튜드"),
    ("xoxz", "엑스오엑스지", "엑스 오 엑스 지"),
    ("bang bang", "bangbang", "뱅 뱅", "뱅뱅", "방방"),
    ("blackhole", "black hole", "블랙홀", "블랙 홀"),
)


def _fold(value: str) -> str:
    return " ".join(value.casefold().replace("\u3000", " ").split())


def _compact(value: str) -> str:
    return "".join(character for character in _fold(value) if character.isalnum())


def _current_hard_title_rejection_reason(title: str) -> str | None:
    """Return a current-policy reason for explicitly non-weekly footage.

    A weekly-show result or winner word alone is not enough: ordinary main-stage
    uploads can mention a later win.  The special MBC winner-afterstage format is
    admitted only by the conjunction of an unreleased marker and an explicit
    first-place marker.  Explicit Korean/English encore labels are always hard
    rejections.  A location-qualified ``Music Bank in <city>`` label denotes
    the broadcaster's global concert/special series rather than the weekly
    Music Bank programme, so it is rejected without treating an ordinary
    ``Music Bank`` title as a special.
    """

    folded = _fold(title)
    if "앵콜" in folded or re.search(
        r"(?<![0-9a-z])encore"
        r"(?:[\s._-]*(?:cam|fancam|fullcam|stage|clip|ver(?:sion)?))?"
        r"(?:[\s._-]*(?:4k|8k|uhd|1080p|2160p|60fps))?"
        r"(?![0-9a-z])",
        folded,
    ) is not None:
        return "encore"
    if _MUSIC_BANK_GLOBAL_SPECIAL_MARKER.search(folded) is not None:
        return "music_bank_global_special"
    if _WINNER_FANCAM_MARKER.search(folded) is not None:
        return "winner_afterstage"
    if any(
        marker in folded for marker in _WINNER_AFTERSTAGE_UNRELEASED_MARKERS
    ) and any(
        pattern.search(folded) is not None
        for pattern in _WINNER_AFTERSTAGE_WIN_MARKERS
    ):
        return "winner_afterstage"
    return None


def _song_aliases(song: str) -> tuple[str, ...]:
    compact_song = _compact(song)
    for aliases in _SONG_ALIAS_GROUPS:
        if compact_song in {_compact(alias) for alias in aliases}:
            return tuple(dict.fromkeys(aliases))
    return (song,)


def _known_song_alias_group(song: str) -> tuple[str, ...] | None:
    compact_song = _compact(song)
    return next(
        (
            aliases
            for aliases in _SONG_ALIAS_GROUPS
            if compact_song in {_compact(alias) for alias in aliases}
        ),
        None,
    )


def _contains_alias(value: str, aliases: Iterable[str]) -> bool:
    folded_value = _fold(value)
    value_tokens = re.findall(r"[0-9a-z]+|[가-힣]+", folded_value)
    for alias in aliases:
        folded_alias = _fold(alias)
        alias_tokens = re.findall(r"[0-9a-z]+|[가-힣]+", folded_alias)
        compact_alias = _compact(alias)
        # Exact normalized tokens support future, unregistered song names
        # across case, whitespace, and punctuation without ever accepting a
        # substring inside another word. This also keeps short Hangul member
        # names safe: "레이" is not the complete token "릴레이".
        if compact_alias in value_tokens:
            return True
        if alias_tokens and any(
            value_tokens[index : index + len(alias_tokens)] == alias_tokens
            for index in range(len(value_tokens) - len(alias_tokens) + 1)
        ):
            return True
    return False


def _music_show_program(title: str) -> str | None:
    """Return one canonical weekly-show key from title metadata."""

    for program, markers in _MUSIC_SHOW_PROGRAM_MARKERS:
        if _contains_alias(title, markers):
            return program
    return None


def canonical_music_show_program(title: str) -> str | None:
    """Expose the acquisition policy's canonical programme identity."""

    if not isinstance(title, str):
        raise TypeError("title must be a string")
    return _music_show_program(title)


def _has_ive_title_signal(title: str) -> bool:
    folded = _fold(title)
    return "아이브" in folded or re.search(
        r"(?<![0-9a-z])ive(?![0-9a-z])", folded
    ) is not None


def _official_channel_signal(channel: str, channel_id: str | None) -> str:
    if channel_id in _TRUSTED_CHANNEL_IDS:
        return "allowlisted_channel_id"
    folded = _fold(channel)
    compact = _compact(channel)
    if any(
        marker in folded or _compact(marker) in compact
        for marker in _TRUSTED_CHANNEL_MARKERS
    ):
        return "trusted_channel_name"
    return "none"


def _broadcast_date(title: str) -> str | None:
    separated = re.search(
        r"(?<!\d)(20\d{2}|\d{2})[._-](\d{1,2})[._-](\d{1,2})(?!\d)",
        title,
    )
    if separated is not None:
        year = int(separated.group(1))
        if year < 100:
            year += 2000
        try:
            return datetime(
                year,
                int(separated.group(2)),
                int(separated.group(3)),
            ).date().isoformat()
        except ValueError:
            pass
    for pattern, date_format in (
        (r"(?<!\d)(20\d{6})(?!\d)", "%Y%m%d"),
        (r"(?<!\d)(\d{6})(?!\d)", "%y%m%d"),
    ):
        for match in re.finditer(pattern, title):
            try:
                return datetime.strptime(match.group(1), date_format).date().isoformat()
            except ValueError:
                continue
    return None


def _metadata_broadcast_date_with_basis(
    entry: Mapping[str, Any],
    title: str,
) -> tuple[str | None, str | None]:
    title_date = _broadcast_date(title)
    if title_date is not None:
        return (title_date, "title")
    upload_date = entry.get("upload_date")
    if isinstance(upload_date, str):
        try:
            return (
                datetime.strptime(upload_date, "%Y%m%d").date().isoformat(),
                "upload_date",
            )
        except ValueError:
            pass
    timestamp = entry.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        try:
            return (
                datetime.fromtimestamp(float(timestamp), tz=UTC).date().isoformat(),
                "timestamp",
            )
        except (OverflowError, OSError, ValueError):
            pass
    return (None, None)


def _metadata_broadcast_date(entry: dict[str, Any], title: str) -> str | None:
    return _metadata_broadcast_date_with_basis(entry, title)[0]


def _metadata_queries(request: AnalysisRequest) -> tuple[str, ...]:
    aliases = _song_aliases(request.song)
    english_song = aliases[0]
    korean_song = next((alias for alias in aliases if any("\uac00" <= c <= "\ud7a3" for c in alias)), request.song)
    if request.edit_mode is EditMode.GROUP:
        subject = "IVE"
        suffix = "stage"
        generic = (
            f'IVE "{english_song}" stage',
            f'아이브 "{korean_song}" 무대',
        )
    else:
        assert request.target_member is not None
        member_aliases = _MEMBER_ALIASES[request.target_member]
        english_member = next(
            (alias for alias in member_aliases if alias.isascii() and len(alias) > 3),
            member_aliases[-1],
        )
        subject = f"IVE {english_member}"
        suffix = "fancam"
        generic = (
            f'{subject} "{english_song}" fancam',
            f'아이브 {request.target_member.value} "{korean_song}" 직캠',
        )
    shows = (
        "M COUNTDOWN 엠카운트다운",
        "Music Bank 뮤직뱅크",
        "Music Core 음악중심",
        "Inkigayo 인기가요",
        "THE SHOW 더쇼",
        "Show Champion 쇼챔피언",
    )
    return (*generic, *(f'{subject} "{english_song}" {show} {suffix}' for show in shows))


class YTDLPClient:
    """Minimal subprocess adapter; injectable so unit tests never use network."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        runner: RunProcess = subprocess.run,
        download_profile: str = _PROFESSIONAL_DOWNLOAD_PROFILE,
    ) -> None:
        if executable is not None:
            command = (executable,)
            display = executable
        else:
            frontend = discover_ytdlp_frontend()
            command = frontend.command if frontend is not None else ()
            display = frontend.display if frontend is not None else ""
        if not command:
            raise AcquisitionError(
                "자동 소스 수집에 yt-dlp가 필요합니다. yt-dlp를 설치한 뒤 다시 시도하세요."
            )
        self.command = command
        self.executable = display
        self._runner = runner
        normalized_profile = str(download_profile).strip()
        if normalized_profile not in _DOWNLOAD_PROFILES:
            raise ValueError(
                f"unsupported yt-dlp download profile: {normalized_profile}"
            )
        self.download_profile = normalized_profile
        self.download_format = _download_format_for_profile(normalized_profile)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        command = (
            *self.command,
            "--no-config",
            "--no-cookies",
            "--ignore-errors",
            "--flat-playlist",
            "--dump-single-json",
            f"ytsearch{limit}:{query}",
        )
        try:
            completed = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AcquisitionError(f"YouTube 검색 실패 ({query}): {exc}") from exc
        if completed.returncode != 0 and not completed.stdout.strip():
            detail = (completed.stderr or "yt-dlp returned no metadata").strip().splitlines()[-1]
            raise AcquisitionError(f"YouTube 검색 실패 ({query}): {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AcquisitionError(f"YouTube 검색 응답을 읽지 못했습니다 ({query}).") from exc
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        return [entry for entry in entries if isinstance(entry, dict)]

    def playlist_catalog(self, playlist_id: str) -> dict[str, Any]:
        """Return flat metadata for one explicitly pinned public playlist."""

        if playlist_id != _IVE_STAGE_PLAYLIST_ID:
            raise AcquisitionError("지원하지 않는 플레이리스트 카탈로그입니다.")
        command = (
            *self.command,
            "--no-config",
            "--no-cookies",
            "--ignore-errors",
            "--flat-playlist",
            "--dump-single-json",
            _IVE_STAGE_PLAYLIST_URL,
        )
        try:
            completed = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AcquisitionError(
                f"IVE STAGE 플레이리스트 메타데이터 수집 실패: {exc}"
            ) from exc
        if completed.returncode != 0 and not completed.stdout.strip():
            detail = (
                completed.stderr or "yt-dlp returned no playlist metadata"
            ).strip().splitlines()[-1]
            raise AcquisitionError(
                f"IVE STAGE 플레이리스트 메타데이터 수집 실패: {detail}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AcquisitionError(
                "IVE STAGE 플레이리스트 메타데이터 응답을 읽지 못했습니다."
            ) from exc
        if not isinstance(payload, dict):
            raise AcquisitionError(
                "IVE STAGE 플레이리스트 메타데이터가 객체가 아닙니다."
            )
        payload["_crosscut_ytdlp_returncode"] = completed.returncode
        return payload

    def download(self, candidate: AcquisitionCandidate, directory: Path) -> Path:
        videos_directory = directory / "videos"
        videos_directory.mkdir(parents=True, exist_ok=True)
        if _cached_video_paths(videos_directory, candidate.video_id):
            raise AcquisitionError(
                f"{candidate.title}: 인증되지 않은 동일 ID 캐시 파일이 "
                "남아 있어 다운로드를 중단했습니다."
            )
        template = str(videos_directory / f"{candidate.video_id}.%(ext)s")
        command = (
            *self.command,
            "--no-config",
            "--no-cookies",
            "--no-playlist",
            "--no-overwrites",
            "--continue",
            "--no-warnings",
            "--quiet",
            "--format",
            self.download_format,
            "--merge-output-format",
            "mp4",
            "--output",
            template,
            "--print",
            "after_move:filepath",
            candidate.webpage_url,
        )
        try:
            completed = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AcquisitionError(f"{candidate.title}: 다운로드 실패 ({exc})") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or "yt-dlp download failed").strip().splitlines()[-1]
            raise AcquisitionError(f"{candidate.title}: 다운로드 실패 ({detail})")
        printed_paths = [Path(line.strip()).expanduser() for line in completed.stdout.splitlines() if line.strip()]
        for path in reversed(printed_paths):
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
                return path.resolve()
        cached = _cached_video_path(videos_directory, candidate.video_id)
        if cached is None:
            raise AcquisitionError(f"{candidate.title}: 다운로드 파일을 찾지 못했습니다.")
        return cached


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_catalog_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _normalized_catalog_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    video_id = entry.get("id")
    if not isinstance(video_id, str) or _VIDEO_ID.fullmatch(video_id) is None:
        return None
    normalized: dict[str, Any] = {"id": video_id}
    for key in (
        "title",
        "channel",
        "channel_id",
        "uploader",
        "uploader_id",
        "upload_date",
    ):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    for key in ("webpage_url", "url"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            normalized[key] = value
    for key in ("duration", "timestamp", "release_timestamp"):
        value = entry.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            normalized[key] = value
    return normalized


def _normalized_catalog_object(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("id") != _IVE_STAGE_PLAYLIST_ID:
        raise AcquisitionError(
            "IVE STAGE 플레이리스트 ID가 고정 계약과 다릅니다."
        )
    # The playlist owner's channel ID authenticates only this catalog. It is
    # deliberately never copied into an entry or used as source-channel trust.
    if payload.get("channel_id") != _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID:
        raise AcquisitionError(
            "IVE STAGE 플레이리스트 소유자 채널 ID가 고정 계약과 다릅니다."
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise AcquisitionError(
            "IVE STAGE 플레이리스트 영상 목록을 읽지 못했습니다."
        )
    entries = [
        normalized
        for entry in raw_entries
        if isinstance(entry, dict)
        and (normalized := _normalized_catalog_entry(entry)) is not None
    ]
    if not entries:
        raise AcquisitionError("IVE STAGE 플레이리스트가 비어 있습니다.")
    return {
        "schema_version": _CATALOG_CACHE_SCHEMA_VERSION,
        "playlist": {
            "id": _IVE_STAGE_PLAYLIST_ID,
            "owner_channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
            "owner_name": _IVE_STAGE_PLAYLIST_OWNER_NAME,
        },
        "raw_entry_count": len(raw_entries),
        "normalization_rejected_count": len(raw_entries) - len(entries),
        "entry_count": len(entries),
        "entries": entries,
    }


def _validated_cached_catalog_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AcquisitionError("카탈로그 객체가 JSON 객체가 아닙니다.")
    playlist = payload.get("playlist")
    raw_entries = payload.get("entries")
    raw_count = payload.get("raw_entry_count")
    rejected_count = payload.get("normalization_rejected_count")
    entry_count = payload.get("entry_count")
    if (
        payload.get("schema_version") != _CATALOG_CACHE_SCHEMA_VERSION
        or playlist
        != {
            "id": _IVE_STAGE_PLAYLIST_ID,
            "owner_channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
            "owner_name": _IVE_STAGE_PLAYLIST_OWNER_NAME,
        }
        or not isinstance(raw_entries, list)
        or not isinstance(raw_count, int)
        or isinstance(raw_count, bool)
        or not isinstance(rejected_count, int)
        or isinstance(rejected_count, bool)
        or not isinstance(entry_count, int)
        or isinstance(entry_count, bool)
    ):
        raise AcquisitionError("카탈로그 객체 계약이 일치하지 않습니다.")
    normalized_entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise AcquisitionError("카탈로그 항목이 JSON 객체가 아닙니다.")
        normalized = _normalized_catalog_entry(entry)
        if normalized is None or normalized != entry:
            raise AcquisitionError("카탈로그 항목이 정규화 계약과 다릅니다.")
        normalized_entries.append(normalized)
    if (
        not normalized_entries
        or entry_count != len(normalized_entries)
        or raw_count != entry_count + rejected_count
        or rejected_count < 0
    ):
        raise AcquisitionError("카탈로그 항목 수 계약이 일치하지 않습니다.")
    expected = {
        "schema_version": _CATALOG_CACHE_SCHEMA_VERSION,
        "playlist": playlist,
        "raw_entry_count": raw_count,
        "normalization_rejected_count": rejected_count,
        "entry_count": entry_count,
        "entries": normalized_entries,
    }
    if payload != expected:
        raise AcquisitionError("카탈로그 객체에 허용되지 않은 필드가 있습니다.")
    return expected


def _catalog_cache_root(request: AnalysisRequest) -> Path:
    if request.catalog_cache_directory is not None:
        return request.catalog_cache_directory
    assert request.source_cache_directory is not None
    return (request.source_cache_directory.parent / "_catalog").resolve()


def _read_catalog_cache(
    root: Path,
    *,
    now: datetime,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "cache_root": str(root),
        "cache_state": "miss",
        "cache_error": None,
    }
    try:
        if root.is_symlink():
            raise AcquisitionError("카탈로그 캐시 루트가 심볼릭 링크입니다.")
        index_path = root / _CATALOG_CACHE_INDEX_FILENAME
        if not index_path.exists():
            return None, diagnostics
        if index_path.is_symlink() or not index_path.is_file():
            raise AcquisitionError("카탈로그 캐시 인덱스가 일반 파일이 아닙니다.")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise AcquisitionError("카탈로그 캐시 인덱스가 JSON 객체가 아닙니다.")
        object_sha256 = index.get("object_sha256")
        expected_relative = (
            f"objects/{object_sha256}.json"
            if isinstance(object_sha256, str)
            else None
        )
        if (
            index.get("schema_version") != _CATALOG_CACHE_SCHEMA_VERSION
            or index.get("cache_policy_version")
            != _CATALOG_CACHE_POLICY_VERSION
            or index.get("playlist_id") != _IVE_STAGE_PLAYLIST_ID
            or index.get("owner_channel_id")
            != _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID
            or not isinstance(object_sha256, str)
            or _CATALOG_OBJECT_SHA256.fullmatch(object_sha256) is None
            or index.get("object_relative_path") != expected_relative
        ):
            raise AcquisitionError("카탈로그 캐시 인덱스 계약이 일치하지 않습니다.")
        fetched_at_value = index.get("fetched_at")
        if not isinstance(fetched_at_value, str):
            raise AcquisitionError("카탈로그 캐시 시각이 없습니다.")
        fetched_at = datetime.fromisoformat(fetched_at_value)
        if fetched_at.tzinfo is None:
            raise AcquisitionError("카탈로그 캐시 시각에 타임존이 없습니다.")
        age_seconds = (now - fetched_at.astimezone(UTC)).total_seconds()
        if age_seconds < -300:
            raise AcquisitionError("카탈로그 캐시 시각이 미래입니다.")
        objects_directory = root / "objects"
        if objects_directory.is_symlink():
            raise AcquisitionError("카탈로그 객체 폴더가 심볼릭 링크입니다.")
        object_path = objects_directory / f"{object_sha256}.json"
        if object_path.is_symlink() or not object_path.is_file():
            raise AcquisitionError("카탈로그 객체 파일이 없거나 링크입니다.")
        if object_path.stat().st_size > _CATALOG_CACHE_OBJECT_MAXIMUM_BYTES:
            raise AcquisitionError("카탈로그 객체가 허용 크기를 넘습니다.")
        raw_object = object_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_object).hexdigest()
        if not hmac.compare_digest(actual_sha256, object_sha256):
            raise AcquisitionError("카탈로그 객체 SHA-256이 일치하지 않습니다.")
        object_payload = _validated_cached_catalog_object(
            json.loads(raw_object.decode("utf-8"))
        )
        if len(object_payload["entries"]) != index.get("entry_count"):
            raise AcquisitionError("카탈로그 인덱스와 객체의 항목 수가 다릅니다.")
        diagnostics.update(
            {
                "cache_state": (
                    "fresh"
                    if age_seconds <= _CATALOG_CACHE_MAX_AGE_SECONDS
                    else "stale"
                ),
                "cache_age_seconds": max(0.0, age_seconds),
                "object_sha256": object_sha256,
                "object_path": str(object_path),
                "fetched_at": fetched_at.astimezone(UTC).isoformat(),
                "entry_count": len(object_payload["entries"]),
                "raw_entry_count": object_payload["raw_entry_count"],
                "normalization_rejected_count": object_payload[
                    "normalization_rejected_count"
                ],
            }
        )
        return object_payload, diagnostics
    except (
        AcquisitionError,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        diagnostics.update(
            {
                "cache_state": "invalid",
                "cache_error": str(exc),
            }
        )
        return None, diagnostics


def _write_catalog_cache(
    root: Path,
    *,
    object_payload: dict[str, Any],
    fetched_at: datetime,
) -> dict[str, Any]:
    if root.is_symlink():
        raise AcquisitionError("카탈로그 캐시 루트가 심볼릭 링크입니다.")
    root.mkdir(parents=True, exist_ok=True)
    objects_directory = root / "objects"
    if objects_directory.is_symlink():
        raise AcquisitionError("카탈로그 객체 폴더가 심볼릭 링크입니다.")
    objects_directory.mkdir(parents=True, exist_ok=True)
    object_bytes = _canonical_json_bytes(object_payload)
    object_sha256 = hashlib.sha256(object_bytes).hexdigest()
    object_path = objects_directory / f"{object_sha256}.json"
    if object_path.exists():
        if object_path.is_symlink() or object_path.read_bytes() != object_bytes:
            raise AcquisitionError(
                "동일 SHA-256 카탈로그 객체가 다른 내용으로 존재합니다."
            )
    else:
        _atomic_catalog_write(object_path, object_bytes)
    index = {
        "schema_version": _CATALOG_CACHE_SCHEMA_VERSION,
        "cache_policy_version": _CATALOG_CACHE_POLICY_VERSION,
        "playlist_id": _IVE_STAGE_PLAYLIST_ID,
        "owner_channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
        "object_sha256": object_sha256,
        "object_relative_path": f"objects/{object_sha256}.json",
        "entry_count": object_payload["entry_count"],
        "fetched_at": fetched_at.astimezone(UTC).isoformat(),
    }
    index_path = root / _CATALOG_CACHE_INDEX_FILENAME
    if index_path.is_symlink():
        raise AcquisitionError("카탈로그 인덱스가 심볼릭 링크입니다.")
    _atomic_catalog_write(index_path, _canonical_json_bytes(index))
    return {
        "object_sha256": object_sha256,
        "object_path": str(object_path),
        "fetched_at": index["fetched_at"],
        "entry_count": object_payload["entry_count"],
    }


def _load_official_ive_stage_catalog(
    *,
    request: AnalysisRequest,
    client: YTDLPClient,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    root = _catalog_cache_root(request)
    cached_object, cache_diagnostics = _read_catalog_cache(
        root,
        now=current_time,
    )
    provenance: dict[str, Any] = {
        "schema_version": _CATALOG_CACHE_SCHEMA_VERSION,
        "cache_policy_version": _CATALOG_CACHE_POLICY_VERSION,
        "playlist_id": _IVE_STAGE_PLAYLIST_ID,
        "playlist_url": _IVE_STAGE_PLAYLIST_URL,
        "pinned_owner_channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
        "pinned_owner_name": _IVE_STAGE_PLAYLIST_OWNER_NAME,
        "playlist_owner_proves_catalog_only": True,
        "playlist_owner_never_grants_source_channel_trust": True,
        "source_entries_require_existing_metadata_song_member_show_channel_filters": True,
        "maximum_fresh_age_seconds": _CATALOG_CACHE_MAX_AGE_SECONDS,
        "minimum_plausible_entry_count": (
            _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT
        ),
        **cache_diagnostics,
        "refresh_attempted": False,
        "network_status": "not_attempted",
        "stale_fallback_used": False,
        "result": "pending",
    }
    if cached_object is not None and cache_diagnostics["cache_state"] == "fresh":
        provenance["result"] = "fresh_hash_valid_cache"
        return list(cached_object["entries"]), provenance

    provenance["refresh_attempted"] = True
    try:
        playlist_loader = getattr(client, "playlist_catalog", None)
        if not callable(playlist_loader):
            raise AcquisitionError(
                "주입된 yt-dlp 클라이언트가 플레이리스트 카탈로그를 "
                "지원하지 않습니다."
            )
        raw_payload = playlist_loader(_IVE_STAGE_PLAYLIST_ID)
        if not isinstance(raw_payload, dict):
            raise AcquisitionError(
                "IVE STAGE 플레이리스트 메타데이터가 JSON 객체가 "
                "아닙니다."
            )
        raw_returncode = raw_payload.get("_crosscut_ytdlp_returncode", 0)
        if (
            not isinstance(raw_returncode, int)
            or isinstance(raw_returncode, bool)
            or raw_returncode != 0
        ):
            raise _CatalogCompletenessError(
                "yt-dlp 플레이리스트 응답이 nonzero 종료라 부분 "
                "메타데이터로 간주합니다."
            )
        object_payload = _normalized_catalog_object(raw_payload)
        entry_count = int(object_payload["entry_count"])
        if entry_count < _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT:
            raise _CatalogCompletenessError(
                "IVE STAGE 카탈로그가 최소 완전성 하한보다 "
                f"작습니다: {entry_count} < "
                f"{_CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT}"
            )
        reported_counts = [
            value
            for key in ("playlist_count", "n_entries")
            if isinstance((value := raw_payload.get(key)), int)
            and not isinstance(value, bool)
            and value > 0
        ]
        if (
            reported_counts
            and object_payload["raw_entry_count"] < max(reported_counts)
        ):
            raise _CatalogCompletenessError(
                "yt-dlp 플레이리스트 응답 항목 수가 보고된 전체 "
                "항목 수보다 작습니다."
            )
        if (
            cached_object is not None
            and entry_count < int(cached_object["entry_count"])
        ):
            raise _CatalogCompletenessError(
                "IVE STAGE 카탈로그 갱신이 이전 해시 검증 캐시보다 "
                f"작습니다: {entry_count} < "
                f"{cached_object['entry_count']}"
            )
        if cached_object is not None:
            prior_video_ids = {
                str(entry["id"]) for entry in cached_object["entries"]
            }
            refreshed_video_ids = {
                str(entry["id"]) for entry in object_payload["entries"]
            }
            missing_prior_video_ids = prior_video_ids - refreshed_video_ids
            if missing_prior_video_ids:
                raise _CatalogCompletenessError(
                    "IVE STAGE 카탈로그 갱신에서 이전 해시 검증 "
                    f"항목 {len(missing_prior_video_ids)}개가 사라졌습니다."
                )
        provenance["network_status"] = "validated_pinned_playlist"
    except (AcquisitionError, OSError, ValueError) as exc:
        provenance["network_status"] = (
            "incomplete_or_regressed"
            if isinstance(exc, _CatalogCompletenessError)
            else "failed_or_untrusted"
        )
        provenance["network_error"] = str(exc)
        if cached_object is not None and cache_diagnostics["cache_state"] == "stale":
            provenance["stale_fallback_used"] = True
            provenance["result"] = "stale_hash_valid_cache_after_refresh_failure"
            return list(cached_object["entries"]), provenance
        provenance["result"] = "catalog_unavailable_query_search_only"
        return [], provenance

    try:
        written = _write_catalog_cache(
            root,
            object_payload=object_payload,
            fetched_at=current_time,
        )
    except (AcquisitionError, OSError, ValueError) as exc:
        provenance["cache_write_error"] = str(exc)
        provenance["result"] = "validated_network_catalog_cache_write_failed"
    else:
        provenance.update(written)
        provenance["cache_state"] = "refreshed"
        provenance["result"] = "validated_network_catalog_cached"
    provenance["raw_entry_count"] = object_payload["raw_entry_count"]
    provenance["normalization_rejected_count"] = object_payload[
        "normalization_rejected_count"
    ]
    return list(object_payload["entries"]), provenance


def _cached_video_path(directory: Path, video_id: str) -> Path | None:
    paths = _cached_video_paths(directory, video_id)
    return paths[0] if paths else None


def _cached_video_paths(directory: Path, video_id: str) -> tuple[Path, ...]:
    # Never follow a user-controlled cache-directory symlink. Otherwise a
    # stale cache entry could make quarantine or yt-dlp writes escape the
    # request's visible ``videos`` directory.
    if directory.is_symlink():
        raise AcquisitionError(
            f"캐시 영상 폴더가 심볼릭 링크여서 안전하게 사용할 수 없습니다: {directory}"
        )
    if not directory.is_dir():
        return ()
    resolved_directory = directory.resolve()
    matches = sorted(
        # Keep entry symlinks unresolved so quarantine renames the link itself,
        # never the file it points at outside the cache.
        resolved_directory / path.name
        for path in directory.glob(f"{video_id}.*")
        if (path.is_file() or path.is_symlink())
        and path.suffix.lower() in _VIDEO_SUFFIXES
    )
    return tuple(matches)


def _metadata_observation_digest(
    *,
    video_id: str,
    title: str,
    channel: str,
    channel_id: str | None,
    duration_seconds: float,
    webpage_url: str,
    broadcast_date: str | None,
    broadcast_program: str | None,
    broadcast_date_basis: str | None,
) -> str:
    """Fingerprint stable media-relevant fields from one discovery result."""

    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "channel_id": channel_id,
        "duration_seconds": duration_seconds,
        "webpage_url": webpage_url,
        "broadcast_date": broadcast_date,
        "broadcast_program": broadcast_program,
        "broadcast_date_basis": broadcast_date_basis,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _metadata_candidate(
    entry: dict[str, Any],
    *,
    request: AnalysisRequest,
    query: str,
    discovery_source: str = "query_search",
    verified_fresh_metadata_origin: str | None = None,
) -> AcquisitionCandidate | None:
    expected_discovery_source = {
        "query_search": "query_search",
        "validated_network_catalog": "official_ive_stage_playlist",
    }.get(verified_fresh_metadata_origin)
    if verified_fresh_metadata_origin is not None and (
        expected_discovery_source is None
        or discovery_source != expected_discovery_source
    ):
        raise ValueError("fresh metadata origin/discovery source mismatch")
    video_id = entry.get("id")
    if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
        return None
    title = entry.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "(untitled)"
    raw_channel_id = entry.get("channel_id") or entry.get("uploader_id")
    channel_id = raw_channel_id.strip() if isinstance(raw_channel_id, str) else None
    channel_value = entry.get("channel") or entry.get("uploader") or channel_id
    channel = channel_value.strip() if isinstance(channel_value, str) else "(unknown)"
    raw_duration = entry.get("duration")
    duration = float(raw_duration) if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool) else math.nan
    webpage_value = entry.get("webpage_url") or entry.get("url")
    webpage_url = (
        webpage_value
        if isinstance(webpage_value, str) and webpage_value.startswith(("https://", "http://"))
        else f"https://www.youtube.com/watch?v={video_id}"
    )

    folded_title = _fold(title)
    rejection_reasons: list[str] = []
    if not (math.isfinite(duration) and duration > 0):
        rejection_reasons.append("duration_unavailable")
        duration = 0.0
    elif duration < _MINIMUM_DURATION_SECONDS:
        rejection_reasons.append("duration_too_short")
    elif duration > _MAXIMUM_DURATION_SECONDS:
        rejection_reasons.append("duration_too_long")
    if not _has_ive_title_signal(title):
        rejection_reasons.append("ive_title_signal_missing")
    requested_aliases = _song_aliases(request.song)
    if not _contains_alias(title, requested_aliases):
        rejection_reasons.append("song_title_signal_missing")
    else:
        requested_known_group = _known_song_alias_group(request.song)
        other_known_song_signal = any(
            known_aliases != requested_known_group
            and _contains_alias(title, known_aliases)
            for known_aliases in _SONG_ALIAS_GROUPS
        )
        if requested_known_group is None and other_known_song_signal:
            # Keep future IVE titles discoverable, but never let a partial
            # unknown input such as "AM" or "DIVE" impersonate a registered
            # title merely because one complete token appears in metadata.
            rejection_reasons.append(
                "requested_song_conflicts_with_known_ive_title"
            )
        elif requested_known_group is not None and other_known_song_signal:
            # A result naming two IVE activities is usually a compilation,
            # relay, or stale-search collision rather than this promotion's
            # canonical weekly stage.
            rejection_reasons.append("other_ive_song_title_signal")

    hard_title_reason = _current_hard_title_rejection_reason(title)
    if hard_title_reason is not None:
        rejection_reasons.append(f"non_music_show_{hard_title_reason}")

    broadcast_program = _music_show_program(title)
    has_music_show_signal = broadcast_program is not None
    has_fancam_signal = any(marker in folded_title for marker in _FANCAM_MARKERS)
    official_channel_signal = _official_channel_signal(channel, channel_id)
    trusted = official_channel_signal == "allowlisted_channel_id"
    fallback_reason = None
    fallback_tier = 0
    for reason, markers in _FALLBACK_MARKERS:
        if any(marker in folded_title for marker in markers):
            tier = {
                "intro_plus": 2,
                "concert_or_festival": 2,
                "dance_practice": 3,
                "stage_mix": 4,
                "music_video": 5,
            }[reason]
            if tier > fallback_tier:
                fallback_reason = reason
                fallback_tier = tier

    if request.edit_mode is EditMode.MEMBER:
        assert request.target_member is not None
        target_member_signal = _contains_alias(
            title,
            _MEMBER_ALIASES[request.target_member],
        )
        if not target_member_signal:
            rejection_reasons.append("target_member_title_signal_missing")
            if any(
                member != request.target_member
                and _contains_alias(title, aliases)
                for member, aliases in _MEMBER_ALIASES.items()
            ):
                rejection_reasons.append(
                    "other_member_title_signal_without_target"
                )
        if not has_fancam_signal:
            rejection_reasons.append("fancam_title_signal_missing")
    elif has_fancam_signal and any(
        _contains_alias(title, aliases) for aliases in _MEMBER_ALIASES.values()
    ):
        rejection_reasons.append("single_member_fancam_not_group_stage")

    if not trusted:
        rejection_reasons.append("official_broadcast_channel_id_unverified")
    # A bare "stage"/"무대"/"풀캠" label also appears on award shows,
    # relay dances, studio-live clips, and festivals.  Even an allowlisted
    # broadcaster upload must name an actual weekly music-show programme.
    if not has_music_show_signal:
        rejection_reasons.append("music_show_title_signal_missing")

    if fallback_reason is not None:
        selection_tier = fallback_tier
    elif trusted and has_music_show_signal:
        selection_tier = 0
    elif trusted or has_music_show_signal:
        selection_tier = 1
    else:
        selection_tier = 6

    if fallback_reason not in (None, "intro_plus"):
        rejection_reasons.append(f"non_music_show_{fallback_reason}")

    trust_score = (
        (
            120
            if official_channel_signal == "allowlisted_channel_id"
            else 85 if official_channel_signal == "trusted_channel_name" else 0
        )
        + (35 if has_music_show_signal else 0)
        + (15 if request.edit_mode is EditMode.MEMBER and has_fancam_signal else 0)
        + (3 if "4k" in folded_title or "2160" in folded_title else 0)
    )
    normalized_title = title.strip()
    broadcast_date, broadcast_date_basis = _metadata_broadcast_date_with_basis(
        entry,
        normalized_title,
    )
    fresh_metadata_identities: tuple[_VerifiedFreshMetadataIdentity, ...] = ()
    if (
        verified_fresh_metadata_origin is not None
        and broadcast_program is not None
        and broadcast_date is not None
        and broadcast_date_basis is not None
    ):
        fresh_metadata_identities = (
            _VerifiedFreshMetadataIdentity(
                video_id=video_id,
                title=normalized_title,
                channel_id=channel_id,
                broadcast_program=broadcast_program,
                broadcast_date=broadcast_date,
                broadcast_date_basis=broadcast_date_basis,
                origin=verified_fresh_metadata_origin,
                _seal=_FRESH_METADATA_IDENTITY_SEAL,
            ),
        )
    metadata_observation_sha256 = _metadata_observation_digest(
        video_id=video_id,
        title=normalized_title,
        channel=channel,
        channel_id=channel_id,
        duration_seconds=duration,
        webpage_url=webpage_url,
        broadcast_date=broadcast_date,
        broadcast_program=broadcast_program,
        broadcast_date_basis=broadcast_date_basis,
    )
    return AcquisitionCandidate(
        video_id=video_id,
        title=normalized_title,
        channel=channel,
        channel_id=channel_id,
        duration_seconds=duration,
        webpage_url=webpage_url,
        queries=(query,),
        selection_tier=selection_tier,
        trust_score=trust_score,
        broadcast_date=broadcast_date,
        broadcast_program=broadcast_program,
        broadcast_date_basis=broadcast_date_basis,
        official_channel_signal=official_channel_signal,
        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        fallback_reason=fallback_reason,
        discovery_sources=(discovery_source,),
        _fresh_metadata_identities=fresh_metadata_identities,
        _metadata_observation_sha256=(metadata_observation_sha256,),
    )


def _content_fingerprint(candidate: AcquisitionCandidate) -> tuple[str, int]:
    folded = _fold(candidate.title)
    folded = re.sub(r"\b(?:4k|8k|1080p|2160p|60fps)\b", "", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return (_compact(folded), int(round(candidate.duration_seconds / 3.0)))


def _candidate_order(candidate: AcquisitionCandidate) -> tuple[Any, ...]:
    return (
        bool(candidate.rejection_reasons),
        candidate.selection_tier,
        -candidate.trust_score,
        _fold(candidate.title),
        candidate.video_id,
    )


def _candidate_metadata_richness(candidate: AcquisitionCandidate) -> int:
    return sum(
        (
            4 if candidate.channel_id is not None else 0,
            2 if candidate.channel != "(unknown)" else 0,
            2 if candidate.duration_seconds > 0 else 0,
            2 if candidate.broadcast_date is not None else 0,
            2 if candidate.broadcast_program is not None else 0,
            1 if candidate.title != "(untitled)" else 0,
            1 if candidate.webpage_url.startswith("https://") else 0,
        )
    )


def _merge_discovery(
    existing: AcquisitionCandidate, discovered: AcquisitionCandidate
) -> AcquisitionCandidate:
    def merge_order(candidate: AcquisitionCandidate) -> tuple[Any, ...]:
        base = _candidate_order(candidate)
        return (*base[:3], -_candidate_metadata_richness(candidate), *base[3:])

    # Query search is inserted first and therefore wins an exact metadata tie;
    # a catalog hit can replace it only when it is substantively richer or
    # passes stricter existing filters. Never merge individual trust fields.
    better = min((existing, discovered), key=merge_order)
    alternate = discovered if better is existing else existing
    use_better_date = better.broadcast_date is not None
    return AcquisitionCandidate(
        video_id=better.video_id,
        title=better.title,
        channel=better.channel,
        channel_id=better.channel_id,
        duration_seconds=better.duration_seconds,
        webpage_url=better.webpage_url,
        queries=tuple(dict.fromkeys((*existing.queries, *discovered.queries))),
        selection_tier=better.selection_tier,
        trust_score=better.trust_score,
        # Date/program are non-trust annotations. They may safely fill a gap
        # without transplanting title, channel, trust, tier, or rejections from
        # the losing candidate.
        broadcast_date=better.broadcast_date or alternate.broadcast_date,
        broadcast_program=(
            better.broadcast_program or alternate.broadcast_program
        ),
        broadcast_date_basis=(
            better.broadcast_date_basis
            if use_better_date
            else alternate.broadcast_date_basis
        ),
        official_channel_signal=better.official_channel_signal,
        rejection_reasons=better.rejection_reasons,
        fallback_reason=better.fallback_reason,
        discovery_sources=tuple(
            dict.fromkeys(
                (*existing.discovery_sources, *discovered.discovery_sources)
            )
        ),
        _fresh_metadata_identities=tuple(
            dict.fromkeys(
                (
                    *existing._fresh_metadata_identities,
                    *discovered._fresh_metadata_identities,
                )
            )
        ),
        _metadata_observation_sha256=tuple(
            sorted(
                set(existing._metadata_observation_sha256)
                | set(discovered._metadata_observation_sha256)
            )
        ),
    )


def _record(candidate: AcquisitionCandidate) -> dict[str, Any]:
    return {
        "video_id": candidate.video_id,
        "title": candidate.title,
        "channel": candidate.channel,
        "channel_id": candidate.channel_id,
        "duration_seconds": candidate.duration_seconds,
        "webpage_url": candidate.webpage_url,
        "query": candidate.query,
        "queries": list(candidate.queries),
        "discovery_sources": list(candidate.discovery_sources),
        "selection_tier": candidate.selection_tier,
        "trust_score": candidate.trust_score,
        "broadcast_date": candidate.broadcast_date,
        "broadcast_program": candidate.broadcast_program,
        "broadcast_date_basis": candidate.broadcast_date_basis,
        "official_channel_signal": candidate.official_channel_signal,
        "fallback_reason": candidate.fallback_reason,
        "rejection_reasons": list(candidate.rejection_reasons),
        "selection_exclusion_reason": None,
        "selected": False,
        "download_status": "not_selected",
        "local_path": None,
        "download_error": None,
        "download_hard_rejection_reasons": [],
        "media_validation": None,
        "aspect_validation": None,
        "content_integrity": None,
        "group_relative_quality": None,
        "group_quality_eligible": None,
        "group_quality_relative_rank": None,
        "group_quality_exclusion_reasons": [],
        "preexisting_cache_quarantined_paths": [],
        "persistent_media_contract_rejection": None,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _requires_automatic_member_portrait_contract(
    request: AnalysisRequest,
) -> bool:
    return (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.MEMBER
        and request.aspect_ratio is AspectRatio.PORTRAIT
    )


def _requires_automatic_media_quality_contract(
    request: AnalysisRequest,
) -> bool:
    return request.source_mode is SourceMode.AUTOMATIC


def _requires_group_relative_quality_gate(request: AnalysisRequest) -> bool:
    """Limit relative top-eight triage to the default automatic GROUP pool.

    Explicit non-default counts remain exact request contracts, while MEMBER
    and LOCAL acquisition retain their independent selection behavior.
    """

    return (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.GROUP
        and request.desired_source_count == DEFAULT_AUTOMATIC_SOURCE_COUNT
    )


def _validate_downloaded_media(path: Path) -> MediaInfo:
    info = probe_media(path)
    if not info.video_streams:
        raise AcquisitionError(f"{path.name}: 다운로드 파일에 비디오 스트림이 없습니다.")
    if not info.audio_streams:
        raise AcquisitionError(f"{path.name}: 다운로드 파일에 오디오 스트림이 없습니다.")
    if info.duration_seconds is None or info.duration_seconds <= 0:
        raise AcquisitionError(f"{path.name}: 다운로드 파일 길이를 확인할 수 없습니다.")
    return info


def _validate_media_for_request(
    path: Path,
    *,
    request: AnalysisRequest,
    media_validator: Callable[[Path], object | None],
    media_probe: Callable[[Path], MediaInfo],
) -> dict[str, Any] | None:
    """Apply base validation plus automatic quality and camera contracts.

    The injected validator intentionally keeps its historical one-argument
    signature.  A validator may return ``MediaInfo`` to avoid a second probe;
    otherwise the request-scoped quality check obtains its own typed probe.
    Local sources retain their existing validation path and do not gain this
    automatic-acquisition contract.
    """

    validated = media_validator(path)
    if not _requires_automatic_media_quality_contract(request):
        return None
    info = validated if isinstance(validated, MediaInfo) else media_probe(path)
    member_portrait = _requires_automatic_member_portrait_contract(request)
    expected_ratio = (
        _AUTOMATIC_MEMBER_PORTRAIT_RATIO
        if member_portrait
        else _AUTOMATIC_GROUP_LANDSCAPE_RATIO
    )
    ratio_tolerance = (
        _AUTOMATIC_MEMBER_PORTRAIT_RATIO_TOLERANCE
        if member_portrait
        else _AUTOMATIC_GROUP_LANDSCAPE_RATIO_TOLERANCE
    )
    base = {
        "required": True,
        "analysis_basis": (
            "first_video_stream_dynamic_portrait_crop_without_upscale"
            if member_portrait
            else "first_video_stream_width_over_height"
        ),
        "expected_width_over_height": expected_ratio,
        "tolerance": ratio_tolerance,
        "source_aspect_must_match_output": not member_portrait,
        "minimum_lossless_crop_headroom_zoom": (
            _MEMBER_MINIMUM_LOSSLESS_CROP_HEADROOM_ZOOM
            if member_portrait
            else None
        ),
        "minimum_short_edge": _AUTOMATIC_MINIMUM_SHORT_EDGE,
        "minimum_long_edge": _AUTOMATIC_MINIMUM_LONG_EDGE,
        "minimum_frame_rate": _AUTOMATIC_MINIMUM_FRAME_RATE,
        "width": None,
        "height": None,
        "actual_short_edge": None,
        "actual_long_edge": None,
        "frame_rate": None,
        "actual_width_over_height": None,
        "passed": False,
        "hard_rejection_reason": None,
    }
    if not info.video_streams:
        reason = (
            "automatic_member_portrait_first_video_stream_missing"
            if member_portrait
            else "automatic_group_landscape_first_video_stream_missing"
        )
        diagnostics = {**base, "hard_rejection_reason": reason}
        raise _MediaContractError(
            f"{path.name}: 자동 소스의 첫 비디오 스트림을 "
            "확인할 수 없습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    stream = info.video_streams[0]
    diagnostics = {
        **base,
        "width": stream.width,
        "height": stream.height,
        "frame_rate": stream.frame_rate,
        "professional_output_width": (
            _MEMBER_PRODUCTION_WIDTH
            if member_portrait
            else _GROUP_PRODUCTION_WIDTH
        ),
        "professional_output_height": (
            _MEMBER_PRODUCTION_HEIGHT
            if member_portrait
            else _GROUP_PRODUCTION_HEIGHT
        ),
        "lossless_crop_headroom_zoom": None,
        "crop_headroom_tier": None,
    }
    if (
        stream.width is None
        or stream.height is None
        or stream.width <= 0
        or stream.height <= 0
    ):
        reason = (
            "automatic_member_portrait_dimensions_unavailable"
            if member_portrait
            else "automatic_group_landscape_dimensions_unavailable"
        )
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 소스의 첫 비디오 스트림 "
            "해상도를 확인할 수 없습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    short_edge = min(stream.width, stream.height)
    long_edge = max(stream.width, stream.height)
    diagnostics["actual_short_edge"] = short_edge
    diagnostics["actual_long_edge"] = long_edge
    output_width = (
        _MEMBER_PRODUCTION_WIDTH if member_portrait else _GROUP_PRODUCTION_WIDTH
    )
    output_height = (
        _MEMBER_PRODUCTION_HEIGHT if member_portrait else _GROUP_PRODUCTION_HEIGHT
    )
    lossless_zoom = min(
        stream.width / output_width,
        stream.height / output_height,
    )
    diagnostics["lossless_crop_headroom_zoom"] = min(
        _PROFESSIONAL_MAXIMUM_LOSSLESS_ZOOM,
        max(0.0, lossless_zoom),
    )
    diagnostics["crop_headroom_tier"] = (
        "uhd_2160_or_higher"
        if lossless_zoom >= 2.0 - 1e-6
        else "qhd_1440_or_higher"
        if lossless_zoom >= 4.0 / 3.0 - 1e-6
        else "full_hd_native"
        if lossless_zoom >= 1.0 - 1e-6
        else "below_production_output"
    )
    if (
        short_edge < _AUTOMATIC_MINIMUM_SHORT_EDGE
        or long_edge < _AUTOMATIC_MINIMUM_LONG_EDGE
    ):
        reason = "automatic_video_resolution_below_minimum"
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 소스 해상도 {stream.width}x{stream.height}가 "
            f"최소 {_AUTOMATIC_MINIMUM_SHORT_EDGE}x"
            f"{_AUTOMATIC_MINIMUM_LONG_EDGE} 변 길이 계약을 벗어났습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    if stream.frame_rate is None or not math.isfinite(stream.frame_rate):
        reason = "automatic_video_frame_rate_unavailable"
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 소스 프레임률을 확인할 수 없습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    if stream.frame_rate < _AUTOMATIC_MINIMUM_FRAME_RATE:
        reason = "automatic_video_frame_rate_below_minimum"
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 소스 프레임률 {stream.frame_rate:.3f}fps가 "
            f"최소 {_AUTOMATIC_MINIMUM_FRAME_RATE:.0f}fps보다 낮습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    actual_ratio = stream.width / stream.height
    diagnostics["actual_width_over_height"] = actual_ratio
    if (
        member_portrait
        and lossless_zoom
        < _MEMBER_MINIMUM_LOSSLESS_CROP_HEADROOM_ZOOM - 1e-6
    ):
        reason = "automatic_member_portrait_crop_headroom_below_production"
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 멤버 소스 {stream.width}x{stream.height}는 "
            f"{_MEMBER_PRODUCTION_WIDTH}x{_MEMBER_PRODUCTION_HEIGHT} 세로 다이나믹 "
            "크롭을 업스케일 없이 만들 수 없습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    if not member_portrait and abs(actual_ratio - expected_ratio) > ratio_tolerance:
        reason = "automatic_group_landscape_aspect_ratio_mismatch"
        diagnostics["hard_rejection_reason"] = reason
        raise _MediaContractError(
            f"{path.name}: 자동 소스 비율 {actual_ratio:.6f}이 "
            "16:9±0.03 계약을 벗어났습니다.",
            hard_rejection_reason=reason,
            diagnostics=diagnostics,
        )
    diagnostics["passed"] = True
    return diagnostics


def _quarantine_invalid_media(path: Path) -> Path | None:
    """Move a bad cache entry aside without reusing or overwriting evidence."""

    if not path.is_file() and not path.is_symlink():
        return None
    candidate = path.with_suffix(path.suffix + ".invalid")
    counter = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_suffix(path.suffix + f".invalid.{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def _content_integrity(path: Path) -> dict[str, Any]:
    """Bind one selected cache file to its acquisition manifest."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise AcquisitionError(
            f"{path.name}: 무결성 해시를 계산하는 동안 캐시 파일이 변경되었습니다."
        )
    return {
        "contract_version": _CONTENT_INTEGRITY_CONTRACT_VERSION,
        "algorithm": "sha256",
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def _content_integrity_matches(path: Path, binding: object) -> bool:
    if not isinstance(binding, dict):
        return False
    expected_size = binding.get("size_bytes")
    expected_digest = binding.get("sha256")
    if (
        binding.get("contract_version") != _CONTENT_INTEGRITY_CONTRACT_VERSION
        or binding.get("algorithm") != "sha256"
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
    ):
        return False
    try:
        if path.stat().st_size != expected_size:
            return False
        actual_digest = _content_integrity(path)["sha256"]
    except (AcquisitionError, OSError):
        return False
    return hmac.compare_digest(actual_digest, expected_digest)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _candidate_media_metadata_identity(
    candidate: AcquisitionCandidate,
) -> dict[str, Any] | None:
    observations = sorted(set(candidate._metadata_observation_sha256))
    if not observations or any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in observations
    ):
        return None
    payload = {
        "schema_version": 1,
        "video_id": candidate.video_id,
        "title": candidate.title,
        "channel": candidate.channel,
        "channel_id": candidate.channel_id,
        "duration_seconds": candidate.duration_seconds,
        "webpage_url": candidate.webpage_url,
        "broadcast_date": candidate.broadcast_date,
        "broadcast_program": candidate.broadcast_program,
        "broadcast_date_basis": candidate.broadcast_date_basis,
        "official_channel_signal": candidate.official_channel_signal,
        "current_metadata_observation_sha256": observations,
    }
    return {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def _client_download_contract(client: object) -> tuple[str, str | None]:
    raw_profile = getattr(client, "download_profile", None)
    profile = (
        raw_profile.strip()
        if isinstance(raw_profile, str) and raw_profile.strip()
        else "injected_client_unspecified_profile"
    )
    raw_format = getattr(client, "download_format", None)
    download_format = (
        raw_format.strip()
        if isinstance(raw_format, str) and raw_format.strip()
        else None
    )
    return (profile, download_format)


def _persistent_media_contract_request_identity(
    *,
    request: AnalysisRequest,
    client: object,
) -> dict[str, Any]:
    member_portrait = _requires_automatic_member_portrait_contract(request)
    profile, download_format = _client_download_contract(client)
    payload = {
        "schema_version": 1,
        "group": "IVE",
        "song": request.song,
        "source_mode": request.source_mode.value,
        "edit_mode": request.edit_mode.value,
        "aspect_ratio": request.aspect_ratio.value,
        "target_member": (
            request.target_member.value if request.target_member else None
        ),
        "automatic_media_quality_contract_version": (
            _AUTOMATIC_MEDIA_QUALITY_CONTRACT_VERSION
            if _requires_automatic_media_quality_contract(request)
            else None
        ),
        "member_portrait_aspect_contract_version": (
            _MEMBER_PORTRAIT_ASPECT_CONTRACT_VERSION
            if member_portrait
            else None
        ),
        "expected_width_over_height": (
            _AUTOMATIC_MEMBER_PORTRAIT_RATIO
            if member_portrait
            else _AUTOMATIC_GROUP_LANDSCAPE_RATIO
        ),
        "aspect_ratio_tolerance": (
            _AUTOMATIC_MEMBER_PORTRAIT_RATIO_TOLERANCE
            if member_portrait
            else _AUTOMATIC_GROUP_LANDSCAPE_RATIO_TOLERANCE
        ),
        "minimum_short_edge": _AUTOMATIC_MINIMUM_SHORT_EDGE,
        "minimum_long_edge": _AUTOMATIC_MINIMUM_LONG_EDGE,
        "minimum_frame_rate": _AUTOMATIC_MINIMUM_FRAME_RATE,
        "download_profile": profile,
        "download_format": download_format,
    }
    return {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def _persistent_rejection_quarantine_path(
    *,
    manifest_path: Path,
    video_id: str,
    artifact: object,
) -> Path | None:
    if not isinstance(artifact, dict):
        return None
    raw_relative_path = artifact.get("relative_path")
    if not isinstance(raw_relative_path, str):
        return None
    relative_path = Path(raw_relative_path)
    if relative_path.is_absolute() or len(relative_path.parts) != 2:
        return None
    if relative_path.parts[0] != "videos":
        return None
    filename = relative_path.parts[1]
    if re.fullmatch(
        rf"{re.escape(video_id)}\.(?:mp4|mkv|mov|webm|m4v)"
        r"\.invalid(?:\.[1-9][0-9]*)?",
        filename,
    ) is None:
        return None
    videos_directory = manifest_path.parent / "videos"
    if videos_directory.is_symlink():
        return None
    quarantine_path = manifest_path.parent / relative_path
    try:
        if quarantine_path.is_symlink() or not quarantine_path.is_file():
            return None
        if quarantine_path.parent.resolve() != videos_directory.resolve():
            return None
    except (OSError, RuntimeError):
        return None
    return quarantine_path


def _quarantine_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.lstat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _build_persistent_media_contract_rejection(
    *,
    candidate: AcquisitionCandidate,
    request: AnalysisRequest,
    client: object,
    cache_directory: Path,
    quarantine_path: Path | None,
    hard_rejection_reason: str,
    diagnostics: dict[str, Any],
    evidence_origin: str = "downloaded_media_contract_failure",
) -> dict[str, Any] | None:
    if (
        quarantine_path is None
        or evidence_origin
        not in {
            "downloaded_media_contract_failure",
            "legacy_candidate_rejection_revalidated",
        }
        or hard_rejection_reason
        not in _PERSISTENT_MEDIA_CONTRACT_REJECTION_REASONS
        or diagnostics.get("passed") is not False
        or diagnostics.get("hard_rejection_reason") != hard_rejection_reason
    ):
        return None
    candidate_identity = _candidate_media_metadata_identity(candidate)
    if candidate_identity is None:
        return None
    try:
        if quarantine_path.is_symlink() or not quarantine_path.is_file():
            return None
        cache_root = cache_directory.resolve()
        resolved_quarantine = quarantine_path.resolve(strict=True)
        relative_path = resolved_quarantine.relative_to(cache_root)
        if relative_path.parent != Path("videos"):
            return None
        expected_path = _persistent_rejection_quarantine_path(
            manifest_path=cache_root / "acquisition-manifest.json",
            video_id=candidate.video_id,
            artifact={"relative_path": str(relative_path)},
        )
        if expected_path is None or expected_path.resolve() != resolved_quarantine:
            return None
        rejected_content_integrity = _content_integrity(resolved_quarantine)
    except (AcquisitionError, OSError, RuntimeError, ValueError):
        return None
    diagnostics_copy = deepcopy(diagnostics)
    return {
        "schema_version": _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION,
        "evidence_origin": evidence_origin,
        "video_id": candidate.video_id,
        "candidate_media_metadata_identity": candidate_identity,
        "request_media_contract_identity": (
            _persistent_media_contract_request_identity(
                request=request,
                client=client,
            )
        ),
        "hard_rejection_reason": hard_rejection_reason,
        "diagnostics": diagnostics_copy,
        "diagnostics_sha256": _canonical_sha256(diagnostics_copy),
        "quarantine_artifact": {
            "relative_path": str(relative_path),
            "content_integrity": rejected_content_integrity,
        },
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _migrate_legacy_persistent_media_contract_rejection(
    *,
    prior_manifest: dict[str, Any] | None,
    manifest_path: Path,
    candidate: AcquisitionCandidate,
    request: AnalysisRequest,
    client: object,
    media_validator: Callable[[Path], object | None],
    media_probe: Callable[[Path], MediaInfo],
) -> dict[str, Any] | None:
    """Bind an older failed candidate record to its still-present artifact.

    Policy-v5 manifests already persisted the exact candidate fields, typed
    media diagnostics, hard reason, and quarantine path, but did not hash the
    rejected bytes.  This migration never trusts that record alone: every
    field must match current discovery metadata, the request/download profile
    must match, and the quarantined bytes must reproduce the identical typed
    contract failure before a new content-bound v1 envelope is issued.
    """

    if (
        prior_manifest is None
        or not _manifest_matches_cache_binding_contract(prior_manifest, request)
        or "persistent_media_contract_rejection_contract_version"
        in prior_manifest
        or "persistent_media_contract_rejections" in prior_manifest
    ):
        return None
    current_profile, _current_format = _client_download_contract(client)
    if prior_manifest.get("download_profile") != current_profile:
        return None
    raw_records = prior_manifest.get("candidates")
    if (
        not isinstance(raw_records, list)
        or len(raw_records) > 512
        or any(not isinstance(record, dict) for record in raw_records)
    ):
        return None
    matches = [
        record
        for record in raw_records
        if record.get("video_id") == candidate.video_id
    ]
    if len(matches) != 1:
        return None
    record = matches[0]
    exact_metadata = {
        "video_id": candidate.video_id,
        "title": candidate.title,
        "channel": candidate.channel,
        "channel_id": candidate.channel_id,
        "duration_seconds": candidate.duration_seconds,
        "webpage_url": candidate.webpage_url,
        "broadcast_date": candidate.broadcast_date,
        "broadcast_program": candidate.broadcast_program,
        "broadcast_date_basis": candidate.broadcast_date_basis,
        "official_channel_signal": candidate.official_channel_signal,
        "fallback_reason": candidate.fallback_reason,
    }
    if any(record.get(key) != value for key, value in exact_metadata.items()):
        return None
    hard_reasons = record.get("download_hard_rejection_reasons")
    diagnostics = record.get("aspect_validation")
    raw_quarantine_path = record.get("quarantined_path")
    if (
        record.get("selected") is not False
        or record.get("download_status") != "failed"
        or record.get("local_path") is not None
        or not isinstance(hard_reasons, list)
        or len(hard_reasons) != 1
        or hard_reasons[0]
        not in _PERSISTENT_MEDIA_CONTRACT_REJECTION_REASONS
        or not isinstance(diagnostics, dict)
        or diagnostics.get("passed") is not False
        or diagnostics.get("hard_rejection_reason") != hard_reasons[0]
        or not isinstance(raw_quarantine_path, str)
        or not isinstance(record.get("download_error"), str)
        or not record["download_error"]
    ):
        return None
    try:
        cache_root = manifest_path.parent.resolve()
        raw_path = Path(raw_quarantine_path)
        if not raw_path.is_absolute():
            return None
        resolved_path = raw_path.resolve(strict=True)
        relative_path = resolved_path.relative_to(cache_root)
    except (OSError, RuntimeError, ValueError):
        return None
    quarantine_path = _persistent_rejection_quarantine_path(
        manifest_path=manifest_path,
        video_id=candidate.video_id,
        artifact={"relative_path": str(relative_path)},
    )
    if quarantine_path is None:
        return None
    try:
        before_identity = _quarantine_file_identity(quarantine_path)
        _validate_media_for_request(
            quarantine_path,
            request=request,
            media_validator=media_validator,
            media_probe=media_probe,
        )
    except _MediaContractError as exc:
        if (
            exc.hard_rejection_reason != hard_reasons[0]
            or exc.diagnostics != diagnostics
        ):
            return None
    except (AcquisitionError, MediaProbeError, OSError, ValueError):
        return None
    else:
        return None
    try:
        if _quarantine_file_identity(quarantine_path) != before_identity:
            return None
    except OSError:
        return None
    evidence = _build_persistent_media_contract_rejection(
        candidate=candidate,
        request=request,
        client=client,
        cache_directory=manifest_path.parent,
        quarantine_path=quarantine_path,
        hard_rejection_reason=hard_reasons[0],
        diagnostics=diagnostics,
        evidence_origin="legacy_candidate_rejection_revalidated",
    )
    if evidence is None:
        return None
    try:
        if _quarantine_file_identity(quarantine_path) != before_identity:
            return None
    except OSError:
        return None
    return evidence


def _validated_persistent_media_contract_rejections(
    *,
    prior_manifest: dict[str, Any] | None,
    manifest_path: Path,
    request: AnalysisRequest,
    client: object,
) -> tuple[_ValidatedPersistentMediaContractRejection, ...]:
    if (
        prior_manifest is None
        or not _manifest_matches_cache_binding_contract(prior_manifest, request)
        or prior_manifest.get(
            "persistent_media_contract_rejection_contract_version"
        )
        != _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
    ):
        return ()
    raw_entries = prior_manifest.get("persistent_media_contract_rejections")
    if (
        not isinstance(raw_entries, list)
        or len(raw_entries) > 512
        or any(not isinstance(entry, dict) for entry in raw_entries)
    ):
        return ()
    expected_request_identity = _persistent_media_contract_request_identity(
        request=request,
        client=client,
    )
    validated: list[_ValidatedPersistentMediaContractRejection] = []
    for raw_entry in raw_entries:
        video_id = raw_entry.get("video_id")
        reason = raw_entry.get("hard_rejection_reason")
        metadata_identity = raw_entry.get("candidate_media_metadata_identity")
        request_identity = raw_entry.get("request_media_contract_identity")
        diagnostics = raw_entry.get("diagnostics")
        diagnostics_sha256 = raw_entry.get("diagnostics_sha256")
        if (
            raw_entry.get("schema_version")
            != _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
            or raw_entry.get("evidence_origin")
            not in {
                "downloaded_media_contract_failure",
                "legacy_candidate_rejection_revalidated",
            }
            or not isinstance(video_id, str)
            or _VIDEO_ID.fullmatch(video_id) is None
            or reason not in _PERSISTENT_MEDIA_CONTRACT_REJECTION_REASONS
            or not isinstance(metadata_identity, dict)
            or metadata_identity.get("algorithm") != "sha256"
            or not isinstance(metadata_identity.get("payload"), dict)
            or metadata_identity["payload"].get("video_id") != video_id
            or not isinstance(metadata_identity.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata_identity["sha256"])
            is None
            or not hmac.compare_digest(
                metadata_identity["sha256"],
                _canonical_sha256(metadata_identity["payload"]),
            )
            or not isinstance(
                metadata_identity["payload"].get(
                    "current_metadata_observation_sha256"
                ),
                list,
            )
            or not metadata_identity["payload"][
                "current_metadata_observation_sha256"
            ]
            or any(
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in metadata_identity["payload"][
                    "current_metadata_observation_sha256"
                ]
            )
            or request_identity != expected_request_identity
            or not isinstance(diagnostics, dict)
            or diagnostics.get("passed") is not False
            or diagnostics.get("hard_rejection_reason") != reason
            or not isinstance(diagnostics_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", diagnostics_sha256) is None
            or not hmac.compare_digest(
                diagnostics_sha256,
                _canonical_sha256(diagnostics),
            )
        ):
            continue
        quarantine_artifact = raw_entry.get("quarantine_artifact")
        quarantine_path = _persistent_rejection_quarantine_path(
            manifest_path=manifest_path,
            video_id=video_id,
            artifact=quarantine_artifact,
        )
        if quarantine_path is None or not isinstance(quarantine_artifact, dict):
            continue
        content_integrity = quarantine_artifact.get("content_integrity")
        try:
            if not _content_integrity_matches(
                quarantine_path,
                content_integrity,
            ):
                continue
            file_identity = _quarantine_file_identity(quarantine_path)
        except (AcquisitionError, OSError):
            continue
        validated.append(
            _ValidatedPersistentMediaContractRejection(
                evidence=deepcopy(raw_entry),
                quarantine_path=quarantine_path,
                quarantine_file_identity=file_identity,
            )
        )
    return tuple(validated)


def _matching_persistent_media_contract_rejection(
    *,
    candidate: AcquisitionCandidate,
    validated_rejections: Sequence[
        _ValidatedPersistentMediaContractRejection
    ],
    request: AnalysisRequest,
    media_validator: Callable[[Path], object | None],
    media_probe: Callable[[Path], MediaInfo],
) -> dict[str, Any] | None:
    candidate_identity = _candidate_media_metadata_identity(candidate)
    if candidate_identity is None:
        return None
    matches = [
        entry
        for entry in validated_rejections
        if entry.evidence.get("video_id") == candidate.video_id
        and entry.evidence.get("candidate_media_metadata_identity")
        == candidate_identity
    ]
    if len(matches) != 1:
        return None
    match = matches[0]
    try:
        _validate_media_for_request(
            match.quarantine_path,
            request=request,
            media_validator=media_validator,
            media_probe=media_probe,
        )
    except _MediaContractError as exc:
        evidence = match.evidence
        if (
            exc.hard_rejection_reason
            != evidence.get("hard_rejection_reason")
            or exc.diagnostics != evidence.get("diagnostics")
        ):
            return None
        try:
            if (
                _quarantine_file_identity(match.quarantine_path)
                != match.quarantine_file_identity
            ):
                return None
        except OSError:
            return None
        return deepcopy(evidence)
    except (AcquisitionError, MediaProbeError, OSError, ValueError):
        return None
    return None


def _upsert_persistent_media_contract_rejection(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    raw_entries = manifest.get("persistent_media_contract_rejections")
    entries = (
        [deepcopy(entry) for entry in raw_entries if isinstance(entry, dict)]
        if isinstance(raw_entries, list)
        else []
    )
    video_id = evidence["video_id"]
    scope_sha256 = evidence["request_media_contract_identity"]["sha256"]
    entries = [
        entry
        for entry in entries
        if not (
            entry.get("video_id") == video_id
            and isinstance(entry.get("request_media_contract_identity"), dict)
            and entry["request_media_contract_identity"].get("sha256")
            == scope_sha256
        )
    ]
    entries.append(deepcopy(evidence))
    entries.sort(
        key=lambda entry: (
            str(entry.get("video_id", "")),
            str(
                entry.get("candidate_media_metadata_identity", {}).get(
                    "sha256", ""
                )
                if isinstance(
                    entry.get("candidate_media_metadata_identity"), dict
                )
                else ""
            ),
        )
    )
    manifest["persistent_media_contract_rejections"] = entries


def _copy_persistent_media_contract_rejection_state(
    *,
    destination: dict[str, Any],
    source: dict[str, Any],
) -> None:
    destination["persistent_media_contract_rejection_contract_version"] = (
        _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
    )
    destination["persistent_media_contract_rejections"] = deepcopy(
        source.get("persistent_media_contract_rejections", [])
    )
    destination["persistent_media_contract_rejection_refresh"] = deepcopy(
        source.get("persistent_media_contract_rejection_refresh")
    )
    destination_summary = destination.get("summary")
    source_summary = source.get("summary")
    if isinstance(destination_summary, dict) and isinstance(source_summary, dict):
        for key in (
            "persistent_media_contract_rejection_count",
            "persistent_media_contract_new_rejection_count",
            "persistent_media_contract_skip_count",
            "persistent_media_contract_migrated_legacy_count",
        ):
            destination_summary[key] = source_summary.get(key, 0)


def _stable_manifest_identity(
    manifest_path: Path,
    expected_manifest: dict[str, Any],
) -> tuple[int, str]:
    before = manifest_path.lstat()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AcquisitionError("수집 매니페스트가 일반 파일이 아닙니다.")
    encoded = manifest_path.read_bytes()
    after = manifest_path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(encoded) != after.st_size
    ):
        raise AcquisitionError("수집 매니페스트를 읽는 동안 변경되었습니다.")
    try:
        current_manifest = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("수집 매니페스트 JSON이 손상되었습니다.") from exc
    if current_manifest != expected_manifest:
        raise AcquisitionError("수집 결과와 저장 매니페스트가 다릅니다.")
    return (len(encoded), hashlib.sha256(encoded).hexdigest())


def _issue_member_acquisition_identity_authority(
    *,
    source_paths: Sequence[Path],
    manifest_path: Path,
    manifest: dict[str, Any],
    fresh_metadata_identities_by_video_id: Mapping[
        str,
        Sequence[_VerifiedFreshMetadataIdentity],
    ] | None = None,
) -> VerifiedMemberAcquisitionIdentityAuthority | None:
    if str(manifest.get("edit_mode", "")).strip().lower() != "member":
        return None
    summary = manifest.get("summary")
    selected = manifest.get("selected")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("content_integrity_contract_version")
        == _CONTENT_INTEGRITY_CONTRACT_VERSION
        and manifest.get("automatic_media_quality_contract_version") == 1
        and manifest.get("member_portrait_aspect_contract_version")
        == _MEMBER_PORTRAIT_ASPECT_CONTRACT_VERSION
        and isinstance(summary, dict)
        and summary.get("status") == "complete"
        and isinstance(selected, list)
        and len(selected) == len(source_paths)
    ):
        raise AcquisitionError(
            "완료·무결성·세로 직캠 계약이 검증된 멤버 수집만 사용할 수 있습니다."
        )
    manifest_size, manifest_sha256 = _stable_manifest_identity(
        manifest_path,
        manifest,
    )
    identities: list[VerifiedAcquisitionSourceIdentity] = []
    seen_video_ids: set[str] = set()
    for source_index, (source, raw_entry) in enumerate(
        zip(source_paths, selected, strict=True)
    ):
        if not isinstance(raw_entry, dict):
            raise AcquisitionError("멤버 수집 항목 형식이 잘못되었습니다.")
        video_id = raw_entry.get("video_id")
        broadcast_date = raw_entry.get("broadcast_date")
        broadcast_date_basis = raw_entry.get("broadcast_date_basis")
        title = raw_entry.get("title")
        channel_id = raw_entry.get("channel_id")
        programme = raw_entry.get("broadcast_program")
        canonical_programme = (
            _music_show_program(title) if isinstance(title, str) else None
        )
        canonical_date = _broadcast_date(title) if isinstance(title, str) else None
        integrity = raw_entry.get("content_integrity")
        media_validation = raw_entry.get("media_validation")
        aspect_validation = raw_entry.get("aspect_validation")
        unresolved_source = Path(source).expanduser()
        source_is_symlink = unresolved_source.is_symlink()
        resolved_source = unresolved_source.resolve(strict=True)
        raw_local_path = raw_entry.get("local_path")
        if not (
            raw_entry.get("source_index") == source_index
            and isinstance(video_id, str)
            and re.fullmatch(r"[0-9A-Za-z_-]{6,32}", video_id)
            and video_id not in seen_video_ids
            and isinstance(raw_local_path, str)
            and Path(raw_local_path).expanduser().resolve(strict=True)
            == resolved_source
            and isinstance(broadcast_date, str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", broadcast_date)
            and isinstance(programme, str)
            and bool(programme.strip())
            and isinstance(media_validation, dict)
            and media_validation.get("passed") is True
            and isinstance(aspect_validation, dict)
            and aspect_validation.get("passed") is True
            and isinstance(integrity, dict)
            and integrity.get("contract_version")
            == _CONTENT_INTEGRITY_CONTRACT_VERSION
            and integrity.get("algorithm") == "sha256"
            and isinstance(integrity.get("size_bytes"), int)
            and not isinstance(integrity.get("size_bytes"), bool)
            and isinstance(integrity.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", integrity["sha256"])
            and not source_is_symlink
            and _content_integrity_matches(resolved_source, integrity)
        ):
            raise AcquisitionError(
                "멤버 수집 방송 슬롯·파일 무결성 계약이 일치하지 않습니다."
            )
        if canonical_programme is None or programme.strip() != canonical_programme:
            raise _MemberAcquisitionIdentityReauthenticationRequired(
                "캐시된 멤버 직캠의 방송일·음방 프로그램을 공식 "
                "메타데이터로 다시 인증해야 합니다."
            )
        if canonical_date is not None:
            metadata_identity_verified = broadcast_date == canonical_date
        else:
            raw_fresh_identities = (
                fresh_metadata_identities_by_video_id.get(video_id, ())
                if fresh_metadata_identities_by_video_id is not None
                and isinstance(video_id, str)
                else ()
            )
            metadata_identity_verified = any(
                isinstance(identity, _VerifiedFreshMetadataIdentity)
                and identity._seal is _FRESH_METADATA_IDENTITY_SEAL
                and identity.video_id == video_id
                and identity.title == title
                and identity.channel_id == channel_id
                and identity.broadcast_program == programme.strip()
                and identity.broadcast_date == broadcast_date
                and identity.broadcast_date_basis == broadcast_date_basis
                and identity.broadcast_date_basis in {"upload_date", "timestamp"}
                for identity in raw_fresh_identities
            )
        if not metadata_identity_verified:
            raise _MemberAcquisitionIdentityReauthenticationRequired(
                "캐시된 멤버 직캠의 방송일·음방 프로그램을 공식 "
                "메타데이터로 다시 인증해야 합니다."
            )
        seen_video_ids.add(video_id)
        identities.append(
            VerifiedAcquisitionSourceIdentity(
                source_index=source_index,
                source_path=str(resolved_source),
                video_id=video_id,
                broadcast_date=broadcast_date,
                broadcast_program=programme.strip(),
                source_size_bytes=integrity["size_bytes"],
                source_sha256=integrity["sha256"],
            )
        )
    group = str(manifest.get("group", "")).strip().upper()
    song = manifest.get("song")
    target_member = manifest.get("target_member")
    if not (
        group == "IVE"
        and isinstance(song, str)
        and bool(song.strip())
        and isinstance(target_member, str)
        and bool(target_member.strip())
    ):
        raise AcquisitionError("멤버 수집 요청 범위가 올바르지 않습니다.")
    return VerifiedMemberAcquisitionIdentityAuthority(
        group=group,
        song=song,
        target_member=target_member,
        manifest_path=str(manifest_path.resolve(strict=True)),
        manifest_size_bytes=manifest_size,
        manifest_sha256=manifest_sha256,
        sources=tuple(identities),
        _seal=_ACQUISITION_IDENTITY_AUTHORITY_SEAL,
    )


def _build_acquisition_artifact(
    *,
    source_paths: Sequence[Path],
    manifest_path: Path,
    manifest: dict[str, Any],
    fresh_metadata_identities_by_video_id: Mapping[
        str,
        Sequence[_VerifiedFreshMetadataIdentity],
    ] | None = None,
) -> AcquisitionArtifact:
    raw_sources = tuple(Path(path).expanduser() for path in source_paths)
    if any(path.is_symlink() for path in raw_sources):
        raise AcquisitionError("수집 소스 파일은 심볼릭 링크일 수 없습니다.")
    normalized_sources = tuple(path.resolve(strict=True) for path in raw_sources)
    authority: VerifiedMemberAcquisitionIdentityAuthority | None = None
    authority_error: str | None = None
    try:
        authority = _issue_member_acquisition_identity_authority(
            source_paths=normalized_sources,
            manifest_path=manifest_path,
            manifest=manifest,
            fresh_metadata_identities_by_video_id=(
                fresh_metadata_identities_by_video_id
            ),
        )
    except _MemberAcquisitionIdentityReauthenticationRequired as exc:
        if str(manifest.get("edit_mode", "")).strip().lower() != "member":
            raise
        authority_error = str(exc)
    return AcquisitionArtifact(
        source_paths=normalized_sources,
        member_anchors=(),
        manifest_path=manifest_path,
        manifest=manifest,
        member_identity_authority=authority,
        member_identity_authority_error=authority_error,
    )


def _is_capture_variant(candidate: AcquisitionCandidate) -> bool:
    folded = _fold(candidate.title)
    return any(
        marker in folded
        for marker in (
            "풀캠",
            "fullcam",
            "full cam",
            "직캠",
            "fancam",
            "choreography",
            "one take",
            "one-take",
            "원테이크",
            "단독샷",
            "항공캠",
            "sky cam",
            "skycam",
            "encore",
            "앵콜",
        )
    )


def _broadcast_slot_key(
    candidate: AcquisitionCandidate,
) -> tuple[str, str] | None:
    if candidate.broadcast_date is None or candidate.broadcast_program is None:
        return None
    return (candidate.broadcast_date, candidate.broadcast_program)


def _ordered_for_download(
    candidates: Sequence[AcquisitionCandidate],
) -> list[AcquisitionCandidate]:
    """Put every canonical broadcast slot before its camera variants.

    Some short IVE promotions do not expose eight distinct date/program slots.
    A fullcam or one-take from an already represented broadcast is still a
    genuinely different camera source and may be needed to reach the
    request-scoped automatic source floor. It must never displace that slot's canonical programme
    cut, so variants are retained only in a second, round-robin lane after all
    canonical slots.
    """

    if not candidates:
        return []

    clean_durations = [
        candidate.duration_seconds
        for candidate in candidates
        if candidate.selection_tier <= 1
        and candidate.fallback_reason is None
        and not _is_capture_variant(candidate)
    ]
    median_duration = statistics.median(
        clean_durations or [candidate.duration_seconds for candidate in candidates]
    )

    def quality_order(candidate: AcquisitionCandidate) -> tuple[Any, ...]:
        return (
            candidate.selection_tier,
            candidate.fallback_reason is not None,
            _is_capture_variant(candidate),
            abs(candidate.duration_seconds - median_duration),
            -candidate.trust_score,
            _fold(candidate.title),
            candidate.video_id,
        )

    ranked = sorted(candidates, key=quality_order)
    canonical_by_slot: list[AcquisitionCandidate] = []
    variants_by_slot: dict[tuple[str, str], list[AcquisitionCandidate]] = {}
    unslotted: list[AcquisitionCandidate] = []
    seen_slots: set[tuple[str, str]] = set()
    for candidate in ranked:
        slot = _broadcast_slot_key(candidate)
        if slot is None:
            unslotted.append(candidate)
        elif slot not in seen_slots:
            seen_slots.add(slot)
            canonical_by_slot.append(candidate)
        else:
            variants_by_slot.setdefault(slot, []).append(candidate)

    # One alternate from each represented programme is preferable to several
    # alternates from a single broadcast.  Preserve deterministic quality
    # ordering within every round.
    variants: list[AcquisitionCandidate] = []
    maximum_variant_count = max(
        (len(items) for items in variants_by_slot.values()),
        default=0,
    )
    ordered_slots = sorted(
        variants_by_slot,
        key=lambda slot: quality_order(variants_by_slot[slot][0]),
    )
    for variant_index in range(maximum_variant_count):
        variants.extend(
            variants_by_slot[slot][variant_index]
            for slot in ordered_slots
            if variant_index < len(variants_by_slot[slot])
        )
    return [*canonical_by_slot, *unslotted, *variants]


def _select_promotion_date_cluster(
    candidates: Sequence[AcquisitionCandidate],
    *,
    minimum_count: int,
) -> tuple[list[AcquisitionCandidate], dict[str, Any]]:
    """Select one promotion run with enough distinct weekly-show slots.

    Dates connected by no more than 28 days form one deterministic promotion
    cluster. The cluster with the broadest distinct broadcast-slot coverage
    wins, then total camera-source count and recency break ties. At least four
    distinct slots are required; camera variants may fill the remaining
    request-scoped automatic source floor. Undated and cross-promotion clips cannot
    silently fill that floor.
    """

    dated: dict[str, list[AcquisitionCandidate]] = {}
    for candidate in candidates:
        if candidate.broadcast_date is not None:
            dated.setdefault(candidate.broadcast_date, []).append(candidate)
    ordered_dates = sorted(dated)
    date_clusters: list[list[str]] = []
    for broadcast_date in ordered_dates:
        if not date_clusters:
            date_clusters.append([broadcast_date])
            continue
        previous = datetime.strptime(
            date_clusters[-1][-1], "%Y-%m-%d"
        ).date()
        current = datetime.strptime(broadcast_date, "%Y-%m-%d").date()
        if (current - previous).days <= _PROMOTION_CLUSTER_MAXIMUM_GAP_DAYS:
            date_clusters[-1].append(broadcast_date)
        else:
            date_clusters.append([broadcast_date])

    cluster_records: list[dict[str, Any]] = []
    cluster_candidates: list[list[AcquisitionCandidate]] = []
    for cluster_index, dates in enumerate(date_clusters):
        date_set = set(dates)
        pool = [
            candidate
            for candidate in candidates
            if candidate.broadcast_date in date_set
        ]
        ordered_pool = _ordered_for_download(pool)
        distinct_slots = {
            slot
            for candidate in ordered_pool
            if (slot := _broadcast_slot_key(candidate)) is not None
        }
        minimum_distinct_slots = min(4, minimum_count)
        cluster_candidates.append(pool)
        cluster_records.append(
            {
                "cluster_index": cluster_index,
                "start_date": dates[0],
                "end_date": dates[-1],
                "dated_candidate_count": len(pool),
                "ordered_source_count": len(ordered_pool),
                # Retain the original diagnostic key for manifest readers,
                # while making its source-level meaning explicit. Exact
                # video/content duplicates were already removed upstream;
                # alternate cameras from one broadcast remain valid sources.
                "deduplicated_stage_count": len(ordered_pool),
                "distinct_broadcast_slot_count": len(distinct_slots),
                "minimum_distinct_broadcast_slot_count": (
                    minimum_distinct_slots
                ),
                "canonical_stage_count": sum(
                    not _is_capture_variant(candidate)
                    for candidate in ordered_pool
                ),
                "capture_variant_stage_count": sum(
                    _is_capture_variant(candidate)
                    for candidate in ordered_pool
                ),
                "qualifies": (
                    len(ordered_pool) >= minimum_count
                    and len(distinct_slots) >= minimum_distinct_slots
                ),
            }
        )

    qualifying = [
        record for record in cluster_records if record["qualifies"]
    ]
    selected_record = (
        max(
            qualifying,
            key=lambda record: (
                record["distinct_broadcast_slot_count"],
                record["ordered_source_count"],
                record["end_date"],
            ),
        )
        if qualifying
        else None
    )
    selected = (
        cluster_candidates[selected_record["cluster_index"]]
        if selected_record is not None
        else []
    )
    selected_ids = {candidate.video_id for candidate in selected}
    diagnostics = {
        "policy_version": _LATEST_SONG_ACQUISITION_POLICY_VERSION,
        "maximum_adjacent_date_gap_days": (
            _PROMOTION_CLUSTER_MAXIMUM_GAP_DAYS
        ),
        "minimum_deduplicated_stage_count": minimum_count,
        "selection_order": (
            "largest_distinct_slot_count_then_source_count_then_latest_end_date"
        ),
        "same_broadcast_slot": "broadcast_date_plus_program",
        "dated_candidate_count": sum(len(items) for items in dated.values()),
        "undated_candidate_count": sum(
            candidate.broadcast_date is None for candidate in candidates
        ),
        "clusters": cluster_records,
        "status": (
            "selected_qualifying_cluster"
            if selected_record is not None
            else "no_qualifying_cluster"
        ),
        "selected_cluster": selected_record,
        "excluded_cross_promotion_video_ids": sorted(
            candidate.video_id
            for candidate in candidates
            if candidate.video_id not in selected_ids
        ),
    }
    return (selected, diagnostics)


def _is_member_closeup_candidate(candidate: AcquisitionCandidate) -> bool:
    folded = _fold(candidate.title)
    return any(marker in folded for marker in _MEMBER_CLOSEUP_MARKERS)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _minimum_acceptable_source_count(request: AnalysisRequest) -> int:
    """Return the fail-closed floor for one automatic acquisition request."""

    return request.automatic_source_count_floor


def _source_count_policy(request: AnalysisRequest) -> dict[str, Any]:
    """Describe target-versus-floor semantics in the persisted manifest."""

    minimum_count = _minimum_acceptable_source_count(request)
    if request.edit_mode is EditMode.MEMBER:
        policy = "automatic_member_target_ten_minimum_four"
    elif minimum_count < request.desired_source_count:
        policy = "default_group_target_ten_minimum_eight"
    else:
        policy = "explicit_group_count_required"
    return {
        "policy": policy,
        "target_source_count": request.desired_source_count,
        "minimum_acceptable_source_count": minimum_count,
        "fallback_allowed": minimum_count < request.desired_source_count,
        "explicit_nondefault_group_count_is_exact": (
            request.edit_mode is EditMode.GROUP
            and minimum_count == request.desired_source_count
        ),
    }


def _group_relative_quality_selection_policy() -> dict[str, Any]:
    """Return the persisted contract for default automatic GROUP triage."""

    return {
        "required": True,
        "scope": "automatic_group_default_ten_only",
        "analysis_basis": (
            "eight_source_relative_frames_at_7.5_to_92.5_percent_"
            "480px_quality_sharpness_exposure_consistency_clipping_"
            "plus_ffprobe_metadata_and_title_structure"
        ),
        "activation": "nine_or_ten_rankable_sources",
        "eligible_source_count": 8,
        "eight_source_pool_policy": "retain_all_after_hard_contract",
        "failure_policy": (
            "retain_materialized_pool_when_fewer_than_nine_are_rankable"
        ),
        "member_and_local_scope": "disabled",
        "raw_frames_persisted": False,
    }


def _manifest_has_current_hash_valid_catalog_provenance(
    manifest: dict[str, Any],
) -> bool:
    provenance = manifest.get("official_ive_stage_playlist_catalog")
    if not isinstance(provenance, dict):
        return False
    result = provenance.get("result")
    expected_cache_state = {
        "validated_network_catalog_cached": "refreshed",
        "fresh_hash_valid_cache": "fresh",
        "stale_hash_valid_cache_after_refresh_failure": "stale",
    }.get(result)
    object_sha256 = provenance.get("object_sha256")
    entry_count = provenance.get("entry_count")
    return (
        expected_cache_state is not None
        and provenance.get("schema_version")
        == _CATALOG_CACHE_SCHEMA_VERSION
        and provenance.get("cache_policy_version")
        == _CATALOG_CACHE_POLICY_VERSION
        and provenance.get("playlist_id") == _IVE_STAGE_PLAYLIST_ID
        and provenance.get("pinned_owner_channel_id")
        == _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID
        and provenance.get("playlist_owner_proves_catalog_only") is True
        and provenance.get("playlist_owner_never_grants_source_channel_trust")
        is True
        and provenance.get(
            "source_entries_require_existing_metadata_song_member_show_channel_filters"
        )
        is True
        and provenance.get("minimum_plausible_entry_count")
        == _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT
        and provenance.get("cache_state") == expected_cache_state
        and isinstance(object_sha256, str)
        and _CATALOG_OBJECT_SHA256.fullmatch(object_sha256) is not None
        and isinstance(entry_count, int)
        and not isinstance(entry_count, bool)
        and entry_count >= _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT
        and (
            result != "stale_hash_valid_cache_after_refresh_failure"
            or provenance.get("stale_fallback_used") is True
        )
    )


def _manifest_matches_cache_binding_contract(
    manifest: dict[str, Any],
    request: AnalysisRequest,
) -> bool:
    """Return whether prior entries may be considered for per-file reuse.

    A completed policy-v4 set is not automatically a valid current selection,
    but its cache-owned path, video ID, size, and SHA-256 can still authenticate
    each clean entry independently during a v5 reselection.  Partial manifests
    are allowed here because every successful download is atomically persisted
    with that exact binding before the next candidate is attempted.
    """

    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        return False
    policy_version = manifest.get("selection_policy_version")
    expected_target = request.target_member.value if request.target_member else None
    has_contract_mismatch = (
        policy_version
        not in {_SELECTION_POLICY_VERSION, _LEGACY_SELECTION_POLICY_VERSION}
        or manifest.get("content_integrity_contract_version")
        != _CONTENT_INTEGRITY_CONTRACT_VERSION
        or (
            _requires_automatic_media_quality_contract(request)
            and manifest.get("automatic_media_quality_contract_version")
            != _AUTOMATIC_MEDIA_QUALITY_CONTRACT_VERSION
        )
        or (
            _requires_automatic_member_portrait_contract(request)
            and manifest.get("member_portrait_aspect_contract_version")
            != _MEMBER_PORTRAIT_ASPECT_CONTRACT_VERSION
        )
        or manifest.get("group") != "IVE"
        or manifest.get("song") != request.song
        or manifest.get("edit_mode") != request.edit_mode.value
        or manifest.get("target_member") != expected_target
        or manifest.get("desired_source_count") != request.desired_source_count
    )
    return not has_contract_mismatch


def _manifest_selected_titles_pass_current_hard_audit(
    manifest: dict[str, Any],
) -> bool:
    selected = manifest.get("selected")
    return isinstance(selected, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("title"), str)
        and bool(str(item["title"]).strip())
        and _current_hard_title_rejection_reason(str(item["title"])) is None
        for item in selected
    )


def _manifest_matches_request_contract(
    manifest: dict[str, Any],
    request: AnalysisRequest,
) -> bool:
    """Return whether a complete prior set is selectable under current policy."""

    summary = manifest.get("summary")
    group_quality_gate = manifest.get("group_relative_quality_gate")
    if (
        not isinstance(summary, dict)
        or not _manifest_matches_cache_binding_contract(manifest, request)
        or not _manifest_selected_titles_pass_current_hard_audit(manifest)
        or (
            _requires_group_relative_quality_gate(request)
            and (
                manifest.get("group_relative_quality_gate_version")
                != GROUP_RELATIVE_QUALITY_GATE_VERSION
                or not isinstance(group_quality_gate, dict)
                or group_quality_gate.get("status")
                not in {
                    "applied_top_eight_selected",
                    "not_applied_pool_already_at_minimum_eight",
                    "not_applied_insufficient_rankable_sources",
                }
            )
        )
    ):
        return False
    return summary.get("status") == "complete"


def _cache_refresh_upgrade_decision(
    *,
    artifact: AcquisitionArtifact,
    request: AnalysisRequest,
) -> dict[str, Any]:
    """Explain whether a valid fallback-sized cache must search for upgrades."""

    summary = artifact.manifest.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    raw_materialized = summary.get("materialized_source_count")
    materialized_count = (
        raw_materialized
        if isinstance(raw_materialized, int)
        and not isinstance(raw_materialized, bool)
        and raw_materialized >= len(artifact.source_paths)
        else len(artifact.source_paths)
    )
    fallback_allowed = (
        _minimum_acceptable_source_count(request)
        < request.desired_source_count
    )
    promotion_selection = artifact.manifest.get(
        "promotion_date_cluster_selection"
    )
    latest_policy_current = (
        artifact.manifest.get("latest_song_acquisition_policy_version")
        == _LATEST_SONG_ACQUISITION_POLICY_VERSION
        and isinstance(promotion_selection, dict)
        and promotion_selection.get("policy_version")
        == _LATEST_SONG_ACQUISITION_POLICY_VERSION
        and promotion_selection.get("status")
        == "selected_qualifying_cluster"
        and isinstance(promotion_selection.get("selected_cluster"), dict)
        and bool(promotion_selection["selected_cluster"].get("qualifies"))
        and _manifest_has_current_hash_valid_catalog_provenance(
            artifact.manifest
        )
    )
    refresh_required = (
        not latest_policy_current
        or (
            fallback_allowed
            and materialized_count < request.desired_source_count
        )
    )
    catalog_policy_current = (
        _manifest_has_current_hash_valid_catalog_provenance(
            artifact.manifest
        )
    )
    if not catalog_policy_current:
        decision = "refresh_metadata_to_retry_official_catalog_policy"
    elif not latest_policy_current:
        decision = "refresh_metadata_to_apply_latest_song_policy"
    elif refresh_required:
        decision = "refresh_metadata_to_upgrade_verified_fallback_pool"
    elif len(artifact.source_paths) < request.desired_source_count:
        decision = "reuse_target_materialized_quality_gate_reduced_pool"
    else:
        decision = "reuse_target_complete_pool"
    return {
        "policy_version": _LATEST_SONG_ACQUISITION_POLICY_VERSION,
        "decision": decision,
        "refresh_required": refresh_required,
        "latest_song_policy_current": latest_policy_current,
        "catalog_policy_current": catalog_policy_current,
        "metadata_refresh_attempted": False,
        "authenticated_cached_source_count": len(artifact.source_paths),
        "prior_materialized_source_count": materialized_count,
        "target_source_count": request.desired_source_count,
        "minimum_acceptable_source_count": (
            _minimum_acceptable_source_count(request)
        ),
        "prior_selected_promotion_cluster": artifact.manifest.get(
            "promotion_date_cluster_selection", {}
        ).get("selected_cluster")
        if isinstance(
            artifact.manifest.get("promotion_date_cluster_selection"), dict
        )
        else None,
        "result": "pending_refresh" if refresh_required else "reused",
        "new_download_count": 0,
    }


def _manifest_refresh_candidates(
    *,
    artifact: AcquisitionArtifact,
    request: AnalysisRequest,
) -> list[AcquisitionCandidate]:
    """Rehydrate authenticated selected and quality-excluded cache bindings."""

    candidates: list[AcquisitionCandidate] = []
    selected = artifact.manifest.get("selected")
    if not isinstance(selected, list):
        return candidates
    excluded = _validated_group_quality_excluded_bindings(
        manifest=artifact.manifest,
        manifest_path=artifact.manifest_path,
        request=request,
    )
    refresh_items = [
        *selected,
        *(item for item, _path, _video_id in excluded),
    ]
    for item in refresh_items:
        if not isinstance(item, dict):
            continue
        query_value = item.get("query")
        query = query_value if isinstance(query_value, str) else "verified-cache"
        candidate = _metadata_candidate(
            {
                "id": item.get("video_id"),
                "title": item.get("title"),
                "channel": item.get("channel"),
                "channel_id": item.get("channel_id"),
                "duration": item.get("duration_seconds"),
                "webpage_url": item.get("webpage_url"),
            },
            request=request,
            query=query,
            discovery_source="verified_cache",
        )
        if candidate is None or not candidate.selectable:
            continue
        stored_date = item.get("broadcast_date")
        stored_program = item.get("broadcast_program")
        stored_date_basis = item.get("broadcast_date_basis")
        candidates.append(
            AcquisitionCandidate(
                video_id=candidate.video_id,
                title=candidate.title,
                channel=candidate.channel,
                channel_id=candidate.channel_id,
                duration_seconds=candidate.duration_seconds,
                webpage_url=candidate.webpage_url,
                queries=candidate.queries,
                selection_tier=candidate.selection_tier,
                trust_score=candidate.trust_score,
                broadcast_date=(
                    stored_date if isinstance(stored_date, str) else candidate.broadcast_date
                ),
                broadcast_program=(
                    stored_program
                    if isinstance(stored_program, str)
                    else candidate.broadcast_program
                ),
                broadcast_date_basis=(
                    stored_date_basis
                    if isinstance(stored_date_basis, str)
                    else candidate.broadcast_date_basis
                ),
                official_channel_signal=candidate.official_channel_signal,
                rejection_reasons=candidate.rejection_reasons,
                fallback_reason=candidate.fallback_reason,
                discovery_sources=candidate.discovery_sources,
            )
        )
    return candidates


def _validated_manifest_selection(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    request: AnalysisRequest,
    minimum_count: int | None = None,
) -> tuple[tuple[dict[str, Any], Path, str], ...] | None:
    selected = manifest.get("selected")
    required_count = (
        _minimum_acceptable_source_count(request)
        if minimum_count is None
        else minimum_count
    )
    if (
        isinstance(required_count, bool)
        or not isinstance(required_count, int)
        or required_count < 0
    ):
        raise ValueError("manifest selection minimum_count must be non-negative")
    if (
        not isinstance(selected, list)
        or not required_count <= len(selected) <= request.desired_source_count
        or any(not isinstance(item, dict) for item in selected)
    ):
        return None
    ordered = sorted(
        selected,
        key=lambda item: (
            item.get("source_index")
            if isinstance(item.get("source_index"), int)
            and not isinstance(item.get("source_index"), bool)
            else -1
        ),
    )
    if [item.get("source_index") for item in ordered] != list(range(len(ordered))):
        return None
    raw_videos_directory = manifest_path.parent / "videos"
    if raw_videos_directory.is_symlink():
        return None
    videos_directory = raw_videos_directory.resolve()
    entries: list[tuple[dict[str, Any], Path, str]] = []
    try:
        for item in ordered:
            raw_path = item.get("local_path")
            video_id = item.get("video_id")
            if not isinstance(raw_path, str) or not isinstance(video_id, str):
                return None
            path = Path(raw_path).resolve()
            if (
                _VIDEO_ID.fullmatch(video_id) is None
                or path.parent != videos_directory
                or path.stem != video_id
                or path.suffix.lower() not in _VIDEO_SUFFIXES
            ):
                return None
            entries.append((item, path, video_id))
    except (OSError, RuntimeError, ValueError):
        return None
    paths = [entry[1] for entry in entries]
    video_ids = [entry[2] for entry in entries]
    if len(set(paths)) != len(paths) or len(set(video_ids)) != len(video_ids):
        return None
    return tuple(entries)


def _validated_group_quality_excluded_bindings(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    request: AnalysisRequest,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    """Authenticate retained GROUP-quality exclusions from a complete cache.

    Relative quality exclusion does not make a downloaded file invalid.  A
    nine-source cache may therefore retain one authenticated ninth file while
    exposing only the top eight to the editor.  Admit those retained bindings
    into a metadata refresh only when the complete current-policy manifest,
    applied quality gate, ranking, counts, cache-owned path, video ID, size,
    and SHA-256 all agree.  Any ambiguity rejects the entire excluded set.
    """

    promotion_selection = manifest.get("promotion_date_cluster_selection")
    latest_song_policy_version = manifest.get(
        "latest_song_acquisition_policy_version"
    )
    selected_cluster = (
        promotion_selection.get("selected_cluster")
        if isinstance(promotion_selection, dict)
        else None
    )
    group_gate = manifest.get("group_relative_quality_gate")
    if (
        not _requires_group_relative_quality_gate(request)
        or manifest.get("selection_policy_version") != _SELECTION_POLICY_VERSION
        or latest_song_policy_version
        not in {
            _LATEST_SONG_ACQUISITION_POLICY_VERSION,
            _LEGACY_LATEST_SONG_ACQUISITION_POLICY_VERSION,
        }
        or not _manifest_matches_request_contract(manifest, request)
        or manifest.get("group_relative_quality_gate_version")
        != GROUP_RELATIVE_QUALITY_GATE_VERSION
        or not isinstance(group_gate, dict)
        or group_gate.get("gate_version")
        != GROUP_RELATIVE_QUALITY_GATE_VERSION
        or group_gate.get("status") != "applied_top_eight_selected"
        or group_gate.get("applied") is not True
        or not isinstance(promotion_selection, dict)
        or promotion_selection.get("policy_version")
        != latest_song_policy_version
        or promotion_selection.get("status")
        != "selected_qualifying_cluster"
        or not isinstance(selected_cluster, dict)
        or selected_cluster.get("qualifies") is not True
    ):
        return ()

    selected_entries = _validated_manifest_selection(
        manifest=manifest,
        manifest_path=manifest_path,
        request=request,
    )
    summary = manifest.get("summary")
    raw_records = manifest.get("candidates")
    ranking = group_gate.get("ranking")
    if (
        selected_entries is None
        or len(selected_entries) != GROUP_RELATIVE_QUALITY_TARGET
        or not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or not isinstance(raw_records, list)
        or not raw_records
        or any(not isinstance(record, dict) for record in raw_records)
        or not isinstance(ranking, list)
        or any(not isinstance(item, dict) for item in ranking)
    ):
        return ()

    records = list(raw_records)
    candidate_ids = [record.get("video_id") for record in records]
    if (
        any(
            not isinstance(video_id, str)
            or _VIDEO_ID.fullmatch(video_id) is None
            for video_id in candidate_ids
        )
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        return ()

    excluded_records = [
        record
        for record in records
        if record.get("download_status") == "downloaded_quality_excluded"
    ]
    if not excluded_records or any(
        record.get("selected") is not False
        or record.get("group_quality_eligible") is not False
        or record.get("rejection_reasons") != []
        or record.get("download_hard_rejection_reasons") != []
        or not isinstance(record.get("group_relative_quality"), dict)
        or not isinstance(record.get("group_quality_exclusion_reasons"), list)
        or not record["group_quality_exclusion_reasons"]
        for record in excluded_records
    ):
        return ()

    selected_ids = {video_id for _item, _path, video_id in selected_entries}
    selected_paths = {path for _item, path, _video_id in selected_entries}
    excluded_ids = {str(record["video_id"]) for record in excluded_records}
    if (
        len(excluded_ids) != len(excluded_records)
        or selected_ids & excluded_ids
        or not (selected_ids | excluded_ids) <= set(candidate_ids)
    ):
        return ()

    materialized_count = len(selected_entries) + len(excluded_records)
    if materialized_count not in {
        GROUP_RELATIVE_QUALITY_TARGET + 1,
        request.desired_source_count,
    }:
        return ()

    def exact_nonnegative_int(value: object, expected: int) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == expected
        )

    if (
        not exact_nonnegative_int(
            summary.get("materialized_source_count"), materialized_count
        )
        or not exact_nonnegative_int(
            summary.get("eligible_source_count"), len(selected_entries)
        )
        or not exact_nonnegative_int(
            summary.get("downloaded_source_count"), len(selected_entries)
        )
        or not exact_nonnegative_int(
            summary.get("group_quality_excluded_source_count"),
            len(excluded_records),
        )
        or not exact_nonnegative_int(
            group_gate.get("input_source_count"), materialized_count
        )
        or not exact_nonnegative_int(
            group_gate.get("selected_source_count"), len(selected_entries)
        )
        or len(ranking) != materialized_count
    ):
        return ()

    ranking_ids = [item.get("video_id") for item in ranking]
    ranking_indices = [item.get("acquisition_source_index") for item in ranking]
    if (
        any(not isinstance(video_id, str) for video_id in ranking_ids)
        or len(ranking_ids) != len(set(ranking_ids))
        or set(ranking_ids) != selected_ids | excluded_ids
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in ranking_indices
        )
        or set(ranking_indices) != set(range(materialized_count))
    ):
        return ()
    eligible_ranking_ids = {
        str(item["video_id"])
        for item in ranking
        if item.get("eligible") is True
    }
    excluded_ranking_ids = {
        str(item["video_id"])
        for item in ranking
        if item.get("eligible") is False
    }
    if (
        eligible_ranking_ids != selected_ids
        or excluded_ranking_ids != excluded_ids
        or len(eligible_ranking_ids) + len(excluded_ranking_ids)
        != len(ranking)
    ):
        return ()

    selected_gate_indices = group_gate.get("selected_acquisition_indices")
    excluded_gate_indices = group_gate.get("excluded_acquisition_indices")
    expected_selected_indices = {
        int(item["acquisition_source_index"])
        for item in ranking
        if item.get("eligible") is True
    }
    expected_excluded_indices = set(range(materialized_count)) - (
        expected_selected_indices
    )
    if (
        not isinstance(selected_gate_indices, list)
        or not isinstance(excluded_gate_indices, list)
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in (*selected_gate_indices, *excluded_gate_indices)
        )
        or len(selected_gate_indices) != len(set(selected_gate_indices))
        or len(excluded_gate_indices) != len(set(excluded_gate_indices))
        or set(selected_gate_indices) != expected_selected_indices
        or set(excluded_gate_indices) != expected_excluded_indices
    ):
        return ()

    ranking_order = {
        str(item["video_id"]): int(item["acquisition_source_index"])
        for item in ranking
    }
    raw_videos_directory = manifest_path.parent / "videos"
    if raw_videos_directory.is_symlink():
        return ()
    videos_directory = raw_videos_directory.resolve()
    bindings: list[tuple[dict[str, Any], Path, str]] = []
    excluded_paths: set[Path] = set()
    try:
        for record in sorted(
            excluded_records,
            key=lambda item: ranking_order[str(item["video_id"])],
        ):
            raw_path = record.get("local_path")
            video_id = str(record["video_id"])
            if not isinstance(raw_path, str):
                return ()
            unresolved_path = Path(raw_path)
            if unresolved_path.is_symlink():
                return ()
            path = unresolved_path.resolve()
            if (
                path.parent != videos_directory
                or path.stem != video_id
                or path.suffix.lower() not in _VIDEO_SUFFIXES
                or not path.is_file()
                or path in selected_paths
                or path in excluded_paths
                or not _content_integrity_matches(
                    path,
                    record.get("content_integrity"),
                )
            ):
                return ()
            excluded_paths.add(path)
            bindings.append((record, path, video_id))
    except (AcquisitionError, OSError, RuntimeError, ValueError):
        return ()
    return tuple(bindings)


def _quarantine_prior_hard_title_rejections(
    *,
    prior_manifest: dict[str, Any] | None,
    manifest_path: Path,
    request: AnalysisRequest,
) -> dict[str, Any] | None:
    """Quarantine only cache entries newly forbidden by the current title gate."""

    if prior_manifest is None or not _manifest_matches_cache_binding_contract(
        prior_manifest,
        request,
    ):
        return None
    entries = _validated_manifest_selection(
        manifest=prior_manifest,
        manifest_path=manifest_path,
        request=request,
        minimum_count=0,
    )
    if entries is None:
        return None
    rejected: list[dict[str, Any]] = []
    clean_title_count = 0
    unauditable_title_count = 0
    for item, path, video_id in entries:
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            unauditable_title_count += 1
            continue
        reason = _current_hard_title_rejection_reason(title)
        if reason is None:
            clean_title_count += 1
            continue
        quarantined = _quarantine_invalid_media(path)
        rejected.append(
            {
                "source_index": item.get("source_index"),
                "video_id": video_id,
                "title": title,
                "hard_rejection_reason": f"non_music_show_{reason}",
                "quarantined_path": (
                    str(quarantined) if quarantined is not None else None
                ),
            }
        )
    return {
        "from_selection_policy_version": prior_manifest.get(
            "selection_policy_version"
        ),
        "current_selection_policy_version": _SELECTION_POLICY_VERSION,
        "audit_policy": _CURRENT_HARD_TITLE_AUDIT_POLICY,
        "audited_selected_count": len(entries),
        "clean_title_count": clean_title_count,
        "unauditable_title_count": unauditable_title_count,
        "hard_rejected_count": len(rejected),
        "hard_rejected_entries": rejected,
        "clean_entries_require_exact_path_video_id_size_sha256_before_reuse": True,
    }


def _authenticated_cached_path(
    *,
    prior_manifest: dict[str, Any] | None,
    manifest_path: Path,
    request: AnalysisRequest,
    candidate: AcquisitionCandidate,
) -> Path | None:
    """Authorize one cache hit only through an exact prior hash binding."""

    if prior_manifest is None or not _manifest_matches_cache_binding_contract(
        prior_manifest,
        request,
    ):
        return None
    entries = _validated_manifest_selection(
        manifest=prior_manifest,
        manifest_path=manifest_path,
        request=request,
        minimum_count=0,
    )
    if entries is None:
        return None
    excluded_entries = _validated_group_quality_excluded_bindings(
        manifest=prior_manifest,
        manifest_path=manifest_path,
        request=request,
    )
    matches = [
        entry
        for entry in (*entries, *excluded_entries)
        if entry[2] == candidate.video_id
    ]
    if len(matches) != 1:
        return None
    item, path, _ = matches[0]
    prior_title = item.get("title")
    if (
        not isinstance(prior_title, str)
        or not prior_title.strip()
        or _current_hard_title_rejection_reason(prior_title) is not None
    ):
        return None
    if not path.is_file() or not _content_integrity_matches(
        path,
        item.get("content_integrity"),
    ):
        return None
    return path


def _reusable_completed_manifest(
    *,
    request: AnalysisRequest,
    manifest_path: Path,
    media_validator: Callable[[Path], object | None],
    media_probe: Callable[[Path], MediaInfo],
) -> AcquisitionArtifact | None:
    """Reuse one already validated deterministic source set without web drift."""

    manifest = _read_manifest(manifest_path)
    if manifest is None or not _manifest_matches_request_contract(
        manifest,
        request,
    ):
        return None
    summary = manifest["summary"]
    entries = _validated_manifest_selection(
        manifest=manifest,
        manifest_path=manifest_path,
        request=request,
    )
    if entries is None:
        return None
    ordered = [entry[0] for entry in entries]
    paths = tuple(entry[1] for entry in entries)
    if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        return None
    for item, path in zip(ordered, paths, strict=True):
        if not _content_integrity_matches(path, item.get("content_integrity")):
            # Scope and video-id checks above ensure an edited manifest cannot
            # make us move an unrelated user file. A mismatched cache object is
            # retained as evidence but cannot be silently rebound to metadata.
            _quarantine_invalid_media(path)
            return None
    try:
        for item, path in zip(ordered, paths, strict=True):
            aspect_validation = _validate_media_for_request(
                path,
                request=request,
                media_validator=media_validator,
                media_probe=media_probe,
            )
            if aspect_validation is not None:
                item["media_validation"] = aspect_validation
                item["aspect_validation"] = aspect_validation
    except (AcquisitionError, MediaProbeError, OSError, ValueError):
        return None
    summary["cache_reuse_count"] = (
        int(summary.get("cache_reuse_count", 0)) + 1
    )
    summary["last_reused_at"] = datetime.now(UTC).isoformat()
    manifest["minimum_acceptable_source_count"] = (
        _minimum_acceptable_source_count(request)
    )
    manifest["minimum_full_quality_source_count"] = (
        _minimum_acceptable_source_count(request)
    )
    manifest["source_count_policy"] = _source_count_policy(request)
    summary["downloaded_source_count"] = len(paths)
    summary["eligible_source_count"] = len(paths)
    summary.setdefault("materialized_source_count", len(paths))
    summary["requested_source_count"] = request.desired_source_count
    summary["minimum_acceptable_source_count"] = (
        _minimum_acceptable_source_count(request)
    )
    summary["source_count_fallback_applied"] = (
        len(paths) < request.desired_source_count
    )
    if (
        manifest.get("selection_policy_version")
        == _LEGACY_SELECTION_POLICY_VERSION
    ):
        manifest["legacy_selection_policy_current_title_audit"] = {
            "legacy_selection_policy_version": _LEGACY_SELECTION_POLICY_VERSION,
            "current_selection_policy_version": _SELECTION_POLICY_VERSION,
            "audit_policy": _CURRENT_HARD_TITLE_AUDIT_POLICY,
            "audited_selected_count": len(entries),
            "passed": True,
            "audited_at": datetime.now(UTC).isoformat(),
            "cache_files_reused_only_after_exact_path_video_id_size_sha256": True,
        }
    _write_manifest(manifest_path, manifest)
    return _build_acquisition_artifact(
        source_paths=paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _upgrade_completed_manifest_missing_group_quality_gate(
    *,
    request: AnalysisRequest,
    manifest_path: Path,
    reporter: EventReporter,
    media_validator: Callable[[Path], object | None],
    media_probe: Callable[[Path], MediaInfo],
    analyzer: GroupSourceQualityAnalyzer,
) -> AcquisitionArtifact | None:
    """Upgrade one trusted ten-file GROUP cache without touching the network.

    The relative GROUP quality gate was added after policy-v4/v5 manifests had
    already authenticated complete ten-file pools. Such a manifest must not
    be sent through metadata discovery merely because its new derived ranking
    is absent: discovery can put a known-bad candidate ahead of the tenth
    cached file and needlessly download it again.

    This migration is deliberately narrower than normal partial-cache reuse.
    Every non-gate request binding, selected title, source index, cache-owned
    MP4 path, content digest, and current media contract must pass before a
    staged manifest is changed. Media files are never quarantined, renamed,
    or deleted here. Only the manifest is atomically replaced after a complete
    top-eight decision has been produced.
    """

    if not _requires_group_relative_quality_gate(request):
        return None
    manifest = _read_manifest(manifest_path)
    if (
        manifest is None
        or not _manifest_matches_cache_binding_contract(manifest, request)
        or manifest.get("latest_song_acquisition_policy_version")
        != _LATEST_SONG_ACQUISITION_POLICY_VERSION
        or not _manifest_selected_titles_pass_current_hard_audit(manifest)
        or manifest.get("group_relative_quality_gate_version") is not None
        or manifest.get("group_relative_quality_gate") is not None
    ):
        return None
    summary = manifest.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or not isinstance(manifest.get("selection_policy"), dict)
    ):
        return None

    entries = _validated_manifest_selection(
        manifest=manifest,
        manifest_path=manifest_path,
        request=request,
        minimum_count=request.desired_source_count,
    )
    if entries is None or len(entries) != request.desired_source_count:
        return None

    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list) or any(
        not isinstance(record, dict) for record in raw_candidates
    ):
        return None
    candidate_ids = [record.get("video_id") for record in raw_candidates]
    if (
        any(not isinstance(video_id, str) for video_id in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        return None
    selected_ids = [video_id for _item, _path, video_id in entries]
    if any(video_id not in candidate_ids for video_id in selected_ids):
        return None

    # Authenticate the exact old bindings once. Unlike the ordinary recovery
    # path, a mismatch simply declines the migration and leaves every file in
    # place; this function never quarantines evidence.
    stable_identities: dict[Path, tuple[int, int, int, int]] = {}
    for item, path, _video_id in entries:
        raw_path = item.get("local_path")
        try:
            if (
                not isinstance(raw_path, str)
                or Path(raw_path).is_symlink()
                or path.suffix.lower() != ".mp4"
                or not path.is_file()
                or not _content_integrity_matches(
                    path,
                    item.get("content_integrity"),
                )
            ):
                return None
            stat = path.stat()
        except OSError:
            return None
        stable_identities[path] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )

    staged = deepcopy(manifest)
    staged_selected = sorted(
        staged["selected"],
        key=lambda item: item["source_index"],
    )
    staged_records_by_id = {
        str(record["video_id"]): record for record in staged["candidates"]
    }
    try:
        for item, (_original, path, video_id) in zip(
            staged_selected,
            entries,
            strict=True,
        ):
            if item.get("video_id") != video_id:
                return None
            media_validation = _validate_media_for_request(
                path,
                request=request,
                media_validator=media_validator,
                media_probe=media_probe,
            )
            if media_validation is not None:
                item["media_validation"] = media_validation
                item["aspect_validation"] = media_validation
                record = staged_records_by_id[video_id]
                record["media_validation"] = media_validation
                record["aspect_validation"] = media_validation
    except (
        AcquisitionError,
        KeyError,
        MediaProbeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None

    reporter.status(
        "기존 검증 캐시 10개의 상대 화질을 비교해 편집 풀을 업그레이드하는 중…"
    )
    try:
        eligible_paths, eligible_entries, gate = (
            _apply_group_relative_quality_gate(
                request=request,
                paths=tuple(entry[1] for entry in entries),
                selected_entries=staged_selected,
                records_by_id=staged_records_by_id,
                media_probe=media_probe,
                analyzer=analyzer,
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if (
        gate is None
        or gate.get("status") != "applied_top_eight_selected"
        or not gate.get("applied")
        or len(eligible_paths) != 8
        or len(eligible_entries) != 8
    ):
        return None

    # Detect ordinary concurrent replacement while the local visual analysis
    # ran. A changed pool is left untouched and must be handled by the normal
    # authenticated acquisition path.
    try:
        for path, expected_identity in stable_identities.items():
            stat = path.stat()
            if (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            ) != expected_identity:
                return None
    except OSError:
        return None

    materialized_count = len(entries)
    eligible_count = len(eligible_paths)
    staged["selection_policy_version"] = _SELECTION_POLICY_VERSION
    staged["group_relative_quality_gate_version"] = (
        GROUP_RELATIVE_QUALITY_GATE_VERSION
    )
    staged["group_relative_quality_gate"] = gate
    staged["selected"] = eligible_entries
    staged["minimum_acceptable_source_count"] = (
        _minimum_acceptable_source_count(request)
    )
    staged["minimum_full_quality_source_count"] = (
        _minimum_acceptable_source_count(request)
    )
    staged["source_count_policy"] = _source_count_policy(request)
    staged["selection_policy"]["group_relative_quality_gate"] = (
        _group_relative_quality_selection_policy()
    )
    staged_summary = staged["summary"]
    staged_summary.update(
        {
            "downloaded_source_count": eligible_count,
            "materialized_source_count": materialized_count,
            "eligible_source_count": eligible_count,
            "group_quality_excluded_source_count": (
                materialized_count - eligible_count
            ),
            "requested_source_count": request.desired_source_count,
            "minimum_acceptable_source_count": (
                _minimum_acceptable_source_count(request)
            ),
            "source_count_fallback_applied": (
                eligible_count < request.desired_source_count
            ),
            "status": "complete",
            "cache_reuse_count": (
                int(staged_summary.get("cache_reuse_count", 0)) + 1
            ),
            "last_reused_at": datetime.now(UTC).isoformat(),
        }
    )
    staged["group_relative_quality_gate_upgrade"] = {
        "upgraded_from_missing_gate": True,
        "content_integrity_revalidated": True,
        "media_contract_revalidated": True,
        "network_search_performed": False,
        "media_download_performed": False,
        "existing_media_files_modified": False,
        "materialized_source_count": materialized_count,
        "eligible_source_count": eligible_count,
        "upgraded_at": datetime.now(UTC).isoformat(),
    }
    _write_manifest(manifest_path, staged)
    reporter.log(
        "완료된 GROUP 캐시를 네트워크 없이 상대 품질 gate v"
        f"{GROUP_RELATIVE_QUALITY_GATE_VERSION}로 업그레이드: "
        f"{materialized_count}개 평가, {eligible_count}개 사용"
    )
    reporter.progress(0.20, "IVE 교차편집 소스 확인 완료")
    return _build_acquisition_artifact(
        source_paths=eligible_paths,
        manifest_path=manifest_path,
        manifest=staged,
    )


def _apply_group_relative_quality_gate(
    *,
    request: AnalysisRequest,
    paths: Sequence[Path],
    selected_entries: Sequence[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    media_probe: Callable[[Path], MediaInfo],
    analyzer: GroupSourceQualityAnalyzer,
) -> tuple[tuple[Path, ...], list[dict[str, Any]], dict[str, Any] | None]:
    """Return the eligible GROUP pool and persistable relative diagnostics."""

    if len(paths) != len(selected_entries):
        raise ValueError("quality gate paths and selected entries must match")
    if not _requires_group_relative_quality_gate(request):
        return (tuple(paths), list(selected_entries), None)

    assessments: list[GroupSourceQualityAssessment] = []
    if len(paths) <= 8:
        assessments = [
            GroupSourceQualityAssessment.unavailable(
                "not_sampled_pool_already_at_minimum_eight"
            )
            for _ in paths
        ]
    else:
        for path, entry in zip(paths, selected_entries, strict=True):
            try:
                info = media_probe(path)
                assessment = analyzer(
                    path,
                    info,
                    title=str(entry.get("title") or ""),
                    channel=str(entry.get("channel") or ""),
                    broadcast_date=(
                        str(entry["broadcast_date"])
                        if entry.get("broadcast_date") is not None
                        else None
                    ),
                )
                if not isinstance(assessment, GroupSourceQualityAssessment):
                    raise TypeError(
                        "group quality analyzer must return "
                        "GroupSourceQualityAssessment"
                    )
            except (MediaProbeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                assessment = GroupSourceQualityAssessment.unavailable(
                    "relative_quality_analysis_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
            assessments.append(assessment)

    ranked_sources = tuple(
        RankedGroupSource(
            acquisition_index=index,
            video_id=str(entry["video_id"]),
            assessment=assessment,
        )
        for index, (entry, assessment) in enumerate(
            zip(selected_entries, assessments, strict=True)
        )
    )
    decision = select_group_quality_pool(ranked_sources)
    selected_index_set = set(decision.selected_acquisition_indices)
    entry_by_acquisition_index: dict[int, dict[str, Any]] = {}
    ranking_records: list[dict[str, Any]] = []
    for acquisition_index, (entry, assessment) in enumerate(
        zip(selected_entries, assessments, strict=True)
    ):
        video_id = str(entry["video_id"])
        record = records_by_id[video_id]
        eligible = acquisition_index in selected_index_set
        exclusion_reasons = list(
            decision.exclusion_reasons.get(acquisition_index, ())
        )
        assessment_payload = assessment.to_dict()
        relative_rank = decision.relative_ranks.get(acquisition_index)
        entry["acquisition_source_index"] = acquisition_index
        entry["group_relative_quality"] = assessment_payload
        entry["group_quality_eligible"] = eligible
        entry["group_quality_relative_rank"] = relative_rank
        entry["group_quality_exclusion_reasons"] = exclusion_reasons
        record["group_relative_quality"] = assessment_payload
        record["group_quality_eligible"] = eligible
        record["group_quality_relative_rank"] = relative_rank
        record["group_quality_exclusion_reasons"] = exclusion_reasons
        record["selected"] = eligible
        if not eligible:
            record["download_status"] = "downloaded_quality_excluded"
        entry_by_acquisition_index[acquisition_index] = entry
        ranking_records.append(
            {
                "acquisition_source_index": acquisition_index,
                "video_id": video_id,
                "title": entry.get("title"),
                "relative_rank": relative_rank,
                "ranking_score_0_100": assessment.ranking_score_0_100,
                "rankable": assessment.rankable,
                "eligible": eligible,
                "exclusion_reasons": exclusion_reasons,
                "duplicate_feed_key": assessment.duplicate_feed_key,
                "structure_signals": list(assessment.structure_signals),
            }
        )

    eligible_paths = tuple(paths[index] for index in decision.selected_acquisition_indices)
    eligible_entries: list[dict[str, Any]] = []
    for source_index, acquisition_index in enumerate(
        decision.selected_acquisition_indices
    ):
        entry = entry_by_acquisition_index[acquisition_index]
        entry["source_index"] = source_index
        eligible_entries.append(entry)
    gate = {
        **decision.gate,
        "ranking": sorted(
            ranking_records,
            key=lambda item: (
                item["relative_rank"] is None,
                item["relative_rank"] or 2**31,
                item["acquisition_source_index"],
            ),
        ),
        "excluded_files_retained_in_visible_cache": True,
        "excluded_files_used_by_editor": False,
    }
    return (eligible_paths, eligible_entries, gate)


def acquire_sources(
    request: AnalysisRequest,
    reporter: EventReporter,
    *,
    client: YTDLPClient | None = None,
    media_validator: Callable[[Path], object | None] = _validate_downloaded_media,
    media_probe: Callable[[Path], MediaInfo] | None = None,
    group_quality_analyzer: GroupSourceQualityAnalyzer | None = None,
) -> AcquisitionArtifact:
    """Search, filter, deduplicate, and validate an ordered IVE source set.

    Automatic member sources remain official portrait full frames. This stage
    never creates identity anchors or authorizes a tracking/crop fallback.
    """

    if request.source_cache_directory is None:
        raise ValueError("automatic source acquisition needs a cache directory")
    cache_directory = request.source_cache_directory
    if cache_directory.is_symlink():
        raise AcquisitionError(
            "소스 캐시 폴더가 심볼릭 링크여서 안전하게 사용할 수 없습니다: "
            f"{cache_directory}"
        )
    cache_directory.mkdir(parents=True, exist_ok=True)
    videos_directory = cache_directory / "videos"
    if videos_directory.is_symlink():
        raise AcquisitionError(
            "캐시 영상 폴더가 심볼릭 링크여서 안전하게 사용할 수 없습니다: "
            f"{videos_directory}"
        )
    manifest_path = cache_directory / "acquisition-manifest.json"
    prior_manifest = _read_manifest(manifest_path)
    scoped_media_probe = media_probe or probe_media
    scoped_group_quality_analyzer = (
        group_quality_analyzer or analyze_group_source_quality
    )
    reusable = _reusable_completed_manifest(
        request=request,
        manifest_path=manifest_path,
        media_validator=media_validator,
        media_probe=scoped_media_probe,
    )
    cache_refresh_upgrade: dict[str, Any] | None = None
    verified_cache_candidates: list[AcquisitionCandidate] = []
    if reusable is not None:
        cache_refresh_upgrade = _cache_refresh_upgrade_decision(
            artifact=reusable,
            request=request,
        )
        member_identity_reauthentication_required = bool(
            request.edit_mode is EditMode.MEMBER
            and reusable.member_identity_authority is None
        )
        if member_identity_reauthentication_required:
            cache_refresh_upgrade.update(
                {
                    "decision": (
                        "refresh_metadata_to_reauthenticate_member_"
                        "broadcast_identities"
                    ),
                    "refresh_required": True,
                    "member_identity_reauthentication_required": True,
                    "member_identity_authority_error": (
                        reusable.member_identity_authority_error
                    ),
                }
            )
            reporter.emit(
                "log",
                level="warning",
                message=(
                    "캐시된 멤버 직캠의 방송일·프로그램 근거를 다시 "
                    "인증해야 경계 오디오 소스를 복구할 수 있어 "
                    "공식 메타데이터를 갱신합니다."
                ),
            )
        reusable.manifest["cache_refresh_upgrade"] = cache_refresh_upgrade
        _write_manifest(manifest_path, reusable.manifest)
        if not cache_refresh_upgrade["refresh_required"]:
            reporter.status(
                f"검증된 IVE 소스 {len(reusable.source_paths)}개 캐시를 재사용합니다."
            )
            reporter.log(
                f"완료된 소스 구성 재사용: {len(reusable.source_paths)}개"
            )
            reporter.progress(0.20, "IVE 교차편집 소스 확인 완료")
            return _build_acquisition_artifact(
                source_paths=reusable.source_paths,
                manifest_path=manifest_path,
                manifest=reusable.manifest,
            )
        prior_manifest = reusable.manifest
        verified_cache_candidates = (
            []
            if member_identity_reauthentication_required
            else _manifest_refresh_candidates(
                artifact=reusable,
                request=request,
            )
        )
        cache_refresh_upgrade["metadata_refresh_attempted"] = True
        reporter.status(
            f"검증된 {len(reusable.source_paths)}개 캐시를 유지하며 "
            f"목표 {request.desired_source_count}개로 업그레이드할 "
            "최신 음방 메타데이터를 확인합니다."
        )
    upgraded = _upgrade_completed_manifest_missing_group_quality_gate(
        request=request,
        manifest_path=manifest_path,
        reporter=reporter,
        media_validator=media_validator,
        media_probe=scoped_media_probe,
        analyzer=scoped_group_quality_analyzer,
    )
    if upgraded is not None:
        return upgraded
    prior_policy_title_audit = _quarantine_prior_hard_title_rejections(
        prior_manifest=prior_manifest,
        manifest_path=manifest_path,
        request=request,
    )
    if (
        prior_policy_title_audit is not None
        and prior_policy_title_audit["hard_rejected_count"]
    ):
        reporter.emit(
            "log",
            level="warning",
            message=(
                "이전 캐시에서 앵콜·1위 후속·글로벌 특집 클립 "
                f"{prior_policy_title_audit['hard_rejected_count']}개를 격리했습니다. "
                "나머지 파일은 경로·영상 ID·크기·SHA-256이 "
                "일치하는 경우에만 재사용합니다."
            ),
        )
    try:
        client = client or YTDLPClient()
    except AcquisitionError:
        if reusable is None or cache_refresh_upgrade is None:
            raise
        cache_refresh_upgrade.update(
            {
                "result": "refresh_unavailable_reused_verified_fallback",
                "result_source_count": len(reusable.source_paths),
            }
        )
        _write_manifest(manifest_path, reusable.manifest)
        reporter.emit(
            "log",
            level="warning",
            message=(
                "최신 메타데이터 갱신을 시작하지 못해 "
                "검증된 fallback 캐시를 그대로 사용합니다."
            ),
        )
        return _build_acquisition_artifact(
            source_paths=reusable.source_paths,
            manifest_path=manifest_path,
            manifest=reusable.manifest,
        )
    validated_persistent_rejections = (
        _validated_persistent_media_contract_rejections(
            prior_manifest=prior_manifest,
            manifest_path=manifest_path,
            request=request,
            client=client,
        )
    )
    queries = _metadata_queries(request)
    search_limit = min(40, max(18, request.desired_source_count * 2))
    reporter.status("IVE 음방 영상 후보를 검색하는 중…")

    by_id: dict[str, AcquisitionCandidate] = {
        candidate.video_id: candidate
        for candidate in verified_cache_candidates
    }
    search_failures: list[dict[str, str]] = []
    for query_index, query in enumerate(queries):
        reporter.log(f"YouTube 메타데이터 검색: {query}")
        try:
            entries = client.search(query, search_limit)
        except AcquisitionError as exc:
            search_failures.append({"query": query, "error": str(exc)})
            reporter.emit("log", level="warning", message=str(exc))
            continue
        for entry in entries:
            candidate = _metadata_candidate(
                entry,
                request=request,
                query=query,
                verified_fresh_metadata_origin="query_search",
            )
            if candidate is None:
                continue
            previous = by_id.get(candidate.video_id)
            by_id[candidate.video_id] = (
                candidate if previous is None else _merge_discovery(previous, candidate)
            )
        reporter.progress(
            0.01 + 0.07 * (query_index + 1) / len(queries),
            "음방 메타데이터를 모으는 중…",
        )

    reporter.status("IVE 공식 STAGE 카탈로그를 확인하는 중…")
    catalog_entries, catalog_provenance = _load_official_ive_stage_catalog(
        request=request,
        client=client,
    )
    catalog_query = f"playlist:{_IVE_STAGE_PLAYLIST_ID}"
    catalog_fresh_metadata_origin = (
        "validated_network_catalog"
        if catalog_provenance.get("network_status")
        == "validated_pinned_playlist"
        and catalog_provenance.get("result")
        in {
            "validated_network_catalog_cached",
            "validated_network_catalog_cache_write_failed",
        }
        else None
    )
    catalog_candidate_count = 0
    catalog_selectable_count = 0
    catalog_duplicate_video_id_count = 0
    catalog_rejection_counts: dict[str, int] = {}
    for entry in catalog_entries:
        candidate = _metadata_candidate(
            entry,
            request=request,
            query=catalog_query,
            discovery_source="official_ive_stage_playlist",
            verified_fresh_metadata_origin=catalog_fresh_metadata_origin,
        )
        if candidate is None:
            continue
        catalog_candidate_count += 1
        if candidate.selectable:
            catalog_selectable_count += 1
        else:
            for reason in candidate.rejection_reasons:
                catalog_rejection_counts[reason] = (
                    catalog_rejection_counts.get(reason, 0) + 1
                )
        previous = by_id.get(candidate.video_id)
        if previous is not None:
            catalog_duplicate_video_id_count += 1
        by_id[candidate.video_id] = (
            candidate
            if previous is None
            else _merge_discovery(previous, candidate)
        )
    catalog_provenance.update(
        {
            "metadata_filter_candidate_count": catalog_candidate_count,
            "metadata_filter_selectable_count": catalog_selectable_count,
            "metadata_filter_rejected_count": (
                catalog_candidate_count - catalog_selectable_count
            ),
            "metadata_filter_rejection_counts": dict(
                sorted(catalog_rejection_counts.items())
            ),
            "exact_video_id_duplicates_with_query_or_cache": (
                catalog_duplicate_video_id_count
            ),
            "new_unique_video_ids_after_query_merge": (
                catalog_candidate_count - catalog_duplicate_video_id_count
            ),
            "merge_order": "after_query_search_richer_whole_candidate_preserved",
        }
    )

    # Collapse mirror/re-upload duplicates after exact video-ID deduplication.
    deduplicated: dict[tuple[str, int], AcquisitionCandidate] = {}
    duplicate_ids: list[str] = []
    verified_cache_ids = {
        candidate.video_id for candidate in verified_cache_candidates
    }
    for candidate in sorted(
        by_id.values(),
        key=lambda candidate: (
            candidate.video_id not in verified_cache_ids,
            _candidate_order(candidate),
        ),
    ):
        fingerprint = _content_fingerprint(candidate)
        if fingerprint in deduplicated:
            duplicate_ids.append(candidate.video_id)
            continue
        deduplicated[fingerprint] = candidate
    candidates = sorted(deduplicated.values(), key=_candidate_order)
    records = [_record(candidate) for candidate in candidates]
    records_by_id = {record["video_id"]: record for record in records}
    selectable = [candidate for candidate in candidates if candidate.selectable]
    selectable, promotion_cluster_selection = _select_promotion_date_cluster(
        selectable,
        minimum_count=_minimum_acceptable_source_count(request),
    )
    promotion_selected_ids = {
        candidate.video_id for candidate in selectable
    }
    for record in records:
        if (
            not record["rejection_reasons"]
            and record["video_id"] not in promotion_selected_ids
        ):
            record["selection_exclusion_reason"] = (
                "outside_selected_promotion_date_cluster"
            )
    primary = [candidate for candidate in selectable if candidate.selection_tier <= 1]
    fallback = [candidate for candidate in selectable if candidate.selection_tier > 1]
    if request.edit_mode is EditMode.MEMBER:
        # Standard vertical performer fancams preserve the full choreography
        # and a consistent output aspect. FaceCam/close-up variants remain a
        # last-resort source only when ten standard official clips are not
        # available.
        primary_standard = [
            candidate for candidate in primary
            if not _is_member_closeup_candidate(candidate)
        ]
        fallback_standard = [
            candidate for candidate in fallback
            if not _is_member_closeup_candidate(candidate)
        ]
        primary_closeup = [
            candidate for candidate in primary
            if _is_member_closeup_candidate(candidate)
        ]
        fallback_closeup = [
            candidate for candidate in fallback
            if _is_member_closeup_candidate(candidate)
        ]
        ordered = [
            *_ordered_for_download(primary_standard),
            *_ordered_for_download(fallback_standard),
            *_ordered_for_download(primary_closeup),
            *_ordered_for_download(fallback_closeup),
        ]
    else:
        ordered = [*_ordered_for_download(primary), *_ordered_for_download(fallback)]
    fresh_metadata_identities_by_video_id = {
        candidate.video_id: candidate._fresh_metadata_identities
        for candidate in ordered
        if candidate._fresh_metadata_identities
    }
    # The primary/fallback and standard/close-up lanes are ranked separately.
    # Their internal ordering has already placed all canonical programme slots
    # before alternate cameras; retaining those alternates is necessary when a
    # short promotion exposes fewer distinct weekly-show slots than the target.

    portrait_contract_required = _requires_automatic_member_portrait_contract(
        request
    )
    automatic_quality_contract_required = (
        _requires_automatic_media_quality_contract(request)
    )
    group_relative_quality_gate_required = (
        _requires_group_relative_quality_gate(request)
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection_policy_version": _SELECTION_POLICY_VERSION,
        "latest_song_acquisition_policy_version": (
            _LATEST_SONG_ACQUISITION_POLICY_VERSION
        ),
        "prior_selection_policy_title_audit": prior_policy_title_audit,
        "cache_refresh_upgrade": cache_refresh_upgrade,
        "content_integrity_contract_version": (
            _CONTENT_INTEGRITY_CONTRACT_VERSION
        ),
        "persistent_media_contract_rejection_contract_version": (
            _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
        ),
        "persistent_media_contract_rejections": [
            deepcopy(entry.evidence)
            for entry in validated_persistent_rejections
        ],
        "persistent_media_contract_rejection_refresh": {
            "contract_version": (
                _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
            ),
            "carried_forward_count": len(validated_persistent_rejections),
            "migrated_legacy_rejection_count": 0,
            "new_rejection_count": 0,
            "skipped_download_count": 0,
            "new_rejections": [],
            "skipped_downloads": [],
        },
        "automatic_media_quality_contract_version": (
            _AUTOMATIC_MEDIA_QUALITY_CONTRACT_VERSION
            if automatic_quality_contract_required
            else None
        ),
        "group_relative_quality_gate_version": (
            GROUP_RELATIVE_QUALITY_GATE_VERSION
            if group_relative_quality_gate_required
            else None
        ),
        "group_relative_quality_gate": None,
        "member_portrait_aspect_contract_version": (
            _MEMBER_PORTRAIT_ASPECT_CONTRACT_VERSION
            if portrait_contract_required
            else None
        ),
        "group": "IVE",
        "song": request.song,
        "edit_mode": request.edit_mode.value,
        "aspect_ratio": request.aspect_ratio.value,
        "target_member": request.target_member.value if request.target_member else None,
        "desired_source_count": request.desired_source_count,
        "minimum_acceptable_source_count": (
            _minimum_acceptable_source_count(request)
        ),
        "minimum_full_quality_source_count": (
            _minimum_acceptable_source_count(request)
        ),
        "source_count_policy": _source_count_policy(request),
        "cache_directory": str(cache_directory),
        "yt_dlp_executable": client.executable,
        "download_profile": getattr(
            client,
            "download_profile",
            "injected_client_unspecified_profile",
        ),
        "download_format_policy": (
            {
                "profile": client.download_profile,
                "maximum_long_edge": _PROFESSIONAL_MAXIMUM_LONG_EDGE,
                "preference": "4k_then_1440p_then_1080p",
                "codec_policy": (
                    "allow_av1_or_vp9_for_source_pixels_with_avc_full_hd_fallback"
                ),
                "production_output": {
                    "group": [
                        _GROUP_PRODUCTION_WIDTH,
                        _GROUP_PRODUCTION_HEIGHT,
                    ],
                    "member": [
                        _MEMBER_PRODUCTION_WIDTH,
                        _MEMBER_PRODUCTION_HEIGHT,
                    ],
                },
                "maximum_recorded_lossless_zoom": (
                    _PROFESSIONAL_MAXIMUM_LOSSLESS_ZOOM
                ),
            }
            if getattr(client, "download_profile", None)
            == _PROFESSIONAL_DOWNLOAD_PROFILE
            else {
                "profile": getattr(
                    client,
                    "download_profile",
                    "injected_client_unspecified_profile",
                ),
                "preference": "client_defined_or_full_hd_compatibility",
            }
        ),
        "network_policy": {
            "cookies": "disabled",
            "drm_bypass": False,
            "uploads": False,
        },
        "queries": list(queries),
        "search_failures": search_failures,
        "official_ive_stage_playlist_catalog": catalog_provenance,
        "deduplicated_video_ids": duplicate_ids,
        "promotion_date_cluster_selection": promotion_cluster_selection,
        "selection_policy": {
            "primary": "trusted_official_broadcast_channel_and_music_show_title",
            "fallback": "official_intro_plus_song_stage_only_after_clean_stages",
            "hard_rejections": [
                "encore",
                "winner_afterstage",
                "music_bank_global_special",
                "concert_or_festival",
                "dance_practice",
                "stage_mix",
                "music_video",
                "other_ive_song_title_signal",
                *(
                    [
                        "automatic_video_resolution_below_minimum",
                        "automatic_video_frame_rate_unavailable",
                        "automatic_video_frame_rate_below_minimum",
                    ]
                    if automatic_quality_contract_required
                    else []
                ),
                *(
                    [
                        "automatic_member_portrait_first_video_stream_missing",
                        "automatic_member_portrait_dimensions_unavailable",
                        "automatic_member_portrait_crop_headroom_below_production",
                    ]
                    if portrait_contract_required
                    else []
                ),
                *(
                    [
                        "automatic_group_landscape_first_video_stream_missing",
                        "automatic_group_landscape_dimensions_unavailable",
                        "automatic_group_landscape_aspect_ratio_mismatch",
                    ]
                    if automatic_quality_contract_required
                    and request.edit_mode is EditMode.GROUP
                    else []
                ),
            ],
            "channel_requirement": "allowlisted_channel_id_required_name_signal_diagnostic_only",
            "official_ive_stage_playlist_catalog": {
                "playlist_id": _IVE_STAGE_PLAYLIST_ID,
                "pinned_owner_channel_id": (
                    _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID
                ),
                "playlist_owner_proves_catalog_only": True,
                "playlist_owner_never_grants_source_channel_trust": True,
                "every_entry_uses_existing_metadata_filters": True,
                "media_download_uses_no_playlist": True,
            },
            "performance_diversity": (
                "all_canonical_broadcast_slots_before_round_robin_"
                "fullcam_or_one_take_floor_completion"
            ),
            "latest_song_title_policy": {
                "unregistered_song": (
                    "exact_normalized_token_sequence_case_space_punctuation"
                ),
                "substring_inside_another_title": "rejected",
                "other_registered_ive_activity_in_same_title": "rejected",
                "promotion_cluster": (
                    "minimum_four_distinct_slots_and_request_scoped_source_floor_then_"
                    "largest_slot_count_source_count_latest_tie"
                ),
            },
            "reference_source": "clean_canonical_stage_near_median_duration",
            "encore_title_policy": {
                "explicit_markers": ["encore", "앵콜"],
                "winner_afterstage": (
                    "unreleased_or_미방분_marker_plus_explicit_first_place_marker"
                ),
                "winner_fancam": (
                    "direct_1위_plus_직캠_structure_with_symbol_or_quality_tag_gap"
                ),
                "ordinary_main_stage_win_mentions_without_those_markers": "allowed",
            },
            "global_special_title_policy": {
                "hard_rejected_patterns": [
                    "Music Bank in <city>",
                    "뮤직뱅크 인 <city>",
                ],
                "reason": "official_global_concert_or_special_not_weekly_music_show",
                "ordinary_weekly_music_bank_title": "allowed",
            },
            "cache_content_binding": "exact_path_sha256_and_size_required_before_any_preexisting_file_reuse",
            "persistent_media_contract_rejection": {
                "contract_version": (
                    _PERSISTENT_MEDIA_CONTRACT_REJECTION_VERSION
                ),
                "skip_requires": (
                    "exact_current_metadata_observations_plus_request_media_"
                    "contract_plus_quarantined_content_sha256_plus_same_"
                    "failure_revalidated"
                ),
                "metadata_or_content_or_mode_or_aspect_change": (
                    "revalidate_by_download"
                ),
                "raw_boolean_or_partial_manifest_evidence": "never_skip",
            },
            "member_camera_preference": (
                "standard_fancam_before_facecam_or_closeup_fallback"
                if request.edit_mode is EditMode.MEMBER
                else None
            ),
            "member_portrait_aspect_contract": (
                {
                    "required": True,
                    "analysis_basis": (
                        "downloaded_or_cached_first_video_stream_dynamic_"
                        "portrait_crop_without_upscale"
                    ),
                    "target_output_width": _MEMBER_PRODUCTION_WIDTH,
                    "target_output_height": _MEMBER_PRODUCTION_HEIGHT,
                    "minimum_lossless_crop_headroom_zoom": (
                        _MEMBER_MINIMUM_LOSSLESS_CROP_HEADROOM_ZOOM
                    ),
                    "accepted_source_orientation": (
                        "portrait_or_landscape_when_dynamic_crop_preserves_"
                        "target_resolution"
                    ),
                    "failure_policy": "quarantine_reject_and_continue_to_next_metadata_candidate",
                }
                if portrait_contract_required
                else None
            ),
            "automatic_media_quality_contract": (
                {
                    "required": True,
                    "analysis_basis": "downloaded_or_authenticated_cached_first_video_stream_ffprobe",
                    "minimum_short_edge": _AUTOMATIC_MINIMUM_SHORT_EDGE,
                    "minimum_long_edge": _AUTOMATIC_MINIMUM_LONG_EDGE,
                    "minimum_frame_rate": _AUTOMATIC_MINIMUM_FRAME_RATE,
                    "group_expected_width_over_height": (
                        _AUTOMATIC_GROUP_LANDSCAPE_RATIO
                    ),
                    "group_ratio_tolerance": (
                        _AUTOMATIC_GROUP_LANDSCAPE_RATIO_TOLERANCE
                    ),
                    "member_expected_width_over_height": (
                        _AUTOMATIC_MEMBER_PORTRAIT_RATIO
                    ),
                    "member_ratio_tolerance": (
                        _AUTOMATIC_MEMBER_PORTRAIT_RATIO_TOLERANCE
                    ),
                    "failure_policy": "quarantine_reject_and_continue_to_next_metadata_candidate",
                }
                if automatic_quality_contract_required
                else None
            ),
            "group_relative_quality_gate": (
                _group_relative_quality_selection_policy()
                if group_relative_quality_gate_required
                else None
            ),
            "duration_seconds": {
                "minimum": _MINIMUM_DURATION_SECONDS,
                "maximum": _MAXIMUM_DURATION_SECONDS,
            },
        },
        "candidates": records,
        "selected": [],
        "summary": {
            "discovered_unique_video_ids": len(by_id),
            "query_search_unique_video_ids_before_catalog": (
                len(by_id)
                - catalog_candidate_count
                + catalog_duplicate_video_id_count
            ),
            "catalog_metadata_candidate_count": catalog_candidate_count,
            "catalog_metadata_selectable_count": catalog_selectable_count,
            "catalog_metadata_rejected_count": (
                catalog_candidate_count - catalog_selectable_count
            ),
            "catalog_exact_video_id_duplicate_count": (
                catalog_duplicate_video_id_count
            ),
            "deduplicated_candidate_count": len(candidates),
            "primary_candidate_count": len(primary),
            "fallback_candidate_count": len(fallback),
            "downloaded_source_count": 0,
            "materialized_source_count": 0,
            "eligible_source_count": 0,
            "group_quality_excluded_source_count": 0,
            "requested_source_count": request.desired_source_count,
            "minimum_acceptable_source_count": (
                _minimum_acceptable_source_count(request)
            ),
            "source_count_fallback_applied": False,
            "failed_download_count": 0,
            "media_contract_hard_rejection_count": 0,
            "aspect_hard_rejection_count": 0,
            "fallback_selected_count": 0,
            "persistent_media_contract_rejection_count": len(
                validated_persistent_rejections
            ),
            "persistent_media_contract_new_rejection_count": 0,
            "persistent_media_contract_skip_count": 0,
            "persistent_media_contract_migrated_legacy_count": 0,
            "status": "metadata_complete",
            "cache_reuse_count": (
                int(
                    prior_manifest.get("summary", {}).get(
                        "cache_reuse_count", 0
                    )
                )
                if cache_refresh_upgrade is not None
                and isinstance(prior_manifest, dict)
                and isinstance(prior_manifest.get("summary"), dict)
                else 0
            ),
        },
    }
    _write_manifest(manifest_path, manifest)

    minimum_acceptable_count = _minimum_acceptable_source_count(request)
    if len(ordered) < minimum_acceptable_count:
        if reusable is not None and cache_refresh_upgrade is not None:
            cache_refresh_upgrade.update(
                {
                    "result": (
                        "refresh_metadata_insufficient_reused_verified_fallback"
                    ),
                    "result_source_count": len(reusable.source_paths),
                    "new_download_count": 0,
                    "catalog_refresh_result": catalog_provenance["result"],
                }
            )
            reusable.manifest[
                "official_ive_stage_playlist_catalog"
            ] = catalog_provenance
            reusable.manifest["search_failures"] = search_failures
            _copy_persistent_media_contract_rejection_state(
                destination=reusable.manifest,
                source=manifest,
            )
            _write_manifest(manifest_path, reusable.manifest)
            reporter.emit(
                "log",
                level="warning",
                message=(
                    "최신 메타데이터에서 8개 이상의 하나의 "
                    "활동 군집을 다시 확보하지 못해 검증된 "
                    "fallback 캐시를 사용합니다."
                ),
            )
            return _build_acquisition_artifact(
                source_paths=reusable.source_paths,
                manifest_path=manifest_path,
                manifest=reusable.manifest,
            )
        manifest["summary"]["status"] = "insufficient_metadata_candidates"
        _write_manifest(manifest_path, manifest)
        raise AcquisitionError(
            "공식 음방/직캠 조건을 만족한 IVE 영상을 "
            f"{len(ordered)}개만 찾았습니다. 최소 "
            f"{minimum_acceptable_count}개가 필요합니다. "
            f"진단: {manifest_path}"
        )

    source_paths: list[Path] = []
    download_failures = 0
    fallback_selected_count = 0
    aspect_hard_rejection_count = 0
    persistent_media_contract_new_rejection_count = 0
    persistent_media_contract_skip_count = 0
    persistent_media_contract_migrated_legacy_count = 0
    authenticated_cache_reuse_ids: set[str] = set()
    new_download_ids: set[str] = set()
    for candidate_index, candidate in enumerate(ordered):
        if len(source_paths) >= request.desired_source_count:
            break
        record = records_by_id[candidate.video_id]
        record["download_status"] = "downloading"
        if candidate.fallback_reason is not None:
            reporter.emit(
                "log",
                level="warning",
                message=(
                    f"일반 음방 후보가 부족해 예비 후보를 시도합니다: "
                    f"{candidate.title} ({candidate.fallback_reason})"
                ),
            )
        reporter.status(
            f"소스 {len(source_paths) + 1}/{request.desired_source_count} 확인 중: "
            f"{candidate.title}"
        )
        try:
            videos_directory = cache_directory / "videos"
            persistent_rejection = (
                _matching_persistent_media_contract_rejection(
                    candidate=candidate,
                    validated_rejections=validated_persistent_rejections,
                    request=request,
                    media_validator=media_validator,
                    media_probe=scoped_media_probe,
                )
            )
            if persistent_rejection is None:
                persistent_rejection = (
                    _migrate_legacy_persistent_media_contract_rejection(
                        prior_manifest=prior_manifest,
                        manifest_path=manifest_path,
                        candidate=candidate,
                        request=request,
                        client=client,
                        media_validator=media_validator,
                        media_probe=scoped_media_probe,
                    )
                )
                if persistent_rejection is not None:
                    _upsert_persistent_media_contract_rejection(
                        manifest,
                        persistent_rejection,
                    )
                    persistent_media_contract_migrated_legacy_count += 1
                    manifest[
                        "persistent_media_contract_rejection_refresh"
                    ]["migrated_legacy_rejection_count"] = (
                        persistent_media_contract_migrated_legacy_count
                    )
            if persistent_rejection is not None:
                raise _PersistentMediaContractRejectionSkip(
                    persistent_rejection
                )
            preexisting_paths = _cached_video_paths(
                videos_directory,
                candidate.video_id,
            )
            authenticated_cached_path = _authenticated_cached_path(
                prior_manifest=prior_manifest,
                manifest_path=manifest_path,
                request=request,
                candidate=candidate,
            )
            for preexisting_path in preexisting_paths:
                if preexisting_path == authenticated_cached_path:
                    continue
                quarantined = _quarantine_invalid_media(preexisting_path)
                if quarantined is not None:
                    record["preexisting_cache_quarantined_paths"].append(
                        str(quarantined)
                    )
            path = (
                authenticated_cached_path
                if authenticated_cached_path is not None
                else client.download(candidate, cache_directory)
            )
            try:
                aspect_validation = _validate_media_for_request(
                    path,
                    request=request,
                    media_validator=media_validator,
                    media_probe=scoped_media_probe,
                )
            except _MediaContractError as exc:
                quarantined = _quarantine_invalid_media(path)
                record["aspect_validation"] = exc.diagnostics
                record["download_hard_rejection_reasons"].append(
                    exc.hard_rejection_reason
                )
                if exc.hard_rejection_reason not in record["rejection_reasons"]:
                    record["rejection_reasons"].append(
                        exc.hard_rejection_reason
                    )
                record["quarantined_path"] = (
                    str(quarantined) if quarantined is not None else None
                )
                persistent_evidence = (
                    _build_persistent_media_contract_rejection(
                        candidate=candidate,
                        request=request,
                        client=client,
                        cache_directory=cache_directory,
                        quarantine_path=quarantined,
                        hard_rejection_reason=exc.hard_rejection_reason,
                        diagnostics=exc.diagnostics,
                    )
                )
                if persistent_evidence is not None:
                    record["persistent_media_contract_rejection"] = (
                        deepcopy(persistent_evidence)
                    )
                    _upsert_persistent_media_contract_rejection(
                        manifest,
                        persistent_evidence,
                    )
                    persistent_media_contract_new_rejection_count += 1
                    refresh_audit = manifest[
                        "persistent_media_contract_rejection_refresh"
                    ]
                    refresh_audit["new_rejection_count"] = (
                        persistent_media_contract_new_rejection_count
                    )
                    refresh_audit["new_rejections"].append(
                        {
                            "video_id": candidate.video_id,
                            "title": candidate.title,
                            "decision": (
                                "downloaded_rejected_and_content_bound"
                            ),
                            "hard_rejection_reason": (
                                exc.hard_rejection_reason
                            ),
                            "candidate_media_metadata_sha256": (
                                persistent_evidence[
                                    "candidate_media_metadata_identity"
                                ]["sha256"]
                            ),
                            "request_media_contract_sha256": (
                                persistent_evidence[
                                    "request_media_contract_identity"
                                ]["sha256"]
                            ),
                            "quarantine_relative_path": (
                                persistent_evidence["quarantine_artifact"][
                                    "relative_path"
                                ]
                            ),
                            "quarantined_content_sha256": (
                                persistent_evidence["quarantine_artifact"][
                                    "content_integrity"
                                ]["sha256"]
                            ),
                        }
                    )
                aspect_hard_rejection_count += 1
                raise AcquisitionError(
                    f"{candidate.title}: 자동 소스 미디어 계약 검증 실패 ({exc})"
                ) from exc
            except (AcquisitionError, MediaProbeError, OSError, ValueError) as exc:
                quarantined = _quarantine_invalid_media(path)
                record["quarantined_path"] = (
                    str(quarantined) if quarantined is not None else None
                )
                raise AcquisitionError(f"{candidate.title}: 미디어 검증 실패 ({exc})") from exc
            content_integrity = _content_integrity(path)
            record["media_validation"] = aspect_validation
            record["aspect_validation"] = aspect_validation
            record["content_integrity"] = content_integrity
            if path in source_paths:
                raise AcquisitionError("중복된 로컬 다운로드 경로")
            source_paths.append(path)
            record["selected"] = True
            record["download_status"] = (
                "cached"
                if authenticated_cached_path is not None
                else "downloaded"
            )
            record["local_path"] = str(path)
            if authenticated_cached_path is not None:
                authenticated_cache_reuse_ids.add(candidate.video_id)
            else:
                new_download_ids.add(candidate.video_id)
            if candidate.fallback_reason is not None:
                fallback_selected_count += 1
            manifest["selected"].append(
                {
                    "source_index": len(source_paths) - 1,
                    "video_id": candidate.video_id,
                    "title": candidate.title,
                    "channel": candidate.channel,
                    "channel_id": candidate.channel_id,
                    "duration_seconds": candidate.duration_seconds,
                    "webpage_url": candidate.webpage_url,
                    "query": candidate.query,
                    "queries": list(candidate.queries),
                    "discovery_sources": list(candidate.discovery_sources),
                    "broadcast_date": candidate.broadcast_date,
                    "broadcast_program": candidate.broadcast_program,
                    "broadcast_date_basis": candidate.broadcast_date_basis,
                    "official_channel_signal": candidate.official_channel_signal,
                    "fallback_reason": candidate.fallback_reason,
                    "download_status": record["download_status"],
                    "local_path": str(path),
                    "media_validation": aspect_validation,
                    "aspect_validation": aspect_validation,
                    "content_integrity": content_integrity,
                }
            )
            reporter.log(f"소스 확보: {candidate.title} · {candidate.channel}")
        except _PersistentMediaContractRejectionSkip as exc:
            evidence = exc.evidence
            reason = str(evidence["hard_rejection_reason"])
            persistent_media_contract_skip_count += 1
            record["selected"] = False
            record["download_status"] = (
                "skipped_persistent_media_contract_rejection"
            )
            record["download_error"] = (
                "동일 메타데이터·요청 계약·격리 콘텐츠에서 이미 "
                f"확정된 미디어 계약 실패: {reason}"
            )
            record["aspect_validation"] = deepcopy(evidence["diagnostics"])
            record["download_hard_rejection_reasons"] = [reason]
            if reason not in record["rejection_reasons"]:
                record["rejection_reasons"].append(reason)
            record["quarantined_path"] = str(
                manifest_path.parent
                / evidence["quarantine_artifact"]["relative_path"]
            )
            record["persistent_media_contract_rejection"] = deepcopy(
                evidence
            )
            refresh_audit = manifest[
                "persistent_media_contract_rejection_refresh"
            ]
            refresh_audit["skipped_download_count"] = (
                persistent_media_contract_skip_count
            )
            refresh_audit["skipped_downloads"].append(
                {
                    "video_id": candidate.video_id,
                    "title": candidate.title,
                    "decision": (
                        "download_skipped_after_exact_evidence_revalidation"
                    ),
                    "evidence_origin": evidence["evidence_origin"],
                    "hard_rejection_reason": reason,
                    "candidate_media_metadata_sha256": evidence[
                        "candidate_media_metadata_identity"
                    ]["sha256"],
                    "request_media_contract_sha256": evidence[
                        "request_media_contract_identity"
                    ]["sha256"],
                    "quarantine_relative_path": evidence[
                        "quarantine_artifact"
                    ]["relative_path"],
                    "quarantined_content_sha256": evidence[
                        "quarantine_artifact"
                    ]["content_integrity"]["sha256"],
                }
            )
            reporter.emit(
                "log",
                level="warning",
                message=(
                    f"재다운로드 생략: {candidate.video_id} · {reason} "
                    "(현재 메타데이터·요청 계약·격리 파일 SHA와 "
                    "실패 재검증 일치)"
                ),
            )
        except AcquisitionError as exc:
            download_failures += 1
            record["selected"] = False
            record["download_status"] = "failed"
            record["download_error"] = str(exc)
            reporter.emit("log", level="warning", message=str(exc))
        manifest["summary"].update(
            {
                "downloaded_source_count": len(source_paths),
                "materialized_source_count": len(source_paths),
                "eligible_source_count": len(source_paths),
                "failed_download_count": download_failures,
                "media_contract_hard_rejection_count": (
                    aspect_hard_rejection_count
                ),
                "aspect_hard_rejection_count": aspect_hard_rejection_count,
                "fallback_selected_count": fallback_selected_count,
                "persistent_media_contract_rejection_count": len(
                    manifest["persistent_media_contract_rejections"]
                ),
                "persistent_media_contract_new_rejection_count": (
                    persistent_media_contract_new_rejection_count
                ),
                "persistent_media_contract_skip_count": (
                    persistent_media_contract_skip_count
                ),
                "persistent_media_contract_migrated_legacy_count": (
                    persistent_media_contract_migrated_legacy_count
                ),
                "status": "downloading",
            }
        )
        _write_manifest(manifest_path, manifest)
        reporter.progress(
            0.08 + 0.12 * (candidate_index + 1) / max(1, len(ordered)),
            "IVE 교차편집 소스를 다운로드하는 중…",
        )

    if len(source_paths) < minimum_acceptable_count:
        if reusable is not None and cache_refresh_upgrade is not None:
            cache_refresh_upgrade.update(
                {
                    "result": (
                        "refresh_downloads_insufficient_reused_verified_fallback"
                    ),
                    "result_source_count": len(reusable.source_paths),
                    "new_download_count": len(new_download_ids),
                    "catalog_refresh_result": catalog_provenance["result"],
                }
            )
            reusable.manifest[
                "official_ive_stage_playlist_catalog"
            ] = catalog_provenance
            reusable.manifest["search_failures"] = search_failures
            _copy_persistent_media_contract_rejection_state(
                destination=reusable.manifest,
                source=manifest,
            )
            _write_manifest(manifest_path, reusable.manifest)
            reporter.emit(
                "log",
                level="warning",
                message=(
                    "최신 활동 군집 다운로드가 8개에 못 미쳐 "
                    "기존 검증 fallback 캐시를 유지합니다."
                ),
            )
            return _build_acquisition_artifact(
                source_paths=reusable.source_paths,
                manifest_path=manifest_path,
                manifest=reusable.manifest,
            )
        manifest["summary"]["status"] = "insufficient_downloaded_sources"
        _write_manifest(manifest_path, manifest)
        raise AcquisitionError(
            f"다운로드 성공 {len(source_paths)}개, 실패 {download_failures}개입니다. "
            f"최소 음방 소스 {minimum_acceptable_count}개를 확보하지 "
            f"못했습니다. 진단: {manifest_path}"
        )
    materialized_source_count = len(source_paths)
    (
        eligible_paths,
        eligible_entries,
        group_relative_quality_gate,
    ) = _apply_group_relative_quality_gate(
        request=request,
        paths=source_paths,
        selected_entries=manifest["selected"],
        records_by_id=records_by_id,
        media_probe=scoped_media_probe,
        analyzer=scoped_group_quality_analyzer,
    )
    source_paths = list(eligible_paths)
    manifest["selected"] = eligible_entries
    manifest["group_relative_quality_gate"] = group_relative_quality_gate
    if group_relative_quality_gate is not None:
        if group_relative_quality_gate.get("applied"):
            reporter.status(
                "소스 화질·노출·선명도 일관성을 비교해 상위 8개 음방을 "
                "편집 풀로 확정했습니다."
            )
            reporter.log(
                "GROUP 상대 품질 게이트 적용: "
                f"{materialized_source_count}개 평가, {len(source_paths)}개 사용"
            )
        else:
            reporter.log(
                "GROUP 상대 품질 게이트 미적용: "
                f"{group_relative_quality_gate.get('status')}"
            )
    manifest["summary"]["status"] = "complete"
    manifest["summary"]["downloaded_source_count"] = len(source_paths)
    manifest["summary"]["materialized_source_count"] = (
        materialized_source_count
    )
    manifest["summary"]["eligible_source_count"] = len(source_paths)
    manifest["summary"]["group_quality_excluded_source_count"] = (
        materialized_source_count - len(source_paths)
    )
    manifest["summary"]["source_count_fallback_applied"] = (
        len(source_paths) < request.desired_source_count
    )
    if cache_refresh_upgrade is None:
        source_pool_refresh_required = (
            _minimum_acceptable_source_count(request)
            < request.desired_source_count
            and materialized_source_count < request.desired_source_count
        )
        catalog_retry_required = not (
            _manifest_has_current_hash_valid_catalog_provenance(manifest)
        )
        refresh_on_next_run = (
            source_pool_refresh_required or catalog_retry_required
        )
        manifest["cache_refresh_upgrade"] = {
            "policy_version": _LATEST_SONG_ACQUISITION_POLICY_VERSION,
            "decision": (
                "refresh_metadata_on_next_run_for_official_catalog"
                if catalog_retry_required
                else "refresh_metadata_on_next_run_for_verified_fallback_pool"
                if source_pool_refresh_required
                else "no_refresh_needed_target_materialized"
            ),
            "refresh_required": refresh_on_next_run,
            "catalog_retry_required": catalog_retry_required,
            "catalog_refresh_result": catalog_provenance["result"],
            "metadata_refresh_attempted": False,
            "authenticated_cached_source_count": 0,
            "prior_materialized_source_count": materialized_source_count,
            "target_source_count": request.desired_source_count,
            "minimum_acceptable_source_count": (
                _minimum_acceptable_source_count(request)
            ),
            "result": (
                "initial_catalog_retry_required"
                if catalog_retry_required
                else "initial_verified_fallback"
                if source_pool_refresh_required
                else "initial_target_materialized"
            ),
            "new_download_count": materialized_source_count,
        }
    else:
        # The relative quality gate changes an excluded record's status to
        # ``downloaded_quality_excluded`` regardless of whether this run
        # downloaded or authenticated it. Preserve actual per-run provenance
        # so a retained ninth binding is neither counted nor treated as a new
        # download during a 9 -> 10 refresh.
        new_download_count = len(new_download_ids)
        cached_reuse_count = len(authenticated_cache_reuse_ids)
        prior_cluster = cache_refresh_upgrade.get(
            "prior_selected_promotion_cluster"
        )
        current_cluster = promotion_cluster_selection.get("selected_cluster")
        prior_start = (
            prior_cluster.get("start_date")
            if isinstance(prior_cluster, dict)
            else None
        )
        prior_end = (
            prior_cluster.get("end_date")
            if isinstance(prior_cluster, dict)
            else None
        )
        current_start = (
            current_cluster.get("start_date")
            if isinstance(current_cluster, dict)
            else None
        )
        current_end = (
            current_cluster.get("end_date")
            if isinstance(current_cluster, dict)
            else None
        )
        switched_cluster = (
            all(
                isinstance(value, str)
                for value in (
                    prior_start,
                    prior_end,
                    current_start,
                    current_end,
                )
            )
            and (current_start > prior_end or prior_start > current_end)
        )
        prior_latest_policy_current = bool(
            cache_refresh_upgrade.get("latest_song_policy_current")
        )
        if switched_cluster:
            refresh_result = "switched_to_newer_qualifying_promotion_cluster"
        elif not prior_latest_policy_current:
            refresh_result = (
                "latest_song_policy_applied_to_target_pool"
                if materialized_source_count >= request.desired_source_count
                else "latest_song_policy_applied_to_verified_fallback"
            )
        elif materialized_source_count >= request.desired_source_count:
            refresh_result = "upgraded_to_target"
        else:
            refresh_result = "retained_verified_fallback_after_refresh"
        cache_refresh_upgrade.update(
            {
                "result": refresh_result,
                "new_download_count": new_download_count,
                "authenticated_cache_reuse_count": cached_reuse_count,
                "result_materialized_source_count": materialized_source_count,
                "result_eligible_source_count": len(source_paths),
                "selected_promotion_cluster": current_cluster,
                "catalog_refresh_result": catalog_provenance["result"],
                "catalog_retry_required": not (
                    _manifest_has_current_hash_valid_catalog_provenance(
                        manifest
                    )
                ),
            }
        )
    _write_manifest(manifest_path, manifest)
    artifact = _build_acquisition_artifact(
        source_paths=tuple(source_paths),
        manifest_path=manifest_path,
        manifest=manifest,
        fresh_metadata_identities_by_video_id=(
            fresh_metadata_identities_by_video_id
        ),
    )
    if (
        request.edit_mode is EditMode.MEMBER
        and artifact.member_identity_authority is None
    ):
        reporter.emit(
            "log",
            level="warning",
            message=(
                "이번 수집에서 멤버 직캠 방송일 근거를 권한으로 확정하지 "
                "못해 경계 오디오 소스의 다중 직캠 복구는 비활성화됩니다. "
                "다음 실행에서 공식 메타데이터 재인증을 시도합니다."
            ),
        )
    return artifact


__all__ = [
    "AcquisitionArtifact",
    "AcquisitionCandidate",
    "AcquisitionError",
    "VerifiedAcquisitionSourceIdentity",
    "VerifiedMemberAcquisitionIdentityAuthority",
    "YTDLPClient",
    "acquire_sources",
    "canonical_music_show_program",
    "is_verified_member_acquisition_identity_authority",
]
