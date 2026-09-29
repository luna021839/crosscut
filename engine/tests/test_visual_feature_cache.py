from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.direct_member_visual import _VideoDescriptorProvider
from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import (
    ShotType,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
)
from crosscut.visual_feature_cache import (
    AnalyzerFingerprint,
    CanonicalFrameIdentity,
    DescriptorRequestIdentity,
    DescriptorRequestLocator,
    PersistentVisualFeatureCache,
    PersistentVisualFeatureCacheStorageUnavailableError,
    PersistentVisualFeatureCacheWriteError,
    PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
    _DecodeEquivalentAnalyzerMigration,
    _decode_equivalent_analyzer_migration_closure,
    build_analyzer_fingerprint,
    python_symbol_dependency_sha256,
)
from crosscut.visual_score import (
    FrameVisualDescriptor,
    _VideoSampler,
    build_visual_timeline,
)


_AUDITED_LEGACY_ANALYZER_DIGEST = (
    "648cfb9801b2de54d4325a7c838e31bbadfbeaa8d71a68c37163d8c41f27b19c"
)
_AUDITED_PREVIOUS_ANALYZER_DIGEST = (
    "14fff1f9ee74dbaed4023c6fbbbd4cd94b67e078a94f3d272885669cfe0e6b30"
)
_AUDITED_RUNTIME_PREVIOUS_ANALYZER_DIGEST = (
    "753e2dbcee35d3bf8c72ed73005341b8b4ec3c142860e869904bc8cf5acac927"
)
_AUDITED_STRICT_CACHE_PREVIOUS_ANALYZER_DIGEST = (
    "eb2ead7937ad96cd7225e536b1527075258de9014be52c82ba8355dfa9e81b33"
)
_AUDITED_SFACE_PREVIOUS_ANALYZER_DIGEST = (
    "0e2da6dae7879dea7d529dc36b87f0f7d87184a07c3e06f4bec15b752dc5ca41"
)
_AUDITED_CURRENT_ANALYZER_DIGEST = (
    "072871c33ea2610cd30f4210b7cc80a35e418ca7c563a4c10ce39ca2e5b52173"
)
_AUDITED_ACTIVE_ANALYZER_DIGEST = (
    "14664ec334ef15c192a496aae269c927d0e11289f882e94217e0632a31a0afde"
)
_AUDITED_LEGACY_DESCRIPTOR_DIGEST = (
    "245edba6559b94d4855f676c3edc957994137fc42d2d2da0e3e0eb9f752a61c0"
)
_AUDITED_PREVIOUS_DESCRIPTOR_DIGEST = (
    "c0de5b604e22d12c713b02edfe2b5a0de8d19a80475b1547b861f975d8978d89"
)
_AUDITED_RUNTIME_PREVIOUS_DESCRIPTOR_DIGEST = (
    "909c66c8dbd903ca6ad1b1cc75c63710838fba88179b2b308f7c964d55f82f84"
)
_AUDITED_STRICT_CACHE_PREVIOUS_DESCRIPTOR_DIGEST = (
    "d62354e350187d5e090d103858d15b3fcc95191703ab048178ca428ca7a6d186"
)
_AUDITED_SFACE_PREVIOUS_DESCRIPTOR_DIGEST = (
    "a2586f534c0ed3a8d7a018cec2a9cf0237598b9b4e51a3c0b84404d6cac0cdec"
)
_AUDITED_CURRENT_DESCRIPTOR_DIGEST = (
    "512bf2d767f5ca035f986f0e4e7b4693bb93e5aff828308537e0d8d795204c1e"
)
_AUDITED_ACTIVE_DESCRIPTOR_DIGEST = (
    "3cdd076cf6c0f001f6d6de4d7228d85d259248e1bddb687c7173690f5bab7d39"
)


def _audited_analyzer_components(descriptor_digest: str) -> dict[str, object]:
    return {
        "cache_contract": "ive_source_local_visual_features_v1",
        "feature_schema_version": 1,
        "descriptor_schema": (
            "frame_visual_descriptor_v1_numeric_geometry_pose_silhouette_"
            "nanodet_formation_motion_no_sface_no_raw_frame"
        ),
        "code_sha256": {
            "pose.py": (
                "2d0fbf3e426039c281a5331e93e567a652f3ee50d5be3744446e07038e741870"
            ),
            "visual_score_descriptor_ast_v1": descriptor_digest,
        },
        "model_asset_sha256_or_state": {
            "face:face_detection_yunet_2023mar.onnx": (
                "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4:"
                "loadable=True"
            ),
            "face:haarcascade_frontalface_default.xml": (
                "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
            ),
            "pose:human_segmentation_pphumanseg_2023mar.onnx": (
                "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24:"
                "loadable=True"
            ),
            "pose:object_detection_nanodet_2022nov.onnx": (
                "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186:"
                "loadable=True"
            ),
            "pose:person_detection_mediapipe_2023mar.onnx": (
                "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f:"
                "loadable=True"
            ),
            "pose:pose_estimation_mediapipe_2023mar.onnx": (
                "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f:"
                "loadable=True"
            ),
        },
        "runtime": {
            "numpy": "2.2.6",
            "opencv": "4.10.0",
            "opencv_build_sha256": (
                "1b2c40cbb30ac378cdcbcd4aeeb68ddbe8727bee4a5ea1a784103455d79d0133"
            ),
            "python_implementation": "CPython",
            "python_version": "3.11.9",
        },
    }


def _audited_active_analyzer_components() -> dict[str, object]:
    components = _audited_analyzer_components(
        _AUDITED_ACTIVE_DESCRIPTOR_DIGEST
    )
    components["descriptor_schema"] = (
        "frame_visual_descriptor_v2_numeric_geometry_pose_silhouette_"
        "nanodet_formation_motion_no_sface_no_raw_frame"
    )
    return components


def _analyzer_from_components(components: dict[str, object]) -> AnalyzerFingerprint:
    digest = hashlib.sha256(
        json.dumps(
            components,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return AnalyzerFingerprint(digest=digest, components=components)


def _analyzer(salt: str = "a") -> AnalyzerFingerprint:
    return AnalyzerFingerprint(
        digest=salt * 64,
        components={
            "descriptor_schema": "test_numeric_descriptor_v1",
            "model_asset_sha256_or_state": {"test": "present"},
            "runtime": {"opencv": cv2.__version__},
            "code_sha256": {"test.py": salt * 64},
        },
    )


def _request(
    *,
    frame_index: int = 30,
    frame_digest: str = "1" * 64,
    following_index: int | None = None,
    following_digest: str | None = None,
    subject: bool = True,
    formation: bool = False,
) -> DescriptorRequestIdentity:
    return DescriptorRequestIdentity(
        primary=CanonicalFrameIdentity(frame_index, 30, 1),
        following=(
            CanonicalFrameIdentity(following_index, 30, 1)
            if following_index is not None
            else None
        ),
        primary_working_frame_sha256=frame_digest,
        following_working_frame_sha256=following_digest,
        include_subject_geometry=subject,
        include_formation_pose=formation,
    )


def _descriptor() -> FrameVisualDescriptor:
    formation = FormationPersonDescriptor(
        bbox_xywh=(0.40, 0.15, 0.20, 0.70),
        centroid_xy=(0.50, 0.50),
        detection_confidence=0.91,
        upper_body_keypoints_xyc=(
            (-0.2, -0.3, 0.9),
            (0.2, -0.3, 0.9),
            (-0.3, -0.1, 0.9),
            (0.3, -0.1, 0.9),
            (-0.4, 0.1, 0.9),
            (0.4, 0.1, 0.9),
            (-0.1, 0.4, 0.9),
            (0.1, 0.4, 0.9),
        ),
    )
    return FrameVisualDescriptor(
        quality_score=0.81,
        composition_score=0.72,
        shot_scale=0.44,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        structure_embedding=(0.1,) * 96,
        motion=(0.02, -0.03),
        primary_face_bbox=(0.43, 0.20, 0.14, 0.20),
        color_embedding=(0.2,) * 8,
        face_detector_backend="yunet",
        primary_face_landmarks_xy=(
            (0.46, 0.26),
            (0.54, 0.26),
            (0.50, 0.30),
            (0.47, 0.35),
            (0.53, 0.35),
        ),
        primary_face_head_roll_radians=0.01,
        primary_face_eye_scale=0.08,
        primary_face_geometry=(
            0.46,
            0.26,
            0.54,
            0.26,
            0.08,
            0.01,
            0.0,
            0.3,
            0.7,
            0.0,
        ),
        pose_embedding=(0.1,) * 60,
        pose_confidence=0.88,
        detected_person_count=1,
        person_layout_embedding=(0.2,) * 134,
        formation_people=(formation,),
        pose_backend="opencv_zoo_mediapipe_pose_2023mar",
        silhouette_embedding=(0.2,) * 121,
        silhouette_bbox_xywh=(0.35, 0.10, 0.30, 0.82),
        silhouette_centroid_xy=(0.50, 0.51),
        silhouette_area_ratio=0.22,
        silhouette_confidence=0.89,
        silhouette_backend="opencv_zoo_pphumanseg_2023mar",
        foreground_obstruction_score=0.0,
        highlight_clipping_ratio=0.0,
    )


def _group_descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.75,
        shot_scale=0.18,
        shot_type=ShotType.GROUP_WIDE,
        face_count=0,
        structure_embedding=(0.1,) * 96,
        motion=(0.0, 0.0),
        color_embedding=(0.2,) * 8,
        face_detector_backend="none",
        pose_embedding=(0.1,) * 96,
        pose_confidence=0.0,
        detected_person_count=0,
        person_layout_embedding=(),
        pose_backend="structure_hog_fallback",
        silhouette_embedding=(),
        silhouette_area_ratio=0.0,
        silhouette_confidence=0.0,
        silhouette_backend="none",
    )


class _FakeCapture:
    def __init__(self, _path: str, *, frame_rate: float = 30.0) -> None:
        self.frame_rate = frame_rate
        self.position = 0
        self.opened = True
        self.read_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.frame_rate
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        elif property_id == cv2.CAP_PROP_POS_MSEC:
            self.position = int(round(value / 1000.0 * self.frame_rate))
        return True

    def read(self):  # type: ignore[no-untyped-def]
        self.read_count += 1
        value = self.position % 251
        return (True, np.full((12, 20, 3), value, dtype=np.uint8))

    def release(self) -> None:
        self.opened = False


class VisualFeatureCacheTests(unittest.TestCase):
    def test_symbol_dependency_fingerprint_uses_versioned_alias_contract(
        self,
    ) -> None:
        self.assertEqual(
            PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
            "python_module_symbol_dependency_ast_v2",
        )

    def test_canonical_frame_identity_removes_rounded_millisecond_collision(
        self,
    ) -> None:
        first = CanonicalFrameIdentity.from_seconds(1.0001, frame_rate=2000.0)
        second = CanonicalFrameIdentity.from_seconds(1.0004, frame_rate=2000.0)
        self.assertEqual(round(1.0001 * 1000), round(1.0004 * 1000))
        self.assertNotEqual(first, second)
        self.assertEqual((first.frame_index, second.frame_index), (2000, 2001))

    def test_analyzer_fingerprint_changes_with_runtime_and_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            code = root / "analyzer.py"
            code.write_text("version = 1\n", encoding="utf-8")
            first = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "1"},
            )
            second = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "2"},
            )
            code.write_text("version = 2\n", encoding="utf-8")
            third = build_analyzer_fingerprint(
                descriptor_schema="schema-v1",
                code_paths=(code,),
                model_assets={"pose": "a" * 64},
                runtime_components={"opencv": "1"},
            )
            self.assertEqual(len({first.digest, second.digest, third.digest}), 3)

    def test_symbol_dependency_fingerprint_ignores_unrelated_selector_code(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module = Path(temporary_directory) / "combined.py"
            module.write_text(
                "CONST = 2\n"
                "def helper(value):\n    return value * CONST\n"
                "def descriptor(value):\n    return helper(value)\n"
                "def selector(value):\n    return value + 1\n",
                encoding="utf-8",
            )
            first = python_symbol_dependency_sha256(
                module,
                root_symbols=("descriptor",),
            )
            module.write_text(
                "CONST = 2\n"
                "def helper(value):\n    return value * CONST\n"
                "def descriptor(value):\n    return helper(value)\n"
                "def selector(value):\n    return value + 999\n",
                encoding="utf-8",
            )
            selector_changed = python_symbol_dependency_sha256(
                module,
                root_symbols=("descriptor",),
            )
            module.write_text(
                "CONST = 3\n"
                "def helper(value):\n    return value * CONST\n"
                "def descriptor(value):\n    return helper(value)\n"
                "def selector(value):\n    return value + 999\n",
                encoding="utf-8",
            )
            descriptor_changed = python_symbol_dependency_sha256(
                module,
                root_symbols=("descriptor",),
            )

            self.assertEqual(first, selector_changed)
            self.assertNotEqual(first, descriptor_changed)

    def test_symbol_dependency_fingerprint_ignores_unreferenced_import_alias(
        self,
    ) -> None:
        variants = (
            (
                "from .dependency import descriptor_helper as helper\n",
                "from .dependency import (\n"
                "    descriptor_helper as helper,\n"
                "    selector_helper as selector,\n"
                ")\n",
            ),
            (
                "import descriptor_dependency as helper\n",
                "import descriptor_dependency as helper, selector_dependency as selector\n",
            ),
        )
        for initial_import, import_with_selector in variants:
            with self.subTest(initial_import=initial_import):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    module = Path(temporary_directory) / "combined.py"
                    module.write_text(
                        initial_import
                        + "def descriptor(value):\n    return helper(value)\n",
                        encoding="utf-8",
                    )
                    before = python_symbol_dependency_sha256(
                        module,
                        root_symbols=("descriptor",),
                    )
                    module.write_text(
                        import_with_selector
                        + "def descriptor(value):\n    return helper(value)\n",
                        encoding="utf-8",
                    )
                    after = python_symbol_dependency_sha256(
                        module,
                        root_symbols=("descriptor",),
                    )
                    self.assertEqual(before, after)

    def test_symbol_dependency_fingerprint_binds_referenced_import_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module = Path(temporary_directory) / "combined.py"

            def fingerprint(import_statement: str, loaded_name: str) -> str:
                module.write_text(
                    import_statement
                    + f"def descriptor(value):\n    return {loaded_name}(value)\n",
                    encoding="utf-8",
                )
                return python_symbol_dependency_sha256(
                    module,
                    root_symbols=("descriptor",),
                )

            baseline = fingerprint(
                "from .dependency import descriptor_helper as helper\n",
                "helper",
            )
            changed_module = fingerprint(
                "from .other_dependency import descriptor_helper as helper\n",
                "helper",
            )
            changed_name = fingerprint(
                "from .dependency import alternate_helper as helper\n",
                "helper",
            )
            changed_asname = fingerprint(
                "from .dependency import descriptor_helper\n",
                "descriptor_helper",
            )
            explicit_same_asname = fingerprint(
                "from .dependency import descriptor_helper as descriptor_helper\n",
                "descriptor_helper",
            )
            changed_level = fingerprint(
                "from ..dependency import descriptor_helper as helper\n",
                "helper",
            )
            changed_import_name = fingerprint(
                "import alternate_dependency as helper\n",
                "helper",
            )

            self.assertEqual(
                len(
                    {
                        baseline,
                        changed_module,
                        changed_name,
                        changed_asname,
                        changed_import_name,
                    }
                ),
                5,
            )
            self.assertNotEqual(changed_asname, explicit_same_asname)
            self.assertNotEqual(baseline, changed_level)

    def test_current_descriptor_dependency_digest_matches_active_schema(
        self,
    ) -> None:
        visual_score_path = (
            Path(__file__).resolve().parents[1] / "crosscut" / "visual_score.py"
        )
        descriptor_ast = python_symbol_dependency_sha256(
            visual_score_path,
            root_symbols=(
                "FrameVisualDescriptor",
                "_VideoSampler",
                "_descriptor_cache_payload",
                "_descriptor_from_cache_payload",
                "describe_frame",
            ),
        )
        self.assertEqual(
            descriptor_ast,
            "155011228942eed6ebf67363e423d55f46f499ce72335a81c1039ea3f44028a8",
        )
        # Reader behavior is part of the audited endpoint, even if the import
        # alias and sampler AST remain unchanged. Older constants above retain
        # their meaning for explicit predecessor/schema rejection tests.
        dependency = visual_score_path.with_name("seek_locality.py")
        combined = hashlib.sha256(json.dumps({
            "contract": "visual-score-with-seek-locality-v1",
            "descriptor_ast_sha256": descriptor_ast,
            "seek_locality_sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
            "shared_source_frames_sha256": hashlib.sha256(
                visual_score_path.with_name("shared_source_frames.py").read_bytes()).hexdigest(),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(
            combined,
            "a9f194987374a325c822af71736eaacc9bd5d72060d5866c1a4e3c6caacfa1ea",
        )

    def test_measured_face_scale_numeric_change_has_no_equivalent_migration(self):
        # This revision changes actual descriptor values. Older decoder-only
        # equivalence edges must not relabel those old measurements as current.
        from crosscut.visual_score import _visual_analyzer_fingerprint
        previous = "5a8c720efe6d42869df46685bb2aec9650308a1c42b455e0c1101349e072b048"
        self.assertEqual(_decode_equivalent_analyzer_migration_closure(previous), ())
        # A later cache-capacity-only change may reuse this corrected numeric
        # generation, but must still not cross the earlier face-scale change.
        closure = _decode_equivalent_analyzer_migration_closure(_visual_analyzer_fingerprint().digest)
        self.assertEqual(tuple(item.legacy_analyzer_digest for item in closure), (
            "fd673124f0f2cde3116f368e0dbdf854edc140d9df967debb9859fb1b7b0bc78",
            "2b3a32dfd5b5edafd38cccb76a009f99d0a1c3f1b72df7f682c70fd79aac0762",
            "b4d75c9d934f0791937f74eed357e8bfc3862f521733b9e9e42bfd4515bb496a",
            "ed00a2703e2cf1459adf8fa983e195288ca6d6b309aad0aa5159da1d1eb64e10",
            "9e5522429b560ee3494f932dc0fa24ec9381f54349e9c6ddd37771fec501589b", previous))

    def test_shared_source_readers_migrate_exact_full_and_content_records_only(self):
        from dataclasses import replace
        old_components = _audited_active_analyzer_components()
        old_components['code_sha256'].update({
            'visual_score_descriptor_ast_v1': '022e714b516d1b08226a11219e8001349995fe4d753e3e74ecc4288b5f846cc6',
            'pose.py': 'cab4b390073348daf33181a2fa0c8a171058e33f8c21223d501370ee23f17b21',
        })
        new_components = json.loads(json.dumps(old_components))
        new_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            'ccc993e4a2ae1da9d62a34cc2b56e36c322779f34e931b1694e6339547968192')
        old, current = map(_analyzer_from_components, (old_components, new_components))
        self.assertEqual(old.digest, 'b4d75c9d934f0791937f74eed357e8bfc3862f521733b9e9e42bfd4515bb496a')
        self.assertEqual(current.digest, '2b3a32dfd5b5edafd38cccb76a009f99d0a1c3f1b72df7f682c70fd79aac0762')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.write_bytes(b'original')
            previous = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=old)
            full = _request()
            content = replace(full, include_pose=False)
            self.assertTrue(previous.put(full, {'measurement': .123, 'mode': 'full'}))
            self.assertTrue(previous.put(content, {'measurement': .123, 'mode': 'content'}))
            changed = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertEqual(changed.get_by_locator(full.locator), {'measurement': .123, 'mode': 'full'})
            self.assertEqual(changed.get_by_locator(content.locator), {'measurement': .123, 'mode': 'content'})
            self.assertEqual(changed.diagnostics()['decode_free_legacy_hit_count'], 2)
            for part, name, value in (('code_sha256', 'visual_score_descriptor_ast_v1', '0'*64),
                    ('runtime', 'opencv', 'different'), ('model_asset_sha256_or_state', 'new', 'unverified')):
                bad_components = json.loads(json.dumps(new_components))
                bad_components[part][name] = value
                bad = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source,
                    analyzer=_analyzer_from_components(bad_components))
                self.assertIsNone(bad.get_by_locator(full.locator))
            source.write_bytes(b'replaced')
            replaced = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertIsNone(replaced.get_by_locator(full.locator))

    def test_short_gap_rgb_optimization_reuses_exact_prior_features_only(self):
        from dataclasses import replace
        from crosscut.visual_score import _visual_analyzer_fingerprint
        current = _visual_analyzer_fingerprint()
        components = json.loads(json.dumps(current.components))
        components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            'ccc993e4a2ae1da9d62a34cc2b56e36c322779f34e931b1694e6339547968192')
        previous = _analyzer_from_components(components)
        self.assertEqual(previous.digest, '2b3a32dfd5b5edafd38cccb76a009f99d0a1c3f1b72df7f682c70fd79aac0762')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.write_bytes(b'fixed source')
            old = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=previous)
            requests = (_request(), replace(_request(), include_pose=False))
            for request in requests:
                self.assertTrue(old.put(request, {'numeric': .123, 'pose': request.include_pose}))
            new = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            for request in requests:
                self.assertEqual(new.get_by_locator(request.locator), {'numeric': .123, 'pose': request.include_pose})
            self.assertEqual(new.diagnostics()['decode_free_legacy_hit_count'], 2)
            for part, name in (('code_sha256', 'visual_score_descriptor_ast_v1'),
                    ('code_sha256', 'pose.py'), ('runtime', 'numpy'),
                    ('model_asset_sha256_or_state', 'face:face_detection_yunet_2023mar.onnx')):
                invalid = json.loads(json.dumps(current.components))
                invalid[part][name] = '0'*64
                cache = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source,
                    analyzer=_analyzer_from_components(invalid))
                self.assertIsNone(cache.get_by_locator(_request().locator))

    def test_content_feature_split_reuses_unchanged_full_mode_only(self):
        from dataclasses import replace
        old_components = _audited_active_analyzer_components()
        old_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "4af3b6fe5a7c63ce3a752566a8c32747f8181f05c2bc80f2640d48cc5380d084")
        new_components = json.loads(json.dumps(old_components))
        new_components['code_sha256'].update({
            'visual_score_descriptor_ast_v1': '022e714b516d1b08226a11219e8001349995fe4d753e3e74ecc4288b5f846cc6',
            'pose.py': 'cab4b390073348daf33181a2fa0c8a171058e33f8c21223d501370ee23f17b21',
        })
        old, current = map(_analyzer_from_components, (old_components, new_components))
        self.assertEqual(old.digest, 'ed00a2703e2cf1459adf8fa983e195288ca6d6b309aad0aa5159da1d1eb64e10')
        self.assertEqual(current.digest, 'b4d75c9d934f0791937f74eed357e8bfc3862f521733b9e9e42bfd4515bb496a')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.write_bytes(b'original')
            previous = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=old)
            request = _request()
            self.assertTrue(previous.put(request, {'measurement': .123}))
            changed = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertEqual(changed.get_by_locator(request.locator), {'measurement': .123})
            self.assertEqual(changed.diagnostics()['decode_free_legacy_hit_count'], 1)
            self.assertIsNone(changed.get_by_locator(replace(request.locator, include_pose=False)))
            for part, name, value in (('code_sha256', 'pose.py', '0'*64),
                    ('runtime', 'numpy', 'different'), ('model_asset_sha256_or_state', 'new', 'unverified')):
                bad_components = json.loads(json.dumps(new_components))
                bad_components[part][name] = value
                bad = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source,
                    analyzer=_analyzer_from_components(bad_components))
                self.assertIsNone(bad.get_by_locator(request.locator))
            source.write_bytes(b'replaced')
            replaced = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertIsNone(replaced.get_by_locator(request.locator))

    def test_sparse_materialization_reuses_only_exact_numeric_predecessor(self):
        old_components = _audited_active_analyzer_components()
        new_components = _audited_active_analyzer_components()
        old_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "619a7e5f5ca525a2f95833e8bf648af3a9012474dc44cef34d502fd5af532b82")
        new_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "4af3b6fe5a7c63ce3a752566a8c32747f8181f05c2bc80f2640d48cc5380d084")
        old, current = map(_analyzer_from_components, (old_components, new_components))
        self.assertEqual(current.digest, "ed00a2703e2cf1459adf8fa983e195288ca6d6b309aad0aa5159da1d1eb64e10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.mp4'; source.write_bytes(b'same-source')
            request = _request()
            previous = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=old)
            self.assertTrue(previous.put(request, {'value':.123}))
            changed = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertEqual(changed.get_by_locator(request.locator), {'value':.123})
            self.assertEqual(changed.diagnostics()['decode_free_legacy_hit_count'], 1)
            new_components['runtime']['numpy'] = 'different-runtime'
            incompatible = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source,
                analyzer=_analyzer_from_components(new_components))
            self.assertIsNone(incompatible.get_by_locator(request.locator))
            source.write_bytes(b'changed-source')
            replaced = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertIsNone(replaced.get_by_locator(request.locator))

    def test_larger_working_pixel_lru_reuses_only_exact_corrected_generation(self):
        old_components = _audited_active_analyzer_components()
        new_components = _audited_active_analyzer_components()
        old_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "d19d38dd50805b619e036a4faeeeb44e0190c2543ed09565c79bb684913011a3")
        new_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "619a7e5f5ca525a2f95833e8bf648af3a9012474dc44cef34d502fd5af532b82")
        previous, current = map(_analyzer_from_components, (old_components,new_components))
        self.assertEqual(previous.digest, "5a8c720efe6d42869df46685bb2aec9650308a1c42b455e0c1101349e072b048")
        self.assertEqual(current.digest, "9e5522429b560ee3494f932dc0fa24ec9381f54349e9c6ddd37771fec501589b")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.mp4'; source.write_bytes(b'unchanged-source')
            request = _request()
            old = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=previous)
            self.assertTrue(old.put(request, {'value':.123}))
            new = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertEqual(new.get_by_locator(request.locator), {'value':.123})
            self.assertEqual(new.diagnostics()['decode_free_legacy_hit_count'], 1)

    def test_h264_locality_exact_forward_migration_preserves_records(self) -> None:
        previous_components = _audited_active_analyzer_components()
        current_components = _audited_active_analyzer_components()
        previous_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "9f3590797903f8bbb52fd57d992b7d7f93b5992081acb4b2c659c3948dee8834"
        )
        current_components['code_sha256']['visual_score_descriptor_ast_v1'] = (
            "14ec585b0d692c05498594a529632dca6c89be35fcba77ac9b67999c64da1d5f"
        )
        previous = _analyzer_from_components(previous_components)
        current = _analyzer_from_components(current_components)
        self.assertEqual(previous.digest, "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509")
        self.assertEqual(current.digest, "0d665ac9a9b9ef183e0c666196d63cfd84c77f35fbe3d4c59fac1350c4fa1e8d")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"same-source")
            request = _request()
            old = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=previous)
            self.assertTrue(old.put(request, {"value": 1.0}))
            new = PersistentVisualFeatureCache(source_cache_directory=root/'cache', source_path=source, analyzer=current)
            self.assertEqual(new.get_by_locator(request.locator), {"value": 1.0})
            self.assertEqual(new.diagnostics()["decode_free_hit_count"], 1)

    def test_seek_locality_endpoint_preserves_audited_predecessors(self) -> None:
        current = "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509"
        closure = _decode_equivalent_analyzer_migration_closure(current)
        self.assertEqual(tuple(item.legacy_analyzer_digest for item in closure), (
            _AUDITED_ACTIVE_ANALYZER_DIGEST,
            "0bb5d3e1ae287ec7b59697727580b219dae5b857f6a0a25d8a55db5d59ec9921",
        ))
        self.assertEqual(closure[0].edges[0][1].legacy_descriptor_digest,
                         _AUDITED_ACTIVE_DESCRIPTOR_DIGEST)
        self.assertEqual(_decode_equivalent_analyzer_migration_closure("f" * 64), ())

    def test_active_schema_does_not_reuse_incomplete_v1_descriptor(self) -> None:
        previous = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        active = _analyzer_from_components(_audited_active_analyzer_components())
        self.assertEqual(previous.digest, _AUDITED_CURRENT_ANALYZER_DIGEST)
        self.assertEqual(active.digest, _AUDITED_ACTIVE_ANALYZER_DIGEST)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=previous,
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=active,
            )

            self.assertIsNone(warm.get_by_locator(request.locator))
            self.assertEqual(warm.diagnostics()["decode_free_legacy_hit_count"], 0)

    def test_content_reuse_migration_keeps_exact_previous_v2_records(self) -> None:
        components = _audited_active_analyzer_components()
        components["code_sha256"]["visual_score_descriptor_ast_v1"] = (
            "fe78347f24e895c2cc0ce57e7490706c46b9dd9ae08a3c87871a120838121ebd"
        )
        previous = _analyzer_from_components(components)
        active = _analyzer_from_components(_audited_active_analyzer_components())
        self.assertEqual(previous.digest, "0bb5d3e1ae287ec7b59697727580b219dae5b857f6a0a25d8a55db5d59ec9921")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v2")
            request = _request()
            cold = PersistentVisualFeatureCache(source_cache_directory=root / "cache", source_path=source, analyzer=previous)
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(source_cache_directory=root / "cache", source_path=source, analyzer=active)
            self.assertEqual(warm.get_by_locator(request.locator), {"value": 1.0})
            self.assertEqual(warm.diagnostics()["decode_free_legacy_hit_count"], 1)

    def test_content_hit_persists_each_exact_locator_for_decode_free_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content")
            with patch("crosscut.visual_score.cv2.VideoCapture", side_effect=lambda path: _FakeCapture(path)), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint", return_value=_analyzer()
            ), patch("crosscut.visual_score.describe_frame", return_value=_descriptor()) as describe:
                cold = _VideoSampler(source, feature_cache_directory=root / "cache")
                first = cold.feature_descriptor_at(1.0)
                second = cold.feature_descriptor_at(1.001)
                self.assertEqual(cold.describe_frame_call_count, 1)
                self.assertEqual(cold.persistent_feature_cache_diagnostics()["write_count"], 2)
                cold.close()
                warm = _VideoSampler(source, feature_cache_directory=root / "cache")
                self.assertEqual(warm.feature_descriptor_at(1.0), first)
                self.assertEqual(warm.feature_descriptor_at(1.001), second)
                self.assertEqual(warm.frame_decode_count, 0)
                warm.close()
            self.assertEqual(describe.call_count, 1)

    def test_symbol_dependency_fingerprint_binds_helper_body_and_constant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module = Path(temporary_directory) / "combined.py"

            def fingerprint(constant: int, operator: str) -> str:
                module.write_text(
                    "CONST = "
                    f"{constant}\n"
                    "def helper(value):\n"
                    f"    return value {operator} CONST\n"
                    "def descriptor(value):\n"
                    "    return helper(value)\n",
                    encoding="utf-8",
                )
                return python_symbol_dependency_sha256(
                    module,
                    root_symbols=("descriptor",),
                )

            baseline = fingerprint(2, "*")
            changed_body = fingerprint(2, "+")
            changed_constant = fingerprint(3, "*")
            self.assertEqual(
                len({baseline, changed_body, changed_constant}),
                3,
            )

    def test_atomic_round_trip_binds_source_frame_options_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            request = _request(
                following_index=31,
                following_digest="2" * 64,
                formation=True,
            )
            payload = {
                "quality": 0.8,
                "vector": [0.1, 0.2, 0.3],
                "backend": "test",
            }
            self.assertTrue(cache.put(request, payload))
            self.assertEqual(cache.get(request), payload)
            stale_analyzer = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer("b"),
            )
            self.assertIsNone(stale_analyzer.get(request))
            self.assertIsNone(
                cache.get(
                    _request(
                        following_index=31,
                        following_digest="3" * 64,
                        formation=True,
                    )
                )
            )
            self.assertIsNone(
                cache.get(
                    _request(
                        following_index=31,
                        following_digest="2" * 64,
                        formation=False,
                    )
                )
            )
            records = tuple((root / "cache" / "derived").rglob("*.json"))
            self.assertEqual(len(records), 1)
            raw_record = records[0].read_text(encoding="utf-8")
            self.assertNotIn(str(source), raw_record)
            self.assertNotIn("source-content-v1", raw_record)
            record = json.loads(raw_record)
            self.assertFalse(record["privacy"]["raw_frames_persisted"])
            self.assertFalse(
                record["privacy"]["face_identity_embeddings_persisted"]
            )

            # A valid JSON edit without a matching checksum is corruption, not
            # evidence that can flow into timeline scoring.
            record["numeric_features"]["quality"] = 0.1
            records[0].write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(cache.get(request))
            self.assertEqual(cache.diagnostics()["invalid_record_count"], 1)

    def test_direct_derived_root_reuses_real_records_decode_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            direct_derived_root = root / "mounted-derived-cache"
            direct_derived_root.mkdir()
            unused_legacy_root = root / "unused-source-cache"
            request = _request(
                following_index=31,
                following_digest="2" * 64,
                formation=True,
            )
            payload = {"quality": 0.8, "vector": [0.1, 0.2, 0.3]}

            cold = PersistentVisualFeatureCache(
                source_cache_directory=unused_legacy_root,
                source_path=source,
                analyzer=_analyzer(),
                derived_cache_directory=direct_derived_root,
            )
            self.assertTrue(cold.put(request, payload))
            self.assertFalse(unused_legacy_root.exists())
            records = tuple(
                (direct_derived_root / "visual-v1").rglob("*.json")
            )
            self.assertEqual(len(records), 1)

            warm = PersistentVisualFeatureCache(
                source_cache_directory=unused_legacy_root,
                source_path=source,
                analyzer=_analyzer(),
                derived_cache_directory=direct_derived_root,
            )
            self.assertEqual(warm.get_by_locator(request.locator), payload)
            diagnostics = warm.diagnostics()
            self.assertEqual(diagnostics["hit_count"], 1)
            self.assertEqual(diagnostics["decode_free_hit_count"], 1)
            self.assertEqual(diagnostics["miss_count"], 0)
            self.assertFalse(unused_legacy_root.exists())

    def test_direct_derived_root_rejects_symlink_missing_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            existing_directory = root / "real-derived-cache"
            existing_directory.mkdir()
            symlink = root / "derived-cache-link"
            symlink.symlink_to(existing_directory, target_is_directory=True)
            existing_file = root / "derived-cache-file"
            existing_file.write_text("not a directory", encoding="utf-8")
            common = {
                "source_cache_directory": root / "unused-source-cache",
                "source_path": source,
                "analyzer": _analyzer(),
            }

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                PersistentVisualFeatureCache(
                    **common,
                    derived_cache_directory=symlink,
                )
            with self.assertRaisesRegex(ValueError, "must exist"):
                PersistentVisualFeatureCache(
                    **common,
                    derived_cache_directory=root / "missing-derived-cache",
                )
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                PersistentVisualFeatureCache(
                    **common,
                    derived_cache_directory=existing_file,
                )

    def test_direct_derived_root_preserves_visual_child_symlink_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            direct_derived_root = root / "mounted-derived-cache"
            direct_derived_root.mkdir()
            redirected_visual_root = root / "redirected-visual-cache"
            redirected_visual_root.mkdir()
            (direct_derived_root / "visual-v1").symlink_to(
                redirected_visual_root,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ValueError, "cache path is a symlink"):
                PersistentVisualFeatureCache(
                    source_cache_directory=root / "unused-source-cache",
                    source_path=source,
                    analyzer=_analyzer(),
                    derived_cache_directory=direct_derived_root,
                )

    def test_runtime_direct_root_loss_raises_distinct_storage_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            direct_derived_root = root / "mounted-derived-cache"
            direct_derived_root.mkdir()
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "unused-source-cache",
                source_path=source,
                analyzer=_analyzer(),
                derived_cache_directory=direct_derived_root,
            )
            detached_root = root / "detached-derived-cache"
            direct_derived_root.rename(detached_root)

            with self.assertRaises(
                PersistentVisualFeatureCacheStorageUnavailableError
            ) as raised:
                cache.get_by_locator(_request().locator)

            self.assertNotIn(str(root), str(raised.exception))
            self.assertEqual(cache.diagnostics()["storage_io_failure_count"], 1)
            self.assertEqual(cache.diagnostics()["atomic_write_failure_count"], 0)

    def test_atomic_record_write_failure_raises_distinct_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            direct_derived_root = root / "mounted-derived-cache"
            direct_derived_root.mkdir()
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "unused-source-cache",
                source_path=source,
                analyzer=_analyzer(),
                derived_cache_directory=direct_derived_root,
            )

            with patch(
                "crosscut.visual_feature_cache.os.replace",
                side_effect=OSError("simulated atomic replace failure"),
            ), self.assertRaises(PersistentVisualFeatureCacheWriteError) as raised:
                cache.put(_request(), {"quality": 0.8})

            self.assertNotIn(str(root), str(raised.exception))
            self.assertEqual(cache.diagnostics()["storage_io_failure_count"], 0)
            self.assertEqual(cache.diagnostics()["atomic_write_failure_count"], 1)
            self.assertEqual(cache.diagnostics()["write_count"], 0)

    def test_changed_source_never_reuses_old_content_address(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"AAAA")
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            request = _request()
            self.assertTrue(cache.put(request, {"value": 1.0}))
            original_mtime = source.stat().st_mtime_ns
            source.write_bytes(b"BBBB")
            os.utime(source, ns=(original_mtime, original_mtime))
            self.assertIsNone(cache.get(request))
            self.assertTrue(cache.diagnostics()["disabled_by_source_change"])
            replacement = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            self.assertNotEqual(cache.source_sha256, replacement.source_sha256)
            self.assertIsNone(replacement.get(request))

    def test_sampler_reuses_full_descriptor_without_persisting_sface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"fake-video-source")
            cache_root = root / "cache"
            captures: list[_FakeCapture] = []

            def capture(path: str) -> _FakeCapture:
                value = _FakeCapture(path)
                captures.append(value)
                return value

            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=capture,
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_descriptor(),
            ) as describe:
                first = _VideoSampler(
                    source,
                    feature_cache_directory=cache_root,
                )
                cold = first.feature_descriptor_at(
                    1.0,
                    following_seconds=1.08,
                    include_formation_pose=True,
                )
                first.close()
                second = _VideoSampler(
                    source,
                    feature_cache_directory=cache_root,
                )
                warm = second.feature_descriptor_at(
                    1.0,
                    following_seconds=1.08,
                    include_formation_pose=True,
                )
                diagnostics = second.feature_cache_diagnostics()
                second.close()
            self.assertEqual(cold, warm)
            self.assertEqual(describe.call_count, 1)
            self.assertEqual(len(captures), 2)
            self.assertGreater(captures[0].read_count, 0)
            self.assertEqual(captures[1].read_count, 0)
            assert diagnostics is not None
            self.assertEqual(diagnostics["hit_count"], 1)
            self.assertEqual(diagnostics["decode_free_hit_count"], 1)
            self.assertEqual(diagnostics["source_frame_decode_count"], 0)
            records = tuple((cache_root / "derived").rglob("*.json"))
            self.assertTrue(records)
            joined = b"".join(path.read_bytes() for path in records)
            self.assertNotIn(b"SFACE_PRIVATE_SENTINEL", joined)
            self.assertNotIn(source.read_bytes(), joined)

    def test_decode_free_locator_fails_closed_on_ambiguous_frame_digests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=_analyzer(),
            )
            first = _request(frame_digest="1" * 64)
            second = _request(frame_digest="2" * 64)
            self.assertTrue(cache.put(first, {"value": 1.0}))
            self.assertTrue(cache.put(second, {"value": 2.0}))
            locator = DescriptorRequestLocator(
                primary=first.primary,
                following=first.following,
                include_subject_geometry=first.include_subject_geometry,
                include_formation_pose=first.include_formation_pose,
            )

            self.assertIsNone(cache.get_by_locator(locator))
            self.assertEqual(cache.get(first), {"value": 1.0})
            self.assertEqual(cache.get(second), {"value": 2.0})
            self.assertEqual(cache.diagnostics()["ambiguous_locator_count"], 1)

    def test_over_budget_locator_preflight_reads_no_json_and_uses_exact_lookup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            cache_root = root / "cache"
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=cache_root,
                source_path=source,
                analyzer=_analyzer(),
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            shard = cold._root / "ff"
            shard.mkdir(exist_ok=True)
            (shard / f"{'f' * 64}.json").write_text("{}", encoding="utf-8")

            warm = PersistentVisualFeatureCache(
                source_cache_directory=cache_root,
                source_path=source,
                analyzer=_analyzer(),
            )
            with patch(
                "crosscut.visual_feature_cache."
                "_MAXIMUM_DECODE_FREE_LOCATOR_RECORDS",
                1,
            ), patch.object(
                warm,
                "_read_record",
                wraps=warm._read_record,
            ) as read_record:
                self.assertIsNone(warm.get_by_locator(request.locator))
                read_record.assert_not_called()
                self.assertEqual(warm.get(request), {"value": 1.0})
                self.assertEqual(read_record.call_count, 1)
                diagnostics = warm.diagnostics()

            self.assertEqual(
                diagnostics["locator_index_status"],
                "over_budget_exact_decode_fallback",
            )
            self.assertEqual(diagnostics["locator_index_record_budget"], 1)
            self.assertEqual(
                diagnostics["locator_index_budget_fallback_count"],
                1,
            )
            self.assertEqual(diagnostics["locator_index_build_count"], 0)

            restarted = PersistentVisualFeatureCache(
                source_cache_directory=cache_root,
                source_path=source,
                analyzer=_analyzer(),
            )
            with patch(
                "crosscut.visual_feature_cache."
                "_MAXIMUM_DECODE_FREE_LOCATOR_RECORDS",
                1,
            ), patch.object(
                restarted,
                "_preflight_locator_record_paths",
                wraps=restarted._preflight_locator_record_paths,
            ) as preflight, patch.object(
                restarted,
                "_read_record",
                wraps=restarted._read_record,
            ) as read_record:
                self.assertIsNone(restarted.get_by_locator(request.locator))
                preflight.assert_not_called()
                read_record.assert_not_called()
                restarted_diagnostics = restarted.diagnostics()
            self.assertEqual(
                restarted_diagnostics["locator_index_status"],
                "over_budget_process_memo_exact_decode_fallback",
            )
            self.assertEqual(
                restarted_diagnostics["locator_index_process_memo_hit_count"],
                1,
            )

    def test_exactly_audited_cache_only_analyzer_migration_reuses_record(
        self,
    ) -> None:
        previous = _analyzer_from_components(
            _audited_analyzer_components(
                _AUDITED_STRICT_CACHE_PREVIOUS_DESCRIPTOR_DIGEST
            )
        )
        current = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        self.assertEqual(
            previous.digest,
            _AUDITED_STRICT_CACHE_PREVIOUS_ANALYZER_DIGEST,
        )
        self.assertEqual(current.digest, _AUDITED_CURRENT_ANALYZER_DIGEST)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=previous,
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=current,
            )

            self.assertEqual(
                warm.get_by_locator(request.locator),
                {"value": 1.0},
            )
            diagnostics = warm.diagnostics()
            self.assertEqual(diagnostics["decode_free_hit_count"], 1)
            self.assertEqual(diagnostics["decode_free_legacy_hit_count"], 1)

    def test_current_analyzer_transitively_reuses_all_audited_generations_in_order(
        self,
    ) -> None:
        generations = {
            "oldest": _analyzer_from_components(
                _audited_analyzer_components(_AUDITED_LEGACY_DESCRIPTOR_DIGEST)
            ),
            "older": _analyzer_from_components(
                _audited_analyzer_components(_AUDITED_PREVIOUS_DESCRIPTOR_DIGEST)
            ),
            "nearest": _analyzer_from_components(
                _audited_analyzer_components(
                    _AUDITED_RUNTIME_PREVIOUS_DESCRIPTOR_DIGEST
                )
            ),
            "strict_previous": _analyzer_from_components(
                _audited_analyzer_components(
                    _AUDITED_STRICT_CACHE_PREVIOUS_DESCRIPTOR_DIGEST
                )
            ),
            "sface_previous": _analyzer_from_components(
                _audited_analyzer_components(
                    _AUDITED_SFACE_PREVIOUS_DESCRIPTOR_DIGEST
                )
            ),
            "current": _analyzer_from_components(
                _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
            ),
        }
        self.assertEqual(
            [generations[name].digest for name in generations],
            [
                _AUDITED_LEGACY_ANALYZER_DIGEST,
                _AUDITED_PREVIOUS_ANALYZER_DIGEST,
                _AUDITED_RUNTIME_PREVIOUS_ANALYZER_DIGEST,
                _AUDITED_STRICT_CACHE_PREVIOUS_ANALYZER_DIGEST,
                _AUDITED_SFACE_PREVIOUS_ANALYZER_DIGEST,
                _AUDITED_CURRENT_ANALYZER_DIGEST,
            ],
        )
        cases = (
            (("oldest",), "oldest"),
            (("oldest", "older"), "older"),
            (("oldest", "older", "nearest"), "nearest"),
            (
                ("oldest", "older", "nearest", "strict_previous"),
                "strict_previous",
            ),
            (
                (
                    "oldest",
                    "older",
                    "nearest",
                    "strict_previous",
                    "sface_previous",
                ),
                "sface_previous",
            ),
            (
                (
                    "oldest",
                    "older",
                    "nearest",
                    "strict_previous",
                    "sface_previous",
                    "current",
                ),
                "current",
            ),
        )
        for present, expected_label in cases:
            with self.subTest(present=present):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source.mp4"
                    source.write_bytes(b"source-content-v1")
                    request = _request()
                    cache_root = root / "cache"
                    for label in present:
                        writer = PersistentVisualFeatureCache(
                            source_cache_directory=cache_root,
                            source_path=source,
                            analyzer=generations[label],
                        )
                        self.assertTrue(writer.put(request, {"generation": label}))
                    warm = PersistentVisualFeatureCache(
                        source_cache_directory=cache_root,
                        source_path=source,
                        analyzer=generations["current"],
                    )

                    self.assertEqual(
                        warm.get_by_locator(request.locator),
                        {"generation": expected_label},
                    )
                    expected_legacy_order = tuple(
                        label
                        for label in (
                            "sface_previous",
                            "strict_previous",
                            "nearest",
                            "older",
                            "oldest",
                        )
                        if label in present
                    )
                    self.assertEqual(
                        tuple(path.name for path in warm._lookup_roots),
                        (
                            _AUDITED_CURRENT_ANALYZER_DIGEST,
                            *(
                                generations[label].digest
                                for label in expected_legacy_order
                            ),
                        ),
                    )
                    self.assertEqual(
                        warm.diagnostics()["decode_free_legacy_hit_count"],
                        int(expected_label != "current"),
                    )

    def test_transitive_legacy_root_ambiguity_still_fails_closed(self) -> None:
        legacy = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_LEGACY_DESCRIPTOR_DIGEST)
        )
        current = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            cache_root = root / "cache"
            first = _request(frame_digest="1" * 64)
            second = _request(frame_digest="2" * 64)
            legacy_cache = PersistentVisualFeatureCache(
                source_cache_directory=cache_root,
                source_path=source,
                analyzer=legacy,
            )
            self.assertTrue(legacy_cache.put(first, {"value": 1.0}))
            self.assertTrue(legacy_cache.put(second, {"value": 2.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=cache_root,
                source_path=source,
                analyzer=current,
            )

            self.assertIsNone(warm.get_by_locator(first.locator))
            self.assertEqual(warm.diagnostics()["ambiguous_locator_count"], 1)

    def test_migration_closure_rejects_cycle_duplicate_and_broken_chain(
        self,
    ) -> None:
        current_digest = "a" * 64
        first_legacy_digest = "b" * 64
        second_legacy_digest = "c" * 64
        shared_legacy_digest = "d" * 64

        def edge(
            legacy_analyzer_digest: str,
            legacy_descriptor_digest: str,
            current_descriptor_digest: str,
        ) -> _DecodeEquivalentAnalyzerMigration:
            return _DecodeEquivalentAnalyzerMigration(
                legacy_analyzer_digest=legacy_analyzer_digest,
                descriptor_component_name="visual_score_descriptor_ast_v1",
                legacy_descriptor_digest=legacy_descriptor_digest,
                current_descriptor_digest=current_descriptor_digest,
            )

        direct = edge(first_legacy_digest, "1" * 64, "2" * 64)
        graphs = (
            (
                "cycle",
                {
                    current_digest: (direct,),
                    first_legacy_digest: (
                        edge(current_digest, "2" * 64, "1" * 64),
                    ),
                },
                "cycle",
            ),
            (
                "duplicate",
                {current_digest: (direct, direct)},
                "duplicate or ambiguous",
            ),
            (
                "diamond",
                {
                    current_digest: (
                        direct,
                        edge(second_legacy_digest, "3" * 64, "2" * 64),
                    ),
                    first_legacy_digest: (
                        edge(shared_legacy_digest, "4" * 64, "1" * 64),
                    ),
                    second_legacy_digest: (
                        edge(shared_legacy_digest, "4" * 64, "3" * 64),
                    ),
                },
                "duplicate or ambiguous",
            ),
            (
                "broken_chain",
                {
                    current_digest: (direct,),
                    first_legacy_digest: (
                        edge(second_legacy_digest, "3" * 64, "4" * 64),
                    ),
                },
                "chain endpoints do not match",
            ),
        )
        for label, graph, error in graphs:
            with self.subTest(graph=label), patch(
                "crosscut.visual_feature_cache."
                "_DECODE_EQUIVALENT_ANALYZER_MIGRATIONS",
                graph,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    _decode_equivalent_analyzer_migration_closure(
                        current_digest
                    )

    def test_transitive_migration_rejects_noncanonical_intermediate_endpoint(
        self,
    ) -> None:
        legacy = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_LEGACY_DESCRIPTOR_DIGEST)
        )
        current = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        forged_intermediate_descriptor = "f" * 64
        graph = {
            _AUDITED_PREVIOUS_ANALYZER_DIGEST: (
                _DecodeEquivalentAnalyzerMigration(
                    legacy_analyzer_digest=_AUDITED_LEGACY_ANALYZER_DIGEST,
                    descriptor_component_name="visual_score_descriptor_ast_v1",
                    legacy_descriptor_digest=_AUDITED_LEGACY_DESCRIPTOR_DIGEST,
                    current_descriptor_digest=_AUDITED_PREVIOUS_DESCRIPTOR_DIGEST,
                ),
            ),
            _AUDITED_RUNTIME_PREVIOUS_ANALYZER_DIGEST: (
                _DecodeEquivalentAnalyzerMigration(
                    legacy_analyzer_digest=_AUDITED_PREVIOUS_ANALYZER_DIGEST,
                    descriptor_component_name="visual_score_descriptor_ast_v1",
                    legacy_descriptor_digest=_AUDITED_PREVIOUS_DESCRIPTOR_DIGEST,
                    current_descriptor_digest=forged_intermediate_descriptor,
                ),
            ),
            _AUDITED_CURRENT_ANALYZER_DIGEST: (
                _DecodeEquivalentAnalyzerMigration(
                    legacy_analyzer_digest=(
                        _AUDITED_RUNTIME_PREVIOUS_ANALYZER_DIGEST
                    ),
                    descriptor_component_name="visual_score_descriptor_ast_v1",
                    legacy_descriptor_digest=forged_intermediate_descriptor,
                    current_descriptor_digest=_AUDITED_CURRENT_DESCRIPTOR_DIGEST,
                ),
            ),
        }
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "crosscut.visual_feature_cache."
            "_DECODE_EQUIVALENT_ANALYZER_MIGRATIONS",
            graph,
        ):
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=legacy,
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=current,
            )

            self.assertIsNone(warm.get_by_locator(request.locator))
            self.assertEqual(warm.diagnostics()["invalid_record_count"], 1)

    def test_previous_generation_retains_audited_legacy_lookup(self) -> None:
        legacy = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_LEGACY_DESCRIPTOR_DIGEST)
        )
        previous = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_PREVIOUS_DESCRIPTOR_DIGEST)
        )
        self.assertEqual(legacy.digest, _AUDITED_LEGACY_ANALYZER_DIGEST)
        self.assertEqual(previous.digest, _AUDITED_PREVIOUS_ANALYZER_DIGEST)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=legacy,
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=previous,
            )

            self.assertEqual(
                warm.get_by_locator(request.locator),
                {"value": 1.0},
            )
            diagnostics = warm.diagnostics()
            self.assertEqual(diagnostics["decode_free_hit_count"], 1)
            self.assertEqual(diagnostics["decode_free_legacy_hit_count"], 1)

    def test_analyzer_migration_rejects_unknown_and_reverse_directions(
        self,
    ) -> None:
        previous = _analyzer_from_components(
            _audited_analyzer_components(
                _AUDITED_RUNTIME_PREVIOUS_DESCRIPTOR_DIGEST
            )
        )
        current = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        unknown = _analyzer_from_components(
            _audited_analyzer_components("f" * 64)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            previous_cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "legacy-cache",
                source_path=source,
                analyzer=previous,
            )
            self.assertTrue(previous_cache.put(request, {"value": 1.0}))
            unknown_cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "legacy-cache",
                source_path=source,
                analyzer=unknown,
            )
            self.assertIsNone(unknown_cache.get_by_locator(request.locator))

            current_cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "reverse-cache",
                source_path=source,
                analyzer=current,
            )
            self.assertTrue(current_cache.put(request, {"value": 2.0}))
            reverse_cache = PersistentVisualFeatureCache(
                source_cache_directory=root / "reverse-cache",
                source_path=source,
                analyzer=previous,
            )
            self.assertIsNone(reverse_cache.get_by_locator(request.locator))

    def test_analyzer_migration_rejects_every_non_descriptor_tamper(
        self,
    ) -> None:
        mutations = {
            "cache_contract": lambda value: value.__setitem__(
                "cache_contract", "tampered-cache-contract"
            ),
            "schema": lambda value: value.__setitem__(
                "descriptor_schema", "tampered-schema"
            ),
            "feature_schema": lambda value: value.__setitem__(
                "feature_schema_version", 2
            ),
            "model": lambda value: value[
                "model_asset_sha256_or_state"
            ].__setitem__(
                "face:haarcascade_frontalface_default.xml", "f" * 64
            ),
            "runtime": lambda value: value["runtime"].__setitem__(
                "opencv", "tampered-runtime"
            ),
            "pose_code": lambda value: value["code_sha256"].__setitem__(
                "pose.py", "f" * 64
            ),
            "descriptor_code": lambda value: value["code_sha256"].__setitem__(
                "visual_score_descriptor_ast_v1", "f" * 64
            ),
        }
        current = _analyzer_from_components(
            _audited_analyzer_components(_AUDITED_CURRENT_DESCRIPTOR_DIGEST)
        )
        for label, mutate in mutations.items():
            with self.subTest(tamper=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "source.mp4"
                    source.write_bytes(b"source-content-v1")
                    request = _request()
                    previous_components = json.loads(
                        json.dumps(
                            _audited_analyzer_components(
                                _AUDITED_RUNTIME_PREVIOUS_DESCRIPTOR_DIGEST
                            )
                        )
                    )
                    mutate(previous_components)
                    tampered_previous = AnalyzerFingerprint(
                        digest=_AUDITED_RUNTIME_PREVIOUS_ANALYZER_DIGEST,
                        components=previous_components,
                    )
                    cold = PersistentVisualFeatureCache(
                        source_cache_directory=root / "cache",
                        source_path=source,
                        analyzer=tampered_previous,
                    )
                    self.assertTrue(cold.put(request, {"value": 1.0}))
                    warm = PersistentVisualFeatureCache(
                        source_cache_directory=root / "cache",
                        source_path=source,
                        analyzer=current,
                    )
                    self.assertIsNone(
                        warm.get_by_locator(request.locator),
                    )
                    self.assertEqual(
                        warm.diagnostics()["invalid_record_count"],
                        1,
                    )

    def test_analyzer_migration_rejects_tampered_current_endpoint(
        self,
    ) -> None:
        previous = _analyzer_from_components(
            _audited_analyzer_components(
                _AUDITED_RUNTIME_PREVIOUS_DESCRIPTOR_DIGEST
            )
        )
        tampered_current_components = _audited_analyzer_components(
            _AUDITED_CURRENT_DESCRIPTOR_DIGEST
        )
        tampered_current_components["runtime"]["opencv"] = "tampered-runtime"
        tampered_current = AnalyzerFingerprint(
            digest=_AUDITED_CURRENT_ANALYZER_DIGEST,
            components=tampered_current_components,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"source-content-v1")
            request = _request()
            cold = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=previous,
            )
            self.assertTrue(cold.put(request, {"value": 1.0}))
            warm = PersistentVisualFeatureCache(
                source_cache_directory=root / "cache",
                source_path=source,
                analyzer=tampered_current,
            )
            self.assertIsNone(warm.get_by_locator(request.locator))
            self.assertEqual(warm.diagnostics()["invalid_record_count"], 1)

    def test_direct_member_provider_reuses_interrupted_run_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"fake-video-source")
            cache_root = root / "cache"
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path),
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_descriptor(),
            ) as describe:
                cold_provider = _VideoDescriptorProvider(
                    (source,),
                    feature_cache_directory=cache_root,
                )
                cold = cold_provider(0, source, 1.0)
                cold_diagnostics = cold_provider.cache_diagnostics()
                cold_provider.close()

                warm_provider = _VideoDescriptorProvider(
                    (source,),
                    feature_cache_directory=cache_root,
                )
                warm = warm_provider(0, source, 1.0)
                warm_diagnostics = warm_provider.cache_diagnostics()
                warm_provider.close()

            self.assertEqual(cold, warm)
            self.assertEqual(describe.call_count, 1)
            cold_persistent = cold_diagnostics[
                "persistent_visual_feature_cache"
            ]
            warm_persistent = warm_diagnostics[
                "persistent_visual_feature_cache"
            ]
            self.assertEqual(cold_persistent["write_count"], 1)
            self.assertEqual(warm_persistent["hit_count"], 1)
            self.assertTrue(
                warm_persistent["interrupted_run_records_reusable"]
            )

    def test_sampler_frame_lru_uses_canonical_frames_not_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.mp4"
            source.write_bytes(b"fake-video-source")
            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path, frame_rate=2000.0),
            ):
                sampler = _VideoSampler(source)
                first = sampler.frame_at(1.0001)
                second = sampler.frame_at(1.0004)
                sampler.close()
            assert first is not None and second is not None
            self.assertFalse(np.array_equal(first, second))

    def test_warm_features_rebuild_identical_timeline_instead_of_caching_dp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = (root / "a.mp4", root / "b.mp4")
            sources[0].write_bytes(b"fake-source-a")
            sources[1].write_bytes(b"fake-source-b")
            arguments = {
                "sources": sources,
                "canonical_spans": tuple(
                    (float(index + 1), float(index + 2)) for index in range(8)
                ),
                "audio_offsets": (0.0, 0.0),
                "sync_scores": (0.9, 0.8),
                "scene_cut_times": ((), ()),
                "preferred_source_indices": (0, 1, 0, 1, 0, 1, 0, 1),
                "feature_cache_directory": root / "cache",
            }

            def select_again(candidates_by_segment, **_kwargs):  # type: ignore[no-untyped-def]
                choices = tuple(
                    TimelineChoice(
                        candidate=candidates[index % len(candidates)],
                        emission_score=1.0,
                        transition_score=0.0,
                        cumulative_score=float(index + 1),
                        used_group_wide_fallback=False,
                    )
                    for index, candidates in enumerate(candidates_by_segment)
                )
                return (
                    (
                        TimelineResult(
                            mode=TimelineMode.GENERAL,
                            target_member=None,
                            choices=choices,
                            total_score=float(len(choices)),
                        ),
                    ),
                    ("test",),
                )

            with patch(
                "crosscut.visual_score.cv2.VideoCapture",
                side_effect=lambda path: _FakeCapture(path),
            ), patch(
                "crosscut.visual_score._visual_analyzer_fingerprint",
                return_value=_analyzer(),
            ), patch(
                "crosscut.visual_score.describe_frame",
                return_value=_group_descriptor(),
            ) as describe, patch(
                "crosscut.visual_score._select_safe_cadence_candidates",
                side_effect=select_again,
            ) as select_timeline:
                cold = build_visual_timeline(**arguments)
                cold_call_count = describe.call_count
                describe.reset_mock()
                warm = build_visual_timeline(**arguments)
                warm_call_count = describe.call_count
            self.assertGreater(cold_call_count, 0)
            self.assertEqual(warm_call_count, 0)
            self.assertEqual(select_timeline.call_count, 2)
            self.assertEqual(cold.to_dict(), warm.to_dict())


if __name__ == "__main__":
    unittest.main()
