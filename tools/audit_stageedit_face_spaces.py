"""Bounded source-native comparison, not an identity accuracy certificate.

Reuses exact cached observations/clocks and the user's existing local banks.
Never trains, modifies a gallery, or serializes query/reference embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import cv2

from crosscut.ive_face_identity_gallery import (
    classify_open_set_ive_identity, load_ive_face_identity_gallery_artifact,
)
from crosscut.ive_local_arcface_gallery import LocalIVEArcFaceGallery
from crosscut.member_state_graph import load_member_state_graph_artifact
from crosscut.native_face_descriptor import (
    ARCFACE_MODEL_NAME, NativeArcFaceR50, NativeFaceDescriptorSpace,
)
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.face_identity import _validate_normalized_face


def select_observations(observations, *, per_source=3, minimum_face_height=.15):
    """Even time quantiles of sufficiently large faces, before identity scores."""
    if type(per_source) is not int or not 1 <= per_source <= 6:
        raise ValueError("per_source must be between one and six")
    if not 0 < minimum_face_height <= 1:
        raise ValueError("minimum_face_height must be in (0, 1]")
    candidates = {}
    for obs in observations:
        box = obs.face_bbox
        if box is not None and box.bottom - box.top >= minimum_face_height:
            candidates.setdefault(obs.source_id, []).append(obs)
    selected = []
    for source_id in sorted(candidates):
        ordered = sorted(candidates[source_id],
            key=lambda obs: (obs.timestamp_seconds, obs.observation_id))
        # Collapse nearby observations of the same shot; do not sample dozens
        # of almost identical frames merely because the decision grid is dense.
        spaced = []
        for obs in ordered:
            if not spaced or obs.timestamp_seconds - spaced[-1].timestamp_seconds >= 1.:
                spaced.append(obs)
        count = min(per_source, len(spaced))
        indices = ([len(spaced) // 2] if count == 1 else
            [round(i * (len(spaced) - 1) / (count - 1)) for i in range(count)])
        selected.extend(spaced[i] for i in indices)
    return tuple(selected)


def _iou(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def observed_face_geometry(face, width, height):
    """Intersect the detector rectangle, never fabricate off-screen landmarks.

    ArcFace alignment uses the FIVE ORIGINAL points, not the rectangle. A
    detector box spilling by a few pixels cannot invalidate otherwise fully
    observed eyes/nose/mouth. Missing landmark evidence still stays missing.
    """
    raw = tuple(float(v) for v in (face[0]/width, face[1]/height, face[2]/width, face[3]/height))
    left, top = max(0., raw[0]), max(0., raw[1])
    right, bottom = min(1., raw[0]+raw[2]), min(1., raw[1]+raw[3])
    box = (left, top, right-left, bottom-top)
    landmarks = tuple((float(face[k]/width), float(face[k+1]/height)) for k in range(4, 14, 2))
    _validate_normalized_face(box, landmarks)
    return box, landmarks, raw


def _file_identity(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise OSError("source must be a regular nonsymlink file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def verify_source(source, *, device, check=lambda: None, root=Path("/Volumes/CrossCutMedia")):
    """Hash the current original; a retained request's digest is not live proof."""
    path = Path(source["path"])
    before = _file_identity(path)
    if (not path.is_relative_to(root) or before[0] != device or before[2] != source["size_bytes"]):
        raise OSError("source differs from the retained request")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            check()
            digest.update(chunk)
    if _file_identity(path) != before or digest.hexdigest() != source["media_content_sha256"]:
        raise OSError("source content differs from the retained request")
    return before


def render_observation_sheet(report_path, request_path, output_path):
    """Review retained scalar decisions against their pixels, without new NN calls."""
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads(Path(report_path).read_bytes())
    request_bytes = Path(request_path).read_bytes()
    if hashlib.sha256(request_bytes).hexdigest() != report["request_file_sha256"]:
        raise ValueError("contact sheet request changed")
    rows = [row for row in report["rows"] if row["state"] == "OBSERVED"]
    if not 1 <= len(rows) <= 192 or Path(output_path).exists():
        raise ValueError("contact sheet needs bounded rows and a new output")
    sources = {source["source_id"]: source for source in json.loads(request_bytes)["sources"]}
    sheet = Image.new("RGB", (1280, ((len(rows)+3)//4)*246), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 15)
    for i, row in enumerate(rows):
        capture = cv2.VideoCapture(sources[row["source_id"]]["path"])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["native_frame"])
            ok, pixels = capture.read()
            if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES))-1 != row["native_frame"]:
                raise OSError("contact sheet returned another native frame")
        finally:
            capture.release()
        image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        image.thumbnail((320, 180))
        x, y = (i % 4)*320, (i // 4)*246
        sheet.paste(image, (x, y))
        s, a = row["sface"], row["arcface"]
        draw.text((x+4, y+182), f'{i+1:02d} {row["source_id"]} {row["master_seconds"]:.2f}s', font=font, fill="white")
        draw.text((x+4, y+202), f'S: {s["accepted_member"] or "UNKNOWN"} {s["top_similarity"]:.3f}', font=font, fill="white")
        draw.text((x+4, y+222), f'A: {a["member"] or "UNKNOWN"} {a["best_similarity"]:.3f}', font=font, fill="white")
    sheet.save(output_path, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "member-graph", "member-bank", "yujin-bank", "arcface-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=3)
    parser.add_argument("--all-cached-faces", action="store_true",
        help="Explicit face-only corpus extraction; no pose/visual reanalysis.")
    parser.add_argument("--maximum-face-observations", type=int, default=2048)
    parser.add_argument("--arcface-only", action="store_true")
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--ssd-device", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.maximum_face_observations <= 4096:
        parser.error("maximum-face-observations must be between one and4096")
    if not 0 < args.seconds <= (600 if args.all_cached_faces else 180):
        parser.error("seconds exceeds the explicitly selected observation mode")
    started = time.monotonic()
    volume = Path("/Volumes/CrossCutMedia")

    def live():
        if not os.path.ismount(volume) or volume.stat().st_dev != args.ssd_device:
            raise OSError("authorized SSD was disconnected or replaced")
        if time.monotonic() - started > args.seconds:
            raise TimeoutError("bounded source-native face comparison exhausted its time budget")

    live()
    if not args.output.is_relative_to(volume) or args.output.exists():
        parser.error("output must be a new file on the authorized SSD")
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    sources = {source["source_id"]: source for source in request["sources"]}
    graph = load_member_state_graph_artifact(args.member_graph)
    selected = (tuple(sorted((obs for obs in graph.graph.observations if obs.face_bbox is not None),
        key=lambda obs: (obs.source_id, obs.timestamp_seconds, obs.observation_id)))
        if args.all_cached_faces else select_observations(graph.graph.observations, per_source=args.per_source))
    if not selected or len(sources) > 32 or len(selected) > (args.maximum_face_observations if args.all_cached_faces else 192):
        raise ValueError("face observation count exceeds this explicit bounded mode")
    installed = request["gallery_installation"]
    sface_gallery = None if args.arcface_only else load_ive_face_identity_gallery_artifact(
        Path(installed["gallery_artifact_path"]),
        expected_content_sha256=installed["gallery_content_sha256"])
    gallery = LocalIVEArcFaceGallery.from_existing_photo_banks(args.member_bank, args.yujin_bank)
    arc = NativeArcFaceR50(args.arcface_model)
    space = NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME, 512, arc.asset_sha256,
        hashlib.sha256(Path("engine/crosscut/native_face_descriptor.py").read_bytes()).hexdigest())
    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create("engine/crosscut/models/face_detection_yunet_2023mar.onnx",
        "", (960, 540), .75, .3, 5000)
    recognizer = None if args.arcface_only else cv2.FaceRecognizerSF.create(
        "engine/crosscut/models/face_recognition_sface_2021dec.onnx", "")
    rows = []
    capture = None
    current_source = None
    source_stat = None
    calls = dict(decoded_frames=0, detector=0, arcface=0, sface=0)
    try:
        for obs in selected:
            live()
            source = sources[obs.source_id]
            path = Path(source["path"])
            if not path.is_relative_to(volume):
                raise ValueError("source moved outside the authorized SSD")
            if obs.source_id != current_source:
                if capture is not None:
                    capture.release()
                    if source_stat != _file_identity(last_path):
                        raise OSError("source changed during comparison")
                source_stat = verify_source(source, device=args.ssd_device, check=live)
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise OSError("cannot open retained source")
                current_source = obs.source_id
                last_path = path
            clock = ReferenceSourceClock.from_mapping(source["clock"])
            source_seconds = clock.master_to_source(obs.timestamp_seconds)
            native_frame = round(source_seconds * source["frame_rate"])
            current = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if current <= native_frame and native_frame-current <= 30:
                for _ in range(native_frame-current):
                    if not capture.grab():
                        raise OSError("native source advance rejected")
            elif not capture.set(cv2.CAP_PROP_POS_FRAMES, native_frame):
                raise OSError("native source seek rejected")
            ok, pixels = capture.read()
            if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1 != native_frame:
                raise OSError("native source seek returned another frame")
            calls["decoded_frames"] += 1
            row = dict(source_id=obs.source_id, observation_id=obs.observation_id,
                media_content_sha256=source["media_content_sha256"],
                master_seconds=obs.timestamp_seconds, source_seconds=source_seconds,
                native_frame=native_frame,
                cached_known_mass=1.-obs.member_probabilities.unknown_mass)
            small = cv2.resize(pixels, (960, 540), interpolation=cv2.INTER_AREA)
            _, faces = detector.detect(small)
            calls["detector"] += 1
            expected = (obs.face_bbox.left, obs.face_bbox.top, obs.face_bbox.right, obs.face_bbox.bottom)
            ranked = []
            for face in (() if faces is None else faces):
                box = (face[0]/960, face[1]/540, (face[0]+face[2])/960, (face[1]+face[3])/540)
                ranked.append((_iou(expected, box), face))
            ranked.sort(key=lambda item: -item[0])
            if not ranked or ranked[0][0] < .25:
                row["state"] = "CACHED_FACE_NOT_REOBSERVED"
                rows.append(row)
                continue
            overlap, face = ranked[0]
            face = face.copy()
            height, width = pixels.shape[:2]
            face[:14:2] *= width/960
            face[1:14:2] *= height/540
            try:
                box, landmarks, raw_box = observed_face_geometry(face, width, height)
            except ValueError:
                row["state"] = "FIVE_FACE_LANDMARKS_NOT_FULLY_OBSERVED"
                rows.append(row)
                continue
            arc_vector = arc.embedding(pixels, bbox_xywh=box, landmarks_xy=landmarks)
            calls["arcface"] += 1
            hint = gallery.classify(arc_vector, descriptor_space=space)
            sface_result = None
            if recognizer is not None:
                sface_vector = recognizer.feature(recognizer.alignCrop(pixels, face)).ravel()
                calls["sface"] += 1
                sface_result = classify_open_set_ive_identity(sface_gallery.gallery,
                    tuple(float(v) for v in sface_vector)).to_scalar_dict()
                del sface_vector
            row.update(state="OBSERVED", cached_bbox_iou=float(overlap), face_bbox_xywh=box,
                detector_bbox_xywh=raw_box, alignment_landmarks_changed=False,
                arcface=hint, sface=sface_result)
            rows.append(row)
            del arc_vector, pixels, small
            if len(rows) % 100 == 0:
                print(json.dumps(dict(completed=len(rows), selected=len(selected), calls=calls,
                    elapsed_seconds=time.monotonic()-started)), flush=True)
    finally:
        if capture is not None:
            capture.release()
            if source_stat != _file_identity(last_path):
                raise OSError("source changed during comparison")
    live()
    report = dict(contract="bounded_source_native_sface_arcface_comparison_v1",
        request_file_sha256=hashlib.sha256(request_bytes).hexdigest(),
        graph_content_sha256=graph.content_sha256, selected_count=len(selected),
        selection=("every_cached_primary_face_exact_source_time" if args.all_cached_faces else
            "time_quantiles_per_source_face_height_at_least_015_one_second_separation"),
        calls=calls, rows=rows, elapsed_seconds=time.monotonic()-started,
        current_source_contents_verified_against_request=True,
        raw_query_or_reference_vectors_serialized=False, gallery_changed=False,
        identity_accuracy_measured=False, professional_quality_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(args.output), calls=calls,
        selected_count=len(selected), elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
