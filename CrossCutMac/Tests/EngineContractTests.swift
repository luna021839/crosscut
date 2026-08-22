import XCTest
@testable import CrossCutMac

final class EngineContractTests: XCTestCase {
    private func makeExecutableScript(_ body: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-engine-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        let script = directory.appendingPathComponent("probe.sh")
        try ("#!/bin/sh\n" + body + "\n").write(
            to: script,
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: script.path
        )
        return script
    }

    func testRequestUsesEngineContractKeys() throws {
        let box = try XCTUnwrap(
            NormalizedBoundingBox(left: 0.1, top: 0.2, right: 0.7, bottom: 0.9)
        )
        let selection = try XCTUnwrap(MemberAnchorSelection(timeSeconds: 12.5, bbox: box))
        let request = EditRequest(
            song: "I AM",
            targetMember: .anYujin,
            editMode: .member,
            aspectRatio: .portrait,
            sources: [URL(fileURLWithPath: "/tmp/stage-a.mp4")],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            memberAnchors: [MemberAnchorRequest(sourceIndex: 0, selection: selection)]
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )

        XCTAssertEqual(object["song"] as? String, "I AM")
        XCTAssertEqual(object["target_member"] as? String, "안유진")
        XCTAssertEqual(object["edit_mode"] as? String, "member")
        XCTAssertEqual(object["aspect_ratio"] as? String, "9:16")
        XCTAssertEqual(object["output_directory"] as? String, "/tmp/output")
        let anchors = try XCTUnwrap(object["member_anchors"] as? [[String: Any]])
        XCTAssertEqual(anchors.count, 1)
        XCTAssertEqual(anchors[0]["source_index"] as? Int, 0)
        XCTAssertEqual(anchors[0]["time_seconds"] as? Double, 12.5)
        XCTAssertEqual(anchors[0]["bbox_xyxy"] as? [Double], [0.1, 0.2, 0.7, 0.9])
    }

    func testGroupRequestAlwaysOmitsSelectionsAsEmptyArray() throws {
        let box = try XCTUnwrap(
            NormalizedBoundingBox(left: 0.1, top: 0.2, right: 0.7, bottom: 0.9)
        )
        let selection = try XCTUnwrap(MemberAnchorSelection(timeSeconds: 4, bbox: box))
        let request = EditRequest(
            song: "LOVE DIVE",
            targetMember: .gaeul,
            editMode: .group,
            aspectRatio: .landscape,
            sources: [URL(fileURLWithPath: "/tmp/stage.mp4")],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            memberAnchors: [MemberAnchorRequest(sourceIndex: 0, selection: selection)]
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        XCTAssertEqual((object["member_anchors"] as? [Any])?.count, 0)
    }

    func testAspectFitCoordinateMappingExcludesLetterboxBars() throws {
        let imageSize = CGSize(width: 1_920, height: 1_080)
        let containerSize = CGSize(width: 800, height: 600)
        let fitted = AspectFitCoordinateMapper.fittedRect(imageSize: imageSize, in: containerSize)

        XCTAssertEqual(fitted.origin.x, 0, accuracy: 0.0001)
        XCTAssertEqual(fitted.origin.y, 75, accuracy: 0.0001)
        XCTAssertEqual(fitted.width, 800, accuracy: 0.0001)
        XCTAssertEqual(fitted.height, 450, accuracy: 0.0001)

        let box = try XCTUnwrap(
            AspectFitCoordinateMapper.boundingBox(
                from: CGPoint(x: 80, y: 120),
                to: CGPoint(x: 720, y: 525),
                imageSize: imageSize,
                in: containerSize
            )
        )
        XCTAssertEqual(box.left, 0.1, accuracy: 0.0001)
        XCTAssertEqual(box.top, 0.1, accuracy: 0.0001)
        XCTAssertEqual(box.right, 0.9, accuracy: 0.0001)
        XCTAssertEqual(box.bottom, 1.0, accuracy: 0.0001)

        let restored = AspectFitCoordinateMapper.rect(
            for: box,
            imageSize: imageSize,
            in: containerSize
        )
        XCTAssertEqual(restored.minX, 80, accuracy: 0.0001)
        XCTAssertEqual(restored.minY, 120, accuracy: 0.0001)
        XCTAssertEqual(restored.maxX, 720, accuracy: 0.0001)
        XCTAssertEqual(restored.maxY, 525, accuracy: 0.0001)
    }

    func testProgressEventAcceptsPercentAndNormalizesIt() throws {
        let data = Data(#"{"type":"progress","percent":42,"message":"tracking"}"#.utf8)
        let event = try JSONDecoder().decode(EngineEvent.self, from: data)

        XCTAssertEqual(event.progress, 0.42)
        XCTAssertEqual(event.message, "tracking")
    }

    func testResultEventAcceptsMultipleOutputShapes() throws {
        let data = Data(
            #"{"type":"result","output_path":"/tmp/a.mp4","outputs":["/tmp/b.mp4"]}"#.utf8
        )
        let event = try JSONDecoder().decode(EngineEvent.self, from: data)

        XCTAssertEqual(event.outputPaths, ["/tmp/b.mp4", "/tmp/a.mp4"])
    }

    func testAugmentedPathAlwaysIncludesHomebrewMediaTools() {
        let path = EngineLocator.augmentedPath(existing: "/custom/bin:/usr/bin")
        let parts = path.split(separator: ":").map(String.init)

        XCTAssertTrue(parts.contains("/opt/homebrew/bin"))
        XCTAssertTrue(parts.contains("/custom/bin"))
        XCTAssertEqual(parts.filter { $0 == "/usr/bin" }.count, 1)
    }

    func testRuntimeProbeRequiresMachineReadableFullReady() {
        XCTAssertTrue(
            EngineLocator.runtimeProbeReady(
                from: Data(#"{"full_ready":true}"#.utf8)
            )
        )
        XCTAssertFalse(
            EngineLocator.runtimeProbeReady(
                from: Data(#"{"full_ready":false}"#.utf8)
            )
        )
        XCTAssertFalse(
            EngineLocator.runtimeProbeReady(
                from: Data(#"{"mode_readiness":{"group":true}}"#.utf8)
            )
        )
        XCTAssertTrue(
            EngineLocator.runtimeProbeReady(
                from: Data("ignored warning\n{\"full_ready\":true}\n".utf8)
            )
        )
    }

    func testRuntimeProbeHonorsJSONResultAndExitStatus() throws {
        let ready = try makeExecutableScript(
            "printf '%s\\n' '{\"full_ready\":true}'; exit 0"
        )
        let falseReport = try makeExecutableScript(
            "printf '%s\\n' '{\"full_ready\":false}'; exit 0"
        )
        let failed = try makeExecutableScript(
            "printf '%s\\n' '{\"full_ready\":true}'; exit 1"
        )
        let engine = URL(fileURLWithPath: "/tmp/engine/crosscut_cli.py")

        XCTAssertTrue(
            EngineLocator.supportsEngineModules(
                pythonURL: ready,
                engineURL: engine,
                timeout: 1
            )
        )
        XCTAssertFalse(
            EngineLocator.supportsEngineModules(
                pythonURL: falseReport,
                engineURL: engine,
                timeout: 1
            )
        )
        XCTAssertFalse(
            EngineLocator.supportsEngineModules(
                pythonURL: failed,
                engineURL: engine,
                timeout: 1
            )
        )
    }

    func testRuntimeProbeTimesOutInsteadOfBlockingLocator() throws {
        let hanging = try makeExecutableScript(
            "trap 'exit 0' TERM; while :; do :; done"
        )

        XCTAssertFalse(
            EngineLocator.supportsEngineModules(
                pythonURL: hanging,
                engineURL: URL(fileURLWithPath: "/tmp/engine/crosscut_cli.py"),
                timeout: 0.05
            )
        )
    }

    func testApplicationBundleContainsCleanRunnableEngineSource() throws {
        let resources = try XCTUnwrap(Bundle.main.resourceURL)
        let engineDirectory = resources.appendingPathComponent("engine", isDirectory: true)
        let script = engineDirectory.appendingPathComponent("crosscut_cli.py")
        let package = engineDirectory.appendingPathComponent("crosscut", isDirectory: true)

        XCTAssertTrue(FileManager.default.isReadableFile(atPath: script.path))
        XCTAssertTrue(
            FileManager.default.isReadableFile(
                atPath: package.appendingPathComponent("__init__.py").path
            )
        )
        XCTAssertTrue(
            FileManager.default.isReadableFile(
                atPath: engineDirectory
                    .appendingPathComponent("crosscut/models/face_detection_yunet_2023mar.onnx")
                    .path
            )
        )
        XCTAssertTrue(
            FileManager.default.isReadableFile(
                atPath: engineDirectory
                    .appendingPathComponent("crosscut/models/LICENSE.yunet").path
            )
        )
        let sfaceAssetPaths = [
            "crosscut/models/face_recognition_sface_2021dec.onnx",
            "crosscut/models/LICENSE.opencv-zoo-sface",
        ]
        for relativePath in sfaceAssetPaths {
            XCTAssertTrue(
                FileManager.default.isReadableFile(
                    atPath: engineDirectory.appendingPathComponent(relativePath).path
                ),
                "Missing bundled SFace asset: \(relativePath)"
            )
        }
        let poseAssetPaths = [
            "crosscut/models/person_detection_mediapipe_2023mar.onnx",
            "crosscut/models/pose_estimation_mediapipe_2023mar.onnx",
            "crosscut/models/human_segmentation_pphumanseg_2023mar.onnx",
            "crosscut/models/LICENSE.opencv-zoo-mediapipe-pose",
            "crosscut/models/LICENSE.opencv-zoo-pphumanseg",
            "crosscut/models/PROVENANCE.pose.md",
        ]
        for relativePath in poseAssetPaths {
            XCTAssertTrue(
                FileManager.default.isReadableFile(
                    atPath: engineDirectory.appendingPathComponent(relativePath).path
                ),
                "Missing bundled pose asset: \(relativePath)"
            )
        }

        let located = try XCTUnwrap(
            EngineLocator.locate(fileManager: .default, bundle: .main)
        )
        XCTAssertEqual(located.standardizedFileURL, script.standardizedFileURL)

        let enumerator = try XCTUnwrap(
            FileManager.default.enumerator(
                at: engineDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        )
        let relativePaths = enumerator.compactMap { item -> String? in
            guard let url = item as? URL else { return nil }
            return String(url.path.dropFirst(engineDirectory.path.count + 1))
        }

        XCTAssertGreaterThan(relativePaths.filter { $0.hasSuffix(".py") }.count, 5)
        XCTAssertFalse(relativePaths.contains { $0.contains("__pycache__") })
        XCTAssertFalse(relativePaths.contains { $0.hasSuffix(".pyc") })
        XCTAssertFalse(relativePaths.contains { $0.hasPrefix("tests/") })
        XCTAssertTrue(
            relativePaths.allSatisfy {
                $0 == "crosscut_cli.py"
                    || $0 == "crosscut"
                    || ($0.hasPrefix("crosscut/") && $0.hasSuffix(".py"))
                    || $0 == "crosscut/models"
                    || $0 == "crosscut/models/face_detection_yunet_2023mar.onnx"
                    || $0 == "crosscut/models/LICENSE.yunet"
                    || sfaceAssetPaths.contains($0)
                    || poseAssetPaths.contains($0)
            }
        )
    }
}
