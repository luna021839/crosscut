import CryptoKit
import XCTest
@testable import CrossCutMac

final class SceneReviewProposalTests: XCTestCase {
    private func eventObject() -> [String: Any] {
        ["type": "scene_review_proposal", "scene_review_proposal": [
            "contract": "normal_scene_review_proposal_v1", "status": "REVIEW_REQUIRED",
            "artifact_role": "scene_review_only", "quality_approved": false,
            "delivery_completion_message_allowed": false,
            "video_path": "/tmp/review.mp4", "manifest_path": "/tmp/review.json",
            "video_sha256": String(repeating: "a", count: 64),
            "manifest_sha256": String(repeating: "b", count: 64)]]
    }
    private func decode(_ value: [String: Any]) throws -> EngineEvent {
        try JSONDecoder().decode(EngineEvent.self, from: JSONSerialization.data(withJSONObject: value))
    }
    func testSeparateEventCannotAuthorizeDeliveryOrProgress() throws {
        let value = try decode(eventObject())
        XCTAssertEqual(value.sceneReviewProposal?.isReviewOnly, true)
        XCTAssertTrue(value.outputPaths.isEmpty)
        XCTAssertNil(value.artifactResult)
        XCTAssertNil(value.progress)
    }
    func testApprovalRoleStatusAndHashMutationsReject() throws {
        let mutations: [(String, Any)] = [
            ("quality_approved", true), ("delivery_completion_message_allowed", true),
            ("status", "APPROVED"), ("artifact_role", "delivery"), ("video_sha256", "bad")]
        for (key, value) in mutations {
            var object = eventObject()
            var payload = object["scene_review_proposal"] as! [String: Any]
            payload[key] = value; object["scene_review_proposal"] = payload
            XCTAssertEqual(try decode(object).sceneReviewProposal?.isReviewOnly, false)
        }
    }
    @MainActor
    func testLiveFileHashBindingAndReplacement() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let video = root.appendingPathComponent("review.mp4")
        let manifest = root.appendingPathComponent("review.json")
        // Hash/disposition unit fixture, not a decoded-media QA fixture.
        let bytes = Data("HASH_BINDING_FIXTURE_NOT_REAL_VIDEO".utf8)
        try bytes.write(to: video)
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        var event = eventObject()
        var payload = event["scene_review_proposal"] as! [String: Any]
        payload["video_path"] = video.path; payload["manifest_path"] = manifest.path
        payload["video_sha256"] = hash(bytes)
        var document = payload
        document["technical_qa"] = ["technical_only": true, "professional_qa_passed": false,
                                   "issue_codes": []] as [String: Any]
        let data = try JSONSerialization.data(withJSONObject: document)
        try data.write(to: manifest);payload["manifest_sha256"] = hash(data)
        event["scene_review_proposal"] = payload
        let proposal = try XCTUnwrap(decode(event).sceneReviewProposal)
        XCTAssertTrue(EngineRunner.validateSceneReviewProposal(proposal, directory: root))
        XCTAssertFalse(EngineRunner.validateSceneReviewProposal(proposal, directory: root.appendingPathComponent("other")))
        try Data("REPLACED".utf8).write(to: video)
        XCTAssertFalse(EngineRunner.validateSceneReviewProposal(proposal, directory: root))
    }
}
