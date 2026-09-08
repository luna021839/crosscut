import XCTest
@testable import CrossCutMac

final class EngineProcessIntegrationTests: XCTestCase {
    @MainActor
    private func runFixture(_ program: String, mode: EditMode) async throws -> EngineRunner {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-ipc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("fixture.py")
        try program.write(to: script, atomically: true, encoding: .utf8)
        let runner = EngineRunner(
            locateEngine: { script },
            preparationProbe: { _, _, _ in
                .ready(URL(fileURLWithPath: "/usr/bin/python3"))
            }
        )
        let request = EditRequest(
            song: "I AM",
            targetMember: .gaeul,
            editMode: mode,
            aspectRatio: mode == .member ? .portrait : .landscape,
            sources: [root.appendingPathComponent("a.mp4"), root.appendingPathComponent("b.mp4")],
            outputDirectory: root.appendingPathComponent("output", isDirectory: true)
        )
        runner.start(request: request)
        let deadline = Date().addingTimeInterval(5)
        while runner.state.isRunning && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        if runner.state.isRunning {
            runner.cancel()
            let cancellationDeadline = Date().addingTimeInterval(3)
            while runner.state.isRunning && Date() < cancellationDeadline {
                try await Task.sleep(nanoseconds: 10_000_000)
            }
            XCTFail("Fixture engine did not finish within its bounded timeout")
        }
        // The real runner serializes and removes a request; source files are
        // intentionally absent because this fixture never imports the engine.
        if let requestLog = runner.logs.first(where: { $0.message.hasPrefix("REQUEST_FILE:") }) {
            let path = String(requestLog.message.dropFirst("REQUEST_FILE:".count))
            XCTAssertFalse(FileManager.default.fileExists(atPath: path))
        } else {
            XCTFail("Fixture did not receive the real serialized request")
        }
        return runner
    }

    @MainActor
    func testFragmentedPipeEventsDoNotCompletePreflightOnlyGroupRun() async throws {
        let runner = try await runFixture(#"""
import json, os, sys, time
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
assert request['edit_mode'] == 'group'
print('REQUEST_FILE:' + request_path, flush=True)
print('not a JSON event', flush=True)
payload = json.dumps({
    'type': 'joint_candidate_terminal',
    'contract': 'joint_candidate_terminal_v1',
    'candidate_index': 0, 'status': 'decoded_review_passed',
    'elapsed_seconds': 1.0, 'decoded_pass_count': 8,
    'final_media_authority': False
}).encode('utf-8') + b'\n'
os.write(1, payload[:17])
time.sleep(0.02)
os.write(1, payload[17:])
os.write(1, b'{"type":"progress","percent":"42"}')
"""#, mode: .group)

        guard case .failed = runner.state else {
            return XCTFail("Preflight-only process exit must not deliver a video")
        }
        XCTAssertEqual(runner.progress, 0.42)
        XCTAssertTrue(runner.logs.contains { $0.message.contains("후보 1 사전 영상 QA 통과") })
        XCTAssertTrue(runner.logs.contains { $0.message == "not a JSON event" })
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertNil(runner.renderedVideoURL)
    }

    @MainActor
    func testFinalUnterminatedErrorEventIsDrainedBeforeMemberProcessExit() async throws {
        let runner = try await runFixture(#"""
import json, os, sys, time
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
assert request['edit_mode'] == 'member'
print('REQUEST_FILE:' + request_path, flush=True)
print('fixture stderr diagnostic', file=sys.stderr, flush=True)
payload = json.dumps({'type': 'error', 'message': '멤버 시험 오류'}, ensure_ascii=False).encode('utf-8')
split = payload.index('멤'.encode('utf-8')) + 1
os.write(1, payload[:split])
time.sleep(0.02)
os.write(1, payload[split:])
sys.exit(7)
"""#, mode: .member)

        XCTAssertEqual(runner.state, .failed("멤버 시험 오류"))
        XCTAssertTrue(runner.logs.contains { $0.message == "fixture stderr diagnostic" })
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertNil(runner.renderedVideoURL)
    }
}
