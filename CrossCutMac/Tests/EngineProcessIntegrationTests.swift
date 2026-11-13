import XCTest
@testable import CrossCutMac

final class EngineProcessIntegrationTests: XCTestCase {
    @MainActor
    private func runFixture(
        _ program: String,
        mode: EditMode,
        sourceMode: SourceMode = .local,
        outputDrainGraceNanoseconds: UInt64 = 5_000_000_000,
        cancelWhenReadyFileExists: Bool = false,
        prepareOutputs: ((URL) throws -> [URL])? = nil,
        whileRunning: ((EngineRunner, URL) throws -> Void)? = nil
    ) async throws -> EngineRunner {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-ipc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("fixture.py")
        try program.write(to: script, atomically: true, encoding: .utf8)
        let outputDirectory = root.appendingPathComponent("output", isDirectory: true)
        if let prepareOutputs {
            let files = try prepareOutputs(outputDirectory)
            try JSONSerialization.data(withJSONObject: files.map(\.path))
                .write(to: root.appendingPathComponent("fixture-outputs.json"), options: .atomic)
        }
        let runner = EngineRunner(
            locateEngine: { script },
            preparationProbe: { _, _, _ in
                .ready(URL(fileURLWithPath: "/usr/bin/python3"))
            },
            outputDrainGraceNanoseconds: outputDrainGraceNanoseconds
        )
        let request = EditRequest(
            song: "XOXZ",
            targetMember: .gaeul,
            editMode: mode,
            aspectRatio: mode == .member ? .portrait : .landscape,
            sources: [root.appendingPathComponent("a.mp4"), root.appendingPathComponent("b.mp4")],
            outputDirectory: outputDirectory,
            sourceMode: sourceMode
        )
        runner.start(request: request)
        let deadline = Date().addingTimeInterval(5)
        while runner.state.isRunning && Date() < deadline {
            XCTAssertTrue(runner.outputURLs.isEmpty, "Artifacts must stay private until successful final validation")
            try whileRunning?(runner, root)
            if cancelWhenReadyFileExists,
               FileManager.default.fileExists(atPath: root.appendingPathComponent("ready-to-cancel").path) {
                runner.cancel()
            }
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
    func testFlushedStatusArrivesWhileTheEngineIsStillWaiting() async throws {
        var observedWhileWaiting = false
        let runner = try await runFixture(#"""
import json, pathlib, sys, time
root = pathlib.Path(__file__).parent
request_path = sys.argv[sys.argv.index('--request') + 1]
print('REQUEST_FILE:' + request_path, flush=True)
print(json.dumps({'type': 'status', 'message': 'LIVE_STATUS_BEFORE_EXIT'}), flush=True)
print('LIVE_STDERR_BEFORE_EXIT', file=sys.stderr, flush=True)
(root / 'status-was-written').write_text('ready')
deadline = time.monotonic() + 2
while not (root / 'release-engine').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
(root / 'engine-finished-waiting').write_text('finished')
"""#, mode: .group, whileRunning: { runner, root in
            if runner.statusMessage == "LIVE_STATUS_BEFORE_EXIT",
               runner.logs.contains(where: { $0.message == "LIVE_STDERR_BEFORE_EXIT" }),
               FileManager.default.fileExists(atPath: root.appendingPathComponent("status-was-written").path),
               !FileManager.default.fileExists(atPath: root.appendingPathComponent("engine-finished-waiting").path) {
                observedWhileWaiting = true
                try Data([1]).write(to: root.appendingPathComponent("release-engine"))
            }
        })
        XCTAssertTrue(runner.logs.contains { $0.message == "LIVE_STATUS_BEFORE_EXIT" })
        XCTAssertTrue(observedWhileWaiting, "Flushed progress must be received before EOF, while the engine waits for the UI")
    }

    @MainActor
    func testInheritedPipeAfterParentExitCannotBecomeSuccessfulDelivery() async throws {
        try await assertInheritedPipeFails(requestCancellation: false)
    }

    @MainActor
    func testInheritedPipeCannotBeReportedAsCleanCancellation() async throws {
        try await assertInheritedPipeFails(requestCancellation: true)
    }

    @MainActor
    private func assertInheritedPipeFails(requestCancellation: Bool) async throws {
        var fixtureRoot: URL?
        let runner = try await runFixture(#"""
import json, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).parent
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'child-held-result.mp4'
video.write_bytes(b'x')
child_code = "import pathlib, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(0.8); pathlib.Path(sys.argv[2]).write_text('done')"
subprocess.Popen([sys.executable, '-c', child_code, str(root / 'child-ready'), str(root / 'child-finished')])
deadline = time.monotonic() + 2
while not (root / 'child-ready').exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('Inherited-pipe fixture child did not become ready')
    time.sleep(0.005)
print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
(root / 'ready-to-cancel').write_text('ready')
"""#, mode: .group, outputDrainGraceNanoseconds: 150_000_000,
            cancelWhenReadyFileExists: requestCancellation,
            whileRunning: { _, root in fixtureRoot = root })
        if case .failed(let message) = runner.state {
            XCTAssertTrue(message.contains("하위 작업"))
        } else {
            XCTFail("Unclosed inherited output streams must fail explicitly, not deliver the pending video")
        }
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertNil(runner.renderedVideoURL)
        let root = try XCTUnwrap(fixtureRoot)
        let childFinished = root.appendingPathComponent("child-finished")
        XCTAssertFalse(FileManager.default.fileExists(atPath: childFinished.path), "Runner must stop waiting before the inherited pipe's child exits")
        // The child's ready handshake proves its TERM-ignore handler is active
        // before cancellation can race the parent exit. Otherwise legitimate
        // group cancellation kills the child, closes both pipes and invalidates
        // this test's deliberately unclosed-stream premise.
        // The disposable child exits naturally; do not signal a dead parent's
        // process group or remove the fixture directory while the child uses it.
        let deadline = Date().addingTimeInterval(2)
        while !FileManager.default.fileExists(atPath: childFinished.path) && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: childFinished.path))
    }

    func testLineBufferPreservesSplitUnicodeAndBoundsUnterminatedInput() throws {
        let bytes = Data("한글\nlast".utf8)
        var buffer = EngineOutputLineBuffer()
        buffer.append(Data(bytes.prefix(1)))
        XCTAssertNil(try buffer.nextLine())
        buffer.append(Data(bytes.dropFirst(1).prefix(2)))
        XCTAssertNil(try buffer.nextLine())
        buffer.append(Data(bytes.dropFirst(3)))
        XCTAssertEqual(try buffer.nextLine(), "한글")
        XCTAssertNil(try buffer.nextLine())
        XCTAssertEqual(try buffer.finish(), ["last"])
        var oversized = EngineOutputLineBuffer(maximumLineBytes: 4)
        oversized.append(Data("12345".utf8))
        XCTAssertThrowsError(try oversized.nextLine())
        var invalid = EngineOutputLineBuffer()
        invalid.append(Data([0xff]))
        XCTAssertNil(try invalid.nextLine())
        XCTAssertThrowsError(try invalid.finish())
    }

    @MainActor
    func testMalformedOutputCannotPromoteResultOrOverwriteOriginalError() async throws {
        for originalError in [false, true] {
            let program = #"""
import json, os, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'malformed-result.mp4'
video.write_bytes(b'x')
print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
"""# + (originalError ? "\nprint(json.dumps({'type': 'error', 'message': 'original pipeline failure'}), flush=True)\n" : "\n")
                + (originalError ? "os.write(1, b'\\xff\\n')\n" : "os.write(1, b'\\xff')\n")
            let runner = try await runFixture(program, mode: .group)
            guard case .failed(let message) = runner.state else {
                XCTFail("Unreadable stdout cannot authorize the pending result")
                continue
            }
            if originalError { XCTAssertEqual(message, "original pipeline failure") }
            else { XCTAssertTrue(message.contains("UTF-8")) }
            XCTAssertTrue(runner.outputURLs.isEmpty)
            XCTAssertNil(runner.renderedVideoURL)
        }
    }

    @MainActor
    func testAutomaticBareVideoResultCannotBypassDeliveryQA() async throws {
        for mode in [EditMode.group, .member] {
            let runner = try await runFixture(#"""
import json, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'unreviewed.mp4'
video.write_bytes(b'x')
print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
"""#, mode: mode, sourceMode: .automatic)
            guard case .failed = runner.state else {
                XCTFail("Automatic \(mode) must require delivery and final QA evidence")
                continue
            }
            XCTAssertTrue(runner.outputURLs.isEmpty)
            XCTAssertNil(runner.renderedVideoURL)
        }
    }

    @MainActor
    func testResultFollowedByEngineErrorDoesNotExposeArtifacts() async throws {
        let runner = try await runFixture(#"""
import json, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'rejected.mp4'
video.write_bytes(b'x')
print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
print(json.dumps({'type': 'error', 'message': 'final QA failed'}), flush=True)
sys.exit(7)
"""#, mode: .group)
        XCTAssertEqual(runner.state, .failed("final QA failed"))
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertNil(runner.renderedVideoURL)
    }

    @MainActor
    func testLocalSuccessfulResultPreservesLegacyCompatibility() async throws {
        let runner = try await runFixture(#"""
import json, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'local-result.mp4'
video.write_bytes(b'x')
print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
"""#, mode: .group)
        XCTAssertEqual(runner.state, .completed)
        XCTAssertEqual(runner.outputURLs.count, 1)
        XCTAssertEqual(runner.renderedVideoURL?.lastPathComponent, "local-result.mp4")
    }

    @MainActor
    func testCancellationIgnoresLateResultFromTerminationHandler() async throws {
        let runner = try await runFixture(#"""
import json, pathlib, signal, sys, time
request_path = sys.argv[sys.argv.index('--request') + 1]
with open(request_path) as source:
    request = json.load(source)
print('REQUEST_FILE:' + request_path, flush=True)
video = pathlib.Path(request['output_directory']) / 'cancelled.mp4'
video.write_bytes(b'x')
def terminated(signum, frame):
    print(json.dumps({'type': 'result', 'output': str(video)}), flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, terminated)
pathlib.Path(__file__).with_name('ready-to-cancel').write_text('ready')
time.sleep(10)
"""#, mode: .group, cancelWhenReadyFileExists: true)
        XCTAssertEqual(runner.state, .cancelled)
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertNil(runner.renderedVideoURL)
    }

    @MainActor
    func testAutomaticDeliveryMetadataPassPublishesOnlyItsCoreFiles() async throws {
        let fixtures = EngineContractTests(selector: NSSelectorFromString("testReportedDeliveryUsesOnlyTheExplicitArtifactSet"))
        for mode in [EditMode.group, .member] {
            var expected: [URL] = []
            let runner = try await runFixture(#"""
import json, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
print('REQUEST_FILE:' + request_path, flush=True)
with open(pathlib.Path(__file__).with_name('fixture-outputs.json')) as source:
    files = json.load(source)
print(json.dumps({'type': 'result', 'outputs': files}), flush=True)
"""#, mode: mode, sourceMode: .automatic, prepareOutputs: { directory in
                expected = try mode == .group
                    ? fixtures.makeCompletedAutomaticGroupOutputSet(in: directory, timestamp: "20260908-120002", song: "XOXZ")
                    : fixtures.makeCompletedAutomaticMemberOutputSet(in: directory, timestamp: "20260908-120002", song: "XOXZ")
                let extra = directory.appendingPathComponent("unreviewed-preview.mp4")
                try Data([1]).write(to: extra)
                return expected + [extra]
            })
            XCTAssertEqual(runner.state, .completed)
            XCTAssertEqual(runner.outputURLs, expected.map(\.standardizedFileURL))
            XCTAssertEqual(runner.renderedVideoURL, expected.last?.standardizedFileURL)
        }
    }

    @MainActor
    func testAutomaticReviewResultRetainsDraftStateAcrossTheRealProcessPipe() async throws {
        let fixtures = EngineContractTests(selector: NSSelectorFromString("testReviewReadyGroupAndMemberArePlayableButNeverApproved"))
        for mode in [EditMode.group, .member] {
            let runner = try await runFixture(#"""
import json, pathlib, sys
request_path = sys.argv[sys.argv.index('--request') + 1]
print('REQUEST_FILE:' + request_path, flush=True)
files = json.loads(pathlib.Path(__file__).with_name('fixture-outputs.json').read_text())
analysis = json.loads(pathlib.Path(files[0]).read_text())
print(json.dumps({'type': 'result', 'outputs': files, 'artifact_result': analysis['post_render_qa']['review_first_disposition']}), flush=True)
"""#, mode: mode, sourceMode: .automatic, prepareOutputs: { directory in
                try fixtures.makeReviewReadyOutputSet(in: directory, mode: mode)
            })
            XCTAssertEqual(runner.state, .reviewReady)
            XCTAssertNotNil(runner.renderedVideoURL)
            XCTAssertTrue(runner.statusMessage.contains("최종 승인 아님"))
        }
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
