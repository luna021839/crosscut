import XCTest
@testable import CrossCutMac

final class EngineProgressTests: XCTestCase {
    private func event(_ json: String) throws -> EngineEvent {
        try JSONDecoder().decode(EngineEvent.self, from: Data(json.utf8))
    }

    func testSelectionStageShowsCountsWithoutOverallCompletion() throws {
        let e = try event(#"{"type":"joint_visual_selection_progress","contract":"joint_selection_progress_v1","phase":"profile_complete","completed_boundary_layers":119,"total_boundary_layers":119,"profile_index":2,"total_profiles":3}"#)
        XCTAssertEqual(e.workStatus?.message, "컷 후보 조합 분석 · 설정 2/3 · 경계 119/119")
        XCTAssertNil(e.progress)
        XCTAssertTrue(e.outputPaths.isEmpty)
        XCTAssertEqual(e.workStatus?.shouldLog, true)
    }

    func testProofProgressDecodesBoundaryAndBridgeCompletion() throws {
        let e = try event(#"{"type":"joint_visual_selection_progress","contract":"joint_selection_progress_v1","phase":"proof_boundary_complete","proof_candidate_ordinal":2,"proof_candidate_pool_count":24,"completed_proof_boundary_count":43,"total_proof_boundary_count":119,"completed_proof_bridge_count":0,"total_proof_bridge_count":118,"proof_failure_code":null}"#)
        XCTAssertEqual(e.workStatus?.message, "후보 연결 검증 2/24 · 컷 43/119 · 구간 0/118")
        XCTAssertEqual(e.workStatus?.shouldLog, false)
        XCTAssertEqual(e.workStatus?.isWarning, false)
    }

    func testFailedProofReviewIsWarning() throws {
        let e = try event(#"{"type":"joint_visual_selection_progress","contract":"joint_selection_progress_v1","phase":"proof_review_complete","proof_candidate_ordinal":2,"proof_candidate_pool_count":24,"completed_proof_boundary_count":119,"total_proof_boundary_count":119,"completed_proof_bridge_count":118,"total_proof_bridge_count":118,"proof_failure_code":"authoritative_review_rejected"}"#)
        XCTAssertEqual(e.workStatus?.isWarning, true)
        XCTAssertEqual(e.workStatus?.shouldLog, true)
        XCTAssertTrue(e.workStatus?.message.contains("탈락") == true)
    }

    func testTerminalCandidateIsPreflightOnly() throws {
        let e = try event(#"{"type":"joint_candidate_terminal","contract":"joint_candidate_terminal_v1","candidate_index":0,"status":"decoded_review_passed","elapsed_seconds":41.25,"decoded_pass_count":8,"final_media_authority":false}"#)
        XCTAssertTrue(e.workStatus?.message.contains("후보 1 사전 영상 QA 통과") == true)
        XCTAssertTrue(e.workStatus?.message.contains("사전 통과 8개") == true)
        XCTAssertEqual(e.workStatus?.isWarning, false)
        XCTAssertNil(e.progress)
        XCTAssertTrue(e.outputPaths.isEmpty)
    }

    func testImpossibleCountsAndUnknownContractsAreRejected() {
        for json in [
            #"{"type":"joint_visual_selection_progress","contract":"joint_selection_progress_v1","phase":"profile_complete","completed_boundary_layers":120,"total_boundary_layers":119,"profile_index":1,"total_profiles":3}"#,
            #"{"type":"joint_candidate_terminal","contract":"joint_candidate_terminal_v1","candidate_index":-1,"status":"decoded_review_passed","elapsed_seconds":1,"decoded_pass_count":8}"#,
            #"{"type":"joint_candidate_terminal","contract":"unknown","candidate_index":0,"status":"decoded_review_passed","elapsed_seconds":1,"decoded_pass_count":8}"#,
        ] {
            XCTAssertThrowsError(try event(json))
        }
    }

    func testNumericStringProgressAndNonfiniteInput() throws {
        XCTAssertEqual(try event(#"{"type":"progress","percent":"42"}"#).progress, 0.42)
        XCTAssertNil(try event(#"{"type":"progress","percent":"nan"}"#).progress)
        XCTAssertNil(try event(#"{"type":"progress","percent":"inf"}"#).progress)
    }

    @MainActor
    func testPendingLookRemainsInProgressAndVisible() throws {
        let runner = EngineRunner(locateEngine: { nil })
        let e = try event(#"{"type":"joint_candidate_look_pending","contract":"joint_candidate_look_pending_v1","candidate_index":12,"pending_candidate_count":13,"elapsed_seconds":4}"#)
        runner.handle(e)
        XCTAssertEqual(runner.statusMessage, "후보 13 · 두 번째 색감 보정 대기 · 대기 13개")
        XCTAssertEqual(runner.state, .idle)
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertEqual(e.workStatus?.isWarning, false)
        XCTAssertNil(e.progress)
    }

    @MainActor
    func testJointSearchDoesNotPresentStaleNinetyEightPercentAsOverallCompletion() throws {
        let runner = EngineRunner(locateEngine: { nil })
        runner.handle(try event(#"{"type":"progress","value":0.9824}"#))
        XCTAssertFalse(runner.progressIsIndeterminate)
        runner.handle(try event(#"{"type":"joint_visual_selection_progress","contract":"joint_selection_progress_v1","phase":"profile_start","completed_boundary_layers":0,"total_boundary_layers":119,"profile_index":1,"total_profiles":3}"#))
        XCTAssertTrue(runner.progressIsIndeterminate)
        XCTAssertEqual(runner.progress, 0.9824)
        XCTAssertTrue(runner.outputURLs.isEmpty)
        runner.handle(try event(#"{"type":"progress","value":0.99,"message":"최종 렌더 진행"}"#))
        XCTAssertFalse(runner.progressIsIndeterminate)
        XCTAssertEqual(runner.progress, 0.99)
    }

    @MainActor
    func testRunnerDisplaysWorkWithoutExposingOutputsOrChangingRunState() throws {
        let runner = EngineRunner(locateEngine: { nil })
        let stateBefore = runner.state
        let progressBefore = runner.progress
        let e = try event(#"{"type":"joint_candidate_terminal","contract":"joint_candidate_terminal_v1","candidate_index":0,"status":"decoded_review_passed","elapsed_seconds":41.25,"decoded_pass_count":8,"output":"/tmp/unreviewed.mp4"}"#)
        runner.handle(e)
        XCTAssertEqual(runner.statusMessage, e.workStatus?.message)
        XCTAssertEqual(runner.state, stateBefore)
        XCTAssertEqual(runner.progress, progressBefore)
        XCTAssertTrue(runner.outputURLs.isEmpty)
        XCTAssertEqual(runner.logs.last?.message, e.workStatus?.message)
    }
}
