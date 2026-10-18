import Combine
import CryptoKit
import Darwin
import Foundation

enum EngineProcessTree {
    private static let defaultForceKillDelayNanoseconds: UInt64 = 2_000_000_000

    @discardableResult
    static func requestTermination(
        _ process: Process,
        forceKillDelayNanoseconds: UInt64 = defaultForceKillDelayNanoseconds
    ) -> Bool {
        guard process.isRunning else { return true }
        let processID = process.processIdentifier
        guard processID > 0 else {
            process.terminate()
            return false
        }

        let processGroupID = getpgid(processID)
        guard processGroupID == processID else {
            process.terminate()
            return false
        }
        guard Darwin.kill(-processGroupID, SIGTERM) == 0 else {
            process.terminate()
            return false
        }

        Task.detached(priority: .utility) { [process] in
            try? await Task.sleep(nanoseconds: forceKillDelayNanoseconds)
            // Only force-kill while the exact Process object we signalled is
            // still the live group leader. This keeps the delayed destructive
            // signal from ever targeting a numerically reused process group.
            if process.isRunning,
               process.processIdentifier == processID,
               getpgid(processID) == processGroupID,
               Darwin.kill(-processGroupID, 0) == 0 {
                _ = Darwin.kill(-processGroupID, SIGKILL)
            }
        }
        return true
    }

    @discardableResult
    static func forceTermination(_ process: Process) -> Bool {
        guard process.isRunning else { return true }
        let processID = process.processIdentifier
        guard processID > 0, getpgid(processID) == processID else {
            return false
        }
        return Darwin.kill(-processID, SIGKILL) == 0
    }
}

enum EngineRunState: Equatable {
    case idle
    case preparing
    case running
    case completed
    case reviewReady
    case failed(String)
    case cancelled

    var isRunning: Bool {
        switch self {
        case .preparing, .running: true
        default: false
        }
    }
}

private enum EngineOutputReadError: Error, LocalizedError {
    case lineTooLong
    case invalidUTF8
    case backlogExceeded

    var errorDescription: String? {
        switch self {
        case .lineTooLong: "엔진 출력 한 줄이 허용 크기를 초과했습니다."
        case .invalidUTF8: "엔진 출력에 올바르지 않은 UTF-8 데이터가 있습니다."
        case .backlogExceeded: "엔진 출력 대기열이 허용 크기를 초과했습니다."
        }
    }
}

/// Decode only complete lines, so a multibyte character may cross any chunk.
/// A final unterminated line is delivered at EOF, not discarded.
struct EngineOutputLineBuffer {
    private var pending = Data()
    private let maximumLineBytes: Int

    init(maximumLineBytes: Int = 8 * 1_024 * 1_024) {
        precondition(maximumLineBytes > 0)
        self.maximumLineBytes = maximumLineBytes
    }

    mutating func append(_ data: Data) {
        pending.append(data)
    }

    mutating func nextLine() throws -> String? {
        if let newline = pending.firstIndex(of: 0x0a) {
            let line = try decode(pending[..<newline])
            pending.removeSubrange(pending.startIndex...newline)
            return line
        }
        guard pending.count <= maximumLineBytes else { throw EngineOutputReadError.lineTooLong }
        return nil
    }

    mutating func finish() throws -> [String] {
        guard !pending.isEmpty else { return [] }
        let line = try decode(pending)
        pending.removeAll()
        return [line]
    }

    private func decode(_ bytes: Data) throws -> String {
        guard bytes.count <= maximumLineBytes else { throw EngineOutputReadError.lineTooLong }
        guard let line = String(data: bytes, encoding: .utf8) else { throw EngineOutputReadError.invalidUTF8 }
        return line
    }
}

private final class EnginePipeReadState: @unchecked Sendable {
    enum ReadResult { case data(Data), retry, end }

    let handle: FileHandle
    private let lock = NSLock()
    private var closed = false

    init(_ handle: FileHandle) throws {
        self.handle = handle
        let descriptor = handle.fileDescriptor
        let flags = fcntl(descriptor, F_GETFL)
        guard flags >= 0, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
    }

    func readAvailable() throws -> ReadResult {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return .end }
        var bytes = Data(count: 64 * 1_024)
        let count = bytes.withUnsafeMutableBytes {
            Darwin.read(handle.fileDescriptor, $0.baseAddress!, $0.count)
        }
        if count < 0 {
            let code = errno
            if code == EINTR || code == EAGAIN { return .retry }
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(code))
        }
        if count == 0 { return .end }
        bytes.removeSubrange(count..<bytes.count)
        return .data(bytes)
    }

    func close() {
        handle.readabilityHandler = nil
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        try? handle.close()
    }
}

private func enginePipeChunks(from handle: FileHandle) -> AsyncThrowingStream<Data, Error> {
    AsyncThrowingStream(bufferingPolicy: .bufferingOldest(128)) { continuation in
        let reader: EnginePipeReadState
        do { reader = try EnginePipeReadState(handle) }
        catch {
            try? handle.close()
            continuation.finish(throwing: error)
            return
        }
        continuation.onTermination = { _ in reader.close() }
        handle.readabilityHandler = { _ in
            do {
                switch try reader.readAvailable() {
                case .data(let data):
                    if case .dropped = continuation.yield(data) {
                        continuation.finish(throwing: EngineOutputReadError.backlogExceeded)
                    }
                case .retry: break
                case .end: continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}

private final class EngineOutputDrainState: @unchecked Sendable {
    private let lock = NSLock()
    private var remainingStreams = 2
    private var expired = false

    func streamFinished() {
        lock.lock()
        defer { lock.unlock() }
        remainingStreams -= 1
    }

    func expireIfPending() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard remainingStreams > 0 else { return false }
        expired = true
        return true
    }

    var timedOut: Bool {
        lock.lock()
        defer { lock.unlock() }
        return expired
    }
}

struct EngineLogEntry: Identifiable, Equatable {
    enum Level: String {
        case info
        case warning
        case error
        case engine
    }

    let id = UUID()
    let timestamp: Date
    let level: Level
    let message: String
}

enum EnginePreparationResult: Equatable, Sendable {
    case ready(URL)
    case pythonUnavailable
    case localGroupUnavailable
    case localMemberTrackingUnavailable
    case automaticAcquisitionUnavailable
}

typealias EnginePreparationProbe = @Sendable (
    _ engineURL: URL,
    _ sourceMode: String,
    _ editMode: String
) -> EnginePreparationResult

private let defaultEnginePreparationProbe: EnginePreparationProbe = {
    engineURL, sourceMode, editMode in
    guard let pythonURL = EngineLocator.locatePython(near: engineURL) else {
        return .pythonUnavailable
    }
    let mode: EditMode = editMode == EditMode.member.rawValue ? .member : .group
    if sourceMode == SourceMode.automatic.rawValue {
        guard EngineLocator.supportsAutomaticAcquisition(
            pythonURL: pythonURL,
            engineURL: engineURL,
            editMode: mode
        ) else {
            return .automaticAcquisitionUnavailable
        }
    } else {
        guard EngineLocator.supportsLocalEditing(
            pythonURL: pythonURL,
            engineURL: engineURL,
            editMode: mode
        ) else {
            return mode == .member
                ? .localMemberTrackingUnavailable
                : .localGroupUnavailable
        }
    }
    return .ready(pythonURL)
}

enum CompletedEngineOutputMode: String, Equatable, Sendable {
    case group
    case member

    init(_ editMode: EditMode) {
        self = editMode == .member ? .member : .group
    }
}

struct CompletedEngineOutputSet: Equatable, Sendable {
    let editMode: CompletedEngineOutputMode
    let timestamp: String
    let outputURLs: [URL]
    let artifactRole: String
}

enum RenderedVideoOutputDiscovery {
    private static let videoExtensions = Set(["mp4", "mov", "m4v"])

    static func firstValidVideo(in urls: [URL]) -> URL? {
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
        ]
        return urls.lazy
            .map(\.standardizedFileURL)
            .first { url in
                guard videoExtensions.contains(url.pathExtension.lowercased()),
                      let values = try? url.resourceValues(forKeys: keys)
                else {
                    return false
                }
                return values.isRegularFile == true
                    && values.isSymbolicLink != true
                    && (values.fileSize ?? 0) > 0
            }
    }
}

enum CompletedEngineOutputSetDiscovery {
    private static let selectionAnalysisBasis =
        "sequential_decoded_proxy_candidate_rerank_v1"
    private static let proxyAnalysisBasisV2 =
        "decoded_output_proxy_all_transition_quality_review_v2"
    private static let proxyAnalysisBasisV3 =
        "decoded_output_proxy_domain_aware_quality_review_v3"
    private static let automaticIVEGroupAppResultExposureContract =
        "automatic_ive_group_app_result_exposure_v1"
    private static let automaticIVEGroupPostRenderQAContract =
        "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
    private static let automaticIVEMemberAppResultExposureContract =
        "automatic_ive_member_app_result_exposure_v1"
    private static let automaticIVEMemberPostRenderQAContract =
        "automatic_ive_member_mandatory_post_render_qa_v1"
    private static let automaticIVEMemberPostRenderQAAnalysis =
        "automatic_ive_member_final_media_and_pixel_provenance_qa_v1"
    private static let automaticIVEMemberRenderReceiptContract =
        "automatic_ive_member_exact_source_time_pixel_provenance_receipt_v1"

    private struct ProxyMetadata: Decodable, Equatable {
        let path: String?
        let temporaryProxyRemovedAfterReview: Bool

        enum CodingKeys: String, CodingKey {
            case path
            case temporaryProxyRemovedAfterReview =
                "temporary_proxy_removed_after_review"
        }
    }

    private struct ProxyReview: Decodable, Equatable {
        let schemaVersion: Int
        let analysis: String
        let proxy: ProxyMetadata
        let transitionCount: Int
        let reviewedBoundaryCount: Int
        let hardCutCount: Int
        let dissolveCount: Int
        let passed: Bool

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case analysis
            case proxy
            case transitionCount = "transition_count"
            case reviewedBoundaryCount = "reviewed_boundary_count"
            case hardCutCount = "hard_cut_count"
            case dissolveCount = "dissolve_count"
            case passed
        }

        var isCompleteAndSanitized: Bool {
            let schemaAndAnalysisAreCompatible =
                (schemaVersion == 2
                    && analysis
                        == CompletedEngineOutputSetDiscovery.proxyAnalysisBasisV2)
                || (schemaVersion == 3
                    && analysis
                        == CompletedEngineOutputSetDiscovery.proxyAnalysisBasisV3)
            return schemaAndAnalysisAreCompatible
                && proxy.path == nil
                && proxy.temporaryProxyRemovedAfterReview
                && transitionCount >= 0
                && reviewedBoundaryCount == transitionCount
                && hardCutCount >= 0
                && dissolveCount >= 0
                && hardCutCount + dissolveCount == transitionCount
        }
    }

    private struct CandidateReview: Decodable, Equatable {
        let candidateIndex: Int
        let selected: Bool
        let error: String?
        let review: ProxyReview?

        enum CodingKeys: String, CodingKey {
            case candidateIndex = "candidate_index"
            case selected
            case error
            case review
        }
    }

    private struct StoragePolicy: Decodable, Equatable {
        let oneProxyAtATime: Bool
        let temporaryProxyRemovedAfterEachReview: Bool
        let decodedFramesPersisted: Bool

        enum CodingKeys: String, CodingKey {
            case oneProxyAtATime = "one_proxy_at_a_time"
            case temporaryProxyRemovedAfterEachReview =
                "temporary_proxy_removed_after_each_review"
            case decodedFramesPersisted = "decoded_frames_persisted"
        }
    }

    private struct CandidateSelection: Decodable, Equatable {
        let analysisBasis: String
        let storagePolicy: StoragePolicy
        let requestedReviewCount: Int
        let reviewedCandidateCount: Int
        let minimumRequiredCandidateCount: Int
        let minimumCandidateCountSatisfied: Bool
        let successfulReviewCount: Int
        let selectedCandidateIndex: Int
        let passed: Bool
        let candidates: [CandidateReview]
        let completeCandidateCount: Int
        let structurePassingMaterializedCount: Int?
        let releasePolicy: ReviewFirstReleasePolicy?
        let candidatePool: String?
        let selectedPathAuthority: String?
        let completeCandidates: [FamilyCompleteCandidate]?
        let exactSuccessfulDecodedReviewCountSatisfied: Bool
        let exactCompleteCandidateCountSatisfied: Bool
        let finalSelectedReview: ProxyReview
        let finalSelectedVersion: String

        enum CodingKeys: String, CodingKey {
            case analysisBasis = "analysis_basis"
            case storagePolicy = "storage_policy"
            case requestedReviewCount = "requested_review_count"
            case reviewedCandidateCount = "reviewed_candidate_count"
            case minimumRequiredCandidateCount =
                "minimum_required_candidate_count"
            case minimumCandidateCountSatisfied =
                "minimum_candidate_count_satisfied"
            case successfulReviewCount = "successful_review_count"
            case selectedCandidateIndex = "selected_candidate_index"
            case passed
            case candidates
            case completeCandidateCount = "complete_candidate_count"
            case structurePassingMaterializedCount =
                "structure_passing_materialized_count"
            case exactSuccessfulDecodedReviewCountSatisfied =
                "exact_successful_decoded_review_count_satisfied"
            case exactCompleteCandidateCountSatisfied =
                "exact_complete_candidate_count_satisfied"
            case finalSelectedReview = "final_selected_review"
            case finalSelectedVersion = "final_selected_version"
            case releasePolicy = "release_policy"
            case candidatePool = "candidate_pool"
            case selectedPathAuthority = "selected_path_authority"
            case completeCandidates = "complete_candidates"
        }

        func isComplete(allowQualityFailure: Bool, renderedFamilyReceipt: PerformerFamilyReceipt? = nil) -> Bool {
            let indices = candidates.map(\.candidateIndex)
            let selected = candidates.filter(\.selected)
            let singleton = releasePolicy?.isReviewFirst == true
                && candidatePool == "bounded_review_first_single_full_song_v1"
                && (["stage_edit_generation", "stage_edit_candidate_reranker", "deterministic_full_song_visual_selector"].contains(selectedPathAuthority ?? "")
                    || (renderedFamilyReceipt?.isBoundReviewEvidence(authority: selectedPathAuthority) == true
                        && completeCandidates?.count == 1
                        && completeCandidates?.first?.candidateOrigin == selectedPathAuthority
                        && completeCandidates?.first?.performerFamilyReview == renderedFamilyReceipt))
            let qualitySingle = releasePolicy?.isQualitySingle == true
            let count = singleton || qualitySingle ? 1 : 8
            return analysisBasis == (singleton ? "bounded_review_first_single_full_song_v1" : CompletedEngineOutputSetDiscovery.selectionAnalysisBasis)
                && storagePolicy.oneProxyAtATime
                && storagePolicy.temporaryProxyRemovedAfterEachReview
                && !storagePolicy.decodedFramesPersisted
                && requestedReviewCount == count
                && reviewedCandidateCount == count
                && minimumRequiredCandidateCount == count
                && minimumCandidateCountSatisfied
                && successfulReviewCount == count
                && completeCandidateCount == count
                && (singleton || (qualitySingle ? (structurePassingMaterializedCount ?? 0) >= 1 : structurePassingMaterializedCount == count))
                && exactSuccessfulDecodedReviewCountSatisfied
                && exactCompleteCandidateCountSatisfied
                && (passed || (singleton && allowQualityFailure))
                && passed == finalSelectedReview.passed
                && candidates.count == count
                && Set(indices).count == count && indices.allSatisfy { $0 >= 0 }
                && (!singleton || indices == [0])
                && selected.count == 1
                && selected[0].candidateIndex == selectedCandidateIndex
                && candidates.allSatisfy {
                    $0.error == nil
                        && ($0.review?.isCompleteAndSanitized == true)
                }
                && (finalSelectedReview.passed || (singleton && allowQualityFailure))
                && finalSelectedReview.isCompleteAndSanitized
                && ["original", "partial_reedit", "windowed_semimarkov_redecode"].contains(finalSelectedVersion)
        }
    }

    private struct PerformerFamilyReceipt: Decodable, Equatable {
        let contract: String
        let planSHA256: String
        let metadataSHA256: String
        let qualityApproved: Bool
        let crossCameraIdentityCertified: Bool
        let bodyPhaseCertified: Bool
        let decodedOutputQualityCertified: Bool

        enum CodingKeys: String, CodingKey {
            case contract
            case planSHA256 = "plan_sha256"
            case metadataSHA256 = "metadata_sha256"
            case qualityApproved = "quality_approved"
            case crossCameraIdentityCertified = "cross_camera_identity_certified"
            case bodyPhaseCertified = "body_phase_certified"
            case decodedOutputQualityCertified = "decoded_output_quality_certified"
        }

        func isBoundReviewEvidence(authority: String?) -> Bool {
            let owners = ["normal_performer_family_camera_owner_v1",
                "normal_source_changing_family_owner_v1", "selected_native_motion_complete_metadata_v1"]
            return authority == contract && owners.contains(contract)
                && CompletedEngineOutputSetDiscovery.isSHA256(planSHA256)
                && CompletedEngineOutputSetDiscovery.isSHA256(metadataSHA256)
                && !qualityApproved && !crossCameraIdentityCertified
                && !bodyPhaseCertified && !decodedOutputQualityCertified
        }
    }

    private struct FamilyCompleteCandidate: Decodable, Equatable {
        let candidateOrigin: String?
        let performerFamilyReview: PerformerFamilyReceipt?
        enum CodingKeys: String, CodingKey {
            case candidateOrigin = "candidate_origin"
            case performerFamilyReview = "performer_family_review"
        }
    }

    private struct ReviewFirstReleasePolicy: Decodable, Equatable {
        let contract: String
        let requiredCompleteCandidateCount: Int?
        let requiredDecodedReviewCount: Int?
        let candidatePool: String?
        let selectionMode: String?
        let sourceCoverageRequirementsChanged: Bool
        let individualCandidateQualityRequirementsChanged: Bool
        let silentUnreviewedFallbackAllowed: Bool?
        let minimumPassingCandidateCount: Int?
        let comparisonCandidateBudget: Int?
        let minimumVisualPassingLearnedCandidateCount: Int?
        let comparisonSelection: String?
        let silentBaselineFallbackAllowed: Bool?
        enum CodingKeys: String, CodingKey {
            case contract
            case requiredCompleteCandidateCount = "required_complete_candidate_count"
            case requiredDecodedReviewCount = "required_decoded_review_count"
            case candidatePool = "candidate_pool"
            case selectionMode = "selection_mode"
            case sourceCoverageRequirementsChanged = "source_coverage_requirements_changed"
            case individualCandidateQualityRequirementsChanged = "individual_candidate_quality_requirements_changed"
            case silentUnreviewedFallbackAllowed = "silent_unreviewed_fallback_allowed"
            case minimumPassingCandidateCount = "minimum_passing_candidate_count"
            case comparisonCandidateBudget = "comparison_candidate_budget"
            case minimumVisualPassingLearnedCandidateCount = "minimum_visual_passing_learned_candidate_count"
            case comparisonSelection = "comparison_selection"
            case silentBaselineFallbackAllowed = "silent_baseline_fallback_allowed"
        }
        var isReviewFirst: Bool {
            contract == "bounded_review_first_single_delivery_v1"
                && requiredCompleteCandidateCount == 1 && requiredDecodedReviewCount == 1
                && candidatePool == "bounded_review_first_single_full_song_v1"
                && selectionMode == "one_complete_full_song_candidate_then_final_qa"
                && !sourceCoverageRequirementsChanged && !individualCandidateQualityRequirementsChanged
                && silentUnreviewedFallbackAllowed == false
        }
        var isQualitySingle: Bool {
            guard minimumPassingCandidateCount == 1, comparisonCandidateBudget == 8,
                  !sourceCoverageRequirementsChanged, !individualCandidateQualityRequirementsChanged else { return false }
            return contract == "joint_quality_gated_single_delivery_v1"
                || (contract == "stage_edit_quality_gated_single_delivery_v1"
                    && minimumVisualPassingLearnedCandidateCount == 1
                    && comparisonSelection == "rank_all_available_decoded_passing_candidates"
                    && silentBaselineFallbackAllowed == false)
        }
    }

    private struct AnalysisRequest: Decodable {
        let group: String
        let song: String?
        let editMode: String
        let sourceMode: String
        let outputDirectory: String
        let aspectRatio: String?
        let targetMember: String?

        enum CodingKeys: String, CodingKey {
            case group
            case song
            case editMode = "edit_mode"
            case sourceMode = "source_mode"
            case outputDirectory = "output_directory"
            case aspectRatio = "aspect_ratio"
            case targetMember = "target_member"
        }
    }

    private struct AppResultExposure: Decodable, Equatable {
        let contract: String
        let videoOutputExposed: Bool
        let deliveryCompletionMessageAllowed: Bool

        enum CodingKeys: String, CodingKey {
            case contract
            case videoOutputExposed = "video_output_exposed"
            case deliveryCompletionMessageAllowed =
                "delivery_completion_message_allowed"
        }

        func authorizes(role: String, for editMode: CompletedEngineOutputMode) -> Bool {
            let expectedContract = switch editMode {
            case .group:
                CompletedEngineOutputSetDiscovery
                    .automaticIVEGroupAppResultExposureContract
            case .member:
                CompletedEngineOutputSetDiscovery
                    .automaticIVEMemberAppResultExposureContract
            }
            return contract == expectedContract
                && videoOutputExposed
                && ["delivery", "review_ready"].contains(role)
                && deliveryCompletionMessageAllowed == (role == "delivery")
        }
    }

    private struct PostRenderQAEvidence: Decodable, Equatable {
        let contract: String
        let required: Bool
        let verdict: String
        let passed: Bool
        let reportPath: String?
        let reportPassed: Bool?
        let contentBindingPassed: Bool?
        let reviewFirstDisposition: EngineArtifactDisposition?
        let promotionBinding: PromotionBinding?

        struct PromotionBinding: Decodable, Equatable { let passed: Bool }

        enum CodingKeys: String, CodingKey {
            case contract
            case required
            case verdict
            case passed
            case reportPath = "report_path"
            case reportPassed = "report_passed"
            case contentBindingPassed = "content_binding_passed"
            case reviewFirstDisposition = "review_first_disposition"
            case promotionBinding = "promotion_binding"
        }

        func authorizes(role: String, for editMode: CompletedEngineOutputMode) -> Bool {
            let expectedContract = switch editMode {
            case .group:
                CompletedEngineOutputSetDiscovery
                    .automaticIVEGroupPostRenderQAContract
            case .member:
                CompletedEngineOutputSetDiscovery
                    .automaticIVEMemberPostRenderQAContract
            }
            guard contract == expectedContract, required else { return false }
            if let disposition = reviewFirstDisposition {
                guard disposition.authorizes(role: role),
                      (editMode == .member ? contentBindingPassed == true : promotionBinding?.passed == true)
                else { return false }
            } else if role != "delivery" { return false }
            if role == "review_ready" {
                return !passed && ["FAIL", "CONDITIONAL"].contains(verdict)
                    && (editMode != .member || reportPassed == false)
            }
            return role == "delivery" && verdict == "PASS" && passed
                && (editMode != .member || (reportPassed == true && contentBindingPassed == true))
        }
    }

    private struct AnalysisCapabilities: Decodable {
        let groupVideoRendered: Bool?
        let memberVideoRendered: Bool?
        let memberDirectFullFrameReady: Bool?

        enum CodingKeys: String, CodingKey {
            case groupVideoRendered = "group_video_rendered"
            case memberVideoRendered = "member_video_rendered"
            case memberDirectFullFrameReady = "member_direct_full_frame_ready"
        }
    }

    private struct Interchange: Decodable {
        let fcpxmlPath: String
        let otioPath: String
        let edlPath: String
        let fidelityAuditPath: String
        let authoritativePixels: String

        enum CodingKeys: String, CodingKey {
            case fcpxmlPath = "fcpxml_path"
            case otioPath = "otio_path"
            case edlPath = "edl_path"
            case fidelityAuditPath = "fidelity_audit_path"
            case authoritativePixels = "authoritative_pixels"
        }
    }

    private struct GroupAnalysisRender: Decodable {
        let videoPath: String
        let planPath: String
        let videoCodec: String
        let interchange: Interchange
        let renderedProxyReview: ProxyReview
        let renderedCandidateSelection: CandidateSelection
        let artifactRole: String
        let appResultExposure: AppResultExposure
        let postRenderQA: PostRenderQAEvidence
        let releasePolicy: ReviewFirstReleasePolicy?
        let candidatePool: String?
        let candidateOrigin: String?
        let performerFamilyReview: PerformerFamilyReceipt?

        enum CodingKeys: String, CodingKey {
            case videoPath = "video_path"
            case planPath = "plan_path"
            case videoCodec = "video_codec"
            case interchange
            case renderedProxyReview = "rendered_proxy_review"
            case renderedCandidateSelection = "rendered_candidate_selection"
            case artifactRole = "artifact_role"
            case appResultExposure = "app_result_exposure"
            case postRenderQA = "post_render_qa"
            case releasePolicy = "release_policy"
            case candidatePool = "candidate_pool"
            case candidateOrigin = "candidate_origin"
            case performerFamilyReview = "performer_family_review"
        }
    }

    private struct MemberAnalysisRender: Decodable {
        let videoPath: String
        let planPath: String
        let videoCodec: String
        let interchange: Interchange
        let directFullFrame: Bool
        let artifactRole: String
        let appResultExposure: AppResultExposure
        let postRenderQA: PostRenderQAEvidence

        enum CodingKeys: String, CodingKey {
            case videoPath = "video_path"
            case planPath = "plan_path"
            case videoCodec = "video_codec"
            case interchange
            case directFullFrame = "direct_full_frame"
            case artifactRole = "artifact_role"
            case appResultExposure = "app_result_exposure"
            case postRenderQA = "post_render_qa"
        }
    }

    private struct InterchangeAudit: Decodable {
        struct Artifact: Decodable {
            let sha256: String
            let sizeBytes: Int

            enum CodingKeys: String, CodingKey {
                case sha256
                case sizeBytes = "size_bytes"
            }
        }

        let schemaVersion: Int
        let analysisBasis: String
        let authoritativeOutput: String
        let outputFrameCount: Int
        let contentSHA256: String
        let artifacts: [String: Artifact]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case analysisBasis = "analysis_basis"
            case authoritativeOutput = "authoritative_output"
            case outputFrameCount = "output_frame_count"
            case contentSHA256 = "content_sha256"
            case artifacts
        }
    }

    private struct GroupAnalysisManifest: Decodable {
        let schemaVersion: Int
        let createdAt: String
        let request: AnalysisRequest
        let renderedProxyReview: ProxyReview
        let renderedCandidateSelection: CandidateSelection
        let render: GroupAnalysisRender
        let capabilities: AnalysisCapabilities
        let artifactRole: String
        let appResultExposure: AppResultExposure
        let postRenderQA: PostRenderQAEvidence

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case createdAt = "created_at"
            case request
            case renderedProxyReview = "rendered_proxy_review"
            case renderedCandidateSelection = "rendered_candidate_selection"
            case render
            case capabilities
            case artifactRole = "artifact_role"
            case appResultExposure = "app_result_exposure"
            case postRenderQA = "post_render_qa"
        }
    }

    private struct MemberDirectFullFrame: Decodable {
        let enabled: Bool
    }

    private struct MemberAnalysisManifest: Decodable {
        let schemaVersion: Int
        let createdAt: String
        let request: AnalysisRequest
        let memberDirectFullFrame: MemberDirectFullFrame
        let render: MemberAnalysisRender
        let capabilities: AnalysisCapabilities
        let artifactRole: String
        let appResultExposure: AppResultExposure
        let postRenderQA: PostRenderQAEvidence

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case createdAt = "created_at"
            case request
            case memberDirectFullFrame = "member_direct_full_frame"
            case render
            case capabilities
            case artifactRole = "artifact_role"
            case appResultExposure = "app_result_exposure"
            case postRenderQA = "post_render_qa"
        }
    }

    private struct MemberPostRenderQAReport: Decodable {
        struct ArtifactSet: Decodable {
            let analysisPath: String
            let renderPlanPath: String
            let videoPath: String

            enum CodingKeys: String, CodingKey {
                case analysisPath = "analysis_path"
                case renderPlanPath = "render_plan_path"
                case videoPath = "video_path"
            }
        }

        struct RenderReceipt: Decodable {
            let contract: String
            let allEligibleSourcesPixelProven: Bool
            let sourceFileIdentitiesStable: Bool
            let artifactContentStable: Bool
            let passed: Bool

            enum CodingKeys: String, CodingKey {
                case contract
                case allEligibleSourcesPixelProven =
                    "all_eligible_sources_pixel_proven"
                case sourceFileIdentitiesStable =
                    "source_file_identities_stable"
                case artifactContentStable = "artifact_content_stable"
                case passed
            }
        }

        let schemaVersion: Int
        let analysis: String
        let artifactSet: ArtifactSet
        let verdict: String
        let passed: Bool
        let issueCount: Int
        let renderReceipt: RenderReceipt

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case analysis
            case artifactSet = "artifact_set"
            case verdict
            case passed
            case issueCount = "issue_count"
            case renderReceipt = "render_receipt"
        }
    }

    private struct RenderPlanSummary: Decodable {
        let aspectRatio: String
        let output: String
        let width: Int
        let height: Int
        let fps: Double
        let renderDuration: Double
        let outputFrameCount: Int

        enum CodingKeys: String, CodingKey {
            case aspectRatio = "aspect_ratio"
            case output
            case width
            case height
            case fps
            case renderDuration = "render_duration"
            case outputFrameCount = "output_frame_count"
        }
    }

    /// Validate the set explicitly reported by this run. Never search the
    /// directory for a replacement result, and expose only its eight core files.
    private static func currentMemberReview(_ reported: [URL], for request: EditRequest) -> CompletedEngineOutputSet? {
        guard request.sourceMode == "automatic", request.editMode == "member",
              request.aspectRatio == "9:16", let target = request.targetMember,
              IVEMember(rawValue: target) != nil else { return nil }
        let directory = URL(fileURLWithPath: request.outputDirectory).standardizedFileURL
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        func json(_ url: URL) -> [String: Any]? {
            guard isNonemptyRegularFile(url, keys: keys),
                  let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                  size <= 32 * 1024 * 1024, let data = try? Data(contentsOf: url),
                  let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
            return value
        }
        func digest(_ url: URL) -> String? {
            guard isNonemptyRegularFile(url, keys: keys), let handle = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? handle.close() }
            var hash = SHA256()
            do { while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) } }
            catch { return nil }
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        let manifests = reported.filter { $0.lastPathComponent.hasSuffix("-analysis.json")
            && samePath($0.deletingLastPathComponent().path, directory) }
        guard manifests.count == 1, let manifestURL = manifests.first,
              let value = json(manifestURL), value["contract"] as? String == "current_member_programme_review_v1",
              value["artifact_role"] as? String == "review_ready", value["professional_quality_approved"] as? Bool == false,
              let configuration = value["request"] as? [String: Any], configuration["group"] as? String == "IVE",
              configuration["song"] as? String == request.song, configuration["target_member"] as? String == target,
              configuration["edit_mode"] as? String == "member", configuration["source_mode"] as? String == "automatic",
              configuration["aspect_ratio"] as? String == "9:16",
              configuration["output_directory"] as? String == request.outputDirectory,
              configuration["source_cache_directory"] as? String == request.sourceCacheDirectory,
              configuration["member_pose_dissolve_enabled"] as? Bool == request.memberPoseDissolveEnabled,
              let sourcePaths = configuration["sources"] as? [String], !sourcePaths.isEmpty,
              request.sources.isEmpty || request.sources == sourcePaths,
              let exposure = value["app_result_exposure"] as? [String: Any],
              exposure["video_output_exposed"] as? Bool == true, exposure["delivery_completion_message_allowed"] as? Bool == false,
              let qa = value["post_render_qa"] as? [String: Any], let state = qa["review_first_disposition"],
              let stateData = try? JSONSerialization.data(withJSONObject: state),
              let disposition = try? JSONDecoder().decode(EngineArtifactDisposition.self, from: stateData),
              disposition.authorizes(role: "review_ready"),
              let inputPath = value["inputs"] as? String, let inputSHA = value["inputs_sha256"] as? String,
              let render = value["render"] as? [String: Any], render["artifact_role"] as? String == "review_ready",
              render["full_AV_decode_exit_code"] as? Int == 0, let videoPath = render["video_path"] as? String,
              let videoSHA = render["movie_sha256"] as? String, isSHA256(videoSHA),
              let planPath = render["plan_path"] as? String, let clockPath = render["frame_clock_artifact"] as? String else { return nil }
        let video = URL(fileURLWithPath: videoPath).standardizedFileURL
        let planURL = URL(fileURLWithPath: planPath).standardizedFileURL
        let inputURL = URL(fileURLWithPath: inputPath).standardizedFileURL
        let cache = URL(fileURLWithPath: request.sourceCacheDirectory).appendingPathComponent("crosscut2", isDirectory: true)
        guard samePath(inputURL.deletingLastPathComponent().path, cache), digest(inputURL) == inputSHA,
              let inputs = json(inputURL), inputs["contract"] as? String == "current_member_numeric_inputs_v1",
              inputs["member"] as? String == target, samePath(planURL.deletingLastPathComponent().path, directory),
              Set(reported.map(\.standardizedFileURL)) == Set([manifestURL.standardizedFileURL, planURL, video]),
              let plan = json(planURL), plan["sources"] as? [String] == sourcePaths,
              plan["output"] as? String == video.path, plan["aspect_ratio"] as? String == "9:16",
              plan["width"] as? Int == 1080, plan["height"] as? Int == 1920,
              plan["member_pose_dissolve_requested"] as? Bool == request.memberPoseDissolveEnabled,
              let frames = plan["output_frame_count"] as? Int, frames > 0,
              let clock = json(URL(fileURLWithPath: clockPath)),
              clock["artifact_contract"] as? String == "render_clock_video_plan_binding_v2",
              clock["video_content_sha256"] as? String == videoSHA,
              let clockPlan = clock["plan"] as? [String: Any], NSDictionary(dictionary: clockPlan).isEqual(to: plan),
              digest(video) == videoSHA,
              let created = value["created_at"] as? String,
              let date = ISO8601DateFormatter().date(from: created) else { return nil }
        if !samePath(video.deletingLastPathComponent().path, directory) {
            guard render["full_film_cache_reused"] as? Bool == true,
                  let movies = inputs["completed_movies"] as? [[String: Any]],
                  movies.contains(where: { ($0["path"] as? String) == video.path && ($0["sha256"] as? String) == videoSHA }) else { return nil }
        }
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0); formatter.dateFormat = "yyyyMMdd-HHmmss"
        return CompletedEngineOutputSet(editMode: .member, timestamp: formatter.string(from: date),
            outputURLs: [manifestURL, planURL, video], artifactRole: "review_ready")
    }

    static func reported(
        _ reportedURLs: [URL],
        for request: EditRequest
    ) -> CompletedEngineOutputSet? {
        if let current = currentMemberReview(reportedURLs, for: request) { return current }
        guard request.sourceMode == SourceMode.automatic.rawValue,
              let requestedMode = EditMode(rawValue: request.editMode) else {
            return nil
        }
        let mode = CompletedEngineOutputMode(requestedMode)
        let song = request.song.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !song.isEmpty else { return nil }
        let directory = URL(fileURLWithPath: request.outputDirectory).standardizedFileURL
        let urls = Set(reportedURLs.map(\.standardizedFileURL)).filter {
            $0.isFileURL && samePath($0.deletingLastPathComponent().path, directory)
        }
        let videoSuffix: String
        switch mode {
        case .group:
            videoSuffix = "-group-crosscut.mp4"
        case .member:
            guard let target = request.targetMember,
                  IVEMember(rawValue: target) != nil else { return nil }
            videoSuffix = "-\(target)-member-crosscut.mp4"
        }
        let reportedVideos = urls.filter { $0.lastPathComponent.hasSuffix(videoSuffix) }
        guard reportedVideos.count == 1, let video = reportedVideos.first else { return nil }
        let prefix = String(video.lastPathComponent.dropLast(videoSuffix.count))
        guard let timestamp = timestamp(in: prefix) else { return nil }
        let groups = Dictionary(grouping: urls, by: \.lastPathComponent)
        guard groups.values.allSatisfy({ $0.count == 1 }) else { return nil }
        let byName = groups.mapValues { $0[0] }
        guard let analysisURL = byName["\(prefix)-analysis.json"] else { return nil }
        let resourceKeys: Set<URLResourceKey> = [
            .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
        ]
        guard isNonemptyRegularFile(analysisURL, keys: resourceKeys) else { return nil }
        switch mode {
        case .group:
            guard let analysis = decode(GroupAnalysisManifest.self, from: analysisURL),
                  analysis.request.song == song,
                  analysis.request.targetMember == request.targetMember else { return nil }
            return completedGroupOutput(
                prefix: prefix, timestamp: timestamp, byName: byName,
                resourceKeys: resourceKeys, outputDirectory: directory
            )
        case .member:
            guard let analysis = decode(MemberAnalysisManifest.self, from: analysisURL),
                  analysis.request.song == song,
                  analysis.request.targetMember == request.targetMember else { return nil }
            return completedMemberOutput(
                prefix: prefix, timestamp: timestamp, analysisURL: analysisURL,
                byName: byName, resourceKeys: resourceKeys, outputDirectory: directory
            )
        }
    }

    static func latest(
        in outputDirectory: URL,
        editMode: CompletedEngineOutputMode
    ) -> CompletedEngineOutputSet? {
        let directory = outputDirectory.standardizedFileURL
        let resourceKeys: Set<URLResourceKey> = [
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
        ]
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: Array(resourceKeys),
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }
        let byName = Dictionary(
            contents.map { ($0.lastPathComponent, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let suffix = "-analysis.json"
        var completed: [CompletedEngineOutputSet] = []
        for analysisURL in contents where analysisURL.lastPathComponent.hasSuffix(suffix) {
            let filename = analysisURL.lastPathComponent
            let prefix = String(filename.dropLast(suffix.count))
            guard let timestamp = timestamp(in: prefix) else { continue }
            let result: CompletedEngineOutputSet?
            switch editMode {
            case .group:
                result = completedGroupOutput(
                    prefix: prefix,
                    timestamp: timestamp,
                    byName: byName,
                    resourceKeys: resourceKeys,
                    outputDirectory: directory
                )
            case .member:
                result = completedMemberOutput(
                    prefix: prefix,
                    timestamp: timestamp,
                    analysisURL: analysisURL,
                    byName: byName,
                    resourceKeys: resourceKeys,
                    outputDirectory: directory
                )
            }
            if let result { completed.append(result) }
        }
        return completed.max { $0.timestamp < $1.timestamp }
    }

    private static func completedGroupOutput(
        prefix: String,
        timestamp: String,
        byName: [String: URL],
        resourceKeys: Set<URLResourceKey>,
        outputDirectory: URL
    ) -> CompletedEngineOutputSet? {
        let expectedNames = [
            "\(prefix)-analysis.json",
            "\(prefix)-render-plan.json",
            "\(prefix)-rendered-candidate-selection.json",
            "\(prefix)-group-timeline.fcpxml",
            "\(prefix)-group-timeline.otio",
            "\(prefix)-group-timeline.edl",
            "\(prefix)-group-timeline-interchange-audit.json",
            "\(prefix)-group-crosscut.mp4",
        ]
        let urls = expectedNames.compactMap { byName[$0]?.standardizedFileURL }
        guard urls.count == expectedNames.count,
              urls.allSatisfy({ isNonemptyRegularFile($0, keys: resourceKeys) }),
              validateGroup(
                analysisURL: urls[0],
                planURL: urls[1],
                selectionURL: urls[2],
                fcpxmlURL: urls[3],
                otioURL: urls[4],
                edlURL: urls[5],
                interchangeAuditURL: urls[6],
                videoURL: urls[7],
                outputDirectory: outputDirectory,
                timestamp: timestamp
              ) else {
            return nil
        }
        guard let validatedAnalysis = decode(GroupAnalysisManifest.self, from: urls[0]) else { return nil }
        return CompletedEngineOutputSet(
            editMode: .group,
            timestamp: timestamp,
            outputURLs: urls,
            artifactRole: validatedAnalysis.artifactRole
        )
    }

    private static func completedMemberOutput(
        prefix: String,
        timestamp: String,
        analysisURL: URL,
        byName: [String: URL],
        resourceKeys: Set<URLResourceKey>,
        outputDirectory: URL
    ) -> CompletedEngineOutputSet? {
        guard let analysis = decode(MemberAnalysisManifest.self, from: analysisURL),
              let targetMember = analysis.request.targetMember,
              IVEMember(rawValue: targetMember) != nil else {
            return nil
        }
        let artifactPrefix = "\(prefix)-\(targetMember)"
        let expectedNames = [
            "\(prefix)-analysis.json",
            "\(artifactPrefix)-render-plan.json",
            "\(artifactPrefix)-member-post-render-qa.json",
            "\(artifactPrefix)-member-timeline.fcpxml",
            "\(artifactPrefix)-member-timeline.otio",
            "\(artifactPrefix)-member-timeline.edl",
            "\(artifactPrefix)-member-timeline-interchange-audit.json",
            "\(artifactPrefix)-member-crosscut.mp4",
        ]
        let urls = expectedNames.compactMap { byName[$0]?.standardizedFileURL }
        guard urls.count == expectedNames.count,
              urls.allSatisfy({ isNonemptyRegularFile($0, keys: resourceKeys) }),
              validateMember(
                analysis: analysis,
                analysisURL: urls[0],
                planURL: urls[1],
                postRenderQAURL: urls[2],
                fcpxmlURL: urls[3],
                otioURL: urls[4],
                edlURL: urls[5],
                interchangeAuditURL: urls[6],
                videoURL: urls[7],
                outputDirectory: outputDirectory,
                timestamp: timestamp
              ) else {
            return nil
        }
        return CompletedEngineOutputSet(
            editMode: .member,
            timestamp: timestamp,
            outputURLs: urls,
            artifactRole: analysis.artifactRole
        )
    }

    private static func timestamp(in prefix: String) -> String? {
        guard prefix.hasPrefix("IVE-"), prefix.count > 16 else { return nil }
        let timestamp = String(prefix.suffix(15))
        let delimiter = prefix.index(prefix.endIndex, offsetBy: -16)
        guard prefix[delimiter] == "-" else { return nil }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.isLenient = false
        guard let date = formatter.date(from: timestamp),
              formatter.string(from: date) == timestamp else {
            return nil
        }
        return timestamp
    }

    private static func isNonemptyRegularFile(
        _ url: URL,
        keys: Set<URLResourceKey>
    ) -> Bool {
        guard let values = try? url.resourceValues(forKeys: keys) else { return false }
        return values.isRegularFile == true
            && values.isSymbolicLink != true
            && (values.fileSize ?? 0) > 0
    }

    private static func decode<Value: Decodable>(
        _ type: Value.Type,
        from url: URL
    ) -> Value? {
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]) else {
            return nil
        }
        return try? JSONDecoder().decode(type, from: data)
    }

    private static func samePath(_ serializedPath: String, _ url: URL) -> Bool {
        URL(fileURLWithPath: serializedPath).standardizedFileURL
            .resolvingSymlinksInPath()
            == url.standardizedFileURL.resolvingSymlinksInPath()
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { character in
            character.isNumber || ("a"..."f").contains(String(character))
        }
    }

    private static func createdAtMatches(
        _ serializedDate: String,
        timestamp: String
    ) -> Bool {
        let filenameFormatter = DateFormatter()
        filenameFormatter.locale = Locale(identifier: "en_US_POSIX")
        filenameFormatter.calendar = Calendar(identifier: .gregorian)
        filenameFormatter.timeZone = TimeZone(secondsFromGMT: 0)
        filenameFormatter.dateFormat = "yyyyMMdd-HHmmss"
        filenameFormatter.isLenient = false
        guard let filenameDate = filenameFormatter.date(from: timestamp) else {
            return false
        }
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let createdAt = isoFormatter.date(from: serializedDate)
            ?? ISO8601DateFormatter().date(from: serializedDate)
        guard let createdAt else { return false }
        return abs(createdAt.timeIntervalSince(filenameDate)) < 1.0
    }

    private static func validateGroup(
        analysisURL: URL,
        planURL: URL,
        selectionURL: URL,
        fcpxmlURL: URL,
        otioURL: URL,
        edlURL: URL,
        interchangeAuditURL: URL,
        videoURL: URL,
        outputDirectory: URL,
        timestamp: String
    ) -> Bool {
        guard let analysis = decode(GroupAnalysisManifest.self, from: analysisURL),
              let plan = decode(RenderPlanSummary.self, from: planURL),
              let selection = decode(CandidateSelection.self, from: selectionURL),
              let interchangeAudit = decode(
                InterchangeAudit.self,
                from: interchangeAuditURL
              ),
              analysis.schemaVersion == 1,
              createdAtMatches(analysis.createdAt, timestamp: timestamp),
              analysis.request.group == "IVE",
              analysis.request.editMode == "group",
              analysis.request.sourceMode == "automatic",
              samePath(analysis.request.outputDirectory, outputDirectory),
              analysis.appResultExposure.authorizes(role: analysis.artifactRole, for: .group),
              analysis.postRenderQA.authorizes(role: analysis.artifactRole, for: .group),
              analysis.render.artifactRole == analysis.artifactRole,
              analysis.render.appResultExposure == analysis.appResultExposure,
              analysis.render.postRenderQA == analysis.postRenderQA,
              analysis.capabilities.groupVideoRendered == true,
              analysis.render.videoCodec == "h264_videotoolbox"
                || analysis.render.videoCodec == "libx264",
              samePath(analysis.render.videoPath, videoURL),
              samePath(analysis.render.planPath, planURL),
              samePath(analysis.render.interchange.fcpxmlPath, fcpxmlURL),
              samePath(analysis.render.interchange.otioPath, otioURL),
              samePath(analysis.render.interchange.edlPath, edlURL),
              samePath(
                analysis.render.interchange.fidelityAuditPath,
                interchangeAuditURL
              ),
              samePath(
                analysis.render.interchange.authoritativePixels,
                videoURL
              ),
              interchangeAudit.schemaVersion == 1,
              interchangeAudit.analysisBasis
                == "reviewed_render_plan_interchange_bundle_v1",
              samePath(interchangeAudit.authoritativeOutput, videoURL),
              interchangeAudit.outputFrameCount == plan.outputFrameCount,
              isSHA256(interchangeAudit.contentSHA256),
              Set(interchangeAudit.artifacts.keys) == Set(["fcpxml", "otio", "edl"]),
              interchangeAudit.artifacts.values.allSatisfy({
                $0.sizeBytes > 0 && isSHA256($0.sha256)
              }),
              selection.isComplete(allowQualityFailure: analysis.artifactRole == "review_ready",
                renderedFamilyReceipt: analysis.render.performerFamilyReview),
              selection.releasePolicy == analysis.render.releasePolicy,
              (selection.releasePolicy == nil || analysis.postRenderQA.reviewFirstDisposition != nil),
              (selection.releasePolicy?.isReviewFirst != true
                || (selection.candidatePool == analysis.render.candidatePool
                    && selection.selectedPathAuthority == analysis.render.candidateOrigin)),
              (selection.releasePolicy?.isQualitySingle != true
                || ((selection.releasePolicy?.contract == "joint_quality_gated_single_delivery_v1"
                    && selection.candidatePool == "joint_decoded_visual_selector"
                    && analysis.render.candidateOrigin == "joint_decoded_visual_selector")
                    || (selection.releasePolicy?.contract == "stage_edit_quality_gated_single_delivery_v1"
                        && ["stage_edit_generation_only", "stage_edit_generation_plus_typed_visual_floor_v1"].contains(selection.candidatePool ?? "")
                        && analysis.render.candidateOrigin == "explicit_stage_edit_generation"))),
              analysis.renderedCandidateSelection == selection,
              analysis.render.renderedCandidateSelection == selection,
              analysis.renderedProxyReview == selection.finalSelectedReview,
              analysis.render.renderedProxyReview == selection.finalSelectedReview,
              plan.aspectRatio == "16:9",
              plan.width == 1_920,
              plan.height == 1_080,
              plan.fps > 0,
              plan.renderDuration > 0,
              plan.outputFrameCount > 0,
              abs(
                plan.renderDuration
                    - Double(plan.outputFrameCount) / plan.fps
              ) < 1e-6,
              samePath(plan.output, videoURL) else {
            return false
        }
        return true
    }

    private static func validateMember(
        analysis: MemberAnalysisManifest,
        analysisURL: URL,
        planURL: URL,
        postRenderQAURL: URL,
        fcpxmlURL: URL,
        otioURL: URL,
        edlURL: URL,
        interchangeAuditURL: URL,
        videoURL: URL,
        outputDirectory: URL,
        timestamp: String
    ) -> Bool {
        guard let plan = decode(RenderPlanSummary.self, from: planURL),
              let postRenderQA = decode(
                MemberPostRenderQAReport.self,
                from: postRenderQAURL
              ),
              let interchangeAudit = decode(
                InterchangeAudit.self,
                from: interchangeAuditURL
              ),
              analysis.schemaVersion == 1,
              createdAtMatches(analysis.createdAt, timestamp: timestamp),
              analysis.request.group == "IVE",
              analysis.request.editMode == "member",
              analysis.request.sourceMode == "automatic",
              analysis.request.aspectRatio == "9:16",
              analysis.request.targetMember.flatMap(IVEMember.init(rawValue:)) != nil,
              samePath(analysis.request.outputDirectory, outputDirectory),
              analysis.memberDirectFullFrame.enabled,
              analysis.appResultExposure.authorizes(role: analysis.artifactRole, for: .member),
              analysis.postRenderQA.authorizes(role: analysis.artifactRole, for: .member),
              samePath(analysis.postRenderQA.reportPath ?? "", postRenderQAURL),
              analysis.render.artifactRole == analysis.artifactRole,
              analysis.render.appResultExposure == analysis.appResultExposure,
              analysis.render.postRenderQA == analysis.postRenderQA,
              analysis.render.directFullFrame,
              analysis.capabilities.memberVideoRendered == true,
              analysis.capabilities.memberDirectFullFrameReady == true,
              analysis.render.videoCodec == "h264_videotoolbox"
                || analysis.render.videoCodec == "libx264",
              samePath(analysis.render.videoPath, videoURL),
              samePath(analysis.render.planPath, planURL),
              samePath(analysis.render.interchange.fcpxmlPath, fcpxmlURL),
              samePath(analysis.render.interchange.otioPath, otioURL),
              samePath(analysis.render.interchange.edlPath, edlURL),
              samePath(
                analysis.render.interchange.fidelityAuditPath,
                interchangeAuditURL
              ),
              samePath(analysis.render.interchange.authoritativePixels, videoURL),
              interchangeAudit.schemaVersion == 1,
              interchangeAudit.analysisBasis
                == "reviewed_render_plan_interchange_bundle_v1",
              samePath(interchangeAudit.authoritativeOutput, videoURL),
              interchangeAudit.outputFrameCount == plan.outputFrameCount,
              isSHA256(interchangeAudit.contentSHA256),
              Set(interchangeAudit.artifacts.keys) == Set(["fcpxml", "otio", "edl"]),
              interchangeAudit.artifacts.values.allSatisfy({
                $0.sizeBytes > 0 && isSHA256($0.sha256)
              }),
              postRenderQA.schemaVersion == 1,
              postRenderQA.analysis == automaticIVEMemberPostRenderQAAnalysis,
              postRenderQA.verdict == analysis.postRenderQA.verdict,
              postRenderQA.passed == analysis.postRenderQA.passed,
              (analysis.artifactRole == "delivery" ? postRenderQA.issueCount == 0 : postRenderQA.issueCount > 0),
              samePath(postRenderQA.artifactSet.analysisPath, analysisURL),
              samePath(postRenderQA.artifactSet.renderPlanPath, planURL),
              samePath(postRenderQA.artifactSet.videoPath, videoURL),
              postRenderQA.renderReceipt.contract
                == automaticIVEMemberRenderReceiptContract,
              postRenderQA.renderReceipt.allEligibleSourcesPixelProven,
              postRenderQA.renderReceipt.sourceFileIdentitiesStable,
              postRenderQA.renderReceipt.artifactContentStable,
              postRenderQA.renderReceipt.passed,
              plan.aspectRatio == "9:16",
              plan.width == 1_080,
              plan.height == 1_920,
              plan.fps > 0,
              plan.renderDuration > 0,
              plan.outputFrameCount > 0,
              abs(
                plan.renderDuration
                    - Double(plan.outputFrameCount) / plan.fps
              ) < 1e-6,
              samePath(plan.output, videoURL) else {
            return false
        }
        return true
    }
}

@MainActor
final class EngineRunner: ObservableObject {
    static let automaticAcquisitionUnavailableMessage =
        "자동 수집 또는 선택한 자동 편집 모드가 준비되지 않았습니다. "
        + "yt-dlp·네트워크·번들 시각 모델과 검증된 StageEdit "
        + "프로덕션 모델 등록 상태를 확인해 주세요. 로컬 원본만 "
        + "확인하려면 ‘내 영상 선택’을 이용할 수 있습니다."
    static let localGroupUnavailableMessage =
        "로컬 단체 편집이 준비되지 않았습니다. FFmpeg/ffprobe와 Python 런타임 진단을 확인해 주세요."
    static let localMemberTrackingUnavailableMessage =
        "로컬 멤버 편집에 필요한 확인 앵커 트래킹을 사용할 수 없습니다. OpenCV Contrib의 CSRT/KCF/MIL 트래커 진단을 확인해 주세요."

    @Published private(set) var state: EngineRunState = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var progressIsIndeterminate = false
    @Published private(set) var statusMessage = "IVE 곡명을 입력하면 음방 10개를 목표로 찾고, 최소 8개부터 편집합니다."
    @Published private(set) var logs: [EngineLogEntry] = []
    @Published private(set) var outputURLs: [URL] = []
    @Published private(set) var renderedVideoURL: URL?
    @Published private(set) var sceneReviewProposalURL: URL?
    private var sceneReviewProposal: SceneReviewProposalDisposition?
    private var sceneReviewProposalDirectory: URL?
    @Published private(set) var engineURL: URL?
    @Published private(set) var cancellationRequested = false
    @Published private(set) var isRestoringCompletedOutputs = false

    private var process: Process?
    private var activeRunID: UUID?
    private var activeRequest: EditRequest?
    private var pendingOutputURLs: [URL] = []
    private var requestFileURL: URL?
    private var reportedError: String?
    private var resultMessage: String?
    private var pendingArtifactResult: EngineArtifactDisposition?
    private var preparationTask: Task<Void, Never>?
    private var activePreparationID: UUID?
    private var completedOutputDiscoveryTask: Task<Void, Never>?
    private var activeCompletedOutputDiscoveryID: UUID?
    private let locateEngine: @Sendable () -> URL?
    private let preparationProbe: EnginePreparationProbe
    private let outputDrainGraceNanoseconds: UInt64

    init(
        locateEngine: @escaping @Sendable () -> URL? = { EngineLocator.locate() },
        preparationProbe: @escaping EnginePreparationProbe = defaultEnginePreparationProbe,
        outputDrainGraceNanoseconds: UInt64 = 5_000_000_000
    ) {
        precondition(outputDrainGraceNanoseconds > 0)
        self.locateEngine = locateEngine
        self.preparationProbe = preparationProbe
        self.outputDrainGraceNanoseconds = outputDrainGraceNanoseconds
        engineURL = locateEngine()
    }

    func refreshEngineLocation() {
        engineURL = locateEngine()
    }

    func restoreLatestCompletedResult(
        in outputDirectory: URL,
        editMode: EditMode,
        reportMissing: Bool = true,
        clearExistingOutputsOnMissing: Bool = false
    ) {
        guard !state.isRunning else { return }

        completedOutputDiscoveryTask?.cancel()
        let discoveryID = UUID()
        activeCompletedOutputDiscoveryID = discoveryID
        isRestoringCompletedOutputs = true
        let directory = outputDirectory.standardizedFileURL
        let outputMode = CompletedEngineOutputMode(editMode)
        completedOutputDiscoveryTask = Task { [weak self] in
            let result = await Task.detached(priority: .utility) {
                CompletedEngineOutputSetDiscovery.latest(
                    in: directory,
                    editMode: outputMode
                )
            }.value
            guard !Task.isCancelled else { return }
            self?.completeCompletedOutputDiscovery(
                result,
                editMode: outputMode,
                discoveryID: discoveryID,
                reportMissing: reportMissing,
                clearExistingOutputsOnMissing: clearExistingOutputsOnMissing
            )
        }
    }

    func start(request: EditRequest) {
        guard !state.isRunning else { return }

        resetForNewRun()
        activeRunID = UUID()
        activeRequest = request
        state = .preparing
        statusMessage = "엔진을 준비하는 중…"

        guard let engineURL = locateEngine() else {
            fail(
                "crosscut_cli.py를 찾지 못했습니다. 저장소의 engine 폴더에 두거나 "
                    + "\(EngineLocator.environmentKey) 환경변수로 경로를 지정해 주세요."
            )
            return
        }
        self.engineURL = engineURL

        let preparationID = UUID()
        activePreparationID = preparationID
        let probe = preparationProbe
        let sourceMode = request.sourceMode
        let editMode = request.editMode
        preparationTask = Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) {
                probe(engineURL, sourceMode, editMode)
            }.value
            guard !Task.isCancelled else { return }
            self?.completePreparation(
                result,
                request: request,
                engineURL: engineURL,
                preparationID: preparationID
            )
        }
    }

    func cancel() {
        guard state.isRunning, !cancellationRequested else { return }
        cancellationRequested = true
        discardReportedOutputs()
        statusMessage = "취소하는 중…"
        appendLog("사용자가 작업 취소를 요청했습니다.", level: .warning)
        if let process {
            let terminatedGroup = EngineProcessTree.requestTermination(process)
            if !terminatedGroup {
                appendLog(
                    "엔진 프로세스 그룹을 확인하지 못해 기본 종료 신호를 보냈습니다.",
                    level: .warning
                )
            }
        } else {
            preparationTask?.cancel()
            preparationTask = nil
            activePreparationID = nil
            cleanupRequestFile()
            activeRunID = nil
            activeRequest = nil
            state = .cancelled
            statusMessage = "작업이 취소되었습니다."
            appendLog(statusMessage, level: .warning)
        }
    }

    func clearLog() {
        guard !state.isRunning else { return }
        logs.removeAll()
    }

    private func resetForNewRun() {
        completedOutputDiscoveryTask?.cancel()
        completedOutputDiscoveryTask = nil
        activeCompletedOutputDiscoveryID = nil
        isRestoringCompletedOutputs = false
        preparationTask?.cancel()
        preparationTask = nil
        activePreparationID = nil
        activeRunID = nil
        activeRequest = nil
        progress = 0
        progressIsIndeterminate = false
        logs.removeAll()
        sceneReviewProposalURL = nil
        sceneReviewProposal = nil
        sceneReviewProposalDirectory = nil
        outputURLs.removeAll()
        renderedVideoURL = nil
        pendingOutputURLs.removeAll()
        cancellationRequested = false
        reportedError = nil
        resultMessage = nil
        pendingArtifactResult = nil
    }

    private func completeCompletedOutputDiscovery(
        _ result: CompletedEngineOutputSet?,
        editMode: CompletedEngineOutputMode,
        discoveryID: UUID,
        reportMissing: Bool,
        clearExistingOutputsOnMissing: Bool
    ) {
        guard activeCompletedOutputDiscoveryID == discoveryID else { return }
        completedOutputDiscoveryTask = nil
        activeCompletedOutputDiscoveryID = nil
        isRestoringCompletedOutputs = false
        guard !state.isRunning else { return }

        guard let result else {
            if clearExistingOutputsOnMissing {
                outputURLs.removeAll()
                renderedVideoURL = nil
                if state == .completed || state == .reviewReady {
                    progress = 0
                    state = .idle
                    statusMessage = "선택한 저장 위치에 완성 결과가 아직 없습니다."
                }
            }
            if reportMissing {
                let modeName = editMode == .member
                    ? "멤버 교차직캠"
                    : "단체 교차편집"
                let message = "선택한 폴더에서 완성된 자동 \(modeName) 결과를 찾지 못했습니다."
                statusMessage = message
                appendLog(message, level: .warning)
            }
            return
        }

        guard result.editMode == editMode else { return }

        outputURLs = result.outputURLs
        renderedVideoURL = RenderedVideoOutputDiscovery.firstValidVideo(
            in: result.outputURLs
        )
        reportedError = nil
        resultMessage = result.artifactRole == "review_ready"
            ? "편집 품질 QA 미통과 검토본을 불러왔습니다."
            : "외부에서 완성된 교차편집 결과를 불러왔습니다."
        cancellationRequested = false
        progress = 1
        progressIsIndeterminate = false
        state = result.artifactRole == "review_ready" ? .reviewReady : .completed
        statusMessage = result.artifactRole == "review_ready" ? "검토용 초안 · 최종 승인 아님" : "교차편집 결과가 준비되었습니다."
        appendLog(
            "완성 결과 복원: \(result.outputURLs.last?.lastPathComponent ?? result.timestamp)",
            level: .info
        )
    }

    private func completePreparation(
        _ result: EnginePreparationResult,
        request: EditRequest,
        engineURL: URL,
        preparationID: UUID
    ) {
        guard activePreparationID == preparationID,
              state == .preparing,
              !cancellationRequested else {
            return
        }
        preparationTask = nil

        switch result {
        case .pythonUnavailable:
            activePreparationID = nil
            fail(
                "Python 3.11+, NumPy 2.0+, OpenCV Contrib 4.10+를 "
                    + "충족하는 런타임을 찾지 못했습니다. "
                    + "CROSSCUT_PYTHON 환경변수로 요구 사항을 갖춘 Python 경로를 "
                    + "지정해 주세요."
            )

        case .localGroupUnavailable:
            activePreparationID = nil
            fail(Self.localGroupUnavailableMessage)

        case .localMemberTrackingUnavailable:
            activePreparationID = nil
            fail(Self.localMemberTrackingUnavailableMessage)

        case .automaticAcquisitionUnavailable:
            activePreparationID = nil
            fail(Self.automaticAcquisitionUnavailableMessage)

        case let .ready(pythonURL):
            do {
                let outputDirectory = URL(
                    fileURLWithPath: request.outputDirectory,
                    isDirectory: true
                )
                try FileManager.default.createDirectory(
                    at: outputDirectory,
                    withIntermediateDirectories: true
                )

                let requestURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("ive-crosscut-\(UUID().uuidString)")
                    .appendingPathExtension("json")
                let encoder = JSONEncoder()
                encoder.outputFormatting = [
                    .prettyPrinted,
                    .sortedKeys,
                    .withoutEscapingSlashes,
                ]
                try encoder.encode(request).write(to: requestURL, options: .atomic)
                requestFileURL = requestURL

                try launch(
                    engineURL: engineURL,
                    pythonURL: pythonURL,
                    requestURL: requestURL
                )
                activePreparationID = nil
            } catch {
                activePreparationID = nil
                cleanupRequestFile()
                fail("엔진을 시작하지 못했습니다: \(error.localizedDescription)")
            }
        }
    }

    private func launch(engineURL: URL, pythonURL: URL, requestURL: URL) throws {
        guard let runID = activeRunID else {
            throw CocoaError(.userCancelled)
        }
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.executableURL = pythonURL
        process.arguments = [
            "-B",
            engineURL.path,
            "analyze",
            "--request",
            requestURL.path,
        ]
        process.currentDirectoryURL = engineURL.deletingLastPathComponent()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PATH"] = EngineLocator.augmentedPath(existing: environment["PATH"])
        process.environment = environment
        let drainState = EngineOutputDrainState()
        let drainGrace = outputDrainGraceNanoseconds

        let stdoutTask = Task.detached(priority: .utility) { [weak self] in
            defer { drainState.streamFinished() }
            do {
                var buffer = EngineOutputLineBuffer()
                for try await bytes in enginePipeChunks(from: stdoutPipe.fileHandleForReading) {
                    buffer.append(bytes)
                    while let line = try buffer.nextLine() {
                        await self?.consumeStandardOutput(line, runID: runID)
                    }
                }
                for line in try buffer.finish() {
                    await self?.consumeStandardOutput(line, runID: runID)
                }
            } catch {
                await self?.recordStreamFailure(
                    "표준 출력 스트림을 읽지 못했습니다: \(error.localizedDescription)", runID: runID
                )
            }
        }

        let stderrTask = Task.detached(priority: .utility) { [weak self] in
            defer { drainState.streamFinished() }
            do {
                var buffer = EngineOutputLineBuffer()
                for try await bytes in enginePipeChunks(from: stderrPipe.fileHandleForReading) {
                    buffer.append(bytes)
                    while let line = try buffer.nextLine() {
                        await self?.consumeStandardError(line, runID: runID)
                    }
                }
                for line in try buffer.finish() {
                    await self?.consumeStandardError(line, runID: runID)
                }
            } catch {
                await self?.recordStreamFailure(
                    "오류 출력 스트림을 읽지 못했습니다: \(error.localizedDescription)", runID: runID
                )
            }
        }

        process.terminationHandler = { [weak self] terminatedProcess in
            Task {
                let deadlineTask = Task.detached(priority: .utility) {
                    do { try await Task.sleep(nanoseconds: drainGrace) }
                    catch { return }
                    if drainState.expireIfPending() {
                        // Stop only our readers. The exited parent's numeric
                        // process group is not authority to kill surviving PIDs.
                        stdoutTask.cancel()
                        stderrTask.cancel()
                    }
                }
                await stdoutTask.value
                await stderrTask.value
                deadlineTask.cancel()
                await self?.finish(
                    exitCode: terminatedProcess.terminationStatus,
                    runID: runID, outputDrainTimedOut: drainState.timedOut
                )
            }
        }

        do {
            try process.run()
        } catch {
            stdoutTask.cancel()
            stderrTask.cancel()
            throw error
        }
        self.process = process
        state = .running
        statusMessage = "소스를 맞추고 교차편집하는 중…"
        appendLog("Python: \(pythonURL.path)", level: .info)
        appendLog("엔진 실행: \(engineURL.path)", level: .info)
    }

    private func consumeStandardOutput(_ line: String, runID: UUID) {
        guard activeRunID == runID else { return }
        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLine.isEmpty else { return }

        guard let data = trimmedLine.data(using: .utf8),
              let event = try? JSONDecoder().decode(EngineEvent.self, from: data) else {
            appendLog(trimmedLine, level: .engine)
            return
        }

        handle(event)
    }

    private func consumeStandardError(_ line: String, runID: UUID) {
        guard activeRunID == runID else { return }
        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLine.isEmpty else { return }
        appendLog(trimmedLine, level: .error)
    }

    private func recordStreamFailure(_ message: String, runID: UUID) {
        guard activeRunID == runID else { return }
        discardReportedOutputs()
        reportedError = reportedError ?? message
        appendLog(message, level: .error)
        if let process { EngineProcessTree.requestTermination(process) }
    }

    func handle(_ event: EngineEvent) {
        let type = event.type.lowercased()

        switch type {
        case "joint_visual_selection_progress", "joint_candidate_terminal", "joint_candidate_look_pending":
            if let work = event.workStatus {
                // A preceding feature-stage value (often 98%) is not an
                // estimate for search, proof, rendering and QA still to come.
                progressIsIndeterminate = true
                statusMessage = work.message
                if work.shouldLog {
                    appendLog(work.message, level: work.isWarning ? .warning : .info)
                }
            }

        case "status":
            statusMessage = event.message ?? event.stage ?? "처리 중…"
            if let message = event.message ?? event.stage {
                appendLog(message, level: .info)
            }

        case "progress":
            if let eventProgress = event.progress {
                progressIsIndeterminate = false
                progress = min(max(eventProgress, 0), 1)
            }
            if let message = event.message ?? event.stage {
                statusMessage = message
            }

        case "log":
            appendLog(
                event.message ?? event.stage ?? "엔진 로그",
                level: Self.logLevel(from: event.level)
            )

        case "scene_review_proposal":
            guard state.isRunning, !cancellationRequested, reportedError == nil,
                  activeRequest?.editMode == EditMode.group.rawValue,
                  let proposal = event.sceneReviewProposal,
                  let directory = activeRequest?.outputDirectory,
                  Self.validateSceneReviewProposal(proposal, directory: URL(fileURLWithPath: directory)) else {
                appendLog("구도 제안의 파일·검증 계약을 확인하지 못했습니다.", level: .warning)
                return
            }
            sceneReviewProposal = proposal
            sceneReviewProposalDirectory = URL(fileURLWithPath: directory)
            sceneReviewProposalURL = URL(fileURLWithPath: proposal.videoPath)
            appendLog("별도 480p 구도 제안이 있습니다. 출연자·동작·박자·편집 품질은 미승인입니다.", level: .warning)
            // Do not set outputURLs, progress=1, completed, or artifactResult.

        case "result":
            guard state.isRunning, !cancellationRequested, reportedError == nil else { return }
            pendingOutputURLs = event.outputPaths.map { URL(fileURLWithPath: $0) }
            pendingArtifactResult = event.artifactResult
            if let message = event.message {
                resultMessage = message
            }

        case "error":
            let message = event.message ?? "엔진에서 알 수 없는 오류를 보고했습니다."
            discardReportedOutputs()
            reportedError = message
            statusMessage = message
            appendLog(message, level: .error)

        default:
            appendLog(event.message ?? "알 수 없는 이벤트: \(event.type)", level: .engine)
        }
    }

    /// Recheck on open as well as receive. A restored final result never
    /// upgrades this sidecar, and a later file replacement invalidates it.
    func verifiedSceneReviewProposalURL() -> URL? {
        guard !cancellationRequested, let proposal = sceneReviewProposal,
              let directory = sceneReviewProposalDirectory,
              Self.validateSceneReviewProposal(proposal, directory: directory) else { return nil }
        return URL(fileURLWithPath: proposal.videoPath)
    }

    static func validateSceneReviewProposal(_ proposal: SceneReviewProposalDisposition, directory: URL) -> Bool {
        guard proposal.isReviewOnly else { return false }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let video = URL(fileURLWithPath: proposal.videoPath).resolvingSymlinksInPath().standardizedFileURL
        let manifest = URL(fileURLWithPath: proposal.manifestPath).resolvingSymlinksInPath().standardizedFileURL
        guard video.deletingLastPathComponent() == root, manifest.deletingLastPathComponent() == root,
              video.pathExtension.lowercased() == "mp4", manifest.pathExtension.lowercased() == "json" else { return false }
        func digest(_ url: URL, maximum: Int) -> String? {
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                  size > 0, size <= maximum, let handle = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? handle.close() }
            var hash = SHA256()
            var read = 0
            do {
                while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                    read += chunk.count
                    guard read <= maximum else { return nil }
                    hash.update(data: chunk)
                }
            } catch { return nil }
            guard read == size else { return nil }
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        guard digest(video, maximum: 512 * 1024 * 1024) == proposal.videoSHA256,
              let manifestSize = try? manifest.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              manifestSize > 0, manifestSize <= 16 * 1024 * 1024,
              let data = try? Data(contentsOf: manifest), data.count == manifestSize,
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == proposal.manifestSHA256,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["contract"] as? String == proposal.contract,
              object["status"] as? String == "REVIEW_REQUIRED",
              object["artifact_role"] as? String == "scene_review_only",
              object["quality_approved"] as? Bool == false,
              object["delivery_completion_message_allowed"] as? Bool == false,
              object["video_sha256"] as? String == proposal.videoSHA256,
              object["video_path"] as? String == proposal.videoPath,
              let technical = object["technical_qa"] as? [String: Any],
              technical["technical_only"] as? Bool == true,
              technical["professional_qa_passed"] as? Bool == false,
              let issues = technical["issue_codes"] as? [String], issues.isEmpty else { return false }
        return true
    }

    private func finish(exitCode: Int32, runID: UUID, outputDrainTimedOut: Bool) async {
        guard activeRunID == runID else { return }
        process = nil
        cleanupRequestFile()

        if outputDrainTimedOut {
            let message = "엔진은 종료했지만 출력 스트림이 닫히지 않았습니다. 하위 작업이 남아 있을 수 있으니 확인이 필요합니다."
            fail(reportedError.map { $0 + "\n" + message } ?? message)
            return
        }

        if cancellationRequested {
            discardReportedOutputs()
            activeRunID = nil
            activeRequest = nil
            state = .cancelled
            statusMessage = "작업이 취소되었습니다."
            appendLog(statusMessage, level: .warning)
            return
        }

        guard exitCode == 0, reportedError == nil else {
            fail(reportedError ?? "엔진이 종료 코드 \(exitCode)로 끝났습니다.")
            return
        }

        guard let request = activeRequest else {
            fail("완성 결과를 현재 실행 요청과 연결하지 못했습니다.")
            return
        }
        let reported = pendingOutputURLs
        let accepted: [URL]
        var finalRole = "delivery"
        if request.sourceMode == SourceMode.automatic.rawValue {
            statusMessage = "최종 결과와 QA 기록을 확인하는 중…"
            let validated = await Task.detached(priority: .utility) {
                CompletedEngineOutputSetDiscovery.reported(reported, for: request)
            }.value
            guard activeRunID == runID, state.isRunning, !cancellationRequested else { return }
            guard let validated else {
                fail("보고된 자동 교차편집 결과의 최종 QA와 전달 기록을 확인하지 못했습니다.")
                return
            }
            accepted = validated.outputURLs
            finalRole = validated.artifactRole
            if let disposition = pendingArtifactResult, !disposition.authorizes(role: finalRole) {
                fail("엔진 결과 상태와 최종 QA의 공개 권한이 일치하지 않습니다.")
                return
            }
        } else {
            accepted = reported
        }
        guard let renderedVideo = RenderedVideoOutputDiscovery.firstValidVideo(in: accepted) else {
            fail(
                "엔진은 정상 종료했지만 읽을 수 있는 비어 있지 않은 "
                    + "결과 영상을 확인하지 못했습니다."
            )
            return
        }
        pendingOutputURLs.removeAll()
        outputURLs = accepted
        renderedVideoURL = renderedVideo
        activeRunID = nil
        activeRequest = nil
        progress = 1
        progressIsIndeterminate = false
        state = finalRole == "review_ready" ? .reviewReady : .completed
        statusMessage = finalRole == "review_ready" ? "검토용 초안 · 최종 승인 아님" : "교차편집 결과가 준비되었습니다."
        if let resultMessage { appendLog(resultMessage, level: .info) }
        appendLog(statusMessage, level: .info)
    }

    private func fail(_ message: String) {
        discardReportedOutputs()
        activeRunID = nil
        activeRequest = nil
        state = .failed(message)
        statusMessage = message
        appendLog(message, level: .error)
    }

    private func discardReportedOutputs() {
        pendingOutputURLs.removeAll()
        outputURLs.removeAll()
        renderedVideoURL = nil
        resultMessage = nil
        pendingArtifactResult = nil
    }

    private func cleanupRequestFile() {
        guard let requestFileURL else { return }
        try? FileManager.default.removeItem(at: requestFileURL)
        self.requestFileURL = nil
    }

    private func appendLog(_ message: String, level: EngineLogEntry.Level) {
        guard !message.isEmpty else { return }
        logs.append(EngineLogEntry(timestamp: Date(), level: level, message: message))
    }

    private static func logLevel(from rawValue: String?) -> EngineLogEntry.Level {
        switch rawValue?.lowercased() {
        case "warn", "warning": .warning
        case "error", "fatal": .error
        case "debug", "engine": .engine
        default: .info
        }
    }
}
