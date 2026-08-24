import Combine
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
    case failed(String)
    case cancelled

    var isRunning: Bool {
        switch self {
        case .preparing, .running: true
        default: false
        }
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

struct CompletedEngineOutputSet: Equatable, Sendable {
    let timestamp: String
    let outputURLs: [URL]
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
        let structurePassingMaterializedCount: Int
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
        }

        var isComplete: Bool {
            let indices = candidates.map(\.candidateIndex)
            let selected = candidates.filter(\.selected)
            return analysisBasis == CompletedEngineOutputSetDiscovery.selectionAnalysisBasis
                && storagePolicy.oneProxyAtATime
                && storagePolicy.temporaryProxyRemovedAfterEachReview
                && !storagePolicy.decodedFramesPersisted
                && requestedReviewCount == 8
                && reviewedCandidateCount == 8
                && minimumRequiredCandidateCount == 8
                && minimumCandidateCountSatisfied
                && successfulReviewCount == 8
                && completeCandidateCount == 8
                && structurePassingMaterializedCount == 8
                && exactSuccessfulDecodedReviewCountSatisfied
                && exactCompleteCandidateCountSatisfied
                && passed
                && candidates.count == 8
                && Set(indices).count == 8
                && selected.count == 1
                && selected[0].candidateIndex == selectedCandidateIndex
                && candidates.allSatisfy {
                    $0.error == nil
                        && ($0.review?.isCompleteAndSanitized == true)
                }
                && finalSelectedReview.passed
                && finalSelectedReview.isCompleteAndSanitized
                && ["original", "partial_reedit"].contains(finalSelectedVersion)
        }
    }

    private struct AnalysisRequest: Decodable {
        let group: String
        let editMode: String
        let sourceMode: String
        let outputDirectory: String

        enum CodingKeys: String, CodingKey {
            case group
            case editMode = "edit_mode"
            case sourceMode = "source_mode"
            case outputDirectory = "output_directory"
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

        var authorizesDelivery: Bool {
            contract
                == CompletedEngineOutputSetDiscovery
                    .automaticIVEGroupAppResultExposureContract
                && videoOutputExposed
                && deliveryCompletionMessageAllowed
        }
    }

    private struct PostRenderQAEvidence: Decodable, Equatable {
        let contract: String
        let required: Bool
        let verdict: String
        let passed: Bool

        var authorizesDelivery: Bool {
            contract
                == CompletedEngineOutputSetDiscovery
                    .automaticIVEGroupPostRenderQAContract
                && required
                && verdict == "PASS"
                && passed
        }
    }

    private struct AnalysisCapabilities: Decodable {
        let groupVideoRendered: Bool

        enum CodingKeys: String, CodingKey {
            case groupVideoRendered = "group_video_rendered"
        }
    }

    private struct AnalysisRender: Decodable {
        struct Interchange: Decodable {
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

        let videoPath: String
        let planPath: String
        let videoCodec: String
        let interchange: Interchange
        let renderedProxyReview: ProxyReview
        let renderedCandidateSelection: CandidateSelection
        let artifactRole: String
        let appResultExposure: AppResultExposure
        let postRenderQA: PostRenderQAEvidence

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

    private struct AnalysisManifest: Decodable {
        let schemaVersion: Int
        let createdAt: String
        let request: AnalysisRequest
        let renderedProxyReview: ProxyReview
        let renderedCandidateSelection: CandidateSelection
        let render: AnalysisRender
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

    static func latest(in outputDirectory: URL) -> CompletedEngineOutputSet? {
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
            guard expectedNames.allSatisfy({ byName[$0] != nil }) else { continue }
            let urls = expectedNames.compactMap { byName[$0]?.standardizedFileURL }
            guard urls.count == expectedNames.count,
                  urls.allSatisfy({ isNonemptyRegularFile($0, keys: resourceKeys) }),
                  validate(
                    analysisURL: urls[0],
                    planURL: urls[1],
                    selectionURL: urls[2],
                    fcpxmlURL: urls[3],
                    otioURL: urls[4],
                    edlURL: urls[5],
                    interchangeAuditURL: urls[6],
                    videoURL: urls[7],
                    outputDirectory: directory,
                    timestamp: timestamp
                  ) else {
                continue
            }
            completed.append(
                CompletedEngineOutputSet(timestamp: timestamp, outputURLs: urls)
            )
        }
        return completed.max { $0.timestamp < $1.timestamp }
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

    private static func validate(
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
        guard let analysis = decode(AnalysisManifest.self, from: analysisURL),
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
              analysis.artifactRole == "delivery",
              analysis.appResultExposure.authorizesDelivery,
              analysis.postRenderQA.authorizesDelivery,
              analysis.render.artifactRole == analysis.artifactRole,
              analysis.render.appResultExposure == analysis.appResultExposure,
              analysis.render.appResultExposure.authorizesDelivery,
              analysis.render.postRenderQA == analysis.postRenderQA,
              analysis.render.postRenderQA.authorizesDelivery,
              analysis.capabilities.groupVideoRendered,
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
              selection.isComplete,
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
    @Published private(set) var statusMessage = "IVE 곡명을 입력하면 음방 10개를 목표로 찾고, 최소 8개부터 편집합니다."
    @Published private(set) var logs: [EngineLogEntry] = []
    @Published private(set) var outputURLs: [URL] = []
    @Published private(set) var renderedVideoURL: URL?
    @Published private(set) var engineURL: URL?
    @Published private(set) var cancellationRequested = false
    @Published private(set) var isRestoringCompletedOutputs = false

    private var process: Process?
    private var requestFileURL: URL?
    private var reportedError: String?
    private var resultMessage: String?
    private var preparationTask: Task<Void, Never>?
    private var activePreparationID: UUID?
    private var completedOutputDiscoveryTask: Task<Void, Never>?
    private var activeCompletedOutputDiscoveryID: UUID?
    private let locateEngine: @Sendable () -> URL?
    private let preparationProbe: EnginePreparationProbe

    init(
        locateEngine: @escaping @Sendable () -> URL? = { EngineLocator.locate() },
        preparationProbe: @escaping EnginePreparationProbe = defaultEnginePreparationProbe
    ) {
        self.locateEngine = locateEngine
        self.preparationProbe = preparationProbe
        engineURL = locateEngine()
    }

    func refreshEngineLocation() {
        engineURL = locateEngine()
    }

    func restoreLatestCompletedGroupResult(
        in outputDirectory: URL,
        reportMissing: Bool = true,
        clearExistingOutputsOnMissing: Bool = false
    ) {
        guard !state.isRunning else { return }

        completedOutputDiscoveryTask?.cancel()
        let discoveryID = UUID()
        activeCompletedOutputDiscoveryID = discoveryID
        isRestoringCompletedOutputs = true
        let directory = outputDirectory.standardizedFileURL
        completedOutputDiscoveryTask = Task { [weak self] in
            let result = await Task.detached(priority: .utility) {
                CompletedEngineOutputSetDiscovery.latest(in: directory)
            }.value
            guard !Task.isCancelled else { return }
            self?.completeCompletedOutputDiscovery(
                result,
                discoveryID: discoveryID,
                reportMissing: reportMissing,
                clearExistingOutputsOnMissing: clearExistingOutputsOnMissing
            )
        }
    }

    func start(request: EditRequest) {
        guard !state.isRunning else { return }

        resetForNewRun()
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
        progress = 0
        logs.removeAll()
        outputURLs.removeAll()
        renderedVideoURL = nil
        cancellationRequested = false
        reportedError = nil
        resultMessage = nil
    }

    private func completeCompletedOutputDiscovery(
        _ result: CompletedEngineOutputSet?,
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
                if state == .completed {
                    progress = 0
                    state = .idle
                    statusMessage = "선택한 저장 위치에 완성 결과가 아직 없습니다."
                }
            }
            if reportMissing {
                let message = "선택한 폴더에서 완성된 자동 그룹 교차편집 결과를 찾지 못했습니다."
                statusMessage = message
                appendLog(message, level: .warning)
            }
            return
        }

        outputURLs = result.outputURLs
        renderedVideoURL = RenderedVideoOutputDiscovery.firstValidVideo(
            in: result.outputURLs
        )
        reportedError = nil
        resultMessage = "외부에서 완성된 교차편집 결과를 불러왔습니다."
        cancellationRequested = false
        progress = 1
        state = .completed
        statusMessage = "교차편집 결과가 준비되었습니다."
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

        let stdoutTask = Task.detached(priority: .utility) { [weak self] in
            do {
                for try await line in stdoutPipe.fileHandleForReading.bytes.lines {
                    await self?.consumeStandardOutput(line)
                }
            } catch {
                await self?.appendLog(
                    "표준 출력 스트림을 읽지 못했습니다: \(error.localizedDescription)",
                    level: .warning
                )
            }
        }

        let stderrTask = Task.detached(priority: .utility) { [weak self] in
            do {
                for try await line in stderrPipe.fileHandleForReading.bytes.lines {
                    await self?.consumeStandardError(line)
                }
            } catch {
                await self?.appendLog(
                    "오류 출력 스트림을 읽지 못했습니다: \(error.localizedDescription)",
                    level: .warning
                )
            }
        }

        process.terminationHandler = { [weak self] terminatedProcess in
            Task {
                await stdoutTask.value
                await stderrTask.value
                await self?.finish(exitCode: terminatedProcess.terminationStatus)
            }
        }

        try process.run()
        self.process = process
        state = .running
        statusMessage = "소스를 맞추고 교차편집하는 중…"
        appendLog("Python: \(pythonURL.path)", level: .info)
        appendLog("엔진 실행: \(engineURL.path)", level: .info)
    }

    private func consumeStandardOutput(_ line: String) {
        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLine.isEmpty else { return }

        guard let data = trimmedLine.data(using: .utf8),
              let event = try? JSONDecoder().decode(EngineEvent.self, from: data) else {
            appendLog(trimmedLine, level: .engine)
            return
        }

        handle(event)
    }

    private func consumeStandardError(_ line: String) {
        let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLine.isEmpty else { return }
        appendLog(trimmedLine, level: .error)
    }

    private func handle(_ event: EngineEvent) {
        let type = event.type.lowercased()

        switch type {
        case "status":
            statusMessage = event.message ?? event.stage ?? "처리 중…"
            if let message = event.message ?? event.stage {
                appendLog(message, level: .info)
            }

        case "progress":
            if let eventProgress = event.progress {
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

        case "result":
            outputURLs = event.outputPaths.map { URL(fileURLWithPath: $0) }
            if let message = event.message {
                resultMessage = message
                statusMessage = message
                appendLog(message, level: .info)
            }

        case "error":
            let message = event.message ?? "엔진에서 알 수 없는 오류를 보고했습니다."
            reportedError = message
            statusMessage = message
            appendLog(message, level: .error)

        default:
            appendLog(event.message ?? "알 수 없는 이벤트: \(event.type)", level: .engine)
        }
    }

    private func finish(exitCode: Int32) {
        process = nil
        cleanupRequestFile()

        if cancellationRequested {
            state = .cancelled
            statusMessage = "작업이 취소되었습니다."
            appendLog(statusMessage, level: .warning)
            return
        }

        guard exitCode == 0, reportedError == nil else {
            fail(reportedError ?? "엔진이 종료 코드 \(exitCode)로 끝났습니다.")
            return
        }

        guard let renderedVideo = RenderedVideoOutputDiscovery.firstValidVideo(
            in: outputURLs
        ) else {
            fail(
                "엔진은 정상 종료했지만 읽을 수 있는 비어 있지 않은 "
                    + "결과 영상을 확인하지 못했습니다."
            )
            return
        }
        renderedVideoURL = renderedVideo
        progress = 1
        state = .completed
        statusMessage = "교차편집 결과가 준비되었습니다."
        appendLog(statusMessage, level: .info)
    }

    private func fail(_ message: String) {
        state = .failed(message)
        statusMessage = message
        appendLog(message, level: .error)
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
