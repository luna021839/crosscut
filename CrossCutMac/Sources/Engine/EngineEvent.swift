import Foundation

struct EngineEvent: Decodable, Equatable {
    let type: String
    let message: String?
    let stage: String?
    let level: String?
    let progress: Double?
    let outputPaths: [String]
    let sourcePaths: [String]
    let acquisitionManifestPath: String?
    let workStatus: EngineWorkStatus?
    let artifactResult: EngineArtifactDisposition?
    let sceneReviewProposal: SceneReviewProposalDisposition?

    enum CodingKeys: String, CodingKey {
        case type
        case message
        case stage
        case level
        case progress
        case value
        case percent
        case output
        case outputPath = "output_path"
        case outputs
        case sourcePaths = "source_paths"
        case acquisitionManifest = "acquisition_manifest"
        case artifactResult = "artifact_result"
        case sceneReviewProposal = "scene_review_proposal"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        message = try container.decodeIfPresent(String.self, forKey: .message)
        stage = try container.decodeIfPresent(String.self, forKey: .stage)
        level = try container.decodeIfPresent(String.self, forKey: .level)

        let rawProgress = try container.decodeFlexibleDouble(forKeys: [.progress, .value, .percent])
        if let rawProgress {
            progress = rawProgress > 1 ? rawProgress / 100 : rawProgress
        } else {
            progress = nil
        }

        var paths = try container.decodeIfPresent([String].self, forKey: .outputs) ?? []
        if let path = try container.decodeIfPresent(String.self, forKey: .outputPath) {
            paths.append(path)
        }
        if let path = try container.decodeIfPresent(String.self, forKey: .output) {
            paths.append(path)
        }
        outputPaths = paths
        sourcePaths = try container.decodeIfPresent([String].self, forKey: .sourcePaths) ?? []
        acquisitionManifestPath = try container.decodeIfPresent(
            String.self,
            forKey: .acquisitionManifest
        )
        workStatus = try EngineWorkStatus.decode(type: type.lowercased(), from: decoder)
        artifactResult = try container.decodeIfPresent(EngineArtifactDisposition.self, forKey: .artifactResult)
        sceneReviewProposal = try container.decodeIfPresent(SceneReviewProposalDisposition.self, forKey: .sceneReviewProposal)
    }
}

/// A draft is playable, but it never grants an approved/delivery UI state.
struct EngineArtifactDisposition: Decodable, Equatable, Sendable {
    let contract: String
    let status: String
    let artifactRole: String
    let videoOutputExposed: Bool
    let deliveryCompletionMessageAllowed: Bool
    let technicalIssueCodes: [String]
    let editorialIssueCodes: [String]

    enum CodingKeys: String, CodingKey {
        case contract, status
        case artifactRole = "artifact_role"
        case videoOutputExposed = "video_output_exposed"
        case deliveryCompletionMessageAllowed = "delivery_completion_message_allowed"
        case technicalIssueCodes = "technical_issue_codes"
        case editorialIssueCodes = "editorial_issue_codes"
    }

    func authorizes(role: String) -> Bool {
        guard contract == "ive_review_first_runtime_v1", artifactRole == role,
              videoOutputExposed, technicalIssueCodes.isEmpty,
              editorialIssueCodes.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else { return false }
        switch role {
        case "delivery":
            return status == "APPROVED" && deliveryCompletionMessageAllowed && editorialIssueCodes.isEmpty
        case "review_ready":
            return status == "REVIEW_READY" && !deliveryCompletionMessageAllowed && !editorialIssueCodes.isEmpty
        default: return false
        }
    }
}

/// Stage counts describe work in progress; they never grant delivery status or
/// set the whole job's progress bar to 100 percent.
struct EngineWorkStatus: Equatable {
    let message: String
    let shouldLog: Bool
    let isWarning: Bool

    private enum Key: String, CodingKey {
        case contract, phase, status
        case completed = "completed_boundary_layers"
        case total = "total_boundary_layers"
        case ordinal = "proof_candidate_ordinal"
        case pool = "proof_candidate_pool_count"
        case boundaries = "completed_proof_boundary_count"
        case boundaryTotal = "total_proof_boundary_count"
        case bridges = "completed_proof_bridge_count"
        case bridgeTotal = "total_proof_bridge_count"
        case profileIndex = "profile_index"
        case profileTotal = "total_profiles"
        case candidateIndex = "candidate_index"
        case passed = "decoded_pass_count"
        case pending = "pending_candidate_count"
        case elapsed = "elapsed_seconds"
        case failure = "proof_failure_code"
    }

    static func decode(type: String, from decoder: Decoder) throws -> Self? {
        guard ["joint_visual_selection_progress", "joint_candidate_terminal", "joint_candidate_look_pending"].contains(type) else {
            return nil
        }
        let c = try decoder.container(keyedBy: Key.self)
        let contract = try c.decode(String.self, forKey: .contract)
        func count(_ key: Key) throws -> Int {
            let value = try c.decode(Int.self, forKey: key)
            guard value >= 0 else {
                throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: "Negative work count")
            }
            return value
        }
        func pair(_ completed: Key, _ total: Key) throws -> String {
            let done = try count(completed)
            let limit = try count(total)
            guard done <= limit else {
                throw DecodingError.dataCorruptedError(forKey: completed, in: c, debugDescription: "Work count exceeds total")
            }
            return "\(done)/\(limit)"
        }
        if type == "joint_candidate_look_pending" {
            guard contract == "joint_candidate_look_pending_v1" else {
                throw DecodingError.dataCorruptedError(forKey: .contract, in: c, debugDescription: "Unknown look continuation contract")
            }
            let index = try count(.candidateIndex)
            guard index < Int.max else {
                throw DecodingError.dataCorruptedError(forKey: .candidateIndex, in: c, debugDescription: "Candidate index overflows")
            }
            return Self(
                message: "후보 \(index + 1) · 두 번째 색감 보정 대기 · 대기 \(try count(.pending))개",
                shouldLog: true,
                isWarning: false
            )
        }
        if type == "joint_visual_selection_progress" {
            guard contract == "joint_selection_progress_v1" else {
                throw DecodingError.dataCorruptedError(forKey: .contract, in: c, debugDescription: "Unknown selection progress contract")
            }
            let phase = try c.decode(String.self, forKey: .phase)
            if phase.hasPrefix("proof_") {
                let candidate = try pair(.ordinal, .pool)
                let boundaries = try pair(.boundaries, .boundaryTotal)
                let bridges = try pair(.bridges, .bridgeTotal)
                let failure = try c.decodeIfPresent(String.self, forKey: .failure)
                let rejected = phase == "proof_failed" || failure != nil
                let label = rejected ? "후보 연결 검증 탈락" : "후보 연결 검증"
                return Self(
                    message: "\(label) \(candidate) · 컷 \(boundaries) · 구간 \(bridges)",
                    shouldLog: ["proof_start", "proof_failed", "proof_review_complete"].contains(phase),
                    isWarning: rejected
                )
            }
            let layers = try pair(.completed, .total)
            let profiles = try pair(.profileIndex, .profileTotal)
            return Self(
                message: "컷 후보 조합 분석 · 설정 \(profiles) · 경계 \(layers)",
                shouldLog: ["profile_start", "profile_complete", "selection_failed"].contains(phase),
                isWarning: phase == "selection_failed"
            )
        }
        guard contract == "joint_candidate_terminal_v1" else {
            throw DecodingError.dataCorruptedError(forKey: .contract, in: c, debugDescription: "Unknown candidate outcome contract")
        }
        let index = try count(.candidateIndex)
        guard index < Int.max else {
            throw DecodingError.dataCorruptedError(forKey: .candidateIndex, in: c, debugDescription: "Candidate index overflows")
        }
        let status = try c.decode(String.self, forKey: .status)
        let passed = status == "decoded_review_passed"
        let labels = [
            "decoded_review_passed": "사전 영상 QA 통과",
            "decoded_review_failed": "사전 영상 QA 탈락",
            "derivation_rejected": "분석에서 제외",
            "timeline_rejected": "편집 구성에서 제외",
            "splice_blocked": "컷 연결에서 제외",
            "proxy_execution_error": "사전 렌더 오류",
            "system_error": "처리 오류",
        ]
        let label = labels[status] ?? "검사 종료"
        let elapsed = try c.decode(Double.self, forKey: .elapsed)
        guard elapsed.isFinite, elapsed >= 0 else {
            throw DecodingError.dataCorruptedError(forKey: .elapsed, in: c, debugDescription: "Invalid elapsed time")
        }
        return Self(
            message: "후보 \(index + 1) \(label) · \(String(format: "%.1f", elapsed))초 · 사전 통과 \(try count(.passed))개",
            shouldLog: true,
            isWarning: !passed
        )
    }
}
private extension KeyedDecodingContainer where Key == EngineEvent.CodingKeys {
    func decodeFlexibleDouble(forKeys keys: [Key]) throws -> Double? {
        for key in keys {
            if let number = try? decode(Double.self, forKey: key), number.isFinite {
                return number
            }
            if let text = try? decode(String.self, forKey: key),
               let number = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)), number.isFinite {
                return number
            }
        }
        return nil
    }
}

/// Explicitly NOT EngineArtifactDisposition: cannot authorize result/delivery.
struct SceneReviewProposalDisposition: Decodable, Equatable, Sendable {
    let contract: String
    let status: String
    let artifactRole: String
    let qualityApproved: Bool
    let deliveryCompletionMessageAllowed: Bool
    let videoPath: String
    let videoSHA256: String
    let manifestPath: String
    let manifestSHA256: String

    enum CodingKeys: String, CodingKey {
        case contract, status
        case artifactRole = "artifact_role"
        case qualityApproved = "quality_approved"
        case deliveryCompletionMessageAllowed = "delivery_completion_message_allowed"
        case videoPath = "video_path"
        case videoSHA256 = "video_sha256"
        case manifestPath = "manifest_path"
        case manifestSHA256 = "manifest_sha256"
    }
    var isReviewOnly: Bool {
        contract == "normal_scene_review_proposal_v1" && status == "REVIEW_REQUIRED"
        && artifactRole == "scene_review_only" && !qualityApproved
        && !deliveryCompletionMessageAllowed
        && [videoSHA256, manifestSHA256].allSatisfy {
            $0.count == 64 && $0.allSatisfy { "0123456789abcdef".contains($0) }
        }
    }
}
