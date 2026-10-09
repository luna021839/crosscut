import Darwin
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

    @discardableResult
    func makeCompletedAutomaticGroupOutputSet(
        in directory: URL,
        timestamp: String,
        song: String = "BANG BANG",
        selectionPassed: Bool = true,
        videoData: Data? = Data([0, 0, 0, 1]),
        proxySchemaVersion: Int = 2,
        mutateAnalysis: ((inout [String: Any]) -> Void)? = nil
    ) throws -> [URL] {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let prefix = "IVE-\(song.replacingOccurrences(of: " ", with: "-"))-\(timestamp)"
        let analysisURL = directory.appendingPathComponent("\(prefix)-analysis.json")
        let planURL = directory.appendingPathComponent("\(prefix)-render-plan.json")
        let selectionURL = directory.appendingPathComponent(
            "\(prefix)-rendered-candidate-selection.json"
        )
        let fcpxmlURL = directory.appendingPathComponent(
            "\(prefix)-group-timeline.fcpxml"
        )
        let otioURL = directory.appendingPathComponent(
            "\(prefix)-group-timeline.otio"
        )
        let edlURL = directory.appendingPathComponent(
            "\(prefix)-group-timeline.edl"
        )
        let interchangeAuditURL = directory.appendingPathComponent(
            "\(prefix)-group-timeline-interchange-audit.json"
        )
        let videoURL = directory.appendingPathComponent("\(prefix)-group-crosscut.mp4")
        let interchangeFiles: [(String, URL, Data)] = [
            ("fcpxml", fcpxmlURL, Data("<fcpxml/>".utf8)),
            ("otio", otioURL, Data("{\"OTIO_SCHEMA\":\"Timeline.1\"}".utf8)),
            ("edl", edlURL, Data("TITLE: TEST\n".utf8)),
        ]
        for (_, url, data) in interchangeFiles {
            try data.write(to: url, options: .atomic)
        }
        let interchangeAudit: [String: Any] = [
            "schema_version": 1,
            "analysis_basis": "reviewed_render_plan_interchange_bundle_v1",
            "authoritative_output": videoURL.path,
            "output_frame_count": 2_997,
            "content_sha256": String(repeating: "d", count: 64),
            "artifacts": Dictionary(
                uniqueKeysWithValues: interchangeFiles.map { name, _, data in
                    (
                        name,
                        [
                            "sha256": String(repeating: "a", count: 64),
                            "size_bytes": data.count,
                        ] as [String: Any]
                    )
                }
            ),
        ]

        let proxyReview: [String: Any] = [
            "schema_version": proxySchemaVersion,
            "analysis": proxySchemaVersion == 3
                ? "decoded_output_proxy_domain_aware_quality_review_v3"
                : "decoded_output_proxy_all_transition_quality_review_v2",
            "proxy": [
                "path": NSNull(),
                "temporary_proxy_removed_after_review": true,
            ],
            "transition_count": 1,
            "reviewed_boundary_count": 1,
            "hard_cut_count": 1,
            "dissolve_count": 0,
            "passed": true,
        ]
        let selectedIndex = 3
        let candidates: [[String: Any]] = (0..<8).map { index in
            [
                "candidate_index": index,
                "selected": index == selectedIndex,
                "error": NSNull(),
                "review": proxyReview,
            ]
        }
        let selection: [String: Any] = [
            "analysis_basis": "sequential_decoded_proxy_candidate_rerank_v1",
            "storage_policy": [
                "one_proxy_at_a_time": true,
                "temporary_proxy_removed_after_each_review": true,
                "decoded_frames_persisted": false,
            ],
            "requested_review_count": 8,
            "reviewed_candidate_count": 8,
            "minimum_required_candidate_count": 8,
            "minimum_candidate_count_satisfied": true,
            "successful_review_count": 8,
            "selected_candidate_index": selectedIndex,
            "passed": selectionPassed,
            "candidates": candidates,
            "complete_candidate_count": 8,
            "structure_passing_materialized_count": 8,
            "exact_successful_decoded_review_count_satisfied": true,
            "exact_complete_candidate_count_satisfied": true,
            "final_selected_review": proxyReview,
            "final_selected_version": "original",
        ]

        let filenameFormatter = DateFormatter()
        filenameFormatter.locale = Locale(identifier: "en_US_POSIX")
        filenameFormatter.calendar = Calendar(identifier: .gregorian)
        filenameFormatter.timeZone = TimeZone(secondsFromGMT: 0)
        filenameFormatter.dateFormat = "yyyyMMdd-HHmmss"
        let creationDate = try XCTUnwrap(filenameFormatter.date(from: timestamp))
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let postRenderQA: [String: Any] = [
            "contract": "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2",
            "required": true,
            "verdict": "PASS",
            "passed": true,
        ]
        let appResultExposure: [String: Any] = [
            "contract": "automatic_ive_group_app_result_exposure_v1",
            "video_output_exposed": true,
            "delivery_completion_message_allowed": true,
        ]
        var analysis: [String: Any] = [
            "schema_version": 1,
            "created_at": isoFormatter.string(from: creationDate),
            "request": [
                "group": "IVE",
                "song": song,
                "edit_mode": "group",
                "source_mode": "automatic",
                "output_directory": directory.path,
            ],
            "artifact_role": "delivery",
            "app_result_exposure": appResultExposure,
            "post_render_qa": postRenderQA,
            "rendered_proxy_review": proxyReview,
            "rendered_candidate_selection": selection,
            "render": [
                "artifact_role": "delivery",
                "app_result_exposure": appResultExposure,
                "post_render_qa": postRenderQA,
                "video_path": videoURL.path,
                "plan_path": planURL.path,
                "video_codec": "libx264",
                "interchange": [
                    "fcpxml_path": fcpxmlURL.path,
                    "otio_path": otioURL.path,
                    "edl_path": edlURL.path,
                    "fidelity_audit_path": interchangeAuditURL.path,
                    "authoritative_pixels": videoURL.path,
                ],
                "rendered_proxy_review": proxyReview,
                "rendered_candidate_selection": selection,
            ],
            "capabilities": ["group_video_rendered": true],
        ]
        mutateAnalysis?(&analysis)
        let fps = 29.97
        let outputFrameCount = 2_997
        let plan: [String: Any] = [
            "aspect_ratio": "16:9",
            "output": videoURL.path,
            "width": 1_920,
            "height": 1_080,
            "fps": fps,
            "render_duration": Double(outputFrameCount) / fps,
            "output_frame_count": outputFrameCount,
        ]

        for (object, url) in [
            (analysis, analysisURL),
            (plan, planURL),
            (selection, selectionURL),
            (interchangeAudit, interchangeAuditURL),
        ] {
            let data = try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: url, options: .atomic)
        }
        if let videoData {
            try videoData.write(to: videoURL, options: .atomic)
        }
        return [
            analysisURL,
            planURL,
            selectionURL,
            fcpxmlURL,
            otioURL,
            edlURL,
            interchangeAuditURL,
            videoURL,
        ]
    }

    @discardableResult
    func makeCompletedAutomaticMemberOutputSet(
        in directory: URL,
        timestamp: String,
        targetMember: String = "가을",
        song: String = "ATTITUDE",
        mutateAnalysis: ((inout [String: Any]) -> Void)? = nil,
        mutatePostRenderQA: ((inout [String: Any]) -> Void)? = nil
    ) throws -> [URL] {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let prefix = "IVE-\(song)-\(timestamp)"
        let memberPrefix = "\(prefix)-\(targetMember)"
        let analysisURL = directory.appendingPathComponent("\(prefix)-analysis.json")
        let planURL = directory.appendingPathComponent(
            "\(memberPrefix)-render-plan.json"
        )
        let postRenderQAURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-post-render-qa.json"
        )
        let fcpxmlURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-timeline.fcpxml"
        )
        let otioURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-timeline.otio"
        )
        let edlURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-timeline.edl"
        )
        let interchangeAuditURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-timeline-interchange-audit.json"
        )
        let videoURL = directory.appendingPathComponent(
            "\(memberPrefix)-member-crosscut.mp4"
        )
        let interchangeFiles: [(String, URL, Data)] = [
            ("fcpxml", fcpxmlURL, Data("<fcpxml/>".utf8)),
            ("otio", otioURL, Data("{\"OTIO_SCHEMA\":\"Timeline.1\"}".utf8)),
            ("edl", edlURL, Data("TITLE: MEMBER TEST\n".utf8)),
        ]
        for (_, url, data) in interchangeFiles {
            try data.write(to: url, options: .atomic)
        }
        try Data([0, 0, 0, 1]).write(to: videoURL, options: .atomic)

        let outputFrameCount = 3_000
        let fps = 30.0
        let interchangeAudit: [String: Any] = [
            "schema_version": 1,
            "analysis_basis": "reviewed_render_plan_interchange_bundle_v1",
            "authoritative_output": videoURL.path,
            "output_frame_count": outputFrameCount,
            "content_sha256": String(repeating: "d", count: 64),
            "artifacts": Dictionary(
                uniqueKeysWithValues: interchangeFiles.map { name, _, data in
                    (
                        name,
                        [
                            "sha256": String(repeating: "a", count: 64),
                            "size_bytes": data.count,
                        ] as [String: Any]
                    )
                }
            ),
        ]
        let plan: [String: Any] = [
            "aspect_ratio": "9:16",
            "output": videoURL.path,
            "width": 1_080,
            "height": 1_920,
            "fps": fps,
            "render_duration": Double(outputFrameCount) / fps,
            "output_frame_count": outputFrameCount,
        ]

        let filenameFormatter = DateFormatter()
        filenameFormatter.locale = Locale(identifier: "en_US_POSIX")
        filenameFormatter.calendar = Calendar(identifier: .gregorian)
        filenameFormatter.timeZone = TimeZone(secondsFromGMT: 0)
        filenameFormatter.dateFormat = "yyyyMMdd-HHmmss"
        let creationDate = try XCTUnwrap(filenameFormatter.date(from: timestamp))
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let exposure: [String: Any] = [
            "contract": "automatic_ive_member_app_result_exposure_v1",
            "video_output_exposed": true,
            "delivery_completion_message_allowed": true,
        ]
        let postRenderQAEvidence: [String: Any] = [
            "contract": "automatic_ive_member_mandatory_post_render_qa_v1",
            "required": true,
            "verdict": "PASS",
            "report_passed": true,
            "content_binding_passed": true,
            "passed": true,
            "report_path": postRenderQAURL.path,
        ]
        let interchange: [String: Any] = [
            "fcpxml_path": fcpxmlURL.path,
            "otio_path": otioURL.path,
            "edl_path": edlURL.path,
            "fidelity_audit_path": interchangeAuditURL.path,
            "authoritative_pixels": videoURL.path,
        ]
        var analysis: [String: Any] = [
            "schema_version": 1,
            "created_at": isoFormatter.string(from: creationDate),
            "request": [
                "group": "IVE",
                "song": song,
                "edit_mode": "member",
                "source_mode": "automatic",
                "output_directory": directory.path,
                "aspect_ratio": "9:16",
                "target_member": targetMember,
            ],
            "member_direct_full_frame": ["enabled": true],
            "artifact_role": "delivery",
            "app_result_exposure": exposure,
            "post_render_qa": postRenderQAEvidence,
            "render": [
                "artifact_role": "delivery",
                "app_result_exposure": exposure,
                "post_render_qa": postRenderQAEvidence,
                "video_path": videoURL.path,
                "plan_path": planURL.path,
                "video_codec": "libx264",
                "direct_full_frame": true,
                "interchange": interchange,
            ],
            "capabilities": [
                "member_video_rendered": true,
                "member_direct_full_frame_ready": true,
            ],
        ]
        mutateAnalysis?(&analysis)

        var postRenderQA: [String: Any] = [
            "schema_version": 1,
            "analysis": "automatic_ive_member_final_media_and_pixel_provenance_qa_v1",
            "artifact_set": [
                "analysis_path": analysisURL.path,
                "render_plan_path": planURL.path,
                "video_path": videoURL.path,
            ],
            "verdict": "PASS",
            "passed": true,
            "issue_count": 0,
            "render_receipt": [
                "contract": "automatic_ive_member_exact_source_time_pixel_provenance_receipt_v1",
                "all_eligible_sources_pixel_proven": true,
                "source_file_identities_stable": true,
                "artifact_content_stable": true,
                "passed": true,
            ],
        ]
        mutatePostRenderQA?(&postRenderQA)

        for (object, url) in [
            (analysis, analysisURL),
            (plan, planURL),
            (postRenderQA, postRenderQAURL),
            (interchangeAudit, interchangeAuditURL),
        ] {
            let data = try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: url, options: .atomic)
        }
        return [
            analysisURL,
            planURL,
            postRenderQAURL,
            fcpxmlURL,
            otioURL,
            edlURL,
            interchangeAuditURL,
            videoURL,
        ]
    }

    private func mutateRender(
        in analysis: inout [String: Any],
        _ mutation: (inout [String: Any]) -> Void
    ) {
        guard var render = analysis["render"] as? [String: Any] else {
            XCTFail("fixture render payload is missing")
            return
        }
        mutation(&render)
        analysis["render"] = render
    }

    func makeReviewReadyOutputSet(in directory: URL, mode: EditMode) throws -> [URL] {
        let files = try mode == .group
            ? makeCompletedAutomaticGroupOutputSet(in: directory, timestamp: "20260908-120002", song: "XOXZ")
            : makeCompletedAutomaticMemberOutputSet(in: directory, timestamp: "20260908-120002", song: "XOXZ")
        func read(_ url: URL) throws -> [String: Any] {
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        }
        func write(_ object: [String: Any], _ url: URL) throws {
            try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url, options: .atomic)
        }
        var analysis = try read(files[0])
        var render = try XCTUnwrap(analysis["render"] as? [String: Any])
        var exposure = try XCTUnwrap(analysis["app_result_exposure"] as? [String: Any])
        var qa = try XCTUnwrap(analysis["post_render_qa"] as? [String: Any])
        exposure["delivery_completion_message_allowed"] = false
        qa["passed"] = false
        qa["verdict"] = "FAIL"
        qa["report_passed"] = false
        qa["content_binding_passed"] = true
        qa["promotion_binding"] = ["passed": true]
        qa["review_first_disposition"] = [
            "contract": "ive_review_first_runtime_v1", "status": "REVIEW_READY",
            "artifact_role": "review_ready", "video_output_exposed": true,
            "delivery_completion_message_allowed": false, "technical_issue_codes": [],
            "editorial_issue_codes": ["source_switch_density_below_hard_minimum"],
        ] as [String: Any]
        analysis["artifact_role"] = "review_ready"
        render["artifact_role"] = "review_ready"
        analysis["app_result_exposure"] = exposure
        render["app_result_exposure"] = exposure
        analysis["post_render_qa"] = qa
        render["post_render_qa"] = qa
        if mode == .group {
            var selection = try read(files[2])
            let policy: [String: Any] = [
                "contract": "bounded_review_first_single_delivery_v1",
                "required_complete_candidate_count": 1, "required_decoded_review_count": 1,
                "candidate_pool": "bounded_review_first_single_full_song_v1",
                "selection_mode": "one_complete_full_song_candidate_then_final_qa",
                "source_coverage_requirements_changed": false,
                "individual_candidate_quality_requirements_changed": false,
                "silent_unreviewed_fallback_allowed": false,
            ]
            var review = try XCTUnwrap(selection["final_selected_review"] as? [String: Any])
            review["passed"] = false
            for key in ["requested_review_count", "reviewed_candidate_count", "minimum_required_candidate_count", "successful_review_count", "complete_candidate_count"] {
                selection[key] = 1
            }
            selection.removeValue(forKey: "structure_passing_materialized_count")
            selection["analysis_basis"] = "bounded_review_first_single_full_song_v1"
            selection["release_policy"] = policy
            selection["candidate_pool"] = "bounded_review_first_single_full_song_v1"
            selection["selected_path_authority"] = "deterministic_full_song_visual_selector"
            selection["selected_candidate_index"] = 0
            selection["passed"] = false
            selection["candidates"] = [["candidate_index": 0, "selected": true, "review": review]]
            selection["final_selected_review"] = review
            analysis["rendered_candidate_selection"] = selection
            render["rendered_candidate_selection"] = selection
            analysis["rendered_proxy_review"] = review
            render["rendered_proxy_review"] = review
            render["release_policy"] = policy
            render["candidate_pool"] = "bounded_review_first_single_full_song_v1"
            render["candidate_origin"] = "deterministic_full_song_visual_selector"
            try write(selection, files[2])
        } else {
            var report = try read(files[2])
            report["passed"] = false
            report["verdict"] = "FAIL"
            report["issue_count"] = 1
            try write(report, files[2])
        }
        analysis["render"] = render
        try write(analysis, files[0])
        return files
    }

    func testReviewReadyGroupAndMemberArePlayableButNeverApproved() throws {
        for mode in [EditMode.group, .member] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let files = try makeReviewReadyOutputSet(in: directory, mode: mode)
            let request = EditRequest(song: "XOXZ", targetMember: .gaeul, editMode: mode,
                aspectRatio: mode == .member ? .portrait : .landscape,
                sources: [], outputDirectory: directory, sourceMode: .automatic)
            let accepted = try XCTUnwrap(CompletedEngineOutputSetDiscovery.reported(files, for: request))
            XCTAssertEqual(accepted.artifactRole, "review_ready")
            var analysis = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: files[0])) as? [String: Any])
            var qa = analysis["post_render_qa"] as! [String: Any]
            var disposition = qa["review_first_disposition"] as! [String: Any]
            disposition["technical_issue_codes"] = ["unclassified_technical_failure"]
            qa["review_first_disposition"] = disposition
            analysis["post_render_qa"] = qa
            var render = analysis["render"] as! [String: Any]
            render["post_render_qa"] = qa
            analysis["render"] = render
            try JSONSerialization.data(withJSONObject: analysis).write(to: files[0], options: .atomic)
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(files, for: request))
        }
    }

    func testReviewReadyFamilyOwnersRequireTheirMatchingCompleteReceipt() throws {
        for owner in ["normal_performer_family_camera_owner_v1",
                      "normal_source_changing_family_owner_v1",
                      "selected_native_motion_complete_metadata_v1"] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let files = try makeReviewReadyOutputSet(in: directory, mode: .group)
            var analysis = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: files[0])) as? [String: Any])
            var selection = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: files[2])) as? [String: Any])
            var render = try XCTUnwrap(analysis["render"] as? [String: Any])
            let receipt: [String: Any] = ["contract": owner,
                "plan_sha256": String(repeating: "a", count: 64),
                "metadata_sha256": String(repeating: "b", count: 64),
                "quality_approved": false, "cross_camera_identity_certified": false,
                "body_phase_certified": false, "decoded_output_quality_certified": false]
            selection["selected_path_authority"] = owner
            selection["complete_candidates"] = [["candidate_origin": owner, "performer_family_review": receipt]]
            render["candidate_origin"] = owner
            render["performer_family_review"] = receipt
            func writeCurrent() throws {
                analysis["rendered_candidate_selection"] = selection
                render["rendered_candidate_selection"] = selection
                analysis["render"] = render
                try JSONSerialization.data(withJSONObject: selection).write(to: files[2], options: .atomic)
                try JSONSerialization.data(withJSONObject: analysis).write(to: files[0], options: .atomic)
            }
            try writeCurrent()
            let request = EditRequest(song: "XOXZ", targetMember: .gaeul, editMode: .group,
                aspectRatio: .landscape, sources: [], outputDirectory: directory, sourceMode: .automatic)
            let accepted = try XCTUnwrap(CompletedEngineOutputSetDiscovery.reported(files, for: request), owner)
            XCTAssertEqual(accepted.artifactRole, "review_ready")
            render.removeValue(forKey: "performer_family_review")
            try writeCurrent()
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(files, for: request), "missing receipt")
            render["performer_family_review"] = receipt
            var changed = receipt
            changed["metadata_sha256"] = String(repeating: "c", count: 64)
            selection["complete_candidates"] = [["candidate_origin": owner, "performer_family_review": changed]]
            try writeCurrent()
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(files, for: request), "different selected metadata")
            changed = receipt
            changed["cross_camera_identity_certified"] = true
            render["performer_family_review"] = changed
            selection["complete_candidates"] = [["candidate_origin": owner, "performer_family_review": changed]]
            try writeCurrent()
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(files, for: request), "invented identity approval")
        }
    }

    func testApprovedSingleDeliveryCohortsDoNotRequireEightFinalists() throws {
        for learned in [false, true] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let files = try makeCompletedAutomaticGroupOutputSet(in: directory, timestamp: "20260908-120003", song: "XOXZ")
            var analysis = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: files[0])) as? [String: Any])
            var selection = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: files[2])) as? [String: Any])
            var render = try XCTUnwrap(analysis["render"] as? [String: Any])
            var policy: [String: Any] = [
                "contract": learned ? "stage_edit_quality_gated_single_delivery_v1" : "joint_quality_gated_single_delivery_v1",
                "minimum_passing_candidate_count": 1, "comparison_candidate_budget": 8,
                "source_coverage_requirements_changed": false, "individual_candidate_quality_requirements_changed": false,
            ]
            if learned {
                policy["minimum_visual_passing_learned_candidate_count"] = 1
                policy["comparison_selection"] = "rank_all_available_decoded_passing_candidates"
                policy["silent_baseline_fallback_allowed"] = false
            }
            let review = try XCTUnwrap(selection["final_selected_review"] as? [String: Any])
            for key in ["requested_review_count", "reviewed_candidate_count", "minimum_required_candidate_count", "successful_review_count", "complete_candidate_count"] {
                selection[key] = 1
            }
            selection["structure_passing_materialized_count"] = 7
            selection["selected_candidate_index"] = 6
            selection["candidates"] = [["candidate_index": 6, "selected": true, "review": review]]
            selection["release_policy"] = policy
            selection["candidate_pool"] = learned ? "stage_edit_generation_only" : "joint_decoded_visual_selector"
            render["candidate_origin"] = learned ? "explicit_stage_edit_generation" : "joint_decoded_visual_selector"
            render["release_policy"] = policy
            var qa = try XCTUnwrap(analysis["post_render_qa"] as? [String: Any])
            qa["promotion_binding"] = ["passed": true]
            qa["review_first_disposition"] = [
                "contract": "ive_review_first_runtime_v1", "status": "APPROVED", "artifact_role": "delivery",
                "video_output_exposed": true, "delivery_completion_message_allowed": true,
                "technical_issue_codes": [], "editorial_issue_codes": [],
            ] as [String: Any]
            analysis["post_render_qa"] = qa
            render["post_render_qa"] = qa
            analysis["rendered_candidate_selection"] = selection
            render["rendered_candidate_selection"] = selection
            analysis["render"] = render
            try JSONSerialization.data(withJSONObject: selection).write(to: files[2], options: .atomic)
            try JSONSerialization.data(withJSONObject: analysis).write(to: files[0], options: .atomic)
            let request = EditRequest(song: "XOXZ", targetMember: .gaeul, editMode: .group, aspectRatio: .landscape,
                sources: [], outputDirectory: directory, sourceMode: .automatic)
            let result = try XCTUnwrap(CompletedEngineOutputSetDiscovery.reported(files, for: request))
            XCTAssertEqual(result.artifactRole, "delivery")
        }
    }

    func testReportedDeliveryUsesOnlyTheExplicitArtifactSet() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-reported-set-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        for mode in [EditMode.group, .member] {
            let directory = root.appendingPathComponent(mode.rawValue)
            let files = try mode == .group
                ? makeCompletedAutomaticGroupOutputSet(in: directory, timestamp: "20260908-120000", song: "XOXZ")
                : makeCompletedAutomaticMemberOutputSet(in: directory, timestamp: "20260908-120000", song: "XOXZ")
            let request = EditRequest(
                song: "XOXZ", targetMember: .gaeul, editMode: mode,
                aspectRatio: mode == .member ? .portrait : .landscape,
                sources: [], outputDirectory: directory, sourceMode: .automatic
            )
            let extraVideo = directory.appendingPathComponent("unreviewed-preview.mp4")
            try Data([1]).write(to: extraVideo)
            let accepted = try XCTUnwrap(
                CompletedEngineOutputSetDiscovery.reported(files + [extraVideo], for: request)
            )
            XCTAssertEqual(accepted.outputURLs, files.map(\.standardizedFileURL))
            // A valid older set in the directory cannot fill missing reported files.
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported([files.last!], for: request))
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(Array(files.dropFirst()), for: request))
        }
    }

    func testReportedDeliveryBindsTheLiveMemberModeAndDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-reported-request-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let files = try makeCompletedAutomaticMemberOutputSet(
            in: root, timestamp: "20260908-120001", song: "XOXZ"
        )
        for (mode, member, directory, song) in [
            (EditMode.member, IVEMember.anYujin, root, "XOXZ"),
            (EditMode.group, IVEMember.gaeul, root, "XOXZ"),
            (EditMode.member, IVEMember.gaeul, root.appendingPathComponent("different"), "XOXZ"),
            (EditMode.member, IVEMember.gaeul, root, "I AM"),
        ] {
            let request = EditRequest(
                song: song, targetMember: member, editMode: mode,
                aspectRatio: .portrait, sources: [], outputDirectory: directory,
                sourceMode: .automatic
            )
            XCTAssertNil(CompletedEngineOutputSetDiscovery.reported(files, for: request))
        }
    }

    private func assertAutomaticGroupOutputIsNotRestorable(
        timestamp: String = "20260823-070707",
        mutateAnalysis: @escaping (inout [String: Any]) -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ive-crosscut-output-rejected-\(UUID().uuidString)"
            )
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let files = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: timestamp,
            mutateAnalysis: mutateAnalysis
        )

        XCTAssertEqual(files.count, 8, file: file, line: line)
        XCTAssertTrue(
            files.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
            file: file,
            line: line
        )
        XCTAssertNil(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .group),
            file: file,
            line: line
        )
    }

    private func assertAutomaticMemberOutputIsNotRestorable(
        timestamp: String = "20260825-120000",
        mutateAnalysis: ((inout [String: Any]) -> Void)? = nil,
        mutatePostRenderQA: ((inout [String: Any]) -> Void)? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ive-crosscut-member-output-rejected-\(UUID().uuidString)"
            )
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let files = try makeCompletedAutomaticMemberOutputSet(
            in: root,
            timestamp: timestamp,
            mutateAnalysis: mutateAnalysis,
            mutatePostRenderQA: mutatePostRenderQA
        )

        XCTAssertEqual(files.count, 8, file: file, line: line)
        XCTAssertTrue(
            files.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
            file: file,
            line: line
        )
        XCTAssertNil(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .member),
            file: file,
            line: line
        )
    }

    func testCompletedOutputDiscoverySeparatesGroupAndMemberResults() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ive-crosscut-mode-separated-output-\(UUID().uuidString)"
            )
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let member = try makeCompletedAutomaticMemberOutputSet(
            in: root,
            timestamp: "20260825-020202"
        )
        let group = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260825-030303"
        )

        let restoredMember = try XCTUnwrap(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .member)
        )
        XCTAssertEqual(restoredMember.editMode, .member)
        XCTAssertEqual(restoredMember.timestamp, "20260825-020202")
        XCTAssertEqual(restoredMember.outputURLs, member.map(\.standardizedFileURL))

        let restoredGroup = try XCTUnwrap(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .group)
        )
        XCTAssertEqual(restoredGroup.editMode, .group)
        XCTAssertEqual(restoredGroup.timestamp, "20260825-030303")
        XCTAssertEqual(restoredGroup.outputURLs, group.map(\.standardizedFileURL))
    }

    func testCompletedMemberOutputDiscoveryRejectsNonDeliveryOrExposureMismatch() throws {
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            analysis["artifact_role"] = "rejected"
            self.mutateRender(in: &analysis) { $0["artifact_role"] = "rejected" }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var exposure = analysis["app_result_exposure"] as? [String: Any] ?? [:]
            exposure["video_output_exposed"] = false
            analysis["app_result_exposure"] = exposure
            self.mutateRender(in: &analysis) {
                $0["app_result_exposure"] = exposure
            }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            self.mutateRender(in: &analysis) { render in
                var exposure = render["app_result_exposure"]
                    as? [String: Any] ?? [:]
                exposure["contract"] = "automatic_ive_group_app_result_exposure_v1"
                render["app_result_exposure"] = exposure
            }
        })
    }

    func testCompletedMemberOutputDiscoveryRejectsWrongOrUnboundMemberQA() throws {
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
            qa["contract"] = "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
            analysis["post_render_qa"] = qa
            self.mutateRender(in: &analysis) { $0["post_render_qa"] = qa }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
            qa["report_passed"] = false
            analysis["post_render_qa"] = qa
            self.mutateRender(in: &analysis) { $0["post_render_qa"] = qa }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
            qa["content_binding_passed"] = false
            analysis["post_render_qa"] = qa
            self.mutateRender(in: &analysis) { $0["post_render_qa"] = qa }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
            qa["report_path"] = "/tmp/unrelated-member-qa.json"
            analysis["post_render_qa"] = qa
            self.mutateRender(in: &analysis) { $0["post_render_qa"] = qa }
        })
    }

    func testCompletedMemberOutputDiscoveryRequiresDirectFullFrameContract() throws {
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            analysis["member_direct_full_frame"] = ["enabled": false]
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            self.mutateRender(in: &analysis) { $0["direct_full_frame"] = false }
        })
        try assertAutomaticMemberOutputIsNotRestorable(mutateAnalysis: { analysis in
            var capabilities = analysis["capabilities"] as? [String: Any] ?? [:]
            capabilities["member_direct_full_frame_ready"] = false
            analysis["capabilities"] = capabilities
        })
    }

    func testCompletedMemberOutputDiscoveryRequiresPassingPixelProvenanceReceipt() throws {
        let receiptMutations: [(inout [String: Any]) -> Void] = [
            { $0["all_eligible_sources_pixel_proven"] = false },
            { $0["source_file_identities_stable"] = false },
            { $0["artifact_content_stable"] = false },
            { $0["passed"] = false },
            { $0["contract"] = "unknown_receipt" },
        ]
        for mutation in receiptMutations {
            try assertAutomaticMemberOutputIsNotRestorable(
                mutatePostRenderQA: { report in
                    var receipt = report["render_receipt"]
                        as? [String: Any] ?? [:]
                    mutation(&receipt)
                    report["render_receipt"] = receipt
                }
            )
        }
    }

    func testCompletedOutputDiscoveryUsesLatestValidSameTimestampSet() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-output-discovery-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }

        let expected = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-010101"
        )
        _ = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-020202",
            selectionPassed: false
        )
        _ = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-030303",
            videoData: nil
        )
        try Data("temporary".utf8).write(
            to: root.appendingPathComponent(".partial-candidate-03.mp4")
        )
        try Data("temporary".utf8).write(
            to: root.appendingPathComponent(".ive-crosscut-candidate-proxy-00.mp4")
        )

        let result = try XCTUnwrap(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .group)
        )
        XCTAssertEqual(result.editMode, .group)
        XCTAssertEqual(result.timestamp, "20260823-010101")
        XCTAssertEqual(
            result.outputURLs,
            expected.map(\.standardizedFileURL)
        )
    }

    func testCompletedOutputDiscoveryAcceptsDomainAwareProxySchemaV3() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-output-v3-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }

        let expected = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-060606",
            proxySchemaVersion: 3
        )

        let result = try XCTUnwrap(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .group)
        )
        XCTAssertEqual(result.editMode, .group)
        XCTAssertEqual(result.timestamp, "20260823-060606")
        XCTAssertEqual(result.outputURLs, expected.map(\.standardizedFileURL))
    }

    func testCompletedOutputDiscoveryRejectsPendingArtifactRoleWithAllEightFiles() throws {
        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            analysis["artifact_role"] = "pending"
            self.mutateRender(in: &analysis) { render in
                render["artifact_role"] = "pending"
            }
        }
    }

    func testCompletedOutputDiscoveryRejectsRejectedArtifactRoleWithAllEightFiles() throws {
        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            analysis["artifact_role"] = "rejected"
            self.mutateRender(in: &analysis) { render in
                render["artifact_role"] = "rejected"
            }
        }
    }

    func testCompletedOutputDiscoveryRejectsComparisonArtifactRoleWithAllEightFiles() throws {
        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            analysis["artifact_role"] = "comparison"
            self.mutateRender(in: &analysis) { render in
                render["artifact_role"] = "comparison"
            }
        }
    }

    func testCompletedOutputDiscoveryRejectsMissingDeliveryEvidence() throws {
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0.removeValue(forKey: "artifact_role") },
            { $0.removeValue(forKey: "app_result_exposure") },
            { $0.removeValue(forKey: "post_render_qa") },
            { analysis in
                self.mutateRender(in: &analysis) {
                    $0.removeValue(forKey: "artifact_role")
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) {
                    $0.removeValue(forKey: "app_result_exposure")
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) {
                    $0.removeValue(forKey: "post_render_qa")
                }
            },
        ]
        for mutation in mutations {
            try assertAutomaticGroupOutputIsNotRestorable(
                mutateAnalysis: mutation
            )
        }
    }

    func testCompletedOutputDiscoveryRejectsWrongOrMissingPostRenderQAContract() throws {
        let mutations: [(inout [String: Any]) -> Void] = [
            { analysis in
                var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
                qa["contract"] = "unknown_contract"
                analysis["post_render_qa"] = qa
                self.mutateRender(in: &analysis) { render in
                    render["post_render_qa"] = qa
                }
            },
            { analysis in
                var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
                qa.removeValue(forKey: "contract")
                analysis["post_render_qa"] = qa
                self.mutateRender(in: &analysis) { render in
                    render["post_render_qa"] = qa
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) { render in
                    var qa = render["post_render_qa"]
                        as? [String: Any] ?? [:]
                    qa["contract"] = "unknown_contract"
                    render["post_render_qa"] = qa
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) { render in
                    var qa = render["post_render_qa"]
                        as? [String: Any] ?? [:]
                    qa.removeValue(forKey: "contract")
                    render["post_render_qa"] = qa
                }
            },
        ]
        for mutation in mutations {
            try assertAutomaticGroupOutputIsNotRestorable(
                mutateAnalysis: mutation
            )
        }
    }

    func testCompletedOutputDiscoveryRejectsMissingOrMismatchedRenderExposure() throws {
        let mutations: [(inout [String: Any]) -> Void] = [
            { analysis in
                self.mutateRender(in: &analysis) {
                    $0.removeValue(forKey: "app_result_exposure")
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) { render in
                    var exposure = render["app_result_exposure"]
                        as? [String: Any] ?? [:]
                    exposure["contract"] = "unknown_contract"
                    render["app_result_exposure"] = exposure
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) { render in
                    var exposure = render["app_result_exposure"]
                        as? [String: Any] ?? [:]
                    exposure["video_output_exposed"] = false
                    render["app_result_exposure"] = exposure
                }
            },
            { analysis in
                self.mutateRender(in: &analysis) { render in
                    var exposure = render["app_result_exposure"]
                        as? [String: Any] ?? [:]
                    exposure["delivery_completion_message_allowed"] = false
                    render["app_result_exposure"] = exposure
                }
            },
        ]
        for mutation in mutations {
            try assertAutomaticGroupOutputIsNotRestorable(
                mutateAnalysis: mutation
            )
        }
    }

    func testCompletedOutputDiscoveryRejectsMismatchedRenderDeliveryEvidence() throws {
        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            self.mutateRender(in: &analysis) { render in
                render["artifact_role"] = "comparison"
            }
        }

        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            self.mutateRender(in: &analysis) { render in
                var qa = render["post_render_qa"] as? [String: Any] ?? [:]
                qa["verdict"] = "FAIL"
                qa["passed"] = false
                render["post_render_qa"] = qa
            }
        }
    }

    func testCompletedOutputDiscoveryRejectsPostRenderQAFailWithAllEightFiles() throws {
        try assertAutomaticGroupOutputIsNotRestorable { analysis in
            var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
            qa["verdict"] = "FAIL"
            qa["passed"] = false
            analysis["post_render_qa"] = qa
            self.mutateRender(in: &analysis) { render in
                render["post_render_qa"] = qa
            }
        }
    }

    func testCompletedOutputDiscoveryRejectsPendingOrRejectedQAVerdicts() throws {
        for verdict in ["PENDING", "REJECTED"] {
            try assertAutomaticGroupOutputIsNotRestorable { analysis in
                var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
                qa["verdict"] = verdict
                // A verdict other than PASS remains ineligible even if a
                // malformed producer also claims the boolean passed flag.
                qa["passed"] = true
                analysis["post_render_qa"] = qa
                self.mutateRender(in: &analysis) { render in
                    render["post_render_qa"] = qa
                }
            }
        }
    }

    func testCompletedOutputDiscoveryRequiresExactExposureAndQAAuthorization() throws {
        let mutations: [(inout [String: Any]) -> Void] = [
            { analysis in
                var exposure = analysis["app_result_exposure"]
                    as? [String: Any] ?? [:]
                exposure["contract"] = "unknown_contract"
                analysis["app_result_exposure"] = exposure
                self.mutateRender(in: &analysis) { render in
                    render["app_result_exposure"] = exposure
                }
            },
            { analysis in
                var exposure = analysis["app_result_exposure"]
                    as? [String: Any] ?? [:]
                exposure["video_output_exposed"] = false
                analysis["app_result_exposure"] = exposure
                self.mutateRender(in: &analysis) { render in
                    render["app_result_exposure"] = exposure
                }
            },
            { analysis in
                var exposure = analysis["app_result_exposure"]
                    as? [String: Any] ?? [:]
                exposure["delivery_completion_message_allowed"] = false
                analysis["app_result_exposure"] = exposure
                self.mutateRender(in: &analysis) { render in
                    render["app_result_exposure"] = exposure
                }
            },
            { analysis in
                var qa = analysis["post_render_qa"] as? [String: Any] ?? [:]
                qa["required"] = false
                analysis["post_render_qa"] = qa
                self.mutateRender(in: &analysis) { render in
                    render["post_render_qa"] = qa
                }
            },
        ]
        for mutation in mutations {
            try assertAutomaticGroupOutputIsNotRestorable(
                mutateAnalysis: mutation
            )
        }
    }

    func testCompletedOutputDiscoveryRejectsEmptyOrSymlinkedVideo() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-output-failclosed-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }

        _ = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-040404",
            videoData: Data()
        )
        var symlinked = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-050505",
            videoData: nil
        )
        let realVideo = root.appendingPathComponent("unrelated.mp4")
        try Data([0, 0, 0, 1]).write(to: realVideo)
        try FileManager.default.createSymbolicLink(
            at: symlinked.removeLast(),
            withDestinationURL: realVideo
        )

        XCTAssertNil(
            CompletedEngineOutputSetDiscovery.latest(in: root, editMode: .group)
        )
    }

    @MainActor
    func testEngineRunnerRestoresDiscoveredOutputsAndClearsThemForEmptyFolder() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-runner-restore-\(UUID().uuidString)")
        let empty = root.appendingPathComponent("empty", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let expected = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260823-060606"
        )
        let runner = EngineRunner(locateEngine: { nil })

        runner.restoreLatestCompletedResult(in: root, editMode: .group)
        for _ in 0..<100 {
            if !runner.isRestoringCompletedOutputs { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(runner.isRestoringCompletedOutputs)
        XCTAssertEqual(runner.state, .completed)
        XCTAssertEqual(runner.progress, 1)
        XCTAssertEqual(runner.outputURLs, expected.map(\.standardizedFileURL))

        runner.restoreLatestCompletedResult(
            in: empty,
            editMode: .group,
            reportMissing: false,
            clearExistingOutputsOnMissing: true
        )
        for _ in 0..<100 {
            if !runner.isRestoringCompletedOutputs { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(runner.isRestoringCompletedOutputs)
        XCTAssertEqual(runner.state, .idle)
        XCTAssertTrue(runner.outputURLs.isEmpty)
    }

    @MainActor
    func testEngineRunnerRestoresGroupAndMemberModesIndependently() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ive-crosscut-runner-mode-restore-\(UUID().uuidString)"
            )
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let member = try makeCompletedAutomaticMemberOutputSet(
            in: root,
            timestamp: "20260825-060606"
        )
        let group = try makeCompletedAutomaticGroupOutputSet(
            in: root,
            timestamp: "20260825-070707"
        )
        let runner = EngineRunner(locateEngine: { nil })

        runner.restoreLatestCompletedResult(in: root, editMode: .member)
        for _ in 0..<100 {
            if !runner.isRestoringCompletedOutputs { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(runner.outputURLs, member.map(\.standardizedFileURL))
        XCTAssertEqual(
            runner.renderedVideoURL,
            member.last?.standardizedFileURL
        )

        runner.restoreLatestCompletedResult(in: root, editMode: .group)
        for _ in 0..<100 {
            if !runner.isRestoringCompletedOutputs { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(runner.outputURLs, group.map(\.standardizedFileURL))
        XCTAssertEqual(
            runner.renderedVideoURL,
            group.last?.standardizedFileURL
        )
    }

    func testLargeMediaDefaultsUseDedicatedExternalVolumeWhenMounted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-storage-\(UUID().uuidString)")
        let external = root.appendingPathComponent("ssd", isDirectory: true)
        let fallback = root.appendingPathComponent("Movies", isDirectory: true)
        try FileManager.default.createDirectory(
            at: external,
            withIntermediateDirectories: true
        )
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }

        let directories = LargeMediaStoragePolicy.defaultDirectories(
            preferredVolume: external,
            fallbackMoviesDirectory: fallback
        )

        XCTAssertTrue(directories.usesPreferredExternalVolume)
        XCTAssertEqual(
            directories.output.standardizedFileURL.path,
            external.appendingPathComponent("CrossCut/Outputs").path
        )
        XCTAssertEqual(
            directories.sourceCache.standardizedFileURL.path,
            external.appendingPathComponent("CrossCut/Sources").path
        )
    }

    func testLargeMediaDefaultsFallBackOnlyWhenExternalVolumeIsUnavailable() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ive-crosscut-storage-\(UUID().uuidString)")
        let missingExternal = root.appendingPathComponent("missing-ssd")
        let fallback = root.appendingPathComponent("Movies", isDirectory: true)

        let directories = LargeMediaStoragePolicy.defaultDirectories(
            preferredVolume: missingExternal,
            fallbackMoviesDirectory: fallback
        )

        XCTAssertFalse(directories.usesPreferredExternalVolume)
        XCTAssertEqual(
            directories.output.standardizedFileURL.path,
            fallback.appendingPathComponent("CrossCut").path
        )
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
        XCTAssertEqual(object["source_mode"] as? String, "local")
        XCTAssertEqual(object["desired_source_count"] as? Int, 10)
        XCTAssertEqual(object["acquire_only"] as? Bool, false)
        XCTAssertEqual(object["member_pose_dissolve_enabled"] as? Bool, true)
        let anchors = try XCTUnwrap(object["member_anchors"] as? [[String: Any]])
        XCTAssertEqual(anchors.count, 1)
        XCTAssertEqual(anchors[0]["source_index"] as? Int, 0)
        XCTAssertEqual(anchors[0]["time_seconds"] as? Double, 12.5)
        XCTAssertEqual(anchors[0]["bbox_xyxy"] as? [Double], [0.1, 0.2, 0.7, 0.9])
    }

    func testCompleteGroupOriginalReferenceRoundTripsWithoutClearingInputAuthority() throws {
        let original = try GroupOriginalReferenceInput(file: URL(fileURLWithPath: "/tmp/complete-original.json"),
            sha256: String(repeating: "a", count: 64))
        let request = EditRequest(song: "XOXZ", targetMember: .gaeul, editMode: .group,
            aspectRatio: .landscape, sources: [], outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            sourceMode: .automatic, groupExternalOriginalReference: original)
        let data = try JSONEncoder().encode(request)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let input = try XCTUnwrap(object["group_external_original_reference"] as? [String: String])
        XCTAssertEqual(input["file"], "/tmp/complete-original.json")
        XCTAssertEqual(input["sha256"], String(repeating: "a", count: 64))
        XCTAssertEqual(try JSONDecoder().decode(EditRequest.self, from: data), request)
        var changed = object
        changed["edit_mode"] = "member"
        XCTAssertThrowsError(try JSONDecoder().decode(EditRequest.self,
            from: JSONSerialization.data(withJSONObject: changed)))
        changed = object
        changed["acquire_only"] = true
        XCTAssertThrowsError(try JSONDecoder().decode(EditRequest.self,
            from: JSONSerialization.data(withJSONObject: changed)))
    }

    func testCompleteGroupOriginalReferenceRejectsMalformedPathsAndContentHashes() throws {
        for hash in ["", String(repeating: "a", count: 63), String(repeating: "A", count: 64),
                     String(repeating: "z", count: 64)] {
            XCTAssertThrowsError(try GroupOriginalReferenceInput(
                file: URL(fileURLWithPath: "/tmp/original.json"), sha256: hash))
        }
        let malformed = ["file": "relative.json", "sha256": String(repeating: "a", count: 64)]
        XCTAssertThrowsError(try JSONDecoder().decode(GroupOriginalReferenceInput.self,
            from: JSONSerialization.data(withJSONObject: malformed)))
    }

    func testLegacyRequestDoesNotImplicitlyActivateExternalOriginalPreservation() throws {
        let request = EditRequest(song: "XOXZ", targetMember: .gaeul, editMode: .group,
            aspectRatio: .landscape, sources: [], outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            sourceMode: .automatic)
        let data = try JSONEncoder().encode(request)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(object["group_external_original_reference"])
        XCTAssertNil(try JSONDecoder().decode(EditRequest.self, from: data).groupExternalOriginalReference)
    }

    func testAutomaticRequestAllowsZeroManualSourcesAndUsesTenByDefault() throws {
        let request = EditRequest(
            song: "ATTITUDE",
            targetMember: .rei,
            editMode: .member,
            aspectRatio: .portrait,
            sources: [],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            sourceMode: .automatic,
            sourceCacheDirectory: URL(fileURLWithPath: "/tmp/ive-cache"),
            catalogCacheDirectory: URL(fileURLWithPath: "/tmp/ive-catalog")
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        XCTAssertEqual(object["source_mode"] as? String, "automatic")
        XCTAssertEqual(object["desired_source_count"] as? Int, 10)
        XCTAssertEqual(object["source_cache_directory"] as? String, "/tmp/ive-cache")
        XCTAssertEqual(object["catalog_cache_directory"] as? String, "/tmp/ive-catalog")
        XCTAssertEqual(object["acquire_only"] as? Bool, false)
        XCTAssertEqual((object["sources"] as? [Any])?.count, 0)
        XCTAssertEqual((object["member_anchors"] as? [Any])?.count, 0)
        XCTAssertEqual(object["member_pose_dissolve_enabled"] as? Bool, true)
    }

    func testMemberPoseDissolveToggleIsMemberOnlyAndDefaultsOn() throws {
        XCTAssertTrue(
            ProjectSettingsControlPolicy.showsMemberPoseDissolve(
                editMode: .member,
                sourceMode: .automatic
            )
        )
        XCTAssertFalse(
            ProjectSettingsControlPolicy.showsMemberPoseDissolve(
                editMode: .group,
                sourceMode: .automatic
            )
        )
        XCTAssertFalse(
            ProjectSettingsControlPolicy.showsMemberPoseDissolve(
                editMode: .member,
                sourceMode: .local
            )
        )

        let disabled = EditRequest(
            song: "ATTITUDE",
            targetMember: .gaeul,
            editMode: .member,
            aspectRatio: .portrait,
            sources: [],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            memberPoseDissolveEnabled: false,
            sourceMode: .automatic
        )
        let disabledObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(disabled))
                as? [String: Any]
        )
        XCTAssertEqual(
            disabledObject["member_pose_dissolve_enabled"] as? Bool,
            false
        )

        var legacyObject = disabledObject
        legacyObject.removeValue(forKey: "member_pose_dissolve_enabled")
        let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
        let decodedLegacy = try JSONDecoder().decode(EditRequest.self, from: legacyData)
        XCTAssertTrue(decodedLegacy.memberPoseDissolveEnabled)
    }

    func testAutomaticSourceCountIsClampedToSupportedQualityRange() {
        let base = (
            song: "I AM",
            targetMember: IVEMember.anYujin,
            editMode: EditMode.group,
            aspectRatio: OutputAspectRatio.landscape,
            sources: [URL](),
            outputDirectory: URL(fileURLWithPath: "/tmp/output")
        )
        let tooLow = EditRequest(
            song: base.song,
            targetMember: base.targetMember,
            editMode: base.editMode,
            aspectRatio: base.aspectRatio,
            sources: base.sources,
            outputDirectory: base.outputDirectory,
            sourceMode: .automatic,
            desiredSourceCount: 2
        )
        let tooHigh = EditRequest(
            song: base.song,
            targetMember: base.targetMember,
            editMode: base.editMode,
            aspectRatio: base.aspectRatio,
            sources: base.sources,
            outputDirectory: base.outputDirectory,
            sourceMode: .automatic,
            desiredSourceCount: 99
        )

        XCTAssertEqual(tooLow.desiredSourceCount, 8)
        XCTAssertEqual(tooHigh.desiredSourceCount, 24)
    }

    func testAutomaticMemberSourceCountIsAlwaysExactlyTen() {
        XCTAssertEqual(EditRequest.automaticMemberSourceCount, 10)
        XCTAssertEqual(EditRequest.minimumAutomaticMemberSourceCount, 4)
        XCTAssertTrue(EditMode.member.detail.contains("소스 하한만 4개"))
        XCTAssertTrue(EditMode.member.detail.contains("전환 밀도를 유지"))
        XCTAssertTrue(EditMode.member.detail.contains("다른 시간대"))
        for requestedCount in [2, 9, 10, 24, 99] {
            let request = EditRequest(
                song: "I AM",
                targetMember: .gaeul,
                editMode: .member,
                aspectRatio: .landscape,
                sources: [],
                outputDirectory: URL(fileURLWithPath: "/tmp/output"),
                sourceMode: .automatic,
                desiredSourceCount: requestedCount
            )

            XCTAssertEqual(request.desiredSourceCount, 10)
            XCTAssertEqual(request.aspectRatio, OutputAspectRatio.portrait.rawValue)
        }
    }

    func testAutomaticModeCanStartWithoutManualVideosButLocalModeCannot() {
        for mode in EditMode.allCases {
            XCTAssertTrue(
                EditStartPolicy.canStart(
                    song: "LOVE DIVE",
                    sourceMode: .automatic,
                    editMode: mode,
                    sourceCount: 0,
                    anchoredSourceCount: 0
                )
            )
            XCTAssertFalse(
                EditStartPolicy.canStart(
                    song: "LOVE DIVE",
                    sourceMode: .local,
                    editMode: mode,
                    sourceCount: 0,
                    anchoredSourceCount: 0
                )
            )
        }

        XCTAssertFalse(
            EditStartPolicy.canStart(
                song: "LOVE DIVE",
                sourceMode: .local,
                editMode: .member,
                sourceCount: 3,
                anchoredSourceCount: 2
            )
        )
        XCTAssertTrue(
            EditStartPolicy.canStart(
                song: "LOVE DIVE",
                sourceMode: .local,
                editMode: .member,
                sourceCount: 3,
                anchoredSourceCount: 3
            )
        )
    }

    func testAutomaticMemberAspectDefaultsToPortraitWithoutManualPreference() {
        XCTAssertEqual(
            MemberAspectRatioPolicy.resolvedRatio(
                current: .landscape,
                editMode: .member,
                sourceMode: .automatic,
                manuallySelectedMemberRatio: nil
            ),
            .portrait
        )
        XCTAssertEqual(
            MemberAspectRatioPolicy.resolvedRatio(
                current: .landscape,
                editMode: .member,
                sourceMode: .local,
                manuallySelectedMemberRatio: nil
            ),
            .landscape
        )
    }

    func testMemberAspectPolicyLocksAutomaticPortraitAndPreservesLocalChoice() {
        XCTAssertEqual(
            MemberAspectRatioPolicy.resolvedRatio(
                current: .portrait,
                editMode: .member,
                sourceMode: .automatic,
                manuallySelectedMemberRatio: .landscape
            ),
            .portrait
        )
        XCTAssertEqual(
            MemberAspectRatioPolicy.resolvedRatio(
                current: .landscape,
                editMode: .member,
                sourceMode: .local,
                manuallySelectedMemberRatio: .portrait
            ),
            .portrait
        )
        XCTAssertEqual(
            MemberAspectRatioPolicy.resolvedRatio(
                current: .portrait,
                editMode: .group,
                sourceMode: .automatic,
                manuallySelectedMemberRatio: .portrait
            ),
            .landscape
        )
    }

    func testAutomaticMemberKeepsMemberPickerEnabledAndAspectPickerLocked() {
        XCTAssertTrue(
            ProjectSettingsControlPolicy.canSelectMember(
                editMode: .member,
                isRunning: false
            )
        )
        XCTAssertFalse(
            ProjectSettingsControlPolicy.canSelectMember(
                editMode: .group,
                isRunning: false
            )
        )
        XCTAssertFalse(
            ProjectSettingsControlPolicy.canSelectMember(
                editMode: .member,
                isRunning: true
            )
        )
        XCTAssertFalse(
            ProjectSettingsControlPolicy.canSelectAspectRatio(
                editMode: .member,
                sourceMode: .automatic,
                isRunning: false
            )
        )
        XCTAssertTrue(
            ProjectSettingsControlPolicy.canSelectAspectRatio(
                editMode: .member,
                sourceMode: .local,
                isRunning: false
            )
        )
    }

    func testAutomaticMemberRequestCannotSerializeLandscape() throws {
        let request = EditRequest(
            song: "I AM",
            targetMember: .gaeul,
            editMode: .member,
            aspectRatio: .landscape,
            sources: [],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            sourceMode: .automatic
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        XCTAssertEqual(object["aspect_ratio"] as? String, "9:16")
    }

    func testAutomaticMemberRequestCannotSerializeConfirmedAnchors() throws {
        let box = try XCTUnwrap(
            NormalizedBoundingBox(left: 0.1, top: 0.2, right: 0.7, bottom: 0.9)
        )
        let selection = try XCTUnwrap(MemberAnchorSelection(timeSeconds: 2, bbox: box))
        let request = EditRequest(
            song: "I AM",
            targetMember: .gaeul,
            editMode: .member,
            aspectRatio: .portrait,
            sources: [],
            outputDirectory: URL(fileURLWithPath: "/tmp/output"),
            memberAnchors: [MemberAnchorRequest(sourceIndex: 0, selection: selection)],
            sourceMode: .automatic
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        XCTAssertEqual((object["member_anchors"] as? [Any])?.count, 0)
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
        XCTAssertNil(object["target_member"])
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

    func testResultEventDecodesExplicitAcquiredSourcesAndManifest() throws {
        let data = Data(
            #"{"type":"result","outputs":["/tmp/compat.mp4"],"source_paths":["/tmp/01.mp4","/tmp/02.mp4"],"acquisition_manifest":"/tmp/acquisition.json"}"#.utf8
        )
        let event = try JSONDecoder().decode(EngineEvent.self, from: data)

        XCTAssertEqual(event.sourcePaths, ["/tmp/01.mp4", "/tmp/02.mp4"])
        XCTAssertEqual(event.acquisitionManifestPath, "/tmp/acquisition.json")
    }

    func testRenderedVideoOutputDiscoveryRequiresNonemptyRegularVideo() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let empty = root.appendingPathComponent("empty.mp4")
        XCTAssertTrue(FileManager.default.createFile(atPath: empty.path, contents: Data()))
        let valid = root.appendingPathComponent("final.mp4")
        XCTAssertTrue(
            FileManager.default.createFile(
                atPath: valid.path,
                contents: Data([0, 0, 0, 24, 102, 116, 121, 112])
            )
        )
        let symlink = root.appendingPathComponent("linked.mp4")
        try FileManager.default.createSymbolicLink(
            at: symlink,
            withDestinationURL: valid
        )
        let analysis = root.appendingPathComponent("analysis.json")
        XCTAssertTrue(
            FileManager.default.createFile(
                atPath: analysis.path,
                contents: Data("{}".utf8)
            )
        )

        XCTAssertNil(
            RenderedVideoOutputDiscovery.firstValidVideo(
                in: [analysis, empty, symlink]
            )
        )
        XCTAssertEqual(
            RenderedVideoOutputDiscovery.firstValidVideo(
                in: [analysis, empty, symlink, valid]
            ),
            valid.standardizedFileURL
        )
    }

    func testAutomaticAcquisitionProbeRequiresRequestedModeAndYTDLP() {
        let ready = Data(
            #"{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true,"automatic_member":false},"runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":true}}}"#.utf8
        )
        XCTAssertTrue(
            EngineLocator.automaticAcquisitionProbeReady(from: ready, editMode: .group)
        )
        XCTAssertFalse(
            EngineLocator.automaticAcquisitionProbeReady(from: ready, editMode: .member)
        )

        let missingYTDLP = Data(
            #"{"automatic_acquisition":{"ready":false},"mode_readiness":{"automatic_group":true,"automatic_member":true}}"#.utf8
        )
        XCTAssertFalse(
            EngineLocator.automaticAcquisitionProbeReady(
                from: missingYTDLP,
                editMode: .group
            )
        )
        XCTAssertFalse(
            EngineLocator.automaticAcquisitionProbeReady(
                from: Data("warning only".utf8),
                editMode: .group
            )
        )
        let missingProfessionalModel = Data(
            #"{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true,"automatic_member":true},"runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":false}}}"#.utf8
        )
        XCTAssertFalse(
            EngineLocator.automaticAcquisitionProbeReady(
                from: missingProfessionalModel,
                editMode: .group
            )
        )
        XCTAssertTrue(
            EngineLocator.automaticAcquisitionProbeReady(
                from: missingProfessionalModel,
                editMode: .member
            )
        )
        let staleEngineWithoutStageEditField = Data(
            #"{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true,"automatic_member":true}}"#.utf8
        )
        XCTAssertFalse(
            EngineLocator.automaticAcquisitionProbeReady(
                from: staleEngineWithoutStageEditField,
                editMode: .group
            )
        )
        for error in ["missing registry", "invalid registry", "corrupt optional checkpoint"] {
            let optional = Data("""
                {"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true},
                 "runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":false,"production_registry_error":"\(error)"}},
                 "mode_capabilities":{"review_first_group":{"ready":true,"quality_approved":false}}}
                """.utf8)
            XCTAssertTrue(EngineLocator.automaticAcquisitionProbeReady(from: optional, editMode: .group))
            XCTAssertFalse(EngineLocator.automaticAcquisitionProbeReady(from: optional, editMode: .group,
                                                                       requireProfessionalModel: true))
        }
    }

    func testAutomaticAcquisitionProbeUsesModeEvenWhenGlobalDoctorExitsNonzero() throws {
        let ready = try makeExecutableScript(
            #"printf '%s\n' '{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true,"automatic_member":false},"runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":true}}}'; exit 0"#
        )
        let globalStrictFailure = try makeExecutableScript(
            #"printf '%s\n' '{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true,"automatic_member":false},"runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":true}}}'; exit 1"#
        )
        let invalid = try makeExecutableScript(
            "printf '%s\\n' 'doctor crashed'; exit 1"
        )
        let engine = ready.deletingLastPathComponent()
            .appendingPathComponent("crosscut_cli.py")

        XCTAssertTrue(
            EngineLocator.supportsAutomaticAcquisition(
                pythonURL: ready,
                engineURL: engine,
                editMode: .group,
                timeout: 1
            )
        )
        XCTAssertFalse(
            EngineLocator.supportsAutomaticAcquisition(
                pythonURL: ready,
                engineURL: engine,
                editMode: .member,
                timeout: 1
            )
        )
        XCTAssertTrue(
            EngineLocator.supportsAutomaticAcquisition(
                pythonURL: globalStrictFailure,
                engineURL: engine,
                editMode: .group,
                timeout: 1
            )
        )
        XCTAssertFalse(
            EngineLocator.supportsAutomaticAcquisition(
                pythonURL: invalid,
                engineURL: engine,
                editMode: .group,
                timeout: 1
            )
        )
    }

    func testLocalEditingProbeRequiresOnlyTheRequestedLocalMode() throws {
        let report = Data(
            #"{"mode_readiness":{"local_group":true,"local_member":false}}"#.utf8
        )
        XCTAssertTrue(
            EngineLocator.localEditingProbeReady(from: report, editMode: .group)
        )
        XCTAssertFalse(
            EngineLocator.localEditingProbeReady(from: report, editMode: .member)
        )

        let globalStrictFailure = try makeExecutableScript(
            #"printf '%s\n' '{"ready":false,"mode_readiness":{"local_group":true,"local_member":false}}'; exit 1"#
        )
        let engine = globalStrictFailure.deletingLastPathComponent()
            .appendingPathComponent("crosscut_cli.py")
        XCTAssertTrue(
            EngineLocator.supportsLocalEditing(
                pythonURL: globalStrictFailure,
                engineURL: engine,
                editMode: .group,
                timeout: 1
            )
        )
        XCTAssertFalse(
            EngineLocator.supportsLocalEditing(
                pythonURL: globalStrictFailure,
                engineURL: engine,
                editMode: .member,
                timeout: 1
            )
        )
    }

    @MainActor
    func testAutomaticAcquisitionFailureMessageHasActionableFallback() {
        let message = EngineRunner.automaticAcquisitionUnavailableMessage
        XCTAssertTrue(message.contains("yt-dlp"))
        XCTAssertTrue(message.contains("StageEdit"))
        XCTAssertTrue(message.contains("내 영상 선택"))
    }

    @MainActor
    func testLocalMemberFailureMessageNamesConfirmedAnchorTracking() {
        let message = EngineRunner.localMemberTrackingUnavailableMessage
        XCTAssertTrue(message.contains("확인 앵커"))
        XCTAssertTrue(message.contains("트래커"))
    }

    @MainActor
    func testEnginePreparationDoesNotBlockUIAndIgnoresLateResultAfterCancel() async throws {
        let engine = URL(fileURLWithPath: "/tmp/crosscut-test-engine.py")
        let runner = EngineRunner(
            locateEngine: { engine },
            preparationProbe: { _, _, _ in
                Thread.sleep(forTimeInterval: 0.30)
                return .pythonUnavailable
            }
        )
        let request = EditRequest(
            song: "I AM",
            targetMember: .anYujin,
            editMode: .group,
            aspectRatio: .landscape,
            sources: [
                URL(fileURLWithPath: "/tmp/a.mp4"),
                URL(fileURLWithPath: "/tmp/b.mp4"),
            ],
            outputDirectory: URL(fileURLWithPath: "/tmp/crosscut-test-output")
        )

        let startTime = Date()
        runner.start(request: request)
        XCTAssertLessThan(Date().timeIntervalSince(startTime), 0.15)
        XCTAssertEqual(runner.state, .preparing)

        // A second start while preparation is active must remain a no-op.
        runner.start(request: request)
        XCTAssertEqual(runner.state, .preparing)

        runner.cancel()
        XCTAssertEqual(runner.state, .cancelled)
        XCTAssertEqual(runner.statusMessage, "작업이 취소되었습니다.")

        try await Task.sleep(nanoseconds: 450_000_000)
        XCTAssertEqual(runner.state, .cancelled)
    }

    @MainActor
    func testOptionalGroupModelAbsencePassesDoctorAndLaunchesWorker() async throws {
        for error in ["missing_registry", "invalid_registry", "corrupt_checkpoint"] {
            let python = try makeExecutableScript("""
                if [ "$3" = "doctor" ]; then
                  printf '%s\\n' '{"automatic_acquisition":{"ready":true},"mode_readiness":{"automatic_group":true},"mode_capabilities":{"review_first_group":{"ready":true,"quality_approved":false}},"runtime_contract":{"stage_edit_generation":{"automatic_group_generation_ready":false,"production_registry_error":"\(error)"}}}'
                  exit 1
                fi
                printf '%s\\n' "$3" > "$(dirname "$0")/worker-entered"
                exit 0
                """)
            let engine = python.deletingLastPathComponent().appendingPathComponent("crosscut_cli.py")
            let marker = python.deletingLastPathComponent().appendingPathComponent("worker-entered")
            let runner = EngineRunner(locateEngine: { engine }, preparationProbe: { engine, _, _ in
                EngineLocator.supportsAutomaticAcquisition(pythonURL: python, engineURL: engine, editMode: .group)
                    ? .ready(python) : .automaticAcquisitionUnavailable
            })
            runner.start(request: EditRequest(song: "synthetic readiness", targetMember: .gaeul,
                editMode: .group, aspectRatio: .landscape, sources: [],
                outputDirectory: python.deletingLastPathComponent().appendingPathComponent("out"), sourceMode: .automatic))
            for _ in 0..<100 {
                if FileManager.default.fileExists(atPath: marker.path) { break }
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            XCTAssertEqual(try String(contentsOf: marker, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), "analyze")
            runner.cancel()
        }
    }

    func testAugmentedPathAlwaysIncludesHomebrewMediaTools() {
        let path = EngineLocator.augmentedPath(existing: "/custom/bin:/usr/bin")
        let parts = path.split(separator: ":").map(String.init)

        XCTAssertTrue(parts.contains("/opt/homebrew/bin"))
        XCTAssertTrue(parts.contains("/custom/bin"))
        XCTAssertEqual(parts.filter { $0 == "/usr/bin" }.count, 1)
    }

    func testRuntimeProbeRequiresMachineReadableBaseRuntimeReady() {
        XCTAssertTrue(
            EngineLocator.runtimeProbeReady(
                from: Data(
                    #"{"base_runtime_ready":true,"full_ready":false}"#.utf8
                )
            )
        )
        XCTAssertFalse(
            EngineLocator.runtimeProbeReady(
                from: Data(
                    #"{"base_runtime_ready":false,"full_ready":true}"#.utf8
                )
            )
        )
        XCTAssertFalse(
            EngineLocator.runtimeProbeReady(
                from: Data(#"{"mode_readiness":{"group":true}}"#.utf8)
            )
        )
        XCTAssertTrue(
            EngineLocator.runtimeProbeReady(
                from: Data(
                    "ignored warning\n{\"base_runtime_ready\":true}\n".utf8
                )
            )
        )
    }

    func testRuntimeProbeHonorsJSONResultAndExitStatus() throws {
        let ready = try makeExecutableScript(
            "printf '%s\\n' '{\"base_runtime_ready\":true,\"full_ready\":false}'; exit 0"
        )
        let falseReport = try makeExecutableScript(
            "printf '%s\\n' '{\"base_runtime_ready\":false,\"full_ready\":true}'; exit 0"
        )
        let failed = try makeExecutableScript(
            "printf '%s\\n' '{\"base_runtime_ready\":true}'; exit 1"
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

    func testPythonLocatorAcceptsBaseRuntimeWhenTrackersAreUnavailable() throws {
        let baseOnly = try makeExecutableScript(
            "printf '%s\\n' '{\"base_runtime_ready\":true,\"full_ready\":false}'; exit 0"
        )
        let previous = ProcessInfo.processInfo.environment[
            EngineLocator.pythonEnvironmentKey
        ]
        setenv(EngineLocator.pythonEnvironmentKey, baseOnly.path, 1)
        defer {
            if let previous {
                setenv(EngineLocator.pythonEnvironmentKey, previous, 1)
            } else {
                unsetenv(EngineLocator.pythonEnvironmentKey)
            }
        }

        let located = EngineLocator.locatePython(
            near: URL(fileURLWithPath: "/tmp/engine/crosscut_cli.py")
        )
        XCTAssertEqual(located?.standardizedFileURL, baseOnly.standardizedFileURL)
    }

    func testRuntimeProbeCannotWriteBytecodeIntoBundledEngine() throws {
        let guardedProbe = try makeExecutableScript(
            "test \"$1\" = '-B' || exit 11; "
                + "test \"$PYTHONDONTWRITEBYTECODE\" = '1' || exit 12; "
                + "printf '%s\\n' '{\"base_runtime_ready\":true}'"
        )

        XCTAssertTrue(
            EngineLocator.supportsEngineModules(
                pythonURL: guardedProbe,
                engineURL: URL(fileURLWithPath: "/tmp/engine/crosscut_cli.py"),
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

    func testProcessTreeCancellationTerminatesMediaChild() throws {
        let program = #"""
import os, signal, subprocess, sys
os.setpgid(0, 0)
def cancel(_signum, _frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, cancel)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
try:
    child.wait()
except KeyboardInterrupt:
    child.wait(timeout=5)
    raise SystemExit(130)
"""#
        let process = Process()
        let stdoutPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", program]
        process.standardOutput = stdoutPipe
        try process.run()
        addTeardownBlock {
            if process.isRunning { process.terminate() }
        }

        let output = stdoutPipe.fileHandleForReading.availableData
        let childText = try XCTUnwrap(String(data: output, encoding: .utf8))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let childPID = try XCTUnwrap(pid_t(childText))
        XCTAssertEqual(getpgid(process.processIdentifier), process.processIdentifier)
        XCTAssertTrue(
            EngineProcessTree.requestTermination(
                process,
                forceKillDelayNanoseconds: 500_000_000
            )
        )
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 130)
        XCTAssertEqual(Darwin.kill(childPID, 0), -1)
        XCTAssertEqual(errno, ESRCH)
    }

    func testProcessTreeCancellationForceKillsStubbornMediaChild() throws {
        let program = #"""
import os, signal, subprocess, sys
os.setpgid(0, 0)
def cancel(_signum, _frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, cancel)
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
    ],
    stdout=subprocess.PIPE,
    text=True,
)
assert child.stdout.readline().strip() == "ready"
print(child.pid, flush=True)
try:
    child.wait()
except KeyboardInterrupt:
    child.wait()
"""#
        let process = Process()
        let stdoutPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", program]
        process.standardOutput = stdoutPipe
        try process.run()
        addTeardownBlock {
            if process.isRunning { process.terminate() }
        }

        let output = stdoutPipe.fileHandleForReading.availableData
        let childText = try XCTUnwrap(String(data: output, encoding: .utf8))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let childPID = try XCTUnwrap(pid_t(childText))
        XCTAssertTrue(
            EngineProcessTree.requestTermination(
                process,
                forceKillDelayNanoseconds: 100_000_000
            )
        )
        process.waitUntilExit()
        XCTAssertNotEqual(process.terminationStatus, 0)
        XCTAssertEqual(Darwin.kill(childPID, 0), -1)
        XCTAssertEqual(errno, ESRCH)
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
        let professionalQualityAssetPaths = [
            "crosscut/models/object_detection_nanodet_2022nov.onnx",
            "crosscut/models/iam_closeup_gold_regression_v1.json",
            "crosscut/models/ive_product_quality_floor_v1.json",
        ]
        for relativePath in professionalQualityAssetPaths {
            XCTAssertTrue(
                FileManager.default.isReadableFile(
                    atPath: engineDirectory.appendingPathComponent(relativePath).path
                ),
                "Missing bundled professional quality asset: \(relativePath)"
            )
        }
        let requiredProfessionalStageEditModules = [
            "crosscut/stage_edit_production_registry.py",
            "crosscut/stage_edit_semimarkov_beam.py",
            "crosscut/stage_edit_visual_candidate_adapter.py",
            "crosscut/stage_edit_dynamic_crop_head_v2.py",
            "crosscut/stage_edit_pipeline_bridge.py",
            "crosscut/stage_edit_transition_authority.py",
        ]
        for relativePath in requiredProfessionalStageEditModules {
            XCTAssertTrue(
                FileManager.default.isReadableFile(
                    atPath: engineDirectory.appendingPathComponent(relativePath).path
                ),
                "Missing bundled professional StageEdit module: \(relativePath)"
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
                    || $0 == "crosscut/licenses"
                    || ["crosscut/licenses/OSNet-MIT.txt", "crosscut/licenses/RTMPose-Apache-2.0.txt", "crosscut/licenses/TransNetV2-MIT.txt"].contains($0)
                    || $0 == "crosscut/models/face_detection_yunet_2023mar.onnx"
                    || $0 == "crosscut/models/LICENSE.yunet"
                    || sfaceAssetPaths.contains($0)
                    || poseAssetPaths.contains($0)
                    || professionalQualityAssetPaths.contains($0)
            }
        )
    }
}
