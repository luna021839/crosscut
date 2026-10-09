import Foundation
import CoreGraphics

enum IVEMember: String, CaseIterable, Codable, Identifiable {
    case anYujin = "안유진"
    case gaeul = "가을"
    case rei = "레이"
    case jangWonyoung = "장원영"
    case liz = "리즈"
    case leeseo = "이서"

    var id: String { rawValue }
}

enum EditMode: String, CaseIterable, Codable, Identifiable {
    case group
    case member

    var id: String { rawValue }

    var title: String {
        switch self {
        case .group: "일반 교차편집"
        case .member: "멤버 교차직캠"
        }
    }

    var detail: String {
        switch self {
        case .group: "박자·포즈·실루엣을 맞춘 교차컷에, 얼굴이 확실한 경계만 짧은 블렌드를 씁니다. 자동 모드는 10개를 목표로 하고 8개부터 진행합니다."
        case .member: "선택 멤버의 공식 세로 직캠을 원본 그대로 연결합니다. 10개를 목표로 찾되 소스 하한만 4개이며, 전환 밀도를 유지한 채 같은 직캠의 다른 시간대도 비인접 구간에 재사용합니다."
        }
    }

    var systemImage: String {
        switch self {
        case .group: "rectangle.stack"
        case .member: "person.crop.rectangle.stack"
        }
    }
}

enum SourceMode: String, CaseIterable, Codable, Identifiable {
    case automatic
    case local

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic: "자동으로 찾기"
        case .local: "내 영상 선택 · 고급"
        }
    }

    var systemImage: String {
        switch self {
        case .automatic: "sparkle.magnifyingglass"
        case .local: "folder"
        }
    }
}

enum OutputAspectRatio: String, CaseIterable, Codable, Identifiable {
    case landscape = "16:9"
    case portrait = "9:16"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .landscape: "가로 16:9"
        case .portrait: "세로 9:16"
        }
    }

    var systemImage: String {
        switch self {
        case .landscape: "rectangle"
        case .portrait: "rectangle.portrait"
        }
    }
}

struct GroupOriginalReferenceInput: Codable, Equatable {
    let file: String
    let sha256: String

    init(file: URL, sha256: String) throws {
        guard file.isFileURL, !file.path.isEmpty,
              sha256.count == 64,
              sha256.allSatisfy({ "0123456789abcdef".contains($0) }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.file = file.standardizedFileURL.path
        self.sha256 = sha256
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let file = try container.decode(String.self, forKey: .file)
        let sha256 = try container.decode(String.self, forKey: .sha256)
        guard file.hasPrefix("/"), sha256.count == 64,
              sha256.allSatisfy({ "0123456789abcdef".contains($0) }) else {
            throw DecodingError.dataCorruptedError(forKey: .sha256, in: container,
                debugDescription: "Complete original needs an absolute file and lowercase SHA-256")
        }
        self.file = file
        self.sha256 = sha256
    }
}

struct EditRequest: Codable, Equatable {
    static let automaticMemberSourceCount = 10
    static let minimumAutomaticMemberSourceCount = 4

    let song: String
    let targetMember: String?
    let editMode: String
    let aspectRatio: String
    let sources: [String]
    let outputDirectory: String
    let memberAnchors: [MemberAnchorRequest]
    let memberPoseDissolveEnabled: Bool
    let sourceMode: String
    let desiredSourceCount: Int
    let sourceCacheDirectory: String
    let catalogCacheDirectory: String?
    let acquireOnly: Bool
    let groupExternalOriginalReference: GroupOriginalReferenceInput?

    init(
        song: String,
        targetMember: IVEMember,
        editMode: EditMode,
        aspectRatio: OutputAspectRatio,
        sources: [URL],
        outputDirectory: URL,
        memberAnchors: [MemberAnchorRequest] = [],
        memberPoseDissolveEnabled: Bool = true,
        sourceMode: SourceMode = .local,
        desiredSourceCount: Int = 10,
        sourceCacheDirectory: URL? = nil,
        catalogCacheDirectory: URL? = nil,
        acquireOnly: Bool = false,
        groupExternalOriginalReference: GroupOriginalReferenceInput? = nil
    ) {
        precondition(groupExternalOriginalReference == nil || (editMode == .group && !acquireOnly),
            "Complete originals are only valid for GROUP renders")
        self.song = song
        self.targetMember = editMode == .member ? targetMember.rawValue : nil
        self.editMode = editMode.rawValue
        if editMode == .group {
            self.aspectRatio = OutputAspectRatio.landscape.rawValue
        } else if sourceMode == .automatic {
            self.aspectRatio = OutputAspectRatio.portrait.rawValue
        } else {
            self.aspectRatio = aspectRatio.rawValue
        }
        self.sources = sources.map(\.path)
        self.outputDirectory = outputDirectory.path
        self.memberAnchors = editMode == .member && sourceMode == .local
            ? memberAnchors
            : []
        self.memberPoseDissolveEnabled = memberPoseDissolveEnabled
        self.sourceMode = sourceMode.rawValue
        self.desiredSourceCount = (
            editMode == .member && sourceMode == .automatic
                ? Self.automaticMemberSourceCount
                : min(max(desiredSourceCount, 8), 24)
        )
        self.sourceCacheDirectory = (sourceCacheDirectory ?? outputDirectory
            .appendingPathComponent("CrossCut Sources", isDirectory: true)).path
        self.catalogCacheDirectory = catalogCacheDirectory?.path
        self.acquireOnly = acquireOnly
        self.groupExternalOriginalReference = groupExternalOriginalReference
    }

    enum CodingKeys: String, CodingKey {
        case song
        case targetMember = "target_member"
        case editMode = "edit_mode"
        case aspectRatio = "aspect_ratio"
        case sources
        case outputDirectory = "output_directory"
        case memberAnchors = "member_anchors"
        case memberPoseDissolveEnabled = "member_pose_dissolve_enabled"
        case sourceMode = "source_mode"
        case desiredSourceCount = "desired_source_count"
        case sourceCacheDirectory = "source_cache_directory"
        case catalogCacheDirectory = "catalog_cache_directory"
        case acquireOnly = "acquire_only"
        case groupExternalOriginalReference = "group_external_original_reference"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        song = try container.decode(String.self, forKey: .song)
        targetMember = try container.decodeIfPresent(String.self, forKey: .targetMember)
        editMode = try container.decode(String.self, forKey: .editMode)
        aspectRatio = try container.decode(String.self, forKey: .aspectRatio)
        sources = try container.decode([String].self, forKey: .sources)
        outputDirectory = try container.decode(String.self, forKey: .outputDirectory)
        memberAnchors = try container.decode(
            [MemberAnchorRequest].self,
            forKey: .memberAnchors
        )
        // Requests saved before this preference existed behaved as though the
        // evidence-gated member dissolve policy was enabled.
        memberPoseDissolveEnabled = try container.decodeIfPresent(
            Bool.self,
            forKey: .memberPoseDissolveEnabled
        ) ?? true
        sourceMode = try container.decode(String.self, forKey: .sourceMode)
        desiredSourceCount = try container.decode(Int.self, forKey: .desiredSourceCount)
        sourceCacheDirectory = try container.decode(
            String.self,
            forKey: .sourceCacheDirectory
        )
        catalogCacheDirectory = try container.decodeIfPresent(
            String.self,
            forKey: .catalogCacheDirectory
        )
        acquireOnly = try container.decode(Bool.self, forKey: .acquireOnly)
        groupExternalOriginalReference = try container.decodeIfPresent(
            GroupOriginalReferenceInput.self, forKey: .groupExternalOriginalReference)
        if groupExternalOriginalReference != nil && (editMode != "group" || acquireOnly) {
            throw DecodingError.dataCorruptedError(forKey: .groupExternalOriginalReference,
                in: container, debugDescription: "Complete originals require a GROUP render")
        }
    }
}

enum EditStartPolicy {
    static func canStart(
        song: String,
        sourceMode: SourceMode,
        editMode: EditMode,
        sourceCount: Int,
        anchoredSourceCount: Int
    ) -> Bool {
        guard !song.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }

        switch (sourceMode, editMode) {
        case (.automatic, .group):
            return true
        case (.automatic, .member):
            return true
        case (.local, .group):
            return sourceCount >= 2
        case (.local, .member):
            return sourceCount >= 2 && anchoredSourceCount == sourceCount
        }
    }
}

struct SelectedVideo: Identifiable, Hashable {
    let url: URL
    var memberAnchor: MemberAnchorSelection?

    init(url: URL, memberAnchor: MemberAnchorSelection? = nil) {
        self.url = url
        self.memberAnchor = memberAnchor
    }

    var id: String { url.standardizedFileURL.path }
    var filename: String { url.lastPathComponent }

    var folderName: String {
        url.deletingLastPathComponent().lastPathComponent
    }

    static func == (lhs: SelectedVideo, rhs: SelectedVideo) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

struct NormalizedBoundingBox: Codable, Hashable {
    let left: Double
    let top: Double
    let right: Double
    let bottom: Double

    init?(left: Double, top: Double, right: Double, bottom: Double) {
        guard [left, top, right, bottom].allSatisfy(\.isFinite),
              left >= 0, top >= 0, right <= 1, bottom <= 1,
              right > left, bottom > top else {
            return nil
        }
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
    }

    var xyxy: [Double] { [left, top, right, bottom] }
}

struct MemberAnchorSelection: Codable, Hashable {
    let timeSeconds: Double
    let bbox: NormalizedBoundingBox

    init?(timeSeconds: Double, bbox: NormalizedBoundingBox) {
        guard timeSeconds.isFinite, timeSeconds >= 0 else { return nil }
        self.timeSeconds = timeSeconds
        self.bbox = bbox
    }

    enum CodingKeys: String, CodingKey {
        case timeSeconds = "time_seconds"
        case bbox
    }
}

struct MemberAnchorRequest: Codable, Equatable {
    let sourceIndex: Int
    let timeSeconds: Double
    let bboxXYXY: [Double]

    init(sourceIndex: Int, selection: MemberAnchorSelection) {
        self.sourceIndex = sourceIndex
        timeSeconds = selection.timeSeconds
        bboxXYXY = selection.bbox.xyxy
    }

    enum CodingKeys: String, CodingKey {
        case sourceIndex = "source_index"
        case timeSeconds = "time_seconds"
        case bboxXYXY = "bbox_xyxy"
    }
}

enum AspectFitCoordinateMapper {
    static func fittedRect(imageSize: CGSize, in containerSize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              containerSize.width > 0, containerSize.height > 0 else {
            return .zero
        }

        let scale = min(
            containerSize.width / imageSize.width,
            containerSize.height / imageSize.height
        )
        let fittedSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: (containerSize.width - fittedSize.width) / 2,
            y: (containerSize.height - fittedSize.height) / 2,
            width: fittedSize.width,
            height: fittedSize.height
        )
    }

    static func normalizedPoint(
        _ point: CGPoint,
        imageSize: CGSize,
        in containerSize: CGSize
    ) -> CGPoint? {
        let rect = fittedRect(imageSize: imageSize, in: containerSize)
        guard !rect.isEmpty else { return nil }
        return CGPoint(
            x: min(max((point.x - rect.minX) / rect.width, 0), 1),
            y: min(max((point.y - rect.minY) / rect.height, 0), 1)
        )
    }

    static func boundingBox(
        from start: CGPoint,
        to end: CGPoint,
        imageSize: CGSize,
        in containerSize: CGSize
    ) -> NormalizedBoundingBox? {
        guard let start = normalizedPoint(start, imageSize: imageSize, in: containerSize),
              let end = normalizedPoint(end, imageSize: imageSize, in: containerSize) else {
            return nil
        }
        return NormalizedBoundingBox(
            left: min(start.x, end.x),
            top: min(start.y, end.y),
            right: max(start.x, end.x),
            bottom: max(start.y, end.y)
        )
    }

    static func rect(
        for box: NormalizedBoundingBox,
        imageSize: CGSize,
        in containerSize: CGSize
    ) -> CGRect {
        let fitted = fittedRect(imageSize: imageSize, in: containerSize)
        return CGRect(
            x: fitted.minX + fitted.width * box.left,
            y: fitted.minY + fitted.height * box.top,
            width: fitted.width * (box.right - box.left),
            height: fitted.height * (box.bottom - box.top)
        )
    }
}
