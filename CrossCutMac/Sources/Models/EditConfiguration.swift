import Foundation

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
        case .group: "박자·포즈·실루엣을 맞춘 교차컷에, 얼굴이 확실한 경계만 짧은 블렌드를 씁니다. 완곡은 9개 이상을 권장합니다."
        case .member: "선택 멤버가 계속 담긴 무편집·고정 직캠을 연결합니다. 추적은 각 앵커에서 앞으로만 진행하며, 완곡은 9개 이상을 권장합니다."
        }
    }

    var systemImage: String {
        switch self {
        case .group: "rectangle.stack"
        case .member: "person.crop.rectangle.stack"
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

struct EditRequest: Codable, Equatable {
    let song: String
    let targetMember: String
    let editMode: String
    let aspectRatio: String
    let sources: [String]
    let outputDirectory: String
    let memberAnchors: [MemberAnchorRequest]

    init(
        song: String,
        targetMember: IVEMember,
        editMode: EditMode,
        aspectRatio: OutputAspectRatio,
        sources: [URL],
        outputDirectory: URL,
        memberAnchors: [MemberAnchorRequest] = []
    ) {
        self.song = song
        self.targetMember = targetMember.rawValue
        self.editMode = editMode.rawValue
        self.aspectRatio = aspectRatio.rawValue
        self.sources = sources.map(\.path)
        self.outputDirectory = outputDirectory.path
        self.memberAnchors = editMode == .member ? memberAnchors : []
    }

    enum CodingKeys: String, CodingKey {
        case song
        case targetMember = "target_member"
        case editMode = "edit_mode"
        case aspectRatio = "aspect_ratio"
        case sources
        case outputDirectory = "output_directory"
        case memberAnchors = "member_anchors"
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
