import AppKit
import AVFoundation
import SwiftUI

struct MemberAnchorEditorSheet: View {
    let video: SelectedVideo
    let member: IVEMember
    let onSave: (MemberAnchorSelection) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var timeSeconds: Double
    @State private var durationSeconds: Double = 0
    @State private var thumbnail: NSImage?
    @State private var boundingBox: NormalizedBoundingBox?
    @State private var dragStart: CGPoint?
    @State private var thumbnailTask: Task<Void, Never>?
    @State private var loadError: String?
    @State private var isLoading = true

    init(
        video: SelectedVideo,
        member: IVEMember,
        onSave: @escaping (MemberAnchorSelection) -> Void
    ) {
        self.video = video
        self.member = member
        self.onSave = onSave
        _timeSeconds = State(initialValue: video.memberAnchor?.timeSeconds ?? 0)
        _boundingBox = State(initialValue: video.memberAnchor?.bbox)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(member.rawValue) 위치 지정")
                    .font(.title2.bold())
                Text("이 장면부터 앞으로만 추적합니다. 가능하면 곡 시작 직후의 멤버가 잘 보이는 장면을 고르고, 머리부터 몸까지 드래그해 감싸 주세요. 뒤쪽 장면을 고르면 모든 직캠 중 가장 늦은 기준 장면 이전은 결과에서 빠집니다.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(video.filename)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            thumbnailCanvas
                .frame(minHeight: 340)

            VStack(spacing: 8) {
                HStack {
                    Text("장면 시각")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text(timeLabel(timeSeconds))
                        .font(.caption.monospacedDigit())
                    Text("/ \(timeLabel(durationSeconds))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $timeSeconds, in: 0...max(durationSeconds, 0.01))
                    .disabled(durationSeconds <= 0)
                    .onChange(of: timeSeconds) {
                        boundingBox = nil
                        scheduleThumbnail(at: timeSeconds)
                    }
            }

            HStack {
                if boundingBox == nil {
                    Label("영상 위에서 대상을 드래그해 주세요.", systemImage: "person.crop.rectangle.badge.plus")
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else {
                    Label("대상 영역이 지정되었습니다.", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                }

                Spacer()

                Button("취소") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("이 위치 사용") {
                    guard let boundingBox,
                          let selection = MemberAnchorSelection(
                              timeSeconds: timeSeconds,
                              bbox: boundingBox
                          ) else { return }
                    onSave(selection)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
                .disabled(boundingBox == nil || thumbnail == nil)
                .keyboardShortcut(.return, modifiers: [.command])
            }
        }
        .padding(22)
        .frame(minWidth: 720, minHeight: 590)
        .task { await prepareVideo() }
        .onDisappear { thumbnailTask?.cancel() }
    }

    @ViewBuilder
    private var thumbnailCanvas: some View {
        GeometryReader { proxy in
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(.black)

                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if let boundingBox {
                        let rect = AspectFitCoordinateMapper.rect(
                            for: boundingBox,
                            imageSize: thumbnail.size,
                            in: proxy.size
                        )
                        ZStack(alignment: .topLeading) {
                            Rectangle()
                                .fill(Color.purple.opacity(0.16))
                            Rectangle()
                                .strokeBorder(Color.purple, lineWidth: 3)
                            Text(member.rawValue)
                                .font(.caption2.bold())
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 3)
                                .background(.purple)
                        }
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                        .allowsHitTesting(false)
                    }

                    Color.clear
                        .contentShape(Rectangle())
                        .gesture(selectionGesture(imageSize: thumbnail.size, containerSize: proxy.size))
                } else if isLoading {
                    ProgressView("장면을 불러오는 중…")
                        .tint(.white)
                        .foregroundStyle(.white)
                } else {
                    ContentUnavailableView(
                        "장면을 표시할 수 없음",
                        systemImage: "exclamationmark.triangle",
                        description: Text(loadError ?? "영상을 다시 선택해 주세요.")
                    )
                    .foregroundStyle(.white)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private func selectionGesture(imageSize: CGSize, containerSize: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { value in
                let fitted = AspectFitCoordinateMapper.fittedRect(
                    imageSize: imageSize,
                    in: containerSize
                )
                if dragStart == nil {
                    guard fitted.contains(value.startLocation) else { return }
                    dragStart = value.startLocation
                }
                guard let dragStart else { return }
                boundingBox = AspectFitCoordinateMapper.boundingBox(
                    from: dragStart,
                    to: value.location,
                    imageSize: imageSize,
                    in: containerSize
                )
            }
            .onEnded { value in
                defer { dragStart = nil }
                guard let dragStart else { return }
                boundingBox = AspectFitCoordinateMapper.boundingBox(
                    from: dragStart,
                    to: value.location,
                    imageSize: imageSize,
                    in: containerSize
                )
            }
    }

    @MainActor
    private func prepareVideo() async {
        do {
            let asset = AVURLAsset(url: video.url)
            let duration = try await asset.load(.duration)
            let seconds = duration.seconds
            durationSeconds = seconds.isFinite && seconds > 0 ? seconds : 0
            timeSeconds = min(max(timeSeconds, 0), durationSeconds)
            await loadThumbnail(at: timeSeconds)
        } catch {
            isLoading = false
            loadError = error.localizedDescription
        }
    }

    @MainActor
    private func scheduleThumbnail(at seconds: Double) {
        thumbnailTask?.cancel()
        thumbnailTask = Task {
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled else { return }
            await loadThumbnail(at: seconds)
        }
    }

    @MainActor
    private func loadThumbnail(at seconds: Double) async {
        isLoading = true
        loadError = nil
        do {
            let asset = AVURLAsset(url: video.url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 1_280, height: 720)
            generator.requestedTimeToleranceBefore = CMTime(seconds: 0.08, preferredTimescale: 600)
            generator.requestedTimeToleranceAfter = CMTime(seconds: 0.08, preferredTimescale: 600)
            let requested = CMTime(seconds: seconds, preferredTimescale: 600)
            let (image, _) = try await generator.image(at: requested)
            guard !Task.isCancelled else { return }
            thumbnail = NSImage(
                cgImage: image,
                size: NSSize(width: image.width, height: image.height)
            )
            isLoading = false
        } catch {
            guard !Task.isCancelled else { return }
            isLoading = false
            loadError = error.localizedDescription
        }
    }

    private func timeLabel(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "00:00.0" }
        let minutes = Int(seconds) / 60
        let remainder = seconds - Double(minutes * 60)
        return String(format: "%02d:%04.1f", minutes, remainder)
    }
}
