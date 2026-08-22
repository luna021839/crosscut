import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var runner = EngineRunner()

    @State private var song = ""
    @State private var targetMember = IVEMember.anYujin
    @State private var editMode = EditMode.group
    @State private var aspectRatio = OutputAspectRatio.landscape
    @State private var selectedVideos: [SelectedVideo] = []
    @State private var outputDirectory = Self.defaultOutputDirectory
    @State private var isChoosingVideos = false
    @State private var isDropTargeted = false
    @State private var validationMessage: String?
    @State private var anchorEditorVideo: SelectedVideo?

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollView {
                HStack(alignment: .top, spacing: 18) {
                    settingsCard
                        .frame(minWidth: 300, maxWidth: 340)

                    sourcesCard
                        .frame(minWidth: 320, maxWidth: .infinity)

                    activityCard
                        .frame(minWidth: 300, maxWidth: 360)
                }
                .padding(22)
            }

            actionBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .fileImporter(
            isPresented: $isChoosingVideos,
            allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie],
            allowsMultipleSelection: true,
            onCompletion: handleFileImport
        )
        .sheet(item: $anchorEditorVideo) { video in
            MemberAnchorEditorSheet(video: video, member: targetMember) { selection in
                guard let index = selectedVideos.firstIndex(where: { $0.id == video.id }) else {
                    return
                }
                selectedVideos[index].memberAnchor = selection
                validationMessage = nil
            }
        }
        .onChange(of: editMode) {
            if editMode == .group {
                aspectRatio = .landscape
            }
        }
        .onChange(of: targetMember) {
            guard editMode == .member else { return }
            for index in selectedVideos.indices {
                selectedVideos[index].memberAnchor = nil
            }
            validationMessage = "멤버가 바뀌어 영상별 대상 위치를 다시 지정해야 합니다."
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [.purple, .pink],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                Image(systemName: "rectangle.stack.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .frame(width: 46, height: 46)

            VStack(alignment: .leading, spacing: 2) {
                Text("CrossCut")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                Text("음방 교차편집 · 멤버 교차직캠")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            engineBadge
        }
        .padding(.horizontal, 22)
        .padding(.top, 16)
        .padding(.bottom, 14)
        .background(.regularMaterial)
        .overlay(alignment: .bottom) { Divider() }
    }

    private var engineBadge: some View {
        Button {
            runner.refreshEngineLocation()
        } label: {
            HStack(spacing: 7) {
                Circle()
                    .fill(runner.engineURL == nil ? Color.orange : Color.green)
                    .frame(width: 8, height: 8)
                Text(runner.engineURL == nil ? "엔진 연결 대기" : "엔진 준비됨")
                    .font(.caption.weight(.medium))
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(.quaternary, in: Capsule())
        }
        .buttonStyle(.plain)
        .help(runner.engineURL?.path ?? "클릭하여 crosscut_cli.py를 다시 찾습니다.")
    }

    private var settingsCard: some View {
        Card(title: "프로젝트 설정", systemImage: "slider.horizontal.3") {
            VStack(alignment: .leading, spacing: 16) {
                labeled("그룹") {
                    HStack(spacing: 8) {
                        Text("IVE")
                            .font(.headline)
                        Text("전용")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.purple)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(.purple.opacity(0.12), in: Capsule())
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                    .frame(height: 32)
                    .background(.quaternary.opacity(0.7), in: RoundedRectangle(cornerRadius: 7))
                }

                labeled("곡명") {
                    TextField("예: I AM", text: $song)
                        .textFieldStyle(.roundedBorder)
                }

                labeled("출력 모드") {
                    Picker("출력 모드", selection: $editMode) {
                        ForEach(EditMode.allCases) { mode in
                            Label(mode.title, systemImage: mode.systemImage)
                                .tag(mode)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    Text(editMode.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                labeled("대상 멤버") {
                    Picker("대상 멤버", selection: $targetMember) {
                        ForEach(IVEMember.allCases) { member in
                            Text(member.rawValue).tag(member)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .disabled(editMode == .group)
                    .opacity(editMode == .group ? 0.55 : 1)
                }

                labeled("화면 비율") {
                    Picker("화면 비율", selection: $aspectRatio) {
                        ForEach(OutputAspectRatio.allCases) { ratio in
                            Label(ratio.title, systemImage: ratio.systemImage)
                                .tag(ratio)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .disabled(editMode == .group)
                    .opacity(editMode == .group ? 0.55 : 1)
                }

                labeled("저장 위치") {
                    HStack(spacing: 8) {
                        Image(systemName: "folder")
                            .foregroundStyle(.secondary)
                        Text(outputDirectory.path)
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 0)
                        Button("변경…", action: chooseOutputDirectory)
                            .controlSize(.small)
                    }
                }
            }
        }
    }

    private var sourcesCard: some View {
        Card(title: "원본 영상", systemImage: "film.stack") {
            VStack(spacing: 12) {
                Button {
                    isChoosingVideos = true
                } label: {
                    VStack(spacing: 9) {
                        Image(systemName: "plus.rectangle.on.folder")
                            .font(.system(size: 28, weight: .light))
                            .foregroundStyle(.purple)
                        Text("영상을 선택하거나 여기에 드래그")
                            .font(.subheadline.weight(.medium))
                        Text(sourceRequirementText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(isDropTargeted ? Color.purple.opacity(0.12) : Color.clear)
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(
                                isDropTargeted ? Color.purple : Color.secondary.opacity(0.35),
                                style: StrokeStyle(lineWidth: 1, dash: [6, 5])
                            )
                    }
                }
                .buttonStyle(.plain)
                .onDrop(
                    of: [UTType.fileURL.identifier],
                    isTargeted: $isDropTargeted,
                    perform: handleDrop
                )

                HStack {
                    Text("\(selectedVideos.count)개 선택")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(selectedVideos.count >= 2 ? Color.secondary : Color.orange)
                    Spacer()
                    if !selectedVideos.isEmpty {
                        Button("모두 지우기") {
                            selectedVideos.removeAll()
                        }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .disabled(runner.state.isRunning)
                    }
                }

                if editMode == .member {
                    VStack(alignment: .leading, spacing: 5) {
                        Label(
                            "각 영상에서 \(targetMember.rawValue) 위치를 한 번씩 지정해야 합니다.",
                            systemImage: "person.crop.rectangle.badge.plus"
                        )
                        Label(
                            "카메라 컷이나 대상 손실 시 다른 사람을 재식별하지 않고 해당 소스 추적을 안전하게 중단합니다.",
                            systemImage: "shield.checkered"
                        )
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if selectedVideos.isEmpty {
                    ContentUnavailableView(
                        "선택한 영상 없음",
                        systemImage: "film",
                        description: Text(emptySourcesDescription)
                    )
                    .frame(maxHeight: .infinity)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 7) {
                            ForEach(selectedVideos) { video in
                                SourceRow(
                                    video: video,
                                    showsMemberAnchor: editMode == .member,
                                    setAnchor: { anchorEditorVideo = video },
                                    remove: { selectedVideos.removeAll { $0.id == video.id } }
                                )
                                .disabled(runner.state.isRunning)
                            }
                        }
                    }
                }
            }
            .frame(minHeight: 420)
        }
    }

    private var activityCard: some View {
        Card(title: "진행 상태", systemImage: "waveform.path.ecg") {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 10) {
                    Image(systemName: stateSymbol)
                        .foregroundStyle(stateColor)
                        .font(.title2)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(stateTitle)
                            .font(.headline)
                        Text(runner.statusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                }

                ProgressView(value: runner.progress)
                    .tint(.purple)
                Text("\(Int(runner.progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)

                Divider()

                HStack {
                    Text("로그")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !runner.logs.isEmpty {
                        Button("지우기") { runner.clearLog() }
                            .buttonStyle(.plain)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .disabled(runner.state.isRunning)
                    }
                }

                if runner.logs.isEmpty {
                    Text("엔진 메시지가 여기에 표시됩니다.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding(.top, 4)
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(runner.logs) { entry in
                                    LogRow(entry: entry)
                                        .id(entry.id)
                                }
                            }
                        }
                        .onChange(of: runner.logs.count) {
                            if let id = runner.logs.last?.id {
                                withAnimation { proxy.scrollTo(id, anchor: .bottom) }
                            }
                        }
                    }
                }

                if !runner.outputURLs.isEmpty {
                    Divider()
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting(runner.outputURLs)
                    } label: {
                        Label("결과 보기", systemImage: "folder.badge.checkmark")
                    }
                }
            }
            .frame(minHeight: 420)
        }
    }

    private var actionBar: some View {
        HStack(spacing: 12) {
            if let validationMessage {
                Label(validationMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else {
                Text("입력 영상은 변경하지 않고 결과를 별도 폴더에 저장합니다.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if runner.state.isRunning {
                Button("취소", role: .cancel) {
                    runner.cancel()
                }
                .keyboardShortcut(.cancelAction)
            }

            Button {
                startAnalysis()
            } label: {
                Label(
                    runner.state.isRunning ? "편집 중…" : "교차편집 시작",
                    systemImage: runner.state.isRunning ? "hourglass" : "sparkles.tv"
                )
                .frame(minWidth: 100)
            }
            .buttonStyle(.borderedProminent)
            .tint(.purple)
            .controlSize(.large)
            .disabled(runner.state.isRunning || song.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedVideos.count < 2)
            .keyboardShortcut(.return, modifiers: [.command])
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 14)
        .background(.regularMaterial)
        .overlay(alignment: .top) { Divider() }
    }

    private func labeled<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func startAnalysis() {
        let trimmedSong = song.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedSong.isEmpty else {
            validationMessage = "곡명을 입력해 주세요."
            return
        }
        guard selectedVideos.count >= 2 else {
            validationMessage = "원본 영상이 2개 이상 필요합니다."
            return
        }
        if editMode == .member {
            let missingCount = selectedVideos.lazy.filter { $0.memberAnchor == nil }.count
            guard missingCount == 0 else {
                validationMessage = "모든 영상에서 \(targetMember.rawValue) 위치를 지정해 주세요. (남은 영상 \(missingCount)개)"
                return
            }
        }

        validationMessage = nil
        let memberAnchors = selectedVideos.enumerated().compactMap { index, video in
            video.memberAnchor.map { MemberAnchorRequest(sourceIndex: index, selection: $0) }
        }
        let request = EditRequest(
            song: trimmedSong,
            targetMember: targetMember,
            editMode: editMode,
            aspectRatio: aspectRatio,
            sources: selectedVideos.map(\.url),
            outputDirectory: outputDirectory,
            memberAnchors: memberAnchors
        )
        runner.start(request: request)
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            addVideos(urls)
        case .failure(let error):
            validationMessage = "영상을 열지 못했습니다: \(error.localizedDescription)"
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        let supportedProviders = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }
        guard !supportedProviders.isEmpty else { return false }

        for provider in supportedProviders {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                let url: URL?
                if let data = item as? Data {
                    url = URL(dataRepresentation: data, relativeTo: nil)
                } else {
                    url = item as? URL
                }

                guard let url else { return }
                Task { @MainActor in addVideos([url]) }
            }
        }
        return true
    }

    private func addVideos(_ urls: [URL]) {
        let allowedExtensions = Set(["mp4", "mov", "m4v", "mkv", "webm", "avi", "ts"])
        let videos = urls
            .filter { allowedExtensions.contains($0.pathExtension.lowercased()) }
            .map { SelectedVideo(url: $0) }

        var knownPaths = Set(selectedVideos.map(\.id))
        for video in videos where knownPaths.insert(video.id).inserted {
            selectedVideos.append(video)
        }
        selectedVideos.sort { $0.filename.localizedStandardCompare($1.filename) == .orderedAscending }

        if videos.count != urls.count {
            validationMessage = "지원하지 않는 파일은 목록에서 제외했습니다."
        } else {
            validationMessage = nil
        }
    }

    private func chooseOutputDirectory() {
        let panel = NSOpenPanel()
        panel.title = "결과를 저장할 폴더 선택"
        panel.prompt = "선택"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = outputDirectory

        guard panel.runModal() == .OK, let url = panel.url else { return }
        outputDirectory = url
    }

    private var stateTitle: String {
        switch runner.state {
        case .idle: "준비"
        case .preparing: "준비 중"
        case .running: "편집 중"
        case .completed: "완료"
        case .failed: "오류"
        case .cancelled: "취소됨"
        }
    }

    private var stateSymbol: String {
        switch runner.state {
        case .idle: "circle.dotted"
        case .preparing: "gearshape.2"
        case .running: "waveform"
        case .completed: "checkmark.circle.fill"
        case .failed: "exclamationmark.octagon.fill"
        case .cancelled: "xmark.circle.fill"
        }
    }

    private var stateColor: Color {
        switch runner.state {
        case .idle: .secondary
        case .preparing, .running: .purple
        case .completed: .green
        case .failed: .red
        case .cancelled: .orange
        }
    }

    private static var defaultOutputDirectory: URL {
        let movies = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
        return movies.appendingPathComponent("CrossCut", isDirectory: true)
    }

    private var sourceRequirementText: String {
        switch editMode {
        case .group:
            "같은 곡의 음방 영상 2개 이상 · 완곡 권장 9개+"
        case .member:
            "같은 곡의 무편집·고정 멤버 직캠 2개 이상 · 완곡 권장 9개+"
        }
    }

    private var emptySourcesDescription: String {
        switch editMode {
        case .group:
            "방송사별 원본 영상을 추가하세요."
        case .member:
            "대상 멤버가 계속 화면에 담긴 무편집·고정 직캠을 추가하세요."
        }
    }
}

private struct Card<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content()
        }
        .padding(16)
        .background(.background, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.separator.opacity(0.65))
        }
        .shadow(color: .black.opacity(0.035), radius: 8, y: 2)
    }
}

private struct SourceRow: View {
    let video: SelectedVideo
    let showsMemberAnchor: Bool
    let setAnchor: () -> Void
    let remove: () -> Void

    var body: some View {
        VStack(spacing: 7) {
            HStack(spacing: 10) {
                Image(systemName: "film")
                    .foregroundStyle(.purple)
                    .frame(width: 24, height: 24)
                    .background(.purple.opacity(0.1), in: RoundedRectangle(cornerRadius: 5))
                VStack(alignment: .leading, spacing: 2) {
                    Text(video.filename)
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(video.folderName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 4)
                Button(action: remove) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("목록에서 제거")
            }

            if showsMemberAnchor {
                Divider()
                HStack(spacing: 7) {
                    if let anchor = video.memberAnchor {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("대상 지정됨 · \(anchor.timeSeconds, specifier: "%.1f")초")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("수정", action: setAnchor)
                            .controlSize(.mini)
                    } else {
                        Image(systemName: "person.crop.rectangle.badge.plus")
                            .foregroundStyle(.orange)
                        Text("대상 위치가 필요합니다")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                        Spacer()
                        Button("대상 지정…", action: setAnchor)
                            .controlSize(.mini)
                    }
                }
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct LogRow: View {
    let entry: EngineLogEntry

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 7) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(entry.timestamp, format: .dateTime.hour().minute().second())
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
            Text(entry.message)
                .font(.caption)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var color: Color {
        switch entry.level {
        case .info: .blue
        case .warning: .orange
        case .error: .red
        case .engine: .secondary
        }
    }
}

#Preview {
    ContentView()
        .frame(width: 1_120, height: 760)
}
