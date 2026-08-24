import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum MemberAspectRatioPolicy {
    static func resolvedRatio(
        current: OutputAspectRatio,
        editMode: EditMode,
        sourceMode: SourceMode,
        manuallySelectedMemberRatio: OutputAspectRatio?
    ) -> OutputAspectRatio {
        guard editMode == .member else { return .landscape }
        if sourceMode == .automatic { return .portrait }
        if let manuallySelectedMemberRatio {
            return manuallySelectedMemberRatio
        }
        return current
    }
}

enum ProjectSettingsControlPolicy {
    static func canSelectMember(editMode: EditMode, isRunning: Bool) -> Bool {
        editMode == .member && !isRunning
    }

    static func canSelectAspectRatio(
        editMode: EditMode,
        sourceMode: SourceMode,
        isRunning: Bool
    ) -> Bool {
        editMode == .member && sourceMode == .local && !isRunning
    }
}

struct ContentView: View {
    private static let defaultStorageDirectories =
        LargeMediaStoragePolicy.defaultDirectories()

    @StateObject private var runner = EngineRunner()

    @State private var song = ""
    @State private var targetMember = IVEMember.anYujin
    @State private var editMode = EditMode.group
    @State private var aspectRatio = OutputAspectRatio.landscape
    @State private var manuallySelectedMemberAspectRatio: OutputAspectRatio?
    @State private var sourceMode = SourceMode.automatic
    @State private var desiredSourceCount = 10
    @State private var selectedVideos: [SelectedVideo] = []
    @State private var outputDirectory = Self.defaultOutputDirectory
    @State private var sourceCacheRootDirectory = Self.defaultSourceCacheRootDirectory
    @State private var isChoosingVideos = false
    @State private var isDropTargeted = false
    @State private var validationMessage: String?
    @State private var anchorEditorVideo: SelectedVideo?

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollView {
                HStack(alignment: .top, spacing: 18) {
                    VStack(spacing: 18) {
                        sourceModeCard
                        settingsCard
                    }
                    .frame(minWidth: 300, maxWidth: 340)

                    Group {
                        if sourceMode == .automatic {
                            automaticSourcesCard
                        } else {
                            sourcesCard
                        }
                    }
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
        .onAppear {
            runner.restoreLatestCompletedGroupResult(
                in: outputDirectory,
                reportMissing: false,
                clearExistingOutputsOnMissing: true
            )
        }
        .onChange(of: outputDirectory) {
            runner.restoreLatestCompletedGroupResult(
                in: outputDirectory,
                reportMissing: false,
                clearExistingOutputsOnMissing: true
            )
        }
        .onChange(of: editMode) {
            applyAspectRatioPolicy()
            validationMessage = nil
        }
        .onChange(of: sourceMode) {
            applyAspectRatioPolicy()
            validationMessage = nil
        }
        .onChange(of: targetMember) {
            guard editMode == .member, sourceMode == .local else { return }
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
                    .fill(runner.engineURL == nil ? Color.orange : Color.blue)
                    .frame(width: 8, height: 8)
                Text(runner.engineURL == nil ? "엔진 연결 대기" : "엔진 연결됨")
                    .font(.caption.weight(.medium))
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(.quaternary, in: Capsule())
        }
        .buttonStyle(.plain)
        .help(
            runner.engineURL.map {
                "엔진 경로 연결됨: \($0.path)\n자동 모드는 시작할 때 yt-dlp 준비 상태를 별도로 검사합니다."
            } ?? "클릭하여 crosscut_cli.py를 다시 찾습니다."
        )
    }

    private var sourceModeCard: some View {
        Card(title: "영상 준비", systemImage: "tray.and.arrow.down") {
            VStack(alignment: .leading, spacing: 9) {
                Picker("영상 준비", selection: $sourceMode) {
                    ForEach(SourceMode.allCases) { mode in
                        Label(mode.title, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .disabled(runner.state.isRunning)

                Text(sourceMode == .automatic
                    ? "곡명을 넣으면 IVE 영상을 찾고 받아 바로 교차편집합니다."
                    : "이미 가지고 있는 원본을 직접 가져옵니다.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
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
                        .disabled(runner.state.isRunning)
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
                    .disabled(runner.state.isRunning)
                    Text(editModeDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                labeled("대상 멤버") {
                    let canSelectMember = ProjectSettingsControlPolicy.canSelectMember(
                        editMode: editMode,
                        isRunning: runner.state.isRunning
                    )
                    Picker("대상 멤버", selection: $targetMember) {
                        ForEach(IVEMember.allCases) { member in
                            Text(member.rawValue).tag(member)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .disabled(!canSelectMember)
                    .opacity(canSelectMember ? 1 : 0.55)
                }

                labeled("화면 비율") {
                    let canSelectAspectRatio = ProjectSettingsControlPolicy.canSelectAspectRatio(
                        editMode: editMode,
                        sourceMode: sourceMode,
                        isRunning: runner.state.isRunning
                    )
                    Picker("화면 비율", selection: aspectRatioSelection) {
                        ForEach(OutputAspectRatio.allCases) { ratio in
                            Label(ratio.title, systemImage: ratio.systemImage)
                                .tag(ratio)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .disabled(!canSelectAspectRatio)
                    .opacity(canSelectAspectRatio ? 1 : 0.55)
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
                            .disabled(runner.state.isRunning)
                    }
                }
            }
        }
    }

    private var aspectRatioSelection: Binding<OutputAspectRatio> {
        Binding(
            get: { aspectRatio },
            set: { selectedRatio in
                if editMode == .member, sourceMode == .local {
                    aspectRatio = selectedRatio
                    manuallySelectedMemberAspectRatio = selectedRatio
                } else {
                    applyAspectRatioPolicy()
                }
            }
        )
    }

    private func applyAspectRatioPolicy() {
        aspectRatio = MemberAspectRatioPolicy.resolvedRatio(
            current: aspectRatio,
            editMode: editMode,
            sourceMode: sourceMode,
            manuallySelectedMemberRatio: manuallySelectedMemberAspectRatio
        )
    }

    private var automaticSourcesCard: some View {
        Card(title: "IVE 영상 자동 수집", systemImage: "sparkle.magnifyingglass") {
            VStack(alignment: .leading, spacing: 14) {
                VStack(spacing: 10) {
                    Image(systemName: editMode == .group
                        ? "rectangle.stack.badge.play"
                        : "person.crop.rectangle.stack")
                        .font(.system(size: 34, weight: .light))
                        .foregroundStyle(.purple)
                    Text(automaticSourceTitle)
                        .font(.headline)
                    Text(automaticWorkflowText)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.purple)
                        .multilineTextAlignment(.center)
                    Text(automaticSourceDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 18)
                .padding(.horizontal, 12)
                .background(.purple.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(.purple.opacity(0.22))
                }

                if editMode == .member {
                    HStack {
                        Label("자동 수집 목표", systemImage: "lock.fill")
                        Spacer()
                        Text(
                            "\(EditRequest.automaticMemberSourceCount)개 목표 · 최소 \(EditRequest.minimumAutomaticMemberSourceCount)개"
                        )
                            .font(.body.monospacedDigit().weight(.semibold))
                            .foregroundStyle(.purple)
                    }
                } else {
                    Stepper(value: $desiredSourceCount, in: 8...24) {
                        HStack {
                            Label("자동 수집 목표", systemImage: "number.square")
                            Spacer()
                            Text(
                                desiredSourceCount == 10
                                    ? "10개 목표 · 최소 8개"
                                    : "\(desiredSourceCount)개 필수"
                            )
                                .font(.body.monospacedDigit().weight(.semibold))
                                .foregroundStyle(.purple)
                        }
                    }
                    .disabled(runner.state.isRunning)
                }

                Text(
                    editMode == .member
                        ? "공식 세로 직캠 10개를 목표로 찾고, 최소 4개만 통과해도 컷 밀도를 줄이지 않습니다. 같은 소스의 다른 시간대를 비인접 구간에 균형 재사용합니다."
                        : desiredSourceCount == 10
                            ? "공식 음방 10개를 먼저 찾고, 검증을 통과한 영상이 8개뿐이면 8개로 진행합니다."
                            : "직접 선택한 \(desiredSourceCount)개는 모두 확보해야 진행합니다."
                )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Label(
                    "시작할 때 엔진과 yt-dlp 자동 수집 준비 상태를 별도로 검사합니다.",
                    systemImage: "checkmark.shield"
                )
                .font(.caption2)
                .foregroundStyle(.secondary)

                Divider()

                VStack(alignment: .leading, spacing: 7) {
                    Label("대용량 원본 저장 위치", systemImage: "externaldrive")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        Text(automaticCacheDirectory.path)
                            .font(.caption2)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                        Spacer(minLength: 0)
                        Button("변경…", action: chooseSourceCacheRootDirectory)
                            .controlSize(.small)
                            .disabled(runner.state.isRunning)
                    }
                }

                if editMode == .member {
                    Label(
                        "\(targetMember.rawValue) 공식 세로 직캠만 수집합니다. 자동 모드는 인물 재식별·임의 크롭 없이 원본 9:16 화면을 그대로 동기화해 교차편집합니다. 직접 앵커와 추적 크롭을 쓰려면 ‘내 영상 선택 · 고급’을 이용하세요.",
                        systemImage: "shield.checkered"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 0)
            }
            .frame(minHeight: 420)
        }
    }

    private var sourcesCard: some View {
        Card(title: "원본 영상", systemImage: "film.stack") {
            VStack(spacing: 12) {
                Button {
                    guard !runner.state.isRunning else { return }
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
                .disabled(runner.state.isRunning)
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

                if !runner.state.isRunning || !runner.outputURLs.isEmpty {
                    Divider()
                    HStack(spacing: 10) {
                        if !runner.state.isRunning {
                            Button {
                                runner.restoreLatestCompletedGroupResult(
                                    in: outputDirectory,
                                    clearExistingOutputsOnMissing: true
                                )
                            } label: {
                                Label(
                                    runner.isRestoringCompletedOutputs
                                        ? "찾는 중…"
                                        : "완성 결과 찾기",
                                    systemImage: "arrow.clockwise.circle"
                                )
                            }
                            .disabled(runner.isRestoringCompletedOutputs)
                        }
                        if let renderedVideoURL = runner.renderedVideoURL {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(renderedVideoURL.path)
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                    .truncationMode(.middle)
                                    .textSelection(.enabled)
                                HStack(spacing: 8) {
                                    Button {
                                        NSWorkspace.shared.open(renderedVideoURL)
                                    } label: {
                                        Label(
                                            "영상 열기",
                                            systemImage: "play.rectangle"
                                        )
                                    }
                                    Button {
                                        NSWorkspace.shared.activateFileViewerSelecting(
                                            [renderedVideoURL]
                                        )
                                    } label: {
                                        Label(
                                            "Finder에서 보기",
                                            systemImage: "folder.badge.checkmark"
                                        )
                                    }
                                }
                            }
                        }
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
                Button(
                    runner.cancellationRequested ? "취소 중…" : "취소",
                    role: .cancel
                ) {
                    runner.cancel()
                }
                .keyboardShortcut(.cancelAction)
                .disabled(runner.cancellationRequested)
            }

            Button {
                startAnalysis()
            } label: {
                Label(
                    runner.state.isRunning ? "편집 중…" : startButtonTitle,
                    systemImage: runner.state.isRunning ? "hourglass" : "sparkles.tv"
                )
                .frame(minWidth: 100)
            }
            .buttonStyle(.borderedProminent)
            .tint(.purple)
            .controlSize(.large)
            .disabled(runner.state.isRunning || !canStart)
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
        guard !runner.state.isRunning else { return }
        let trimmedSong = song.trimmingCharacters(in: .whitespacesAndNewlines)
        if sourceMode == .automatic {
            guard !trimmedSong.isEmpty else {
                validationMessage = "곡명을 입력해 주세요."
                return
            }
            validationMessage = nil
            let request = EditRequest(
                song: trimmedSong,
                targetMember: targetMember,
                editMode: editMode,
                aspectRatio: aspectRatio,
                sources: [],
                outputDirectory: outputDirectory,
                sourceMode: .automatic,
                desiredSourceCount: automaticSourceCount,
                sourceCacheDirectory: automaticCacheDirectory,
                catalogCacheDirectory: automaticCatalogCacheDirectory
            )
            runner.start(request: request)
            return
        }
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
        guard !runner.state.isRunning else { return }
        switch result {
        case .success(let urls):
            addVideos(urls)
        case .failure(let error):
            validationMessage = "영상을 열지 못했습니다: \(error.localizedDescription)"
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard !runner.state.isRunning else { return false }
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
                Task { @MainActor in
                    guard !runner.state.isRunning else { return }
                    addVideos([url])
                }
            }
        }
        return true
    }

    private func addVideos(_ urls: [URL]) {
        guard !runner.state.isRunning else { return }
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
        guard !runner.state.isRunning else { return }
        let panel = NSOpenPanel()
        panel.title = "결과를 저장할 폴더 선택"
        panel.prompt = "선택"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = outputDirectory

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let previousDefaultCacheRoot = outputDirectory
            .appendingPathComponent("CrossCut Sources", isDirectory: true)
            .standardizedFileURL
        let followsOutputDirectory =
            sourceCacheRootDirectory.standardizedFileURL == previousDefaultCacheRoot
        outputDirectory = url
        if followsOutputDirectory {
            sourceCacheRootDirectory = url
                .appendingPathComponent("CrossCut Sources", isDirectory: true)
        }
    }

    private func chooseSourceCacheRootDirectory() {
        guard !runner.state.isRunning else { return }
        let panel = NSOpenPanel()
        panel.title = "IVE 원본 영상을 저장할 폴더 선택"
        panel.prompt = "선택"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = sourceCacheRootDirectory

        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourceCacheRootDirectory = url
    }

    private var startButtonTitle: String {
        guard sourceMode == .automatic else { return "교차편집 시작" }
        switch editMode {
        case .group:
            return automaticSourceCount == 10
                ? "음방 10개를 목표로 찾아서 편집"
                : "음방 \(automaticSourceCount)개 찾아서 편집"
        case .member:
            return "\(targetMember.rawValue) 직캠 \(automaticSourceCount)개 찾아서 편집"
        }
    }

    private var canStart: Bool {
        EditStartPolicy.canStart(
            song: song,
            sourceMode: sourceMode,
            editMode: editMode,
            sourceCount: selectedVideos.count,
            anchoredSourceCount: selectedVideos.lazy.filter { $0.memberAnchor != nil }.count
        )
    }

    private var automaticSourceTitle: String {
        switch editMode {
        case .group:
            return automaticSourceCount == 10
                ? "음방 10개 목표 · 최소 8개로 교차편집"
                : "음방 \(automaticSourceCount)개를 찾아 교차편집"
        case .member:
            return "\(targetMember.rawValue) 직캠 10개 목표 · 최소 4개로 교차편집"
        }
    }

    private var automaticSourceDescription: String {
        switch editMode {
        case .group:
            return "IVE·곡명 제목으로 공식 음방 채널을 우선하고, 같은 곡 영상만 수집한 다음 음원으로 맞추고 편집합니다."
        case .member:
            return "IVE·곡명 제목으로 \(targetMember.rawValue) 공식 세로 직캠만 수집합니다. 원본 화면을 유지하고, 컷 바로 양쪽의 포즈·실루엣·피사체 움직임으로 동작 중 연결을 우선합니다."
        }
    }

    private var editModeDetail: String {
        if sourceMode == .automatic, editMode == .member {
            return "공식 대상 멤버 세로 직캠 10개를 목표로 하되 소스 하한만 4개입니다. 전환 밀도는 그대로 유지하고, 같은 직캠의 다른 시간대를 인접 반복 없이 균형 재사용하며 컷 양쪽 동작 연결을 우선합니다."
        }
        return editMode.detail
    }

    private var automaticWorkflowText: String {
        switch editMode {
        case .group:
            return automaticSourceCount == 10
                ? "곡명 입력 → 공식 음악방송 메타데이터 수집 → 10개 목표·최소 8개 자동 다운로드 → 자동 교차편집"
                : "곡명 입력 → 공식 음악방송 메타데이터 수집 → 음방 \(automaticSourceCount)개 자동 다운로드 → 자동 교차편집"
        case .member:
            return "곡명 입력 → 공식 직캠 10개 목표·최소 4개 수집 → 밀도 유지 시간차 재사용 → 자동 교차직캠"
        }
    }

    private var automaticSourceCount: Int {
        editMode == .member ? EditRequest.automaticMemberSourceCount : desiredSourceCount
    }

    private var automaticCacheDirectory: URL {
        let songComponent = safePathComponent(
            song.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "곡명 미지정" : song
        )
        let modeComponent = editMode == .group
            ? "group"
            : "member-\(safePathComponent(targetMember.rawValue))"
        return sourceCacheRootDirectory
            .appendingPathComponent("IVE", isDirectory: true)
            .appendingPathComponent(songComponent, isDirectory: true)
            .appendingPathComponent(modeComponent, isDirectory: true)
    }

    private var automaticCatalogCacheDirectory: URL {
        sourceCacheRootDirectory
            .appendingPathComponent("IVE", isDirectory: true)
            .appendingPathComponent("_catalog", isDirectory: true)
    }

    private func safePathComponent(_ value: String) -> String {
        let forbidden = CharacterSet(charactersIn: "/:\\?%*|\"<>")
        let pieces = value.components(separatedBy: forbidden)
        let joined = pieces.joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? "untitled" : joined
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
        defaultStorageDirectories.output
    }

    private static var defaultSourceCacheRootDirectory: URL {
        defaultStorageDirectories.sourceCache
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
