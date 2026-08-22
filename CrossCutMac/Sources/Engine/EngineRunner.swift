import Combine
import Foundation

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

@MainActor
final class EngineRunner: ObservableObject {
    @Published private(set) var state: EngineRunState = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var statusMessage = "교차편집할 영상을 추가해 주세요."
    @Published private(set) var logs: [EngineLogEntry] = []
    @Published private(set) var outputURLs: [URL] = []
    @Published private(set) var engineURL: URL?

    private var process: Process?
    private var requestFileURL: URL?
    private var cancellationRequested = false
    private var reportedError: String?
    private var resultMessage: String?

    init() {
        engineURL = EngineLocator.locate()
    }

    func refreshEngineLocation() {
        engineURL = EngineLocator.locate()
    }

    func start(request: EditRequest) {
        guard !state.isRunning else { return }

        resetForNewRun()
        state = .preparing
        statusMessage = "엔진을 준비하는 중…"

        guard let engineURL = EngineLocator.locate() else {
            fail(
                "crosscut_cli.py를 찾지 못했습니다. 저장소의 engine 폴더에 두거나 "
                    + "\(EngineLocator.environmentKey) 환경변수로 경로를 지정해 주세요."
            )
            return
        }
        self.engineURL = engineURL

        guard let pythonURL = EngineLocator.locatePython(near: engineURL) else {
            fail(
                "Python 3.11+, NumPy 2.0+, OpenCV Contrib 4.10+와 "
                    + "사용 가능한 멤버 트래커를 찾지 못했습니다. "
                    + "CROSSCUT_PYTHON 환경변수로 요구 사항을 갖춘 Python 경로를 "
                    + "지정해 주세요."
            )
            return
        }

        do {
            let outputDirectory = URL(fileURLWithPath: request.outputDirectory, isDirectory: true)
            try FileManager.default.createDirectory(
                at: outputDirectory,
                withIntermediateDirectories: true
            )

            let requestURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("ive-crosscut-\(UUID().uuidString)")
                .appendingPathExtension("json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(request).write(to: requestURL, options: .atomic)
            requestFileURL = requestURL

            try launch(engineURL: engineURL, pythonURL: pythonURL, requestURL: requestURL)
        } catch {
            cleanupRequestFile()
            fail("엔진을 시작하지 못했습니다: \(error.localizedDescription)")
        }
    }

    func cancel() {
        guard state.isRunning else { return }
        cancellationRequested = true
        statusMessage = "취소하는 중…"
        appendLog("사용자가 작업 취소를 요청했습니다.", level: .warning)
        process?.terminate()
    }

    func clearLog() {
        guard !state.isRunning else { return }
        logs.removeAll()
    }

    private func resetForNewRun() {
        progress = 0
        logs.removeAll()
        outputURLs.removeAll()
        cancellationRequested = false
        reportedError = nil
        resultMessage = nil
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

        progress = 1
        state = .completed
        let videoExtensions = Set(["mp4", "mov", "m4v"])
        let hasRenderedVideo = outputURLs.contains {
            videoExtensions.contains($0.pathExtension.lowercased())
        }
        if hasRenderedVideo {
            statusMessage = "교차편집 결과가 준비되었습니다."
        } else {
            statusMessage = resultMessage ?? "교차편집 처리가 완료되었습니다."
        }
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
