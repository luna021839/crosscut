import Foundation
import Darwin

enum EngineLocator {
    static let environmentKey = "CROSSCUT_ENGINE"
    static let pythonEnvironmentKey = "CROSSCUT_PYTHON"

    static func locate(
        fileManager: FileManager = .default,
        bundle: Bundle = .main
    ) -> URL? {
        var candidates: [URL] = []

        if let configuredPath = ProcessInfo.processInfo.environment[environmentKey],
           !configuredPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            candidates.append(URL(fileURLWithPath: configuredPath))
        }

        if let bundledScript = bundle.url(
            forResource: "crosscut_cli",
            withExtension: "py",
            subdirectory: "engine"
        ) {
            candidates.append(bundledScript)
        }
        if let resourceURL = bundle.resourceURL {
            // The engine is created by a post-build copy phase, so also check
            // the physical resource path instead of relying only on Bundle's
            // resource lookup table.
            candidates.append(
                resourceURL.appendingPathComponent("engine/crosscut_cli.py")
            )
        }

        let workingDirectory = URL(
            fileURLWithPath: fileManager.currentDirectoryPath,
            isDirectory: true
        )
        candidates.append(workingDirectory.appendingPathComponent("engine/crosscut_cli.py"))
        candidates.append(workingDirectory.appendingPathComponent("../engine/crosscut_cli.py"))
        candidates.append(workingDirectory.appendingPathComponent("../../engine/crosscut_cli.py"))

        #if DEBUG
        // Keep the developer checkout path out of distributable Release binaries.
        let sourceDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Engine
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // CrossCutMac
            .deletingLastPathComponent() // repository root
        candidates.append(sourceDirectory.appendingPathComponent("engine/crosscut_cli.py"))
        #endif

        var seen = Set<String>()
        for candidate in candidates {
            let normalized = candidate.standardizedFileURL
            guard seen.insert(normalized.path).inserted else { continue }
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: normalized.path, isDirectory: &isDirectory),
               !isDirectory.boolValue,
               fileManager.isReadableFile(atPath: normalized.path) {
                return normalized
            }
        }

        return nil
    }

    static func locatePython(
        near engineURL: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        var candidates: [URL] = []
        if let configuredPath = ProcessInfo.processInfo.environment[pythonEnvironmentKey],
           !configuredPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            candidates.append(URL(fileURLWithPath: configuredPath))
        }

        let engineDirectory = engineURL.deletingLastPathComponent()
        candidates.append(engineDirectory.appendingPathComponent(".venv/bin/python3"))
        #if DEBUG
        // A bundled debug engine still uses the workspace's selected runtime.
        let workspace = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(workspace.appendingPathComponent("engine/.venv/bin/python3"))
        #endif
        candidates.append(
            URL(fileURLWithPath: "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3")
        )
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/python3"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/python3"))
        candidates.append(URL(fileURLWithPath: "/usr/bin/python3"))

        var seen = Set<String>()
        for candidate in candidates {
            let normalized = candidate.standardizedFileURL
            guard seen.insert(normalized.path).inserted else { continue }
            if fileManager.isExecutableFile(atPath: normalized.path),
               supportsEngineModules(
                   pythonURL: normalized,
                   engineURL: engineURL
               ) {
                return normalized
            }
        }
        return nil
    }

    static func augmentedPath(existing: String?) -> String {
        let required = [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]
        let existingParts = (existing ?? "").split(separator: ":").map(String.init)
        return (required + existingParts).reduce(into: [String]()) { result, item in
            if !result.contains(item) { result.append(item) }
        }.joined(separator: ":")
    }

    static func supportsEngineModules(
        pythonURL: URL,
        engineURL: URL,
        timeout: TimeInterval = 10
    ) -> Bool {
        guard timeout.isFinite, timeout > 0 else { return false }
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = [
            "-B",
            "-c",
            "import json,sys;sys.path.insert(0,sys.argv[1]);from crosscut.runtime_contract import runtime_contract_report;r=runtime_contract_report();print(json.dumps(r,separators=(',',':')));raise SystemExit(0 if r.get('base_runtime_ready') else 1)",
            engineURL.deletingLastPathComponent().path,
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = environment
        let standardOutput = Pipe()
        process.standardOutput = standardOutput
        process.standardError = FileHandle.nullDevice
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        do {
            try process.run()
            let milliseconds = max(1, Int((timeout * 1_000).rounded()))
            if finished.wait(
                timeout: .now() + .milliseconds(milliseconds)
            ) == .timedOut {
                process.terminate()
                if finished.wait(timeout: .now() + .seconds(1)) == .timedOut {
                    kill(process.processIdentifier, SIGKILL)
                    _ = finished.wait(timeout: .now() + .seconds(1))
                }
                return false
            }
            guard process.terminationStatus == 0 else { return false }
            let data = standardOutput.fileHandleForReading.readDataToEndOfFile()
            return runtimeProbeReady(from: data)
        } catch {
            return false
        }
    }

    static func runtimeProbeReady(from data: Data) -> Bool {
        guard let text = String(data: data, encoding: .utf8) else { return false }
        for line in text.split(whereSeparator: \.isNewline).reversed() {
            guard let lineData = String(line).data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: lineData),
                  let payload = object as? [String: Any],
                  let ready = payload["base_runtime_ready"] as? Bool else {
                continue
            }
            return ready
        }
        return false
    }

    static func supportsAutomaticAcquisition(
        pythonURL: URL,
        engineURL: URL,
        editMode: EditMode,
        timeout: TimeInterval = 10
    ) -> Bool {
        supportsDoctorMode(
            pythonURL: pythonURL,
            engineURL: engineURL,
            timeout: timeout
        ) { data in
            automaticAcquisitionProbeReady(from: data, editMode: editMode)
        }
    }

    static func supportsLocalEditing(
        pythonURL: URL,
        engineURL: URL,
        editMode: EditMode,
        timeout: TimeInterval = 10
    ) -> Bool {
        supportsDoctorMode(
            pythonURL: pythonURL,
            engineURL: engineURL,
            timeout: timeout
        ) { data in
            localEditingProbeReady(from: data, editMode: editMode)
        }
    }

    private static func supportsDoctorMode(
        pythonURL: URL,
        engineURL: URL,
        timeout: TimeInterval,
        resultReady: (Data) -> Bool
    ) -> Bool {
        guard timeout.isFinite, timeout > 0 else { return false }
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = ["-B", engineURL.path, "doctor"]
        process.currentDirectoryURL = engineURL.deletingLastPathComponent()

        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = augmentedPath(existing: environment["PATH"])
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = environment

        let standardOutput = Pipe()
        process.standardOutput = standardOutput
        process.standardError = FileHandle.nullDevice
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        do {
            try process.run()
            let milliseconds = max(1, Int((timeout * 1_000).rounded()))
            if finished.wait(timeout: .now() + .milliseconds(milliseconds)) == .timedOut {
                _ = EngineProcessTree.requestTermination(
                    process,
                    forceKillDelayNanoseconds: 200_000_000
                )
                if finished.wait(timeout: .now() + .seconds(1)) == .timedOut {
                    if !EngineProcessTree.forceTermination(process) {
                        kill(process.processIdentifier, SIGKILL)
                    }
                    _ = finished.wait(timeout: .now() + .seconds(1))
                }
                return false
            }
            return resultReady(
                standardOutput.fileHandleForReading.readDataToEndOfFile()
            )
        } catch {
            return false
        }
    }

    static func automaticAcquisitionProbeReady(from data: Data, editMode: EditMode,
                                               requireProfessionalModel: Bool = false) -> Bool {
        guard let text = String(data: data, encoding: .utf8),
              let openingBrace = text.firstIndex(of: "{") else {
            return false
        }
        let jsonText = String(text[openingBrace...])
        guard let jsonData = jsonText.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: jsonData),
              let payload = object as? [String: Any],
              let modeReadiness = payload["mode_readiness"] as? [String: Any],
              let acquisition = payload["automatic_acquisition"] as? [String: Any],
              acquisition["ready"] as? Bool == true else {
            return false
        }
        let modeKey = editMode == .group ? "automatic_group" : "automatic_member"
        guard modeReadiness[modeKey] as? Bool == true else { return false }
        if editMode == .group {
            let runtimeContract = payload["runtime_contract"] as? [String: Any]
            let stageEdit = runtimeContract?["stage_edit_generation"] as? [String: Any]
            let modelReady = stageEdit?["automatic_group_generation_ready"] as? Bool == true
            let capabilities = payload["mode_capabilities"] as? [String: Any]
            let reviewFirst = capabilities?["review_first_group"] as? [String: Any]
            // Old engines cannot opt into the new baseline contract merely by
            // omitting a model field. Explicit required-model callers stay strict.
            return modelReady || (!requireProfessionalModel && reviewFirst?["ready"] as? Bool == true)
        }
        return true
    }

    static func localEditingProbeReady(from data: Data, editMode: EditMode) -> Bool {
        guard let text = String(data: data, encoding: .utf8),
              let openingBrace = text.firstIndex(of: "{") else {
            return false
        }
        let jsonText = String(text[openingBrace...])
        guard let jsonData = jsonText.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: jsonData),
              let payload = object as? [String: Any],
              let modeReadiness = payload["mode_readiness"] as? [String: Any] else {
            return false
        }
        let modeKey = editMode == .group ? "local_group" : "local_member"
        return modeReadiness[modeKey] as? Bool == true
    }
}
