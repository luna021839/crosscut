import Foundation

/// Keeps downloaded stages, render proxies and final videos off the internal
/// disk whenever the user's dedicated external SSD is mounted.
enum LargeMediaStoragePolicy {
    struct Directories: Equatable {
        let output: URL
        let sourceCache: URL
        let usesPreferredExternalVolume: Bool
    }

    static let preferredVolume = URL(
        fileURLWithPath: "/Volumes/CrossCutMedia",
        isDirectory: true
    )

    static func defaultDirectories(
        fileManager: FileManager = .default,
        preferredVolume: URL = preferredVolume,
        fallbackMoviesDirectory: URL? = nil
    ) -> Directories {
        let external = preferredVolume.standardizedFileURL
        var isDirectory: ObjCBool = false
        let externalIsUsable = fileManager.fileExists(
            atPath: external.path,
            isDirectory: &isDirectory
        ) && isDirectory.boolValue && fileManager.isWritableFile(atPath: external.path)

        if externalIsUsable {
            let root = external.appendingPathComponent(
                "CrossCut",
                isDirectory: true
            )
            return Directories(
                output: root.appendingPathComponent("Outputs", isDirectory: true),
                sourceCache: root.appendingPathComponent("Sources", isDirectory: true),
                usesPreferredExternalVolume: true
            )
        }

        let movies = fallbackMoviesDirectory
            ?? fileManager.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser
        let output = movies.appendingPathComponent("CrossCut", isDirectory: true)
        return Directories(
            output: output,
            sourceCache: output.appendingPathComponent(
                "CrossCut Sources",
                isDirectory: true
            ),
            usesPreferredExternalVolume: false
        )
    }
}
