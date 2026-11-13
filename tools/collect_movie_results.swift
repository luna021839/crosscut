// Collect viewable results as Finder aliases, without copying large movies.
// Usage: swift tools/collect_movie_results.swift DESTINATION PREFERRED... --other-roots DIRECTORY...
import Foundation

enum CollectionError: Error { case usage, unsafeTarget(String), wrongAlias(String) }
let args = Array(CommandLine.arguments.dropFirst())
guard let separator = args.firstIndex(of: "--other-roots"), separator >= 2, separator < args.count - 1 else {
    throw CollectionError.usage
}
let manager = FileManager.default
let destination = URL(fileURLWithPath: args[0], isDirectory: true)
let previous = destination.appendingPathComponent("이전 비교본", isDirectory: true)
let preferred = args[1..<separator].map { URL(fileURLWithPath: $0).standardizedFileURL }
let roots = args[(separator + 1)...].map { URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL }
guard Set(preferred.map(\.lastPathComponent)).count == preferred.count else { throw CollectionError.usage }
try manager.createDirectory(at: destination, withIntermediateDirectories: true)
try manager.createDirectory(at: previous, withIntermediateDirectories: true)

func alias(_ source: URL, in directory: URL, honorRenamedAliases: Bool = false) throws {
    let values = try source.resourceValues(forKeys: [.isRegularFileKey, .isAliasFileKey, .isSymbolicLinkKey])
    guard values.isRegularFile == true, values.isAliasFile != true, values.isSymbolicLink != true,
          source.pathExtension.lowercased() == "mp4" else { throw CollectionError.unsafeTarget(source.path) }
    if honorRenamedAliases {
        let entries = try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isAliasFileKey], options: [.skipsHiddenFiles])
        for entry in entries where (try entry.resourceValues(forKeys: [.isAliasFileKey])).isAliasFile == true {
            if try URL(resolvingAliasFileAt: entry, options: [.withoutUI, .withoutMounting]).standardizedFileURL == source {
                return
            }
        }
    }
    let output = directory.appendingPathComponent(source.lastPathComponent)
    if manager.fileExists(atPath: output.path) {
        let existing = try output.resourceValues(forKeys: [.isAliasFileKey])
        guard existing.isAliasFile == true,
              try URL(resolvingAliasFileAt: output, options: [.withoutUI, .withoutMounting]).standardizedFileURL == source else {
            throw CollectionError.wrongAlias(output.path)
        }
    } else {
        let data = try source.bookmarkData(options: [.suitableForBookmarkFile], includingResourceValuesForKeys: nil, relativeTo: nil)
        try URL.writeBookmarkData(data, to: output)
    }
    guard try URL(resolvingAliasFileAt: output, options: [.withoutUI, .withoutMounting]).standardizedFileURL == source else {
        throw CollectionError.wrongAlias(output.path)
    }
}

for source in preferred { try alias(source, in: destination, honorRenamedAliases: true) }
var count = 0
for root in roots {
    let files = try manager.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
    for source in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        if source.pathExtension.lowercased() != "mp4" || preferred.contains(source.standardizedFileURL) { continue }
        try alias(source.standardizedFileURL, in: previous)
        count += 1
    }
}
print("Verified \(preferred.count) current and \(count) previous Finder aliases at \(destination.path). Movie bytes were not copied or moved.")
