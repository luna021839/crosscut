import Foundation

struct EngineEvent: Decodable, Equatable {
    let type: String
    let message: String?
    let stage: String?
    let level: String?
    let progress: Double?
    let outputPaths: [String]

    enum CodingKeys: String, CodingKey {
        case type
        case message
        case stage
        case level
        case progress
        case value
        case percent
        case output
        case outputPath = "output_path"
        case outputs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        message = try container.decodeIfPresent(String.self, forKey: .message)
        stage = try container.decodeIfPresent(String.self, forKey: .stage)
        level = try container.decodeIfPresent(String.self, forKey: .level)

        let rawProgress = try container.decodeFlexibleDouble(forKeys: [.progress, .value, .percent])
        if let rawProgress {
            progress = rawProgress > 1 ? rawProgress / 100 : rawProgress
        } else {
            progress = nil
        }

        var paths = try container.decodeIfPresent([String].self, forKey: .outputs) ?? []
        if let path = try container.decodeIfPresent(String.self, forKey: .outputPath) {
            paths.append(path)
        }
        if let path = try container.decodeIfPresent(String.self, forKey: .output) {
            paths.append(path)
        }
        outputPaths = paths
    }
}
private extension KeyedDecodingContainer where Key == EngineEvent.CodingKeys {
    func decodeFlexibleDouble(forKeys keys: [Key]) throws -> Double? {
        for key in keys {
            if let number = try decodeIfPresent(Double.self, forKey: key) {
                return number
            }
            if let text = try decodeIfPresent(String.self, forKey: key),
               let number = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return number
            }
        }
        return nil
    }
}
