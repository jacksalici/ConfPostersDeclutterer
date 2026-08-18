// Minimal macOS Vision OCR helper.
// Usage: vision_ocr <image-path> [more-image-paths...]
// Emits one JSON object per line: {"path":…,"lines":[{"text":…,"conf":…,"x":…,"y":…,"w":…,"h":…}]}
// Coordinates are Vision-normalised (0..1), origin bottom-left.

import Foundation
import Vision
import AppKit

func jsonEscape(_ s: String) -> String {
    var out = ""
    for ch in s.unicodeScalars {
        switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if ch.value < 0x20 {
                out += String(format: "\\u%04x", ch.value)
            } else {
                out.unicodeScalars.append(ch)
            }
        }
    }
    return out
}

func recognize(path: String) -> String {
    guard let image = NSImage(contentsOfFile: path),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return "{\"path\":\"\(jsonEscape(path))\",\"error\":\"unreadable-image\",\"lines\":[]}"
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return "{\"path\":\"\(jsonEscape(path))\",\"error\":\"\(jsonEscape(String(describing: error)))\",\"lines\":[]}"
    }
    let observations = request.results ?? []
    var parts: [String] = []
    for obs in observations {
        guard let best = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        parts.append("{\"text\":\"\(jsonEscape(best.string))\",\"conf\":\(best.confidence),"
                     + "\"x\":\(b.origin.x),\"y\":\(b.origin.y),\"w\":\(b.size.width),\"h\":\(b.size.height)}")
    }
    return "{\"path\":\"\(jsonEscape(path))\",\"lines\":[\(parts.joined(separator: ","))]}"
}

let args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty {
    FileHandle.standardError.write("usage: vision_ocr <image> [image...]\n".data(using: .utf8)!)
    exit(2)
}
for path in args {
    print(recognize(path: path))
    fflush(stdout)
}
