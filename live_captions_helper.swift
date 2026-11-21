// Native half of live_captions.py (macOS 13+).
//
//  - Captures everything the Mac is playing and writes it to stdout as raw
//    16 kHz mono float32 PCM. Default is a Core Audio process tap (macOS 14.2+),
//    which records audio only; ScreenCaptureKit (--capture screen) also works
//    but starts a screen capture, which makes DRM-protected video players
//    blank their video.
//  - Reads JSON lines {"zh": "...", "ja": "..."} from stdin and shows them in a
//    click-through subtitle bar that floats above every window, full-screen
//    video players included. With --translate, lines that arrive without "zh"
//    are translated on-device by Apple's Translation framework first.
//
// Build:  swiftc -O -swift-version 5 live_captions_helper.swift -o .build/live_captions_helper
// Flags:  --capture tap|screen|none   audio source (default tap; none = overlay only)
//         --font-size N   Chinese font size in points (default 34)
//         --hold S        seconds a subtitle stays up without a new one (default 6)
//         --bottom N      distance from the bottom of the screen in points (default 90)
//         --translate     translate lines without "zh" with Apple Translation (macOS 15+)
//         --source CODE   spoken language for --translate, or auto (default ja)
//         --show-original show the "ja" line under the Chinese one

import AppKit
import AVFoundation
import CoreAudio
import CoreMedia
import ScreenCaptureKit
import SwiftUI
import Translation

struct Options {
    var fontSize: CGFloat = 34
    var hold: Double = 6
    var bottom: CGFloat = 90
    var capture = "tap"
    var translate = false
    var source = "ja"
    var showOriginal = false

    init(_ args: [String]) {
        var it = args.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--font-size": fontSize = CGFloat(Double(it.next() ?? "") ?? 34)
            case "--hold": hold = Double(it.next() ?? "") ?? 6
            case "--bottom": bottom = CGFloat(Double(it.next() ?? "") ?? 90)
            case "--capture": capture = it.next() ?? "tap"
            case "--translate": translate = true
            case "--source": source = it.next() ?? "ja"
            case "--show-original": showOriginal = true
            default: fputs("helper: ignoring unknown flag \(arg)\n", stderr)
            }
        }
    }
}

// ── System audio capture: Core Audio process tap ─────────────────────────────

final class ProcessTap {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private let queue = DispatchQueue(label: "tap")
    private let out = FileHandle.standardOutput
    private var silentFrames = 0
    private var warnedSilent = false

    func start() throws {
        // Tap the mix of every process; private + unmuted so playback is unaffected
        let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        desc.uuid = UUID()
        desc.isPrivate = true
        desc.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(desc, &tapID), "create process tap")

        // A tap is read through an aggregate device that contains it
        let outputUID = try defaultOutputUID()
        let agg: [String: Any] = [
            kAudioAggregateDeviceNameKey: "live_captions",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: desc.uuid.uuidString]],
        ]
        try check(AudioHardwareCreateAggregateDevice(agg as CFDictionary, &aggID), "create aggregate device")

        var asbd = try property(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription())
        guard let inFmt = AVAudioFormat(streamDescription: &asbd),
              let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                         channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFmt, to: outFmt) else {
            throw NSError(domain: "live_captions", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "unsupported tap format \(asbd)"])
        }

        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { [unowned self] _, input, _, _, _ in
            guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, bufferListNoCopy: input, deallocator: nil),
                  let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity:
                      AVAudioFrameCount(Double(inBuf.frameLength) * 16000 / inFmt.sampleRate) + 32)
            else { return }
            var fed = false
            converter.convert(to: outBuf, error: nil) { _, status in
                if fed { status.pointee = .noDataNow; return nil }  // more arrives next callback
                fed = true
                status.pointee = .haveData
                return inBuf
            }
            let samples = UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength))
            self.warnIfSilent(samples)
            self.out.write(Data(buffer: samples))
        }, "create IO proc")
        try check(AudioDeviceStart(aggID, procID), "start capture")
    }

    // Without permission the tap delivers exact zeros instead of failing.
    private func warnIfSilent(_ samples: UnsafeBufferPointer<Float>) {
        guard !warnedSilent else { return }
        if samples.contains(where: { $0 != 0 }) { silentFrames = 0; return }
        silentFrames += samples.count
        if silentFrames > 16000 * 20 {
            warnedSilent = true
            fputs("helper: 20 s of pure silence — if something is playing, allow your terminal app in "
                  + "System Settings → Privacy & Security → Screen & System Audio Recording.\n", stderr)
        }
    }

    private func defaultOutputUID() throws -> String {
        let device = try property(AudioObjectID(kAudioObjectSystemObject),
                                  kAudioHardwarePropertyDefaultSystemOutputDevice, AudioObjectID(0))
        return try property(device, kAudioDevicePropertyDeviceUID, "" as CFString) as String
    }

    private func property<T>(_ obj: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T) throws -> T {
        var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                              mElement: kAudioObjectPropertyElementMain)
        var value = initial
        var size = UInt32(MemoryLayout<T>.size)
        let status = withUnsafeMutableBytes(of: &value) {
            AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, $0.baseAddress!)
        }
        try check(status, "read property \(selector)")
        return value
    }

    private func check(_ status: OSStatus, _ what: String) throws {
        guard status == noErr else {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "\(what) failed (OSStatus \(status))"])
        }
    }
}

// ── System audio capture: ScreenCaptureKit (blanks DRM video) ────────────────

final class ScreenTap: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "audio")
    private let out = FileHandle.standardOutput
    private var checkedFormat = false

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw NSError(domain: "live_captions", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no display found"])
        }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.sampleRate = 16000            // what Whisper wants; no resampling needed
        cfg.channelCount = 1
        cfg.excludesCurrentProcessAudio = true
        // Audio-only in practice, but SCK always captures video: keep it tiny.
        cfg.width = 2
        cfg.height = 2
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)  // silences "output NOT found" logs
        try await s.startCapture()
        stream = s
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sb.isValid else { return }
        if !checkedFormat, let asbd = sb.formatDescription?.audioStreamBasicDescription {
            checkedFormat = true
            let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
            if asbd.mSampleRate != 16000 || !isFloat || asbd.mBitsPerChannel != 32 {
                fputs("helper: unexpected audio format \(asbd)\n", stderr)
                exit(3)
            }
        }
        // Non-interleaved float32: buffer 0 is the (only) channel.
        try? sb.withAudioBufferList { abl, _ in
            guard let buf = abl.first, let data = buf.mData else { return }
            out.write(Data(bytes: data, count: Int(buf.mDataByteSize)))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("helper: capture stopped: \(error.localizedDescription)\n", stderr)
        exit(4)
    }
}

// ── Subtitle overlay ─────────────────────────────────────────────────────────

final class Overlay {
    private let opts: Options
    private let panel: NSPanel
    private let box = NSView()
    private let zhLabel = NSTextField(wrappingLabelWithString: "")
    private let jaLabel = NSTextField(wrappingLabelWithString: "")
    private var hideWork: DispatchWorkItem?
    private let pad: CGFloat = 14

    init(_ opts: Options) {
        self.opts = opts
        panel = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel],
                        backing: .buffered, defer: false)
        panel.level = .screenSaver                       // above full-screen players
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        panel.ignoresMouseEvents = true                  // clicks pass through to the video
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false

        box.wantsLayer = true
        box.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.55).cgColor
        box.layer?.cornerRadius = 10
        panel.contentView = box

        let shadow = NSShadow()
        shadow.shadowColor = .black
        shadow.shadowBlurRadius = 3
        shadow.shadowOffset = NSSize(width: 0, height: -1)
        for (label, size, weight) in [(zhLabel, opts.fontSize, NSFont.Weight.semibold),
                                      (jaLabel, opts.fontSize * 0.55, NSFont.Weight.regular)] {
            label.font = NSFont.systemFont(ofSize: size, weight: weight)  // falls back to PingFang for CJK
            label.textColor = label === zhLabel ? .white : NSColor(white: 0.85, alpha: 1)
            label.alignment = .center
            label.shadow = shadow
            label.isSelectable = false
            box.addSubview(label)
        }
    }

    func show(zh: String, ja: String) {
        guard let screen = NSScreen.main else { return }
        let ja = opts.showOriginal ? ja : ""
        let maxW = screen.frame.width * 0.8 - 2 * pad
        zhLabel.stringValue = zh
        jaLabel.stringValue = ja
        let zhSize = fit(zhLabel, maxW)
        let jaSize = ja.isEmpty ? .zero : fit(jaLabel, maxW)
        let gap: CGFloat = ja.isEmpty ? 0 : 4
        let w = max(zhSize.width, jaSize.width) + 2 * pad
        let h = zhSize.height + gap + jaSize.height + 2 * pad * 0.6

        jaLabel.isHidden = ja.isEmpty
        jaLabel.frame = NSRect(x: (w - jaSize.width) / 2, y: pad * 0.6, width: jaSize.width, height: jaSize.height)
        zhLabel.frame = NSRect(x: (w - zhSize.width) / 2, y: pad * 0.6 + jaSize.height + gap,
                               width: zhSize.width, height: zhSize.height)
        let f = screen.frame
        panel.setFrame(NSRect(x: f.midX - w / 2, y: f.minY + opts.bottom, width: w, height: h), display: true)
        panel.orderFrontRegardless()

        hideWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.panel.orderOut(nil) }
        hideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + opts.hold, execute: work)
    }

    private func fit(_ label: NSTextField, _ maxW: CGFloat) -> NSSize {
        let s = label.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: maxW, height: .greatestFiniteMagnitude))
        return NSSize(width: ceil(s.width), height: ceil(s.height))
    }
}

// ── On-device translation: Apple Translation framework ──────────────────────
//
// The macOS 15 SDK (Command Line Tools) can't create a TranslationSession
// directly — SwiftUI's .translationTask hands one to a view — so a hidden 1×1
// window hosts that view and its task translates queued lines for the whole run.

let zhHans = Locale.Language(identifier: "zh-Hans")
let pending = AsyncStream<String>.makeStream()
var translatorWindow: NSWindow?

struct TranslatorView: View {
    let source: Locale.Language?

    var body: some View {
        Color.clear.translationTask(source: source, target: zhHans) { session in
            for await ja in pending.stream {
                let zh = (try? await session.translate(ja).targetText) ?? ja
                fputs("          \(zh)\n", stderr)
                await MainActor.run { subtitleBar.show(zh: zh, ja: ja) }
            }
        }
    }
}

func startTranslator(_ code: String) {
    let source = code == "auto" ? nil : Locale.Language(identifier: code)
    Task { @MainActor in
        // Never let the framework ask to download languages: its prompt would
        // appear in the invisible window and block forever.
        if let source {
            let status = await LanguageAvailability().status(from: source, to: zhHans)
            guard status == .installed else {
                fputs(status == .supported
                      ? "helper: download the translation languages first: System Settings → General → "
                        + "Language & Region → Translation Languages → \(code) and Chinese (Simplified).\n"
                      : "helper: Apple Translation can't translate \(code) to Chinese; use --translator claude.\n",
                      stderr)
                exit(5)
            }
        }
        let win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1, height: 1), styleMask: .borderless,
                           backing: .buffered, defer: false)
        win.alphaValue = 0
        win.ignoresMouseEvents = true
        win.contentView = NSHostingView(rootView: TranslatorView(source: source))
        win.orderFrontRegardless()  // the task only runs while its view is on screen
        translatorWindow = win
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

signal(SIGPIPE, SIG_DFL)  // Python went away → die quietly
let opts = Options(CommandLine.arguments)
let app = NSApplication.shared
app.setActivationPolicy(.accessory)  // no Dock icon, never steals focus
let subtitleBar = Overlay(opts)
let processTap = ProcessTap()
let screenTap = ScreenTap()

func captureFailed(_ error: Error) -> Never {
    fputs("helper: cannot capture audio: \(error.localizedDescription)\n"
          + "helper: allow your terminal app in System Settings → Privacy & Security → "
          + "Screen & System Audio Recording, then restart it.\n", stderr)
    exit(2)
}

switch opts.capture {
case "tap":
    do {
        try processTap.start()
        fputs("helper: capturing system audio (process tap)\n", stderr)
    } catch { captureFailed(error) }
case "screen":
    Task {
        do {
            try await screenTap.start()
            fputs("helper: capturing system audio (ScreenCaptureKit)\n", stderr)
        } catch { captureFailed(error) }
    }
default:
    break  // overlay only
}

if opts.translate {
    startTranslator(opts.source)
}

Thread {
    while let line = readLine() {
        guard let data = line.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { continue }
        let ja = msg["ja"] ?? ""
        if let zh = msg["zh"] {
            DispatchQueue.main.async { subtitleBar.show(zh: zh, ja: ja) }
        } else if opts.translate, !ja.isEmpty {
            pending.continuation.yield(ja)
        }
    }
    exit(0)  // stdin closed: Python exited
}.start()

app.run()
