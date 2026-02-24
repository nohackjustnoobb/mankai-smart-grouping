//
//  ContentView.swift
//  benchmark
//
//  Main UI for the benchmark app with directory selection and results.
//

import SwiftUI

struct ContentView: View {
    @State private var runner = BenchmarkRunner()
    @State private var modelsURL: URL?
    @State private var imagesURL: URL?
    @State private var iterations: Int = 1000
    @State private var warmupIterations: Int = 50
    @State private var showExporter = false

    private var canRun: Bool {
        modelsURL != nil && imagesURL != nil && !runner.status.isRunning
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Directory Selection
                    directorySection

                    // Configuration
                    configSection

                    // Run Button & Status
                    controlSection

                    // Results
                    ResultsView(results: runner.results)
                }
                .padding()
            }
            .navigationTitle("MLPackage Benchmark")
        }
    }

    // MARK: - Sections

    private var directorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Directories")
                .font(.headline)

            DirectorySelectorButton(
                title: "Models Directory",
                icon: "cube.box",
                selectedURL: $modelsURL
            )

            DirectorySelectorButton(
                title: "Images Directory",
                icon: "photo.stack",
                selectedURL: $imagesURL
            )
        }
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration")
                .font(.headline)

            VStack(spacing: 16) {
                HStack {
                    Text("Iterations")
                    Spacer()
                    Picker("Iterations", selection: $iterations) {
                        Text("1K").tag(1000)
                        Text("2K").tag(2000)
                        Text("5K").tag(5000)
                        Text("10K").tag(10000)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 220)
                }

                HStack {
                    Text("Warmup")
                    Spacer()
                    Picker("Warmup", selection: $warmupIterations) {
                        Text("10").tag(10)
                        Text("50").tag(50)
                        Text("100").tag(100)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 180)
                }
            }
            .padding()
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var controlSection: some View {
        VStack(spacing: 16) {
            // Progress bar
            if runner.status.isRunning {
                VStack(spacing: 8) {
                    ProgressView(value: runner.progress)
                        .tint(.blue)

                    Text(runner.status.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Run/Cancel button
            Button {
                if runner.status.isRunning {
                    runner.cancel()
                } else {
                    runBenchmark()
                }
            } label: {
                HStack {
                    if runner.status.isRunning {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .scaleEffect(0.8)
                        Text("Cancel")
                    } else {
                        Image(systemName: "play.fill")
                        Text("Run Benchmark")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canRun && !runner.status.isRunning)

            // Status message for error/completion
            if case let .error(message) = runner.status {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding()
                    .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            } else if case .completed = runner.status {
                Label("Benchmark completed successfully", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
                    .padding()
                    .background(.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            // Export button
            if !runner.results.isEmpty {
                Button {
                    showExporter = true
                } label: {
                    Label("Export Results", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                        .padding()
                }
                .buttonStyle(.bordered)
            }
        }
        .fileExporter(
            isPresented: $showExporter,
            document: BenchmarkResultsDocument(results: runner.results),
            contentType: .json,
            defaultFilename: "inference_benchmark_results.json"
        ) { result in
            if case let .failure(error) = result {
                print("Export failed: \(error)")
            }
        }
    }

    // MARK: - Actions

    private func runBenchmark() {
        guard let modelsURL, let imagesURL else { return }

        let config = BenchmarkConfig(
            modelsURL: modelsURL,
            imagesURL: imagesURL,
            iterations: iterations,
            warmupIterations: warmupIterations
        )

        Task {
            await runner.run(config: config)
        }
    }
}

// MARK: - Export Document

import UniformTypeIdentifiers

struct BenchmarkResultsDocument: FileDocument {
    static var readableContentTypes: [UTType] {
        [.json]
    }

    let results: [BenchmarkResult]

    init(results: [BenchmarkResult]) {
        self.results = results
    }

    init(configuration _: ReadConfiguration) throws {
        results = []
    }

    func fileWrapper(configuration _: WriteConfiguration) throws -> FileWrapper {
        let exportData = results.map { result in
            var data: [String: Any] = [
                "modelName": result.modelName,
                "modelPath": result.modelPath,
                "modelType": result.modelType?.rawValue ?? "unknown",
                "modelArchitecture": result.modelArchitecture.rawValue,
                "totalInferences": result.totalInferences,
                "totalTimeMs": result.totalTimeMs,
                "avgTimeMs": result.avgTimeMs,
                "minTimeMs": result.minTimeMs,
                "maxTimeMs": result.maxTimeMs,
                "inferencePerSecond": result.inferencePerSecond,
                "accuracy": result.accuracy ?? 0,
            ]

            // Include model settings if available
            if let settings = result.modelSettings {
                var settingsDict: [String: Any] = [:]
                if let model = settings.model { settingsDict["model"] = model }
                if let lr = settings.learningRate { settingsDict["learningRate"] = lr }
                if let bs = settings.batchSize { settingsDict["batchSize"] = bs }
                if let epochs = settings.epochs { settingsDict["epochs"] = epochs }
                if let patience = settings.patience { settingsDict["patience"] = patience }
                data["modelSettings"] = settingsDict
            }

            return data
        }

        let jsonData = try JSONSerialization.data(withJSONObject: exportData, options: [.prettyPrinted, .sortedKeys])
        return FileWrapper(regularFileWithContents: jsonData)
    }
}
