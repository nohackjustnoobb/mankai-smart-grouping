//
//  BenchmarkRunner.swift
//  benchmark
//
//  Orchestrates the benchmark process with timing metrics.
//

import Combine
import CoreML
import Foundation

// MARK: - Benchmark Runner

/// Orchestrates model benchmarking
@MainActor
@Observable
class BenchmarkRunner {
    // MARK: - Published State

    var status: BenchmarkStatus = .idle
    var results: [BenchmarkResult] = []
    var progress: Double = 0.0

    // MARK: - Private Properties

    private let modelRunner = ModelRunner()
    private let batchSize = 1000

    // MARK: - Accumulated Results (for batch processing)

    private struct AccumulatedResult {
        var times: [Double] = []
        var correctPredictions: Int = 0
        let modelURL: URL
        let name: String
        let modelType: ModelType?
        let modelSettings: ModelSettings?
    }

    // MARK: - Public Methods

    /// Run the benchmark with the given configuration
    func run(config: BenchmarkConfig) async {
        results = []
        progress = 0.0

        do {
            // 1. Find all model URLs (don't load yet)
            status = .loadingModels
            let modelURLs = try findModelURLs(in: config.modelsURL)

            if modelURLs.isEmpty {
                status = .error("No .mlpackage models found in the selected directory")
                return
            }

            // 2. Setup patch generator
            let generator = ImagePatchGenerator(imagesURL: config.imagesURL)
            try generator.loadImageList()

            // 3. Calculate batches
            let totalIterations = config.iterations
            let numBatches = (totalIterations + batchSize - 1) / batchSize

            // 4. Initialize accumulated results for each model
            var accumulatedResults: [AccumulatedResult] = modelURLs.map { modelURL in
                let folderName = modelURL.deletingLastPathComponent().lastPathComponent
                let fileName = modelURL.deletingPathExtension().lastPathComponent
                let name = "\(folderName)/\(fileName)"

                let modelType: ModelType?
                if modelURL.lastPathComponent == ModelType.optimized.rawValue {
                    modelType = .optimized
                } else if modelURL.lastPathComponent == ModelType.standard.rawValue {
                    modelType = .standard
                } else {
                    modelType = nil
                }

                let modelSettings = ModelSettings.load(from: modelURL)

                var result = AccumulatedResult(
                    modelURL: modelURL,
                    name: name,
                    modelType: modelType,
                    modelSettings: modelSettings
                )
                result.times.reserveCapacity(totalIterations)
                return result
            }

            // 5. Process batches
            var iterationsCompleted = 0

            for batchIndex in 0 ..< numBatches {
                let batchStart = batchIndex * batchSize
                let batchEnd = min(batchStart + batchSize, totalIterations)
                let currentBatchSize = batchEnd - batchStart

                // Generate patches for this batch
                status = .generatingPatches
                let patchPairs = try generator.generatePairs(count: currentBatchSize, includeNormalized: true)

                // Run all models on this batch
                for (modelIndex, modelURL) in modelURLs.enumerated() {
                    let name = accumulatedResults[modelIndex].name

                    // Load the model
                    status = .warming(modelName: name)

                    let model: MLModel
                    let inputType: ModelInputType
                    do {
                        model = try await modelRunner.loadModel(from: modelURL)
                        inputType = await modelRunner.detectInputType(model: model)
                    } catch {
                        print("Failed to load model at \(modelURL): \(error)")
                        continue
                    }

                    // Warmup only on first batch
                    if batchIndex == 0 {
                        try await warmup(model: model, inputType: inputType, patches: patchPairs, iterations: config.warmupIterations)
                    }

                    // Run benchmark on this batch
                    for i in 0 ..< currentBatchSize {
                        let globalIteration = iterationsCompleted + i + 1
                        status = .running(modelName: name, iteration: globalIteration, total: totalIterations)

                        let pair = patchPairs[i]

                        // Measure inference time
                        let startTime = CFAbsoluteTimeGetCurrent()
                        let score = try await modelRunner.predict(
                            model: model,
                            inputType: inputType,
                            patch1: pair.patch1,
                            patch2: pair.patch2
                        )
                        let endTime = CFAbsoluteTimeGetCurrent()

                        accumulatedResults[modelIndex].times.append((endTime - startTime) * 1000.0)

                        // Check accuracy
                        let predicted = score > 0.5
                        if predicted == pair.isAdjacent {
                            accumulatedResults[modelIndex].correctPredictions += 1
                        }

                        // Update progress
                        let modelProgress = Double(i + 1) / Double(currentBatchSize)
                        let batchProgress = (Double(batchIndex) + (Double(modelIndex) + modelProgress) / Double(modelURLs.count)) / Double(numBatches)
                        progress = batchProgress
                    }

                    // Model is released when it goes out of scope
                }

                iterationsCompleted += currentBatchSize

                // Patch pairs are released when they go out of scope at end of batch loop
            }

            // 6. Calculate final statistics for each model
            for accumulated in accumulatedResults {
                guard !accumulated.times.isEmpty else { continue }

                let sortedTimes = accumulated.times.sorted()
                let totalTime = accumulated.times.reduce(0, +)
                let avgTime = totalTime / Double(accumulated.times.count)
                let minTime = sortedTimes.first ?? 0
                let maxTime = sortedTimes.last ?? 0
                let accuracy = Double(accumulated.correctPredictions) / Double(totalIterations) * 100.0

                let result = BenchmarkResult(
                    modelName: accumulated.name,
                    modelPath: accumulated.modelURL.path,
                    modelType: accumulated.modelType,
                    modelSettings: accumulated.modelSettings,
                    totalInferences: totalIterations,
                    totalTimeMs: totalTime,
                    avgTimeMs: avgTime,
                    minTimeMs: minTime,
                    maxTimeMs: maxTime,
                    accuracy: accuracy
                )

                results.append(result)
            }

            status = .completed
            progress = 1.0

        } catch {
            status = .error(error.localizedDescription)
        }
    }

    /// Cancel the current benchmark (if any)
    func cancel() {
        // Currently running benchmarks can be cancelled by checking status
        status = .idle
        progress = 0.0
    }

    // MARK: - Private Methods

    /// Find all MLPackage model URLs in a directory (recursively)
    private func findModelURLs(in url: URL) throws -> [URL] {
        let fileManager = FileManager.default

        // Recursively find all .mlpackage directories
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var mlpackageURLs: [URL] = []

        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == "mlpackage" {
                mlpackageURLs.append(fileURL)
                // Skip enumerating inside .mlpackage bundles
                enumerator.skipDescendants()
            }
        }

        return mlpackageURLs
    }

    /// Warmup the model
    private func warmup(model: MLModel, inputType: ModelInputType, patches: [PatchPair], iterations: Int) async throws {
        guard !patches.isEmpty else { return }

        for i in 0 ..< iterations {
            let pair = patches[i % patches.count]
            _ = try await modelRunner.predict(
                model: model,
                inputType: inputType,
                patch1: pair.patch1,
                patch2: pair.patch2
            )
        }
    }
}
