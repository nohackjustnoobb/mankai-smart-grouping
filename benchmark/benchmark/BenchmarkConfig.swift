//
//  BenchmarkConfig.swift
//  benchmark
//
//  Configuration and data types for the benchmark app.
//

import Foundation

// MARK: - Configuration

/// Configuration for a benchmark run
struct BenchmarkConfig {
    /// Directory containing .mlpackage files
    let modelsURL: URL
    /// Directory containing source images
    let imagesURL: URL
    /// Number of benchmark iterations per model
    var iterations: Int = 100
    /// Number of warmup runs before timing
    var warmupIterations: Int = 5
}

// MARK: - Model Settings

/// Training settings loaded from metrics.json
struct ModelSettings: Codable, Equatable {
    let epochs: Int?
    let batchSize: Int?
    let learningRate: Double?
    let model: String?
    let patience: Int?

    enum CodingKeys: String, CodingKey {
        case epochs
        case batchSize = "batch_size"
        case learningRate = "learning_rate"
        case model
        case patience
    }

    /// Load settings from metrics.json in the model's directory
    static func load(from modelURL: URL) -> ModelSettings? {
        let metricsURL = modelURL.deletingLastPathComponent().appendingPathComponent("metrics.json")
        guard FileManager.default.fileExists(atPath: metricsURL.path) else { return nil }

        do {
            let data = try Data(contentsOf: metricsURL)
            let json = try JSONDecoder().decode(MetricsFile.self, from: data)
            return json.parameters
        } catch {
            print("Failed to load metrics.json: \(error)")
            return nil
        }
    }

    /// Wrapper for the full metrics.json structure
    private struct MetricsFile: Codable {
        let parameters: ModelSettings
    }
}

// MARK: - Model Types

/// Types of MLPackage models supported
enum ModelType: String, CaseIterable, Identifiable {
    /// Optimized model with ImageType inputs and float16 precision
    case optimized = "model_optimized.mlpackage"
    /// Standard model with TensorType inputs and float32 precision
    case standard = "model.mlpackage"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .optimized:
            return "Optimized (ImageType, FP16)"
        case .standard:
            return "Standard (TensorType, FP32)"
        }
    }
}

// MARK: - Benchmark Results

/// Result of benchmarking a single model
struct BenchmarkResult: Identifiable {
    var id: String {
        URL(fileURLWithPath: modelPath).deletingLastPathComponent().lastPathComponent
    }

    let modelName: String
    let modelPath: String
    let modelType: ModelType?
    let modelSettings: ModelSettings?
    let totalInferences: Int
    let totalTimeMs: Double
    let avgTimeMs: Double
    let minTimeMs: Double
    let maxTimeMs: Double
    let accuracy: Double?

    var inferencePerSecond: Double {
        guard avgTimeMs > 0 else { return 0 }
        return 1000.0 / avgTimeMs
    }
}

// MARK: - Status

/// Current status of the benchmark
enum BenchmarkStatus: Equatable {
    case idle
    case loadingModels
    case generatingPatches
    case warming(modelName: String)
    case running(modelName: String, iteration: Int, total: Int)
    case completed
    case error(String)

    var isRunning: Bool {
        switch self {
        case .idle, .completed, .error:
            return false
        default:
            return true
        }
    }

    var description: String {
        switch self {
        case .idle:
            return "Ready to benchmark"
        case .loadingModels:
            return "Loading models..."
        case .generatingPatches:
            return "Generating image patches..."
        case let .warming(name):
            return "Warming up \(name)..."
        case let .running(name, iteration, total):
            return "Running \(name): \(iteration)/\(total)"
        case .completed:
            return "Benchmark completed"
        case let .error(message):
            return "Error: \(message)"
        }
    }
}
