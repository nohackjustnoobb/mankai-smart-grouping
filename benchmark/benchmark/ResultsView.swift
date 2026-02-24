//
//  ResultsView.swift
//  benchmark
//
//  Displays benchmark results in a formatted view.
//

import SwiftUI

/// Displays benchmark results for all models
struct ResultsView: View {
    let results: [BenchmarkResult]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Results")
                .font(.headline)

            if results.isEmpty {
                ContentUnavailableView(
                    "No Results",
                    systemImage: "chart.bar",
                    description: Text("Run a benchmark to see results")
                )
            } else {
                ForEach(results) { result in
                    ResultCard(result: result)
                }
            }
        }
    }
}

/// Card view for a single benchmark result
struct ResultCard: View {
    let result: BenchmarkResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.modelName)
                        .font(.headline)

                    if let type = result.modelType {
                        Text(type.displayName)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Text(result.modelArchitecture.displayName)
                        .font(.caption)
                        .foregroundStyle(.purple)

                    // Model architecture from settings
                    if let settings = result.modelSettings, let model = settings.model {
                        Text(model.capitalized)
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }
                }

                Spacer()

                // Main metric: inferences per second
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f", result.inferencePerSecond))
                        .font(.title2.bold())
                        .foregroundStyle(.blue)
                    Text("inf/sec")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            // Model Settings (if available)
            if let settings = result.modelSettings {
                HStack(spacing: 16) {
                    if let lr = settings.learningRate {
                        SettingBadge(label: "LR", value: String(format: "%.0e", lr))
                    }
                    if let bs = settings.batchSize {
                        SettingBadge(label: "Batch", value: "\(bs)")
                    }
                    if let epochs = settings.epochs {
                        SettingBadge(label: "Epochs", value: "\(epochs)")
                    }
                }
            }

            Divider()

            // Metrics grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible()),
            ], spacing: 12) {
                MetricView(title: "Avg", value: formatTime(result.avgTimeMs))
                MetricView(title: "Min", value: formatTime(result.minTimeMs))
                MetricView(title: "Max", value: formatTime(result.maxTimeMs))
            }

            // Additional info
            HStack {
                Label("\(result.totalInferences) inferences", systemImage: "number")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                if let accuracy = result.accuracy {
                    Label(String(format: "%.1f%% accurate", accuracy), systemImage: "checkmark.circle")
                        .font(.caption)
                        .foregroundStyle(accuracy >= 80 ? .green : (accuracy >= 60 ? .orange : .red))
                }
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func formatTime(_ ms: Double) -> String {
        if ms < 1 {
            return String(format: "%.2f ms", ms)
        } else if ms < 100 {
            return String(format: "%.1f ms", ms)
        } else {
            return String(format: "%.0f ms", ms)
        }
    }
}

/// Single metric display
struct MetricView: View {
    let title: String
    let value: String

    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.subheadline.monospacedDigit())
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

/// Badge for displaying a model setting
struct SettingBadge: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.quaternary, in: Capsule())
    }
}
