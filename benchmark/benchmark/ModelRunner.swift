//
//  ModelRunner.swift
//  benchmark
//
//  Handles loading and running MLPackage models, supporting both ImageType and TensorType inputs.
//

import CoreGraphics
import CoreImage
import CoreML
import Foundation

// MARK: - Model Input Type

enum ModelInputType {
    case imageType // Model expects CGImage/CVPixelBuffer inputs
    case tensorType // Model expects MLMultiArray inputs
}

// MARK: - Model Runner

/// Loads and runs MLPackage models for benchmarking
actor ModelRunner {
    private let context = CIContext()

    /// Load an MLModel from a URL
    func loadModel(from url: URL) async throws -> MLModel {
        // Configure for best performance
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Compile and load the model
        let compiledURL = try await MLModel.compileModel(at: url)
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        return model
    }

    /// Detect the input type of a model
    func detectInputType(model: MLModel) -> ModelInputType {
        let description = model.modelDescription

        guard let inputDescription = description.inputDescriptionsByName["image1"] else {
            // Fallback to tensor type if we can't determine
            return .tensorType
        }

        if inputDescription.type == .image {
            return .imageType
        } else {
            return .tensorType
        }
    }

    /// Run inference on a patch pair
    func predict(
        model: MLModel,
        inputType: ModelInputType,
        patch1: ImagePatch,
        patch2: ImagePatch
    ) throws -> Float {
        let provider: MLFeatureProvider

        switch inputType {
        case .imageType:
            provider = try createImageInput(patch1: patch1, patch2: patch2)
        case .tensorType:
            provider = try createTensorInput(patch1: patch1, patch2: patch2)
        }

        let output = try model.prediction(from: provider)

        // Extract the adjacency score
        guard let scoreFeature = output.featureValue(for: "adjacency_score") else {
            throw ModelRunnerError.outputNotFound
        }

        // Handle different output types
        if let multiArray = scoreFeature.multiArrayValue {
            return multiArray[0].floatValue
        } else if scoreFeature.type == .double {
            return Float(scoreFeature.doubleValue)
        } else {
            throw ModelRunnerError.unexpectedOutputType
        }
    }

    // MARK: - Private Methods

    /// Create input for ImageType models
    private func createImageInput(patch1: ImagePatch, patch2: ImagePatch) throws -> MLFeatureProvider {
        // Convert CGImage to CVPixelBuffer
        let pixelBuffer1 = try createPixelBuffer(from: patch1.image)
        let pixelBuffer2 = try createPixelBuffer(from: patch2.image)

        let features: [String: MLFeatureValue] = [
            "image1": MLFeatureValue(pixelBuffer: pixelBuffer1),
            "image2": MLFeatureValue(pixelBuffer: pixelBuffer2),
        ]

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Create input for TensorType models
    private func createTensorInput(patch1: ImagePatch, patch2: ImagePatch) throws -> MLFeatureProvider {
        guard let tensor1 = patch1.normalizedTensor,
              let tensor2 = patch2.normalizedTensor
        else {
            throw ModelRunnerError.missingNormalizedTensor
        }

        let features: [String: MLFeatureValue] = [
            "image1": MLFeatureValue(multiArray: tensor1),
            "image2": MLFeatureValue(multiArray: tensor2),
        ]

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Create CVPixelBuffer from CGImage
    private func createPixelBuffer(from image: CGImage) throws -> CVPixelBuffer {
        let width = image.width
        let height = image.height

        var pixelBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw ModelRunnerError.failedToCreatePixelBuffer
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let ciImage = CIImage(cgImage: image)
        context.render(ciImage, to: buffer)

        return buffer
    }
}

// MARK: - Model Info

/// Information about a loaded model
struct ModelInfo {
    let url: URL
    let name: String
    let model: MLModel
    let inputType: ModelInputType
    let modelType: ModelType?

    var displayName: String {
        if let type = modelType {
            return "\(name) (\(type.displayName))"
        }
        return name
    }
}

// MARK: - Errors

enum ModelRunnerError: LocalizedError {
    case outputNotFound
    case unexpectedOutputType
    case missingNormalizedTensor
    case failedToCreatePixelBuffer

    var errorDescription: String? {
        switch self {
        case .outputNotFound:
            return "Model output 'adjacency_score' not found"
        case .unexpectedOutputType:
            return "Unexpected output type from model"
        case .missingNormalizedTensor:
            return "Normalized tensor not available for TensorType model"
        case .failedToCreatePixelBuffer:
            return "Failed to create pixel buffer from image"
        }
    }
}
