//
//  ImagePatchGenerator.swift
//  benchmark
//
//  Generates 224×224 patch pairs from images for both Siamese and Merged model architectures.
//

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Constants

private let kPatchSize: Int = 224
private let kMean: [Float] = [0.485, 0.456, 0.406]
private let kStd: [Float] = [0.229, 0.224, 0.225]

// MARK: - Image Patch

/// A single image patch ready for model input
struct ImagePatch {
    /// The CGImage (224x224 RGB)
    let image: CGImage
    /// Pre-computed normalized tensor for TensorType models
    let normalizedTensor: MLMultiArray?
}

/// A pair of patches for adjacency detection
struct PatchPair {
    let patch1: ImagePatch
    let patch2: ImagePatch
    let isAdjacent: Bool
    let sourceImageName: String
}

/// A merged patch for merged classifier adjacency detection
struct MergedPatchPair {
    /// The merged image (two patches side-by-side, resized to 224x224)
    let mergedPatch: ImagePatch
    let isAdjacent: Bool
    let sourceImageName: String
}

// MARK: - Generator

/// Generates patch pairs from images for benchmarking
class ImagePatchGenerator {
    private let imagesURL: URL
    private var imageURLs: [URL] = []

    init(imagesURL: URL) {
        self.imagesURL = imagesURL
    }

    /// Load all image URLs from the directory
    func loadImageList() throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(
            at: imagesURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )

        let imageExtensions = Set(["jpg", "jpeg", "png", "webp"])
        imageURLs = contents.filter { url in
            imageExtensions.contains(url.pathExtension.lowercased())
        }

        if imageURLs.isEmpty {
            throw PatchGeneratorError.noImagesFound
        }
    }

    var imageCount: Int {
        imageURLs.count
    }

    /// Generate a batch of patch pairs for benchmarking
    /// - Parameters:
    ///   - count: Number of pairs to generate
    ///   - includeNormalized: Whether to include pre-normalized tensors
    /// - Returns: Array of patch pairs
    func generatePairs(count: Int, includeNormalized: Bool = true) throws -> [PatchPair] {
        guard !imageURLs.isEmpty else {
            throw PatchGeneratorError.noImagesFound
        }

        var pairs: [PatchPair] = []
        pairs.reserveCapacity(count)

        for i in 0 ..< count {
            let url = imageURLs[i % imageURLs.count]
            let isPositive = i % 2 == 0 // Alternate between positive and negative

            let pair = try generatePair(
                primaryURL: url,
                isPositive: isPositive,
                includeNormalized: includeNormalized
            )
            pairs.append(pair)
        }

        return pairs
    }

    /// Generate a batch of merged patch pairs for merged classifier benchmarking
    /// - Parameters:
    ///   - count: Number of pairs to generate
    ///   - includeNormalized: Whether to include pre-normalized tensors
    /// - Returns: Array of merged patch pairs
    func generateMergedPairs(count: Int, includeNormalized: Bool = true) throws -> [MergedPatchPair] {
        guard !imageURLs.isEmpty else {
            throw PatchGeneratorError.noImagesFound
        }

        var pairs: [MergedPatchPair] = []
        pairs.reserveCapacity(count)

        for i in 0 ..< count {
            let url = imageURLs[i % imageURLs.count]
            let isPositive = i % 2 == 0

            let pair = try generateMergedPair(
                primaryURL: url,
                isPositive: isPositive,
                includeNormalized: includeNormalized
            )
            pairs.append(pair)
        }

        return pairs
    }

    /// Generate a single pair from an image
    private func generatePair(primaryURL: URL, isPositive: Bool, includeNormalized: Bool) throws -> PatchPair {
        // Load primary image
        guard let imageSource = CGImageSourceCreateWithURL(primaryURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
        else {
            throw PatchGeneratorError.failedToLoadImage(primaryURL.lastPathComponent)
        }

        let width = image.width
        let height = image.height
        let midX = width / 2

        // Extract left and right patches
        let leftRect = CGRect(x: 0, y: 0, width: midX, height: height)
        let rightRect = CGRect(x: midX, y: 0, width: width - midX, height: height)

        guard let leftCrop = image.cropping(to: leftRect),
              let rightCrop = image.cropping(to: rightRect)
        else {
            throw PatchGeneratorError.failedToCrop
        }

        // Crop to 224x224, maintaining edge alignment
        let leftPatch = try cropToPatchSize(leftCrop, alignRight: true)

        let patch2Image: CGImage
        let isAdjacent: Bool

        if isPositive {
            // Positive: use right half of same image
            patch2Image = try cropToPatchSize(rightCrop, alignRight: false)
            isAdjacent = true
        } else {
            // Negative: use left half of a different image
            let randomURL = imageURLs.randomElement()!
            guard let randomSource = CGImageSourceCreateWithURL(randomURL as CFURL, nil),
                  let randomImage = CGImageSourceCreateImageAtIndex(randomSource, 0, nil)
            else {
                throw PatchGeneratorError.failedToLoadImage(randomURL.lastPathComponent)
            }

            let randomMidX = randomImage.width / 2
            let randomLeftRect = CGRect(x: 0, y: 0, width: randomMidX, height: randomImage.height)
            guard let randomLeftCrop = randomImage.cropping(to: randomLeftRect) else {
                throw PatchGeneratorError.failedToCrop
            }

            patch2Image = try cropToPatchSize(randomLeftCrop, alignRight: false)
            isAdjacent = false
        }

        // Create patches with optional normalized tensors
        let patch1 = try ImagePatch(
            image: leftPatch,
            normalizedTensor: includeNormalized ? createNormalizedTensor(from: leftPatch) : nil
        )
        let patch2 = try ImagePatch(
            image: patch2Image,
            normalizedTensor: includeNormalized ? createNormalizedTensor(from: patch2Image) : nil
        )

        return PatchPair(
            patch1: patch1,
            patch2: patch2,
            isAdjacent: isAdjacent,
            sourceImageName: primaryURL.lastPathComponent
        )
    }

    /// Generate a single merged pair from an image (for merged classifier)
    /// Concatenates two patches side-by-side and resizes to 224x224
    private func generateMergedPair(primaryURL: URL, isPositive: Bool, includeNormalized: Bool) throws -> MergedPatchPair {
        // Load primary image
        guard let imageSource = CGImageSourceCreateWithURL(primaryURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
        else {
            throw PatchGeneratorError.failedToLoadImage(primaryURL.lastPathComponent)
        }

        let width = image.width
        let height = image.height
        let midX = width / 2

        // Extract left and right patches
        let leftRect = CGRect(x: 0, y: 0, width: midX, height: height)
        let rightRect = CGRect(x: midX, y: 0, width: width - midX, height: height)

        guard let leftCrop = image.cropping(to: leftRect),
              let rightCrop = image.cropping(to: rightRect)
        else {
            throw PatchGeneratorError.failedToCrop
        }

        // Crop to 224x224, maintaining edge alignment
        let leftPatch = try cropToPatchSize(leftCrop, alignRight: true)

        let patch2Image: CGImage
        let isAdjacent: Bool

        if isPositive {
            patch2Image = try cropToPatchSize(rightCrop, alignRight: false)
            isAdjacent = true
        } else {
            let randomURL = imageURLs.randomElement()!
            guard let randomSource = CGImageSourceCreateWithURL(randomURL as CFURL, nil),
                  let randomImage = CGImageSourceCreateImageAtIndex(randomSource, 0, nil)
            else {
                throw PatchGeneratorError.failedToLoadImage(randomURL.lastPathComponent)
            }

            let randomMidX = randomImage.width / 2
            let randomLeftRect = CGRect(x: 0, y: 0, width: randomMidX, height: randomImage.height)
            guard let randomLeftCrop = randomImage.cropping(to: randomLeftRect) else {
                throw PatchGeneratorError.failedToCrop
            }

            patch2Image = try cropToPatchSize(randomLeftCrop, alignRight: false)
            isAdjacent = false
        }

        // Merge the two patches side-by-side and resize to 224x224
        let mergedImage = try mergeAndResize(left: leftPatch, right: patch2Image)

        let mergedPatch = try ImagePatch(
            image: mergedImage,
            normalizedTensor: includeNormalized ? createNormalizedTensor(from: mergedImage) : nil
        )

        return MergedPatchPair(
            mergedPatch: mergedPatch,
            isAdjacent: isAdjacent,
            sourceImageName: primaryURL.lastPathComponent
        )
    }

    /// Merge two patches side-by-side and resize to 224x224
    private func mergeAndResize(left: CGImage, right: CGImage) throws -> CGImage {
        // Draw left and right side-by-side
        let combinedWidth = left.width + right.width
        let combinedHeight = max(left.height, right.height)

        guard let ctx = CGContext(
            data: nil,
            width: combinedWidth,
            height: combinedHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PatchGeneratorError.failedToCreateContext
        }

        ctx.draw(left, in: CGRect(x: 0, y: 0, width: left.width, height: left.height))
        ctx.draw(right, in: CGRect(x: left.width, y: 0, width: right.width, height: right.height))

        guard let combined = ctx.makeImage() else {
            throw PatchGeneratorError.failedToCreateContext
        }

        // Resize the combined image to 224x224
        return try resizeImage(combined, to: CGSize(width: kPatchSize, height: kPatchSize))
    }

    /// Crop/resize image to 224x224
    private func cropToPatchSize(_ image: CGImage, alignRight: Bool) throws -> CGImage {
        let sourceWidth = image.width
        let sourceHeight = image.height

        // If source is larger than target, crop; otherwise resize
        var cropX = 0
        var cropY = 0
        var cropWidth = sourceWidth
        var cropHeight = sourceHeight

        if sourceWidth > kPatchSize {
            cropWidth = kPatchSize
            cropX = alignRight ? (sourceWidth - kPatchSize) : 0
        }

        let cropRect = CGRect(x: cropX, y: cropY, width: cropWidth, height: cropHeight)

        // Crop if needed
        let croppedImage: CGImage
        if cropWidth != sourceWidth || cropHeight != sourceHeight {
            guard let cropped = image.cropping(to: cropRect) else {
                throw PatchGeneratorError.failedToCrop
            }
            croppedImage = cropped
        } else {
            croppedImage = image
        }

        // Resize to exactly 224x224
        return try resizeImage(croppedImage, to: CGSize(width: kPatchSize, height: kPatchSize))
    }

    /// Resize image using Core Graphics
    private func resizeImage(_ image: CGImage, to size: CGSize) throws -> CGImage {
        let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )

        guard let ctx = context else {
            throw PatchGeneratorError.failedToCreateContext
        }

        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(origin: .zero, size: size))

        guard let resized = ctx.makeImage() else {
            throw PatchGeneratorError.failedToResize
        }

        return resized
    }

    /// Create normalized MLMultiArray for TensorType models
    private func createNormalizedTensor(from image: CGImage) throws -> MLMultiArray {
        // Create MLMultiArray with shape [1, 3, 224, 224]
        let shape: [NSNumber] = [1, 3, kPatchSize as NSNumber, kPatchSize as NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        // Extract pixel data
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PatchGeneratorError.failedToCreateContext
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Normalize and fill MLMultiArray
        // Layout: [batch, channel, height, width]
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let pixelIndex = (y * width + x) * bytesPerPixel

                // RGB channels
                for c in 0 ..< 3 {
                    let pixelValue = Float(pixelData[pixelIndex + c]) / 255.0
                    let normalized = (pixelValue - kMean[c]) / kStd[c]

                    // Index: batch * (C*H*W) + channel * (H*W) + y * W + x
                    let arrayIndex = c * (height * width) + y * width + x
                    ptr[arrayIndex] = normalized
                }
            }
        }

        return array
    }
}

// MARK: - Errors

enum PatchGeneratorError: LocalizedError {
    case noImagesFound
    case failedToLoadImage(String)
    case failedToCrop
    case failedToCreateContext
    case failedToResize

    var errorDescription: String? {
        switch self {
        case .noImagesFound:
            return "No images found in the specified directory"
        case let .failedToLoadImage(name):
            return "Failed to load image: \(name)"
        case .failedToCrop:
            return "Failed to crop image"
        case .failedToCreateContext:
            return "Failed to create graphics context"
        case .failedToResize:
            return "Failed to resize image"
        }
    }
}
