//
//  DirectoryPicker.swift
//  benchmark
//
//  SwiftUI wrapper for UIDocumentPickerViewController to select directories.
//

import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// SwiftUI view that presents a directory picker
struct DirectoryPicker: UIViewControllerRepresentable {
    @Binding var selectedURL: URL?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_: UIDocumentPickerViewController, context _: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: DirectoryPicker

        init(_ parent: DirectoryPicker) {
            self.parent = parent
        }

        func documentPicker(_: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }

            // Start accessing security-scoped resource
            guard url.startAccessingSecurityScopedResource() else {
                print("Failed to access security-scoped resource")
                return
            }

            parent.selectedURL = url
            parent.dismiss()
        }

        func documentPickerWasCancelled(_: UIDocumentPickerViewController) {
            parent.dismiss()
        }
    }
}

// MARK: - Directory Selector Button

/// A button that opens a directory picker and shows the selected path
struct DirectorySelectorButton: View {
    let title: String
    let icon: String
    @Binding var selectedURL: URL?
    @State private var showPicker = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                showPicker = true
            } label: {
                HStack {
                    Image(systemName: icon)
                        .font(.title2)
                        .frame(width: 32)

                    VStack(alignment: .leading, spacing: 4) {
                        Text(title)
                            .font(.headline)

                        if let url = selectedURL {
                            Text(url.lastPathComponent)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        } else {
                            Text("Tap to select...")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }

                    Spacer()

                    if selectedURL != nil {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Image(systemName: "chevron.right")
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(.plain)
        }
        .sheet(isPresented: $showPicker) {
            DirectoryPicker(selectedURL: $selectedURL)
        }
    }
}
