#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace Mimir::DatasetTraversal {
namespace fs = std::filesystem;

inline void reportSkipped(const fs::path& path, const std::string& reason) {
    std::cerr << "[dataset] Chemin ignoré: " << path.string() << " (" << reason << ")\n";
}

// Resolve identity only. Keep the original access path for I/O and pairing:
// normalizing link/../ before following the link would change its meaning.
inline fs::path cacheIdentity(const fs::path& path) {
    if (path.empty()) return {};
    std::error_code error;
    auto resolved = fs::weakly_canonical(path, error);
    return error ? fs::path() : resolved;
}

// A false visitor result stops the whole scan. The two paths deliberately have
// different roles: entry.path() is the local alias, resolved is its target.
template <class Visitor>
void forEachRegularFile(const fs::path& root, Visitor&& visitor) {
    std::error_code error;
    const auto resolved_root = fs::canonical(root, error);
    if (error) throw fs::filesystem_error("Cannot resolve dataset directory", root, error);
    if (!fs::is_directory(resolved_root, error)) {
        if (!error) error = std::make_error_code(std::errc::not_a_directory);
        throw fs::filesystem_error("Cannot open dataset directory", root, error);
    }

    struct Frame {
        fs::directory_iterator iterator;
        fs::path resolved;
    };
    fs::directory_iterator first(root, error);
    if (error) throw fs::filesystem_error("Cannot read dataset directory", root, error);
    std::vector<Frame> frames;
    frames.push_back({std::move(first), resolved_root});
    const fs::directory_iterator end;

    while (!frames.empty()) {
        auto& frame = frames.back();
        if (frame.iterator == end) {
            frames.pop_back();
            continue;
        }
        const auto entry = *frame.iterator;
        const auto parent = frame.resolved;
        frame.iterator.increment(error);
        if (error) {
            reportSkipped(parent, error.message());
            frame.iterator = end;
        }

        const auto status = entry.status(error); // follows file and directory links
        if (error || !fs::exists(status)) {
            reportSkipped(entry.path(), error ? error.message() : "cible absente");
            continue;
        }
        if (!fs::is_directory(status) && !fs::is_regular_file(status)) continue;

        const bool is_link = entry.is_symlink(error);
        if (error) {
            reportSkipped(entry.path(), error.message());
            continue;
        }
        fs::path resolved = parent / entry.path().filename();
        if (is_link) {
            resolved = fs::canonical(entry.path(), error);
            if (error) {
                reportSkipped(entry.path(), error.message());
                continue;
            }
        }
        if (fs::is_regular_file(status)) {
            if (!visitor(entry, resolved)) return;
            continue;
        }

        // Only ancestors are excluded. Separate aliases of the same directory
        // must retain their own local sidecars and dataset keys.
        const bool cycle = std::any_of(frames.begin(), frames.end(), [&](const Frame& ancestor) {
            if (ancestor.resolved == resolved) return true;
            if (!is_link) return false;
            std::error_code equivalent_error;
            return fs::equivalent(ancestor.resolved, resolved, equivalent_error);
        });
        if (cycle) {
            reportSkipped(entry.path(), "boucle de liens vers un dossier parent");
            continue;
        }
        fs::directory_iterator child(entry.path(), error);
        if (error) {
            reportSkipped(entry.path(), error.message());
            continue;
        }
        frames.push_back({std::move(child), std::move(resolved)});
    }
}
} // namespace Mimir::DatasetTraversal
