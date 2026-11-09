#include "test_utils.hpp"
#include "Helpers.hpp"

#include <chrono>
#include <fstream>
#include <set>
#include <stdexcept>

namespace {
struct TemporaryTree {
    fs::path root = fs::temp_directory_path() / ("mimir_dataset_symlinks_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryTree() { fs::create_directories(root); }
    ~TemporaryTree() { std::error_code ec; fs::remove_all(root, ec); }
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << text;
    require(bool(output), "Cannot write synthetic fixture: " + path.string());
}

void writeImage(const fs::path& path, unsigned char red = 180) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << "P6\n2 1\n255\n";
    for (int pixel = 0; pixel < 2; ++pixel) {
        const char rgb[] = {char(red), char(90), char(30)};
        output.write(rgb, sizeof(rgb));
    }
    require(bool(output), "Cannot write synthetic image: " + path.string());
}

DatasetItem& itemNamed(std::vector<DatasetItem>& items, const std::string& name) {
    for (auto& item : items) if (item.name == name) return item;
    throw std::runtime_error("Missing item: " + name);
}

std::set<std::string> itemNames(const std::vector<DatasetItem>& items) {
    std::set<std::string> result;
    for (const auto& item : items) result.insert(item.name);
    require(result.size() == items.size(), "Dataset has duplicate lexical item names");
    return result;
}

std::string cacheHash(const fs::path& path) {
    std::ifstream input(path);
    json document;
    input >> document;
    return document.at("dataset_hash").get<std::string>();
}
}

int main() {
    try {
        TemporaryTree fixture;
        const auto external = fixture.root / "external";
        writeImage(external / "source.ppm");
        std::error_code symlink_error;
        fs::create_symlink(external / "source.ppm", fixture.root / "symlink_probe", symlink_error);
        if (symlink_error) {
#ifdef _WIN32
            std::cout << "SKIP: symbolic links are unavailable: " << symlink_error.message() << '\n';
            return 77;
#else
            throw std::runtime_error("Cannot create symbolic-link fixture: " + symlink_error.message());
#endif
        }

        // Links retain their local names so local captions pair with external images.
        const auto files = fixture.root / "files";
        fs::create_directories(files);
        fs::create_symlink(fs::path("../external/source.ppm"), files / "relative.ppm");
        fs::create_symlink(external / "source.ppm", files / "absolute.ppm");
        fs::create_symlink(fs::path("source.ppm"), external / "image_chain.ppm");
        fs::create_symlink(fs::path("../external/image_chain.ppm"), files / "chain.ppm");
        writeText(files / "relative.txt", "Caption relative.");
        writeText(files / "absolute.txt", "Caption absolute.");
        writeText(external / "caption.txt", "Caption chain.");
        fs::create_symlink(fs::path("caption.txt"), external / "caption_chain.txt");
        fs::create_symlink(external / "caption_chain.txt", files / "chain.txt");
        auto linked_files = loadDataset(files.string(), 2, 1, 2);
        require(itemNames(linked_files) == std::set<std::string>{"relative", "absolute", "chain"},
                "Linked files were not paired with their lexical sidecars");
        for (auto& item : linked_files) {
            require(fs::path(item.image_file).parent_path() == files, "Image alias path was canonicalized");
            require(item.loadImageRGB(2, 1) && item.img.size() == 6 && item.img[0] == 180,
                    "Linked image failed lazy decoding");
            require(item.loadText() && item.text && !item.text->empty(), "Linked caption failed lazy loading");
            item.unload();
        }

        // Each alias is a separate logical namespace, even when targets are shared.
        const auto source = fixture.root / "directory_source";
        writeImage(source / "photo.ppm");
        writeText(source / "photo.txt", "Photo caption.");
        writeImage(source / "nested/scene.ppm");
        writeText(source / "nested/scene.txt", "Nested caption.");
        const auto directories = fixture.root / "directories";
        fs::create_directories(directories);
        fs::create_directory_symlink(fs::path("../directory_source"), directories / "relative");
        fs::create_directory_symlink(source, directories / "absolute");
        fs::create_directory_symlink(fs::path("relative"), directories / "chain");
        writeImage(directories / "local/photo.ppm");
        writeText(directories / "local/photo.txt", "Local caption.");

        // File cycles, recursive directory links and missing targets must not hang or abort scans.
        const auto clean_hash = datasetSHA256Hash(directories.string());
        fs::create_directory_symlink(directories, source / "back_to_dataset");
        fs::create_directory_symlink(fs::path("."), source / "self");
        fs::create_directory_symlink(fs::path(".."), source / "nested/up");
        fs::create_directory_symlink(fs::path("missing"), source / "missing_directory");
        fs::create_symlink(fs::path("missing.ppm"), source / "broken.ppm");
        fs::create_symlink(fs::path("cycle_b.txt"), source / "cycle_a.txt");
        fs::create_symlink(fs::path("cycle_a.txt"), source / "cycle_b.txt");
        auto linked_directories = loadDataset(directories.string(), 2, 1, 2);
        const std::set<std::string> directory_names = {
            "relative/photo", "relative/nested/scene", "absolute/photo", "absolute/nested/scene",
            "chain/photo", "chain/nested/scene", "local/photo"
        };
        require(itemNames(linked_directories) == directory_names,
                "Directory links lost nested files, collapsed aliases, or followed a cycle");
        require(datasetSHA256Hash(directories.string()) == clean_hash,
                "Unusable cyclic/broken links changed the usable-file dataset hash");
        require(detectModalities(directories.string()) == (MOD_IMAGE | MOD_TEXT),
                "Modality detection did not follow the same linked-directory traversal");
        const auto seed = simpleDatasetHash(directories.string());
        require(simpleDatasetHash(directories.string()) == seed, "Linked dataset seed is unstable");
        require(itemNamed(linked_directories, "chain/nested/scene").loadText(), "Nested linked caption failed");
        require(*itemNamed(linked_directories, "chain/nested/scene").text == "Nested caption.",
                "Nested linked caption has wrong content");
        itemNamed(linked_directories, "chain/nested/scene").unload();
        const auto root_alias = fixture.root / "root_alias";
        fs::create_directory_symlink(fs::path("directories"), root_alias);
        auto aliased_root = loadDataset(root_alias.string(), 2, 1, 2);
        require(itemNames(aliased_root) == directory_names, "Dataset root symlink was not traversed");
        require(fs::path(itemNamed(aliased_root, "absolute/photo").image_file) == root_alias / "absolute/photo.ppm",
                "Root alias was lost from indexed file paths");

        // Retargeting equal-size/equal-mtime files must invalidate an existing cache.
        const auto replacements = fixture.root / "replacements";
        writeImage(replacements / "a.ppm", 180);
        writeImage(replacements / "b.ppm", 220);
        fs::last_write_time(replacements / "b.ppm", fs::last_write_time(replacements / "a.ppm"));
        require(fs::file_size(replacements / "a.ppm") == fs::file_size(replacements / "b.ppm"),
                "Retarget fixture sizes differ");
        const auto cached_root = fixture.root / "cached_files";
        writeText(cached_root / "image.txt", "Cached caption.");
        fs::create_symlink(replacements / "a.ppm", cached_root / "image.ppm");
        const auto cache = fixture.root / "cache/index.json";
        fs::create_directories(cache.parent_path());
        auto before = loadDatasetCached(cached_root.string(), 2, 1, 2, cache.string());
        require(before.size() == 1, "Initial linked cache has wrong size");
        const auto file_hash = datasetSHA256Hash(cached_root.string());
        const auto stored_hash = cacheHash(cache);
        fs::remove(cached_root / "image.ppm");
        fs::create_symlink(replacements / "b.ppm", cached_root / "image.ppm");
        require(datasetSHA256Hash(cached_root.string()) != file_hash, "File-link retarget did not invalidate hash");
        auto after = loadDatasetCached(cached_root.string(), 2, 1, 2, cache.string());
        require(after.size() == 1 && cacheHash(cache) != stored_hash, "File-link retarget reused stale cache");
        require(after[0].loadImageRGB(2, 1) && after[0].img[0] == 220, "Retargeted image was not decoded");
        after[0].unload();

        // Linked-directory targets and new descendants participate in cache invalidation.
        const auto target_a = fixture.root / "directory_a";
        const auto target_b = fixture.root / "directory_b";
        writeText(target_a / "sample.txt", "AAAA");
        writeText(target_b / "sample.txt", "BBBB");
        fs::last_write_time(target_b / "sample.txt", fs::last_write_time(target_a / "sample.txt"));
        const auto cached_directories = fixture.root / "cached_directories";
        fs::create_directories(cached_directories);
        fs::create_directory_symlink(target_a, cached_directories / "group");
        const auto directory_cache = fixture.root / "cache/directories.json";
        auto original = loadDatasetCached(cached_directories.string(), 2, 1, 1, directory_cache.string());
        require(original.size() == 1, "Linked directory was absent from initial cache");
        const auto directory_hash = datasetSHA256Hash(cached_directories.string());
        const auto stored_directory_hash = cacheHash(directory_cache);
        fs::remove(cached_directories / "group");
        fs::create_directory_symlink(target_b, cached_directories / "group");
        require(datasetSHA256Hash(cached_directories.string()) != directory_hash,
                "Directory retarget with matching child metadata did not invalidate hash");
        auto retargeted = loadDatasetCached(cached_directories.string(), 2, 1, 1, directory_cache.string());
        require(cacheHash(directory_cache) != stored_directory_hash && retargeted.size() == 1,
                "Directory retarget reused stale cache");
        require(retargeted[0].loadText() && *retargeted[0].text == "BBBB", "Wrong retargeted text content");
        retargeted[0].unload();
        writeText(target_b / "nested/new.txt", "New descendant.");
        auto expanded = loadDatasetCached(cached_directories.string(), 2, 1, 1, directory_cache.string());
        require(itemNames(expanded) == std::set<std::string>{"group/sample", "group/nested/new"},
                "Files added through linked directories were not reflected in the cache");

        // The cache may be addressed through the physical target instead of its dataset alias.
        const auto cache_target = fixture.root / "internal_cache_target";
        writeText(cache_target / "sample.txt", "Caption beside cache.");
        const auto cache_root_alias = fixture.root / "internal_cache_alias";
        fs::create_directory_symlink(cache_target, cache_root_alias);
        const auto internal_cache = cache_target / "custom.json";
        auto cached_alias = loadDatasetCached(cache_root_alias.string(), 2, 1, 1, internal_cache.string());
        require(cached_alias.size() == 1, "Cache file was indexed through its directory alias");
        const auto internal_hash = cacheHash(internal_cache);
        const auto excluded_hash = datasetSHA256Hash(cache_root_alias.string(), internal_cache.string());
        auto alias_cache_hit = loadDatasetCached(cache_root_alias.string(), 2, 1, 1, internal_cache.string());
        require(alias_cache_hit.size() == 1 && cacheHash(internal_cache) == internal_hash,
                "Cache inside a linked target invalidated itself");
        require(datasetSHA256Hash(cache_root_alias.string(), internal_cache.string()) == excluded_hash,
                "Physical cache path was not excluded through its alias");

        // COCO's preflight probes and regular sidecar indexing tolerate unusable links too.
        const auto broken_preflight = fixture.root / "broken_preflight";
        writeImage(broken_preflight / "photo.ppm");
        writeText(broken_preflight / "photo.txt", "Fallback text.");
        fs::create_symlink(fs::path("photo.json"), broken_preflight / "photo.json");
        fs::create_directory_symlink(fs::path("annotations"), broken_preflight / "annotations");
        auto ignored_annotation_dir = loadDataset(broken_preflight.string(), 2, 1, 2);
        require(ignored_annotation_dir.size() == 1, "Cyclic annotations directory aborted indexing");
        fs::remove(broken_preflight / "annotations");
        fs::create_directories(broken_preflight / "annotations");
        fs::create_symlink(fs::path("captions_train2017.json"),
                           broken_preflight / "annotations/captions_train2017.json");
        auto ignored_captions = loadDataset(broken_preflight.string(), 2, 1, 2);
        require(ignored_captions.size() == 1 && ignored_captions[0].annotation_file.empty(),
                "Cyclic COCO captions or image annotation was treated as usable");
        require(ignored_captions[0].loadText() && *ignored_captions[0].text == "Fallback text.",
                "Unusable annotation link suppressed a valid local text sidecar");
        ignored_captions[0].unload();

#ifndef _WIN32
        // Follow a symlink before interpreting '..', as the operating system does for I/O.
        const auto posix_target = fixture.root / "posix_target";
        fs::create_directories(posix_target / "deep");
        writeText(posix_target / "dataset/sample.txt", "Resolved before dotdot.");
        const auto posix_access = fixture.root / "posix_access";
        fs::create_directories(posix_access);
        fs::create_directory_symlink(posix_target / "deep", posix_access / "link");
        const auto path_with_dotdot = posix_access / "link/../dataset";
        require(!fs::exists(posix_access / "dataset"), "Dotdot fixture has a misleading lexical fallback");
        auto through_dotdot = loadDataset(path_with_dotdot.string(), 2, 1, 1);
        require(through_dotdot.size() == 1 && through_dotdot[0].name == "sample" &&
                through_dotdot[0].loadText() && *through_dotdot[0].text == "Resolved before dotdot.",
                "Symlink/../dataset was normalized before following its target");
        through_dotdot[0].unload();
        require(!datasetSHA256Hash(path_with_dotdot.string()).empty(), "Dotdot path failed hashing");
#endif

        std::cout << "DATASET_SYMLINKS_OK files directories aliases roots cycles lazy-load cache-retarget\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Dataset symlink test failed: " << error.what() << '\n';
        return 1;
    }
}
