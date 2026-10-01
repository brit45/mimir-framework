#include "test_utils.hpp"
#include "Helpers.hpp"
#include <chrono>
#include <fstream>

int main() {
    const auto path = fs::temp_directory_path() / ("mimir_rgb_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ppm");
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove(path); } } cleanup{path};
    {
        std::ofstream out(path, std::ios::binary);
        out << "P6\n2 1\n255\n";
        const unsigned char pixels[] = {255, 0, 0, 0, 255, 0};
        out.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    }
    auto& memory = DatasetMemoryManager::instance();
    const auto baseline = memory.getCurrentRAM();
    DatasetItem item;
    item.image_file = path.string();
    item.w = 2;
    item.h = 1;
    TASSERT_TRUE(item.estimateRAMNeeded() == 6);
    TASSERT_TRUE(item.loadImage(2, 1));
    TASSERT_TRUE(item.img_c == 3 && item.img.size() == 6);
    TASSERT_TRUE(item.img[0] > 250 && item.img[1] < 5 && item.img[2] < 5);
    TASSERT_TRUE(item.img[3] < 5 && item.img[4] > 250 && item.img[5] < 5);
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 6);
    TASSERT_TRUE(item.loadImage(2, 1));
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 6);
    TASSERT_TRUE(item.loadImage(4, 2));
    TASSERT_TRUE(item.w == 4 && item.h == 2 && item.img_c == 3 && item.img.size() == 24);
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 24);
    TASSERT_TRUE(!item.loadImage(0, 2));
    item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    TASSERT_TRUE(item.loadImageRGB(2, 1));
    TASSERT_TRUE(item.img_c == 3 && item.img.size() == 6);
    item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    // Le préchargement par lots doit suivre le même contrat RGB.
    std::vector<DatasetItem> items(1);
    items[0].image_file = path.string();
    items[0].w = 2;
    items[0].h = 1;
    DatasetManager manager;
    TASSERT_TRUE(manager.ensureLoaded(items, {0}, 2, 1));
    TASSERT_TRUE(items[0].img_c == 3 && items[0].img.size() == 6);
    TASSERT_TRUE(items[0].img[0] > 250 && items[0].img[1] < 5);
    items[0].unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    return 0;
}
