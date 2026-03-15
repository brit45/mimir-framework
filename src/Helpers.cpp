#include "Helpers.hpp"

DatasetItem::~DatasetItem() {
    // Libérer la RAM trackée à la destruction
    unload();
}
