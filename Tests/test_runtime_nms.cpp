#include "test_utils.hpp"

#include "LayerTypes.hpp"
#include "Layers.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include <limits>
#include <vector>

namespace {

bool run_nms(const std::vector<float>& boxes,
             const std::vector<float>& scores,
             const std::vector<float>* classes,
             const Layer& layer,
             std::vector<float>* kept) {
    std::vector<const std::vector<float>*> inputs = {&boxes, &scores};
    if (classes != nullptr) inputs.push_back(classes);
    std::vector<std::vector<float>> outputs;
    const bool ok =
        RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, false);
    if (ok && kept != nullptr) {
        *kept = outputs.empty() ? std::vector<float>{} : outputs[0];
    }
    return ok;
}

}  // namespace

int main() {
    TASSERT_TRUE(LayerRegistry::string_to_type("NMS") == LayerType::NMS);
    TASSERT_TRUE(
        LayerRegistry::string_to_type("NonMaxSuppression") == LayerType::NMS);
    TASSERT_TRUE(LayerRegistry::type_to_string(LayerType::NMS) == "NMS");
    TASSERT_TRUE(
        RuntimeLayerDispatch::cpu_supports_forward_layer_type(LayerType::NMS));
    TASSERT_TRUE(
        !RuntimeLayerDispatch::cpu_supports_backward_layer_type(LayerType::NMS));

    const std::vector<float> boxes = {
        0.0f, 0.0f, 10.0f, 10.0f,
        1.0f, 1.0f, 9.0f, 9.0f,
        20.0f, 20.0f, 30.0f, 30.0f,
    };
    const std::vector<float> scores = {0.9f, 0.8f, 0.7f};

    Layer layer("nms", "NMS", 0);
    layer.nms_iou_threshold = 0.5f;
    layer.nms_score_threshold = 0.0f;
    layer.nms_class_agnostic = true;

    // The second box overlaps the first one and has a lower score.
    {
        std::vector<float> kept;
        TASSERT_TRUE(run_nms(boxes, scores, nullptr, layer, &kept));
        TASSERT_TRUE(kept.size() == 2);
        TASSERT_NEAR(kept[0], 0.0f, 0.0f);
        TASSERT_NEAR(kept[1], 2.0f, 0.0f);
    }

    // Class-aware mode keeps overlapping boxes from different classes.
    {
        const std::vector<float> classes = {0.0f, 1.0f, 0.0f};
        layer.nms_class_agnostic = false;
        std::vector<float> kept;
        TASSERT_TRUE(run_nms(boxes, scores, &classes, layer, &kept));
        TASSERT_TRUE(kept.size() == 3);
        TASSERT_NEAR(kept[0], 0.0f, 0.0f);
        TASSERT_NEAR(kept[1], 1.0f, 0.0f);
        TASSERT_NEAR(kept[2], 2.0f, 0.0f);
    }

    // Score filtering and maximum output count are applied after sorting.
    {
        layer.nms_class_agnostic = true;
        layer.nms_score_threshold = 0.75f;
        layer.nms_max_detections = 1;
        std::vector<float> kept;
        TASSERT_TRUE(run_nms(boxes, scores, nullptr, layer, &kept));
        TASSERT_TRUE(kept.size() == 1);
        TASSERT_NEAR(kept[0], 0.0f, 0.0f);
    }

    // Equal scores are deterministic: lower original index wins first.
    {
        const std::vector<float> tied_scores = {0.5f, 0.5f, 0.5f};
        layer.nms_score_threshold = 0.0f;
        layer.nms_max_detections = 0;
        std::vector<float> kept;
        TASSERT_TRUE(run_nms(boxes, tied_scores, nullptr, layer, &kept));
        TASSERT_TRUE(kept.size() == 2);
        TASSERT_NEAR(kept[0], 0.0f, 0.0f);
        TASSERT_NEAR(kept[1], 2.0f, 0.0f);
    }

    // Non-finite candidates are ignored.
    {
        const std::vector<float> bad_scores = {
            std::numeric_limits<float>::quiet_NaN(), 0.8f, 0.7f};
        std::vector<float> kept;
        TASSERT_TRUE(run_nms(boxes, bad_scores, nullptr, layer, &kept));
        TASSERT_TRUE(kept.size() == 2);
        TASSERT_NEAR(kept[0], 1.0f, 0.0f);
        TASSERT_NEAR(kept[1], 2.0f, 0.0f);
    }

    // Malformed inputs and invalid thresholds are rejected.
    {
        std::vector<float> kept;
        const std::vector<float> malformed_boxes = {0.0f, 0.0f, 1.0f};
        TASSERT_TRUE(!run_nms(malformed_boxes, scores, nullptr, layer, &kept));

        Layer invalid = layer;
        invalid.nms_iou_threshold = 1.5f;
        TASSERT_TRUE(!run_nms(boxes, scores, nullptr, invalid, &kept));
    }

    return 0;
}
