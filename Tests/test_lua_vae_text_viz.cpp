#include "test_utils.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "AsyncMonitor.hpp"
#include <filesystem>
#include <fstream>

struct AsyncMonitorPreviewTestAccess {
    static void attach(AsyncMonitor& monitor) {
        // Exercise the real training -> monitor -> UI mailbox without a display server.
        std::atomic_store(&monitor.viz_, std::make_shared<Visualizer>(
            json{{"visualization", {{"enabled", false}}}}));
    }
    static int verify(AsyncMonitor& monitor) {
        const auto snapshot = monitor.takeVizUpdate();
        TASSERT_TRUE(snapshot.layer_blocks_dirty_ && snapshot.pending_layer_blocks_);
        TASSERT_TRUE(!snapshot.pending_layer_blocks_->empty());
        TASSERT_TRUE(snapshot.context_dirty_ && snapshot.pending_dataset_text_);
        TASSERT_TRUE(snapshot.pending_dataset_text_->raw == "a blue sky above a green tree");
        TASSERT_TRUE(snapshot.pending_dataset_text_->tokens.find("count=") != std::string::npos);
        bool reconstructed = false, latent = false;
        for (const auto& frame : *snapshot.pending_layer_blocks_) {
            TASSERT_TRUE(!frame.pixels.empty() && frame.w > 0 && frame.h > 0);
            reconstructed |= frame.label == "Texte / entrée et reconstruction";
            latent |= frame.label.find("moyenne utilisée par Lumen") != std::string::npos;
        }
        TASSERT_TRUE(reconstructed && latent);
        return 0;
    }
};

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "mimir_vae_text_viz_bridge";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    { std::ofstream text(directory/"context.txt"); text << "a blue sky above a green tree"; }
    LuaScripting lua;
    auto& context = LuaContext::getInstance();
    context.resetRuntimeState();
    lua.setString("TEST_DATASET", directory.string());
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Tokenizer.create(128))
        assert(Mimir.Tokenizer.ensure_vocab_from_text("a blue sky above a green tree"))
        local cfg = Mimir.Architectures.default_config("vae_text")
        for k,v in pairs({seq_len=8,d_model=8,num_layers=1,num_heads=2,mlp_hidden=16,
            latent_tokens=2,proj_dim=8,log_every=100,viz_every_steps=100,
            viz_taps_max_frames=64,autosave_every_epochs=0,validate_items=0,
            stochastic_latent=false}) do cfg[k]=v end
        cfg.vocab_size = Mimir.Tokenizer.vocab_size()
        assert(Mimir.Model.create("vae_text",cfg))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights("xavier",42))
        assert(Mimir.Dataset.load(TEST_DATASET))
    )"));
    context.asyncMonitor = std::make_shared<AsyncMonitor>();
    context.asyncMonitor->configureMetricsCsv((directory/"metrics.csv").string(), false);
    // No Viz: the same training path still succeeds.
    TASSERT_TRUE(lua.executeScript("assert(Mimir.Model.train(1,0.0001))"));
    AsyncMonitorPreviewTestAccess::attach(*context.asyncMonitor);
    // With Viz attached later, one step must publish immediately, despite cadence=100.
    TASSERT_TRUE(lua.executeScript("assert(Mimir.Model.train(1,0.0001))"));
    TASSERT_TRUE(AsyncMonitorPreviewTestAccess::verify(*context.asyncMonitor)==0);
    context.currentModel->modelConfig["viz_taps_force_inference"] = true;
    context.modelConfig["viz_taps_force_inference"] = true;
    TASSERT_TRUE(lua.executeScript("assert(Mimir.Model.train(1,0.0001))"));
    TASSERT_TRUE(AsyncMonitorPreviewTestAccess::verify(*context.asyncMonitor)==0);
    context.resetRuntimeState();
    std::filesystem::remove_all(directory);
    std::cout << "VAE_TEXT_VIZ_BRIDGE_OK\n";
    return 0;
}
