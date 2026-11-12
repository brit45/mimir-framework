#include "test_utils.hpp"
#include "Visualizer.hpp"
#include "viz/AnnotationFeatureOverlay.hpp"
#include "viz/FeatureProbe.hpp"
#include "viz/LumenOutputRole.hpp"
#include "viz/ModelGraphLanes.hpp"
#include "viz/CommandCardLayout.hpp"
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
namespace fs = std::filesystem;
#undef TASSERT_TRUE
#undef TASSERT_NEAR
#define TASSERT_TRUE(x) do { if (!(x)) throw std::runtime_error("FAIL: " #x); } while (0)
#define TASSERT_NEAR(a,b,e) TASSERT_TRUE(std::abs((a)-(b)) <= (e))

#ifdef ENABLE_VIZ
class EventWindow : public VizWindow {
    vizgfx::RenderTexture canvas_{{1400,1800}};
public:
    std::deque<vizgfx::Event> events;
    bool open=true;
    bool isOpen() const override {return open;}
    void close() override {open=false;}
    bool setActive(bool=true) override {return true;}
    void setIcon(const vizgfx::Image&) override {}
    void setFramerateLimit(unsigned) override {}
    void setSize(vizgfx::Vector2u size) override {(void)canvas_.resize(size);}
    vizgfx::Vector2u getSize() const override {return canvas_.getSize();}
    void setView(const vizgfx::View& v) override {canvas_.setView(v);}
    const vizgfx::View& getView() const override {return canvas_.getView();}
    vizgfx::Vector2f mapPixelToCoords(vizgfx::Vector2i p) const override {return {float(p.x),float(p.y)};}
    void setMouseCursor(const vizgfx::Cursor&,int=0) override {}
    bool isKeyPressed(vizgfx::Keyboard::Key) const override {return false;}
    std::optional<vizgfx::Event> pollEvent() override {
        if(events.empty()) return {};
        auto e=events.front();events.pop_front();
        if(const auto* resized=e.getIf<vizgfx::Event::Resized>()) (void)canvas_.resize(resized->size);
        return e;
    }
    void clear(const vizgfx::Color& c=vizgfx::Color::Black) override {canvas_.clear(c);}
    void display() override {canvas_.display();}
    vizgfx::Image captureImage() const override {return canvas_.getTexture().copyToImage();}
protected:
    vizgfx::RenderTarget& renderTarget() override {return canvas_;}
};
struct VisualizerEventTestAccess {
    static void runPanelManager() {
        using P=Visualizer::PanelId;
        constexpr size_t count=static_cast<size_t>(P::Count);
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1000}}}});
        v.setLossLogEnabled(false);
        v.script_scene_["events"] = true;
        auto window=std::make_unique<EventWindow>();auto* events=window.get();events->setSize({1400,1000});
        v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        v.initDefaultPanelsIfNeeded();
        auto draw=[&] {
            v.syncUIView();events->clear(vizgfx::Color(12,14,18));
            if(v.panel_manager_visible_)v.renderPanelManagerOverlay();
            events->display();
        };
        auto mouse=[&](vizgfx::Mouse::Button button,vizgfx::Vector2i point) {
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{button,point});v.processEvents();
        };
        auto click=[&](const std::optional<vizgfx::FloatRect>& bounds) {
            TASSERT_TRUE(bounds);const auto r=*bounds;
            mouse(vizgfx::Mouse::Button::Left,{int(r.position.x+r.size.x*.5f),int(r.position.y+r.size.y*.5f)});
        };
        auto key=[&](vizgfx::Keyboard::Key code) {
            events->events.emplace_back(vizgfx::Event::KeyPressed{code});v.processEvents();
        };
        auto open=[&] {
            mouse(vizgfx::Mouse::Button::Right,{1390,990});
            TASSERT_TRUE(v.panel_manager_visible_);draw();
        };
        auto flags=[&] {
            TASSERT_TRUE(v.isPanelVisible(P::Generated)==v.show_generated_images);
            TASSERT_TRUE(v.isPanelVisible(P::Training)==v.show_training_progress);
            TASSERT_TRUE(v.isPanelVisible(P::Graph)==v.show_loss_graph);
        };
        for(size_t index=0;index<count;++index) {
            v.setPanelVisible(static_cast<P>(index),true);
            auto& panel=v.panels_[index];
            panel.pos={10.f+320.f*float(index%4),10.f+300.f*float(index/4)};
            panel.size={300.f,250.f};
        }
        const auto original_panels=v.panels_;
        // Exercise every real close cross, including the last remaining panel.
        for(size_t index=0;index<count;++index) {
            const auto r=v.panelCloseButtonRect(static_cast<P>(index));
            mouse(vizgfx::Mouse::Button::Left,{int(r.position.x+r.size.x*.5f),int(r.position.y+r.size.y*.5f)});
            TASSERT_TRUE(!v.isPanelVisible(static_cast<P>(index)));flags();
        }
        TASSERT_TRUE(std::none_of(v.panels_.begin(),v.panels_.end(),[](const auto& panel){return panel.visible;}));
        v.pollScriptEvents();
        v.dragging_panel_=v.resizing_panel_=v.dragging_panel_scrollbar_=v.dragging_blocks_scrollbar_=true;
        v.save_chord_armed_=v.save_chord_consumed_=true;
        open();
        TASSERT_TRUE(!v.save_chord_armed_ && !v.save_chord_consumed_);
        TASSERT_TRUE(!v.dragging_panel_ && !v.resizing_panel_ &&
            !v.dragging_panel_scrollbar_ && !v.dragging_blocks_scrollbar_);
        TASSERT_TRUE(!v.training_paused_->load() && !v.stop_training_requested_.load());
        TASSERT_TRUE(v.pollScriptEvents().empty());
        for(size_t index=0;index<count;++index) {
            TASSERT_TRUE(v.panel_manager_row_rects_[index]);
            click(v.panel_manager_row_rects_[index]);draw();
            TASSERT_TRUE(v.isPanelVisible(static_cast<P>(index)) && v.panel_manager_visible_);flags();
            TASSERT_TRUE(v.panels_[index].pos==original_panels[index].pos &&
                v.panels_[index].size==original_panels[index].size);
        }
        TASSERT_TRUE(v.pollScriptEvents().empty());
        // The modal owns keyboard, wheel and pointer events. Model controls,
        // panel scrolling and dragging cannot react through its backdrop.
        const auto scroll=v.panel_scroll_y_;
        const float blocks_scroll=v.blocks_scroll_y_;
        const bool graph=v.show_loss_graph;
        key(vizgfx::Keyboard::Key::G);key(vizgfx::Keyboard::Key::D);key(vizgfx::Keyboard::Key::Z);
        TASSERT_TRUE(v.show_loss_graph==graph && !v.debug_json_dialog_.visible && !v.zoom_active_);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{25,25}});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{25,25}});
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,{25,25}});
        v.processEvents();
        TASSERT_TRUE(v.panel_scroll_y_==scroll && v.blocks_scroll_y_==blocks_scroll);
        TASSERT_TRUE(!v.dragging_panel_ && !v.resizing_panel_ && v.pollScriptEvents().empty());
        // Immediate toggles stay saved when the dialog is dismissed.
        click(v.panel_manager_row_rects_[size_t(P::Context)]);draw();
        TASSERT_TRUE(!v.isPanelVisible(P::Context));key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.panel_manager_visible_ && !v.isPanelVisible(P::Context));
        open();click(v.panel_manager_show_all_rect_);draw();
        TASSERT_TRUE(std::all_of(v.panels_.begin(),v.panels_.end(),[](const auto& panel){return panel.visible;}));flags();

        // Reopening restores reachable geometry without resetting valid peers.
        click(v.panel_manager_row_rects_[size_t(P::Output)]);draw();
        v.panels_[size_t(P::Output)].pos={10000.f,10000.f};
        click(v.panel_manager_row_rects_[size_t(P::Output)]);draw();
        const auto& output=v.panels_[size_t(P::Output)];
        TASSERT_TRUE(output.visible && output.pos.x>=0 && output.pos.y>=0 &&
            output.pos.x+output.size.x<=v.window_width && output.pos.y+output.size.y<=v.window_height);
        TASSERT_TRUE(v.panels_[size_t(P::Blocks)].pos==original_panels[size_t(P::Blocks)].pos);
        click(v.panel_manager_row_rects_[size_t(P::Generated)]);draw();flags();
        json saved;TASSERT_TRUE(v.loadUISettings(saved));
        TASSERT_TRUE(saved.contains("last") && saved["last"]["panels"].size()==count);
        for(size_t index=0;index<count;++index)
            TASSERT_TRUE(saved["last"]["panels"][index]["visible"]==v.panels_[index].visible);
        Visualizer restored(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1000}}}});
        TASSERT_TRUE(restored.applyUILayout(saved["last"]));
        for(size_t index=0;index<count;++index)
            TASSERT_TRUE(restored.isPanelVisible(static_cast<P>(index))==v.isPanelVisible(static_cast<P>(index)));
        TASSERT_TRUE(restored.show_generated_images==v.show_generated_images &&
            restored.show_training_progress==v.show_training_progress && restored.show_loss_graph==v.show_loss_graph);

        // A short window scrolls the section list; keyboard navigation can
        // still reach its final row and the two actions below the viewport.
        for(const auto dimensions:{vizgfx::Vector2u{360,260},vizgfx::Vector2u{640,480}}) {
            events->events.emplace_back(vizgfx::Event::Resized{dimensions});v.processEvents();draw();
            TASSERT_TRUE(v.panel_manager_body_rect_ && v.panel_manager_list_rect_ &&
                v.panel_manager_close_rect_ && v.panel_manager_show_all_rect_ && v.panel_manager_dismiss_rect_);
            const auto body=*v.panel_manager_body_rect_,list=*v.panel_manager_list_rect_;
            TASSERT_TRUE(body.position.x>=0 && body.position.y>=0 &&
                body.position.x+body.size.x<=dimensions.x && body.position.y+body.size.y<=dimensions.y);
            for(const auto* action:{&v.panel_manager_close_rect_,&v.panel_manager_show_all_rect_,&v.panel_manager_dismiss_rect_}) {
                const auto box=**action;
                TASSERT_TRUE(body.contains(box.position) &&
                    body.contains({box.position.x+box.size.x-.1f,box.position.y+box.size.y-.1f}));
                TASSERT_TRUE(!box.findIntersection(list));
            }
            if(dimensions.y==260)TASSERT_TRUE(v.panel_manager_scroll_max_>0);
            const auto old_list_scroll=v.panel_manager_scroll_y_;
            events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,
                {int(list.position.x+list.size.x*.5f),int(list.position.y+list.size.y*.5f)}});
            v.processEvents();draw();
            TASSERT_TRUE(v.panel_manager_scroll_y_>=old_list_scroll);
            v.panel_manager_selected_=0;
            for(size_t index=0;index<count-1;++index)key(vizgfx::Keyboard::Key::Down);
            draw();TASSERT_TRUE(v.panel_manager_selected_==count-1 && v.panel_manager_row_rects_[count-1]);
            const bool before=v.isPanelVisible(P::Output);
            key(vizgfx::Keyboard::Key::Space);draw();TASSERT_TRUE(v.isPanelVisible(P::Output)!=before);flags();
            key(vizgfx::Keyboard::Key::Tab);draw();TASSERT_TRUE(v.panel_manager_selected_==count);
            key(vizgfx::Keyboard::Key::Enter);draw();
            TASSERT_TRUE(std::all_of(v.panels_.begin(),v.panels_.end(),[](const auto& panel){return panel.visible;}));
        }
        events->events.emplace_back(vizgfx::Event::Resized{{1400,1000}});v.processEvents();draw();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_viz_panel_manager.png"));
        v.panel_manager_selected_=count+1;v.pollScriptEvents();
        events->events.emplace_back(vizgfx::Event::KeyPressed{vizgfx::Keyboard::Key::Space});
        events->events.emplace_back(vizgfx::Event::TextEntered{U' '});
        events->events.emplace_back(vizgfx::Event::KeyReleased{vizgfx::Keyboard::Key::Space});
        v.processEvents();TASSERT_TRUE(!v.panel_manager_visible_ && v.pollScriptEvents().empty());
        open();
        click(v.panel_manager_dismiss_rect_);TASSERT_TRUE(!v.panel_manager_visible_);
        open();click(v.panel_manager_close_rect_);TASSERT_TRUE(!v.panel_manager_visible_);
        open();v.pollScriptEvents();
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,{1,1}});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{1,1}});
        v.processEvents();TASSERT_TRUE(!v.panel_manager_visible_ && v.pollScriptEvents().empty());
        // Existing image-selection and other modal right-click gestures keep
        // priority over the main workspace's section manager.
        v.zoom_active_=true;mouse(vizgfx::Mouse::Button::Right,{20,20});
        TASSERT_TRUE(v.zoom_active_ && !v.panel_manager_visible_);v.zoom_active_=false;
        v.snapshot_dialog_visible_=true;mouse(vizgfx::Mouse::Button::Right,{20,20});
        TASSERT_TRUE(v.snapshot_dialog_visible_ && !v.panel_manager_visible_);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.snapshot_dialog_visible_);
        v.shutdown();
    }

    static void runExport3D() {
        using namespace mimir::viz;
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1000}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>();auto* events=window.get();events->setSize({1400,1000});
        v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        auto control=std::make_shared<LatentSpaceControl>();
        control->resetSession("export_fixture","branched_conv4d");v.setLatentSpaceControl(control);
        v.setLatentSpaceVisible(true);
        auto draw=[&] {
            v.syncUIView();events->clear(vizgfx::Color(12,14,18));
            v.renderLatentSpaceOverlay();events->display();
        };
        auto click=[&](const std::optional<vizgfx::FloatRect>& bounds) {
            TASSERT_TRUE(bounds);const auto box=*bounds;
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,
                {int(box.position.x+box.size.x/2),int(box.position.y+box.size.y/2)}});
            v.processEvents();
        };
        auto key=[&](vizgfx::Keyboard::Key code) {
            events->events.emplace_back(vizgfx::Event::KeyPressed{code});v.processEvents();
        };
        auto publish=[&](size_t step,const std::string& label,bool ready=true) {
            LatentSpaceProjection projection;
            projection.generation=control->generation.load();
            projection.correlation_mode=control->correlations_enabled.load();
            projection.neuron_mode=control->neurons_enabled.load();
            projection.live_weight_mode=control->live_weights_enabled.load();
            projection.model_name="export_fixture";projection.architecture="branched_conv4d";
            projection.optimizer_step=step;projection.d_model=64;projection.latent_dim=17;
            projection.feature_dimension=31;projection.effective_rank=2;
            projection.explained_variance_percent={{80.f,15.f,5.f}};
            const std::array<std::array<float,3>,3> positions{{{{-.75f,-.5f,.125f}},
                {{.75f,-.5f,-.125f}},{{0.f,.75f,.25f}}}};
            for(size_t index=0;index<positions.size();++index) {
                LatentSpacePoint point;point.position=positions[index];point.label=label+std::to_string(index);
                point.group=LatentVectorGroup::LayerActivation;point.layer=int(index);point.original_dimension=31;
                point.sampled_dimension=31;point.vector_norm=1.f;point.neuron_kind="canal";
                point.neuron_layer_units=64;point.neuron_unit_index=index;projection.points.push_back(point);
            }
            if(projection.correlation_mode) {
                projection.correlation_epoch=control->correlation_epoch.load();
                projection.correlation_ready=ready;projection.correlation_observations=ready?12:3;
                projection.correlation_training_observations=9;projection.correlation_inference_observations=3;
                projection.correlation_first_step=1;projection.correlation_last_step=step;
                projection.correlation_constant_features=2;projection.correlation_edge_candidates=3;
                projection.correlation_edges={{0,1,.875f},{1,2,-.75f},{0,2,.375f}};
                projection.correlation_candidate_edges=projection.correlation_edges;
                TASSERT_TRUE(control->beginCorrelationProjection(projection.generation,projection.correlation_epoch,
                    step,ready,projection.revision));
            } else {
                control->requestSnapshot();uint64_t generation=0,revision=0;bool neurons=false,live=false;
                TASSERT_TRUE(control->beginCapture(generation,false,control->generation.load(),
                    LatentSpaceControl::Clock::now(),&revision,&neurons,&live));
                LatentSpaceRawSnapshot raw;raw.generation=generation;raw.revision=revision;
                raw.neuron_mode=neurons;raw.live_weight_mode=live;
                TASSERT_TRUE(control->submitRawSnapshot(std::move(raw)));
                const auto captured=control->takeRawSnapshot();TASSERT_TRUE(captured);
                TASSERT_TRUE(control->beginProjection(generation,revision,neurons,live));
                projection.revision=revision;projection.live_weight_epoch=captured->live_weight_epoch;
                if(neurons||live) {
                    projection.neuron_total_units=192;projection.neuron_total_layers=3;
                    projection.neuron_unknown_layers=1;projection.neuron_sampling=true;
                    projection.neuron_links={{0,1,0.f,false,true}};
                }
                if(live) {
                    projection.live_weight_previous_step=step-1;projection.live_weight_baseline_available=true;
                    projection.live_weight_total_parameters=4000;projection.live_weight_sampled=true;
                    LiveWeightCoefficient coefficient;coefficient.key="conv4d:9";coefficient.label="conv4d/weight[9]";
                    coefficient.parameter_index=9;coefficient.value=-.25f;coefficient.delta=-1.234567890123e-40;
                    coefficient.has_delta=true;coefficient.mapped=true;
                    projection.live_weight_coefficients.push_back(coefficient);
                    projection.live_weight_links={{1,2,0}};
                }
            }
            TASSERT_TRUE(control->publishProjection(std::move(projection)));
        };
        auto readDocument=[&] {
            const fs::path path=v.last_latent_space_export_path_;
            TASSERT_TRUE(!path.empty() && path.filename()=="graph.html" && fs::is_regular_file(path));
            TASSERT_TRUE(fs::is_regular_file(path.parent_path()/"vue.png"));
            std::ifstream stream(path.parent_path()/"graph.json");json document;stream>>document;
            TASSERT_TRUE(stream.good()||stream.eof());return document;
        };
        std::set<std::string> paths;
        auto checkNewPath=[&] {
            TASSERT_TRUE(paths.insert(v.last_latent_space_export_path_).second);
            TASSERT_TRUE(!v.dragging_latent_space_ && v.show_latent_space_);
        };
        draw();TASSERT_TRUE(v.last_latent_space_export_rect_);
        TASSERT_TRUE(!v.saveLatentSpaceExport() && !v.latent_space_export_status_.empty());
        key(vizgfx::Keyboard::Key::X);
        TASSERT_TRUE(v.last_latent_space_export_path_.empty() && !fs::exists("exports"));

        publish(17,"displayed point ");
        v.latent_space_yaw_=.375f;v.latent_space_pitch_=-.125f;v.latent_space_zoom_=1.625f;
        v.latent_space_correlation_threshold_=.8125f;draw();
        const auto panel=v.latent_space_displayed_bounds_;
        const auto canvas=v.last_latent_space_canvas_rect_;
        const auto displayedImage=events->captureImage();
        std::vector<vizgfx::Vector2u> markerPixels;
        for(const auto& hit:v.last_latent_space_point_hits_)
            markerPixels.push_back({unsigned(hit.first.position.x+hit.first.size.x*.5f),
                unsigned(hit.first.position.y+hit.first.size.y*.5f)});
        // The model publishes its next step before the event loop reaches the
        // button. Export must preserve the projection and camera actually seen.
        publish(18,"not displayed point ");
        v.latent_space_yaw_=.75f;v.latent_space_correlation_threshold_=.9375f;
        click(v.last_latent_space_export_rect_);checkNewPath();
        auto document=readDocument();
        TASSERT_TRUE(document.at("mode")=="structural" && document.at("model").at("optimizer_step")==17);
        TASSERT_TRUE(document.at("points").at(0).at("label")=="displayed point 0");
        TASSERT_TRUE(document.at("points").at(0).at("position")==json::array({-.75,-.5,.125}));
        TASSERT_TRUE(document.at("view").at("yaw")==.375 && document.at("view").at("pitch")==-.125 &&
            document.at("view").at("zoom")==1.625 && document.at("view").at("correlation_threshold")==.8125);
        TASSERT_TRUE(document.at("view").at("correlation_show_all_candidates")==false);
        TASSERT_TRUE(document.at("view").at("width")==unsigned(canvas.size.x) &&
            document.at("view").at("height")==unsigned(canvas.size.y));
        vizgfx::Image png;
        TASSERT_TRUE(png.loadFromFile(fs::path(v.last_latent_space_export_path_).parent_path()/"vue.png"));
        TASSERT_TRUE(png.getSize().x==unsigned(panel.size.x) && png.getSize().y==unsigned(panel.size.y));
        TASSERT_TRUE(markerPixels.size()==3);
        for(const auto pixel:markerPixels) {
            const auto local=vizgfx::Vector2u{pixel.x-unsigned(panel.position.x),pixel.y-unsigned(panel.position.y)};
            TASSERT_TRUE(png.getPixel(local)==displayedImage.getPixel(pixel));
        }
        key(vizgfx::Keyboard::Key::X);checkNewPath();
        TASSERT_TRUE(readDocument().at("model").at("optimizer_step")==17);

        click(v.last_latent_space_correlations_rect_);
        TASSERT_TRUE(!v.saveLatentSpaceExport()); // Current mode invalidates the old displayed cache.
        draw();TASSERT_TRUE(!v.saveLatentSpaceExport()); // Collection has no projection yet.
        publish(19,"collecting ",false);draw();TASSERT_TRUE(!v.saveLatentSpaceExport());
        publish(20,"correlation ");v.latent_space_correlation_threshold_=.8125f;
        control->correlation_observations.store(23);draw();
        TASSERT_TRUE(!v.latent_space_correlation_show_all_candidates_ &&
            v.last_latent_space_correlation_edge_hits_.size()==1);
        click(v.last_latent_space_correlation_all_rect_);draw();
        TASSERT_TRUE(v.latent_space_correlation_show_all_candidates_ &&
            v.last_latent_space_correlation_edge_hits_.size()==3);
        const auto correlationPanel=v.latent_space_displayed_bounds_;
        const auto correlationImage=events->captureImage();
        const auto belowThresholdEdge=v.last_latent_space_correlation_edge_hits_.back();
        const auto correlationPixel=vizgfx::Vector2u{
            unsigned((belowThresholdEdge.source.x+belowThresholdEdge.target.x)*.5f),
            unsigned((belowThresholdEdge.source.y+belowThresholdEdge.target.y)*.5f)};
        control->correlation_observations.store(31);
        // Like the projection and camera, the candidate toggle belongs to the
        // last displayed frame, even if the live UI state changes before X.
        v.latent_space_correlation_show_all_candidates_=false;
        key(vizgfx::Keyboard::Key::X);checkNewPath();document=readDocument();
        TASSERT_TRUE(document.at("mode")=="correlations" && document.at("correlations").at("observations")==12 &&
            document.at("correlations").at("inference_observations")==3 &&
            document.at("correlations").at("collected_observations")==23);
        TASSERT_TRUE(document.at("links").at("correlations").size()==3 &&
            document.at("links").at("correlations").at(1).at("correlation")==-.75);
        TASSERT_TRUE(document.at("view").at("correlation_show_all_candidates")==true &&
            !v.latent_space_correlation_show_all_candidates_);
        TASSERT_TRUE(png.loadFromFile(fs::path(v.last_latent_space_export_path_).parent_path()/"vue.png"));
        TASSERT_TRUE(png.getPixel({correlationPixel.x-unsigned(correlationPanel.position.x),
            correlationPixel.y-unsigned(correlationPanel.position.y)})==correlationImage.getPixel(correlationPixel));
        control->clearCorrelations();
        TASSERT_TRUE(!v.saveLatentSpaceExport()); // Clearing history invalidates the displayed epoch too.
        publish(20,"fresh correlation ");draw();

        click(v.last_latent_space_neurons_rect_);TASSERT_TRUE(!v.saveLatentSpaceExport());
        publish(21,"neuron ");draw();click(v.last_latent_space_export_rect_);checkNewPath();document=readDocument();
        TASSERT_TRUE(document.at("mode")=="neurons" && document.at("neurons").at("unknown_layers")==1 &&
            document.at("neurons").at("sampling")==true);
        TASSERT_TRUE(document.at("links").at("neurons").at(0).at("aggregated")==true);

        click(v.last_latent_space_live_weights_rect_);TASSERT_TRUE(!v.saveLatentSpaceExport());
        publish(22,"live weight ");draw();key(vizgfx::Keyboard::Key::X);checkNewPath();document=readDocument();
        TASSERT_TRUE(document.at("mode")=="live_weights" && document.at("live_weights").at("previous_step")==21);
        TASSERT_TRUE(document.at("live_weights").at("coefficients").at(0).at("value")==-.25 &&
            document.at("live_weights").at("coefficients").at(0).at("delta").get<double>()==-1.234567890123e-40);
        control->setLiveWeightsEnabled(false);control->setLiveWeightsEnabled(true);
        TASSERT_TRUE(!v.saveLatentSpaceExport()); // A new delta session cannot export the preceding one.
        publish(22,"fresh live weight ");draw();
        TASSERT_TRUE(!v.latent_space_organic_ && v.last_latent_space_organic_rect_);
        key(vizgfx::Keyboard::Key::O);TASSERT_TRUE(v.latent_space_organic_);
        key(vizgfx::Keyboard::Key::O);TASSERT_TRUE(!v.latent_space_organic_);
        click(v.last_latent_space_organic_rect_);TASSERT_TRUE(v.latent_space_organic_);draw();
        const auto organicImage=events->captureImage();
        const auto organicPanel=v.latent_space_displayed_bounds_;
        TASSERT_TRUE(!v.last_latent_space_point_hits_.empty());
        const auto organicHit=v.last_latent_space_point_hits_.front().first;
        const auto organicPixel=vizgfx::Vector2u{unsigned(organicHit.position.x+organicHit.size.x*.5f),
            unsigned(organicHit.position.y+organicHit.size.y*.5f)};
        key(vizgfx::Keyboard::Key::O);TASSERT_TRUE(!v.latent_space_organic_);
        key(vizgfx::Keyboard::Key::X);checkNewPath();document=readDocument();
        TASSERT_TRUE(document.at("view").at("organic")==true);
        TASSERT_TRUE(png.loadFromFile(fs::path(v.last_latent_space_export_path_).parent_path()/"vue.png"));
        TASSERT_TRUE(png.getPixel({organicPixel.x-unsigned(organicPanel.position.x),
            organicPixel.y-unsigned(organicPanel.position.y)})==organicImage.getPixel(organicPixel));

        control->setRealtimeEnabled(false);
        key(vizgfx::Keyboard::Key::B);
        TASSERT_TRUE(v.latent_space_organic_ && control->live_weights_enabled.load() && control->realtime_enabled.load());
        draw();TASSERT_TRUE(v.last_latent_space_brain_rect_);
        key(vizgfx::Keyboard::Key::O);TASSERT_TRUE(!v.latent_space_organic_);
        click(v.last_latent_space_brain_rect_);TASSERT_TRUE(v.latent_space_organic_);

        key(vizgfx::Keyboard::Key::H);TASSERT_TRUE(v.latent_space_heatspots_);draw();
        TASSERT_TRUE(v.last_latent_space_heatspots_rect_ && v.latent_space_heatspot_texture_.getSize().x==64);
        const auto cachedHeatmap=v.latent_space_heatspot_cached_projection_;draw();
        TASSERT_TRUE(v.latent_space_heatspot_cached_projection_==cachedHeatmap);
        key(vizgfx::Keyboard::Key::X);checkNewPath();document=readDocument();
        TASSERT_TRUE(document.at("view").at("heatspots")==true);
        click(v.last_latent_space_heatspots_rect_);TASSERT_TRUE(!v.latent_space_heatspots_);

        // The export action participates in the same wrapped controls layout.
        for(int width:{420,740,1400}) {
            v.window_width=width;events->setSize({unsigned(width),1000});draw();
            const std::optional<vizgfx::FloatRect>* controls[]={&v.last_latent_space_export_rect_, &v.last_latent_space_organic_rect_,
                &v.last_latent_space_realtime_rect_,&v.last_latent_space_correlations_rect_,
                &v.last_latent_space_neurons_rect_,&v.last_latent_space_live_weights_rect_,
                &v.last_latent_space_correlation_all_rect_,
                &v.last_latent_space_correlation_minus_rect_,&v.last_latent_space_correlation_plus_rect_,
                &v.last_latent_space_correlation_clear_rect_};
            for(size_t first=0;first<std::size(controls);++first) {
                TASSERT_TRUE(*controls[first]);const auto bounds=**controls[first];
                TASSERT_TRUE(bounds.position.x>=16.f && bounds.position.x+bounds.size.x<=width-16.f);
                TASSERT_TRUE(bounds.position.y+bounds.size.y<v.last_latent_space_canvas_rect_.position.y);
                for(size_t second=first+1;second<std::size(controls);++second)
                    TASSERT_TRUE(!bounds.findIntersection(**controls[second]));
            }
        }
        // A filesystem failure remains visible and cannot announce a new file.
        fs::rename("exports","exports_saved");
        {std::ofstream block("exports");block<<"regular file blocks export directory";}
        const auto lastPath=v.last_latent_space_export_path_;
        TASSERT_TRUE(!v.saveLatentSpaceExport() && !v.latent_space_export_status_.empty());
        TASSERT_TRUE(v.last_latent_space_export_path_==lastPath);
        fs::remove("exports");fs::rename("exports_saved","exports");
        control->resetSession("replacement","different_architecture");
        TASSERT_TRUE(!v.saveLatentSpaceExport()); // Never export another model's stale frame.
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.show_latent_space_ && !v.saveLatentSpaceExport());
        v.shutdown();
    }
    static void checkCommandCardLabels(const vizgfx::Font& font) {
        const std::array<std::string,10> labels{{"Mettre en pause", "Arrêt demandé",
            "Validation : oui", "Skip : indisponible", "Réglages : manuels", "KL : non",
            "Perte : mse", "Exporter JSON [D]", "Régénérer le rapport HTML",
            "Afficher / actualiser le nuage 3D"}};
        const std::string hint="Entraînement en cours";
        const std::string status="Projection 3D prête · checkpoints/expérience/"+std::string(160,'W')+".json";
        auto measure=[&](const std::string& value,unsigned size,bool bold) {
            vizgfx::Text text(font);text.setCharacterSize(size);
            text.setStyle(bold?vizgfx::Text::Bold:vizgfx::Text::Regular);
            text.setString(vizgfx::String::fromUtf8(value.begin(),value.end()));
            return text.getLocalBounds().size.x;
        };
        auto without_space=[](const std::string& text) {
            std::string result;
            for(unsigned char c:text) if(!std::isspace(c)) result.push_back(static_cast<char>(c));
            return result;
        };
        auto joined=[](const std::vector<std::string>& lines) {
            std::string result;for(const auto& line:lines) result+=line;return result;
        };
        for(float width:{182.f,252.f,402.f,491.f,722.f,1342.f}) {
            const auto layout=mimir::viz::makeCommandCardLayout(width,labels,hint,status,measure);
            for(size_t index=0;index<labels.size();++index) {
                const auto& button=layout.buttons[index];const auto& bounds=button.bounds;
                TASSERT_TRUE(bounds.x>=8.f && bounds.y>=28.f && bounds.width>0 && bounds.height>=34.f);
                TASSERT_TRUE(bounds.x+bounds.width<=width-8.f+.1f);
                TASSERT_TRUE(bounds.y+bounds.height<layout.status_y);
                TASSERT_TRUE(button.text_size==(index==0?14u:12u));
                TASSERT_TRUE(without_space(joined(button.lines))==without_space(labels[index]));
                for(size_t line=0;line<button.lines.size();++line) {
                    TASSERT_TRUE(button.text_x+measure(button.lines[line],button.text_size,true)<=bounds.width-8.f+.1f);
                    vizgfx::Text glyphs(font);glyphs.setCharacterSize(button.text_size);glyphs.setStyle(vizgfx::Text::Bold);
                    glyphs.setString(vizgfx::String::fromUtf8(button.lines[line].begin(),button.lines[line].end()));
                    TASSERT_TRUE(button.text_y+line*button.line_height+glyphs.getLocalBounds().size.y<=bounds.height-4.f+.1f);
                }
                const vizgfx::FloatRect box{{bounds.x,bounds.y},{bounds.width,bounds.height}};
                for(size_t other=index+1;other<labels.size();++other) {
                    const auto& next=layout.buttons[other].bounds;
                    TASSERT_TRUE(!box.findIntersection({{next.x,next.y},{next.width,next.height}}));
                }
            }
            TASSERT_TRUE(layout.buttons[0].bounds.height>=48.f);
            TASSERT_TRUE(without_space(joined(layout.primary_hint_lines))==without_space(hint));
            TASSERT_TRUE(without_space(joined(layout.status_lines))==without_space(status));
            for(const auto& line:layout.status_lines) TASSERT_TRUE(measure(line,11,false)<=width-16.f+.1f);
            TASSERT_TRUE(layout.status_y+layout.status_lines.size()*layout.status_line_height<=layout.height-8.f+.1f);
        }
    }
    static void runGenericProjection() {
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1000}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>();auto* events=window.get();events->setSize({1400,1000});
        v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        v.initDefaultPanelsIfNeeded();using P=Visualizer::PanelId;
        for(auto& panel:v.panels_)panel.visible=false;
        auto& metrics=v.panels_[size_t(P::Metrics)];metrics.visible=true;metrics.pos={10,10};metrics.size={1380,980};
        auto control=std::make_shared<mimir::viz::LatentSpaceControl>();
        control->resetSession("volume_test","custom_conv3d");v.setLatentSpaceControl(control);
        TASSERT_TRUE(!control->realtime_enabled.load() && !control->view_active.load() &&
            !control->neurons_enabled.load() && !control->live_weights_enabled.load());
        auto draw=[&] {v.syncUIView();v.window->clear(vizgfx::Color(12,14,18));v.drawPanelChrome(P::Metrics);v.renderMetrics();};
        auto click=[&](const std::optional<vizgfx::FloatRect>& bounds) {
            TASSERT_TRUE(bounds);const auto b=*bounds;
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,
                {int(b.position.x+b.size.x/2),int(b.position.y+b.size.y/2)}});
            v.processEvents();
        };
        auto key=[&](vizgfx::Keyboard::Key value) {events->events.emplace_back(vizgfx::Event::KeyPressed{value});v.processEvents();};
        draw();
        TASSERT_TRUE(v.last_commands_rect_ && v.last_latent_space_button_rect_);
        const auto button=*v.last_latent_space_button_rect_;
        TASSERT_TRUE(button.size.x>0 && button.size.y>0);
        TASSERT_TRUE(v.last_commands_rect_->contains(button.position));
        TASSERT_TRUE(v.last_commands_rect_->contains({button.position.x+button.size.x-.1f,button.position.y+button.size.y-.1f}));
        click(v.last_latent_space_button_rect_);
        TASSERT_TRUE(v.show_latent_space_ && control->view_active.load() && control->requested.load());
        uint64_t generation=0;TASSERT_TRUE(control->beginCapture(generation));
        mimir::viz::LatentSpaceRawSnapshot raw;
        raw.generation=generation;raw.model_name="volume_test";raw.architecture="custom_conv3d";raw.optimizer_step=12;
        const std::vector<std::vector<float>> values{{1.f,.2f},{.1f,1.f,-.3f},{-.7f,.3f,.8f,.4f},{.2f,-.6f,.1f,.9f,.4f}};
        const mimir::viz::LatentVectorGroup groups[]={mimir::viz::LatentVectorGroup::ConvolutionWeight,
            mimir::viz::LatentVectorGroup::ParameterWeight,mimir::viz::LatentVectorGroup::LayerActivation,
            mimir::viz::LatentVectorGroup::NormWeight};
        for(size_t i=0;i<values.size();++i) {
            mimir::viz::LatentVectorSample sample;sample.values=values[i];sample.label="volume/filter "+std::to_string(i);
            sample.group=groups[i];sample.layer=int(i);sample.original_dimension=sample.values.size();
            raw.vectors.push_back(std::move(sample));
        }
        TASSERT_TRUE(control->submitRawSnapshot(std::move(raw)));
        const auto captured=control->takeRawSnapshot();TASSERT_TRUE(captured);
        auto projection=mimir::viz::projectLatentSpace3D(*captured);
        TASSERT_TRUE(projection.points.size()==4 && projection.feature_dimension==5);
        TASSERT_TRUE(control->publishProjection(std::move(projection)));
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_canvas_rect_.size.x>0 && v.last_latent_space_canvas_rect_.size.y>0);
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()==4 && v.last_latent_space_close_rect_);
        TASSERT_TRUE(v.last_latent_space_realtime_rect_);
        const auto realtime_bounds=*v.last_latent_space_realtime_rect_;
        const vizgfx::Vector2f realtime_center{realtime_bounds.position.x+realtime_bounds.size.x*.5f,
            realtime_bounds.position.y+realtime_bounds.size.y*.5f};
        TASSERT_TRUE(!v.last_latent_space_canvas_rect_.contains(realtime_center));
        TASSERT_TRUE(!v.last_latent_space_close_rect_->contains(realtime_center));
        TASSERT_TRUE(realtime_bounds.position.y+realtime_bounds.size.y<v.last_latent_space_canvas_rect_.position.y);
        const float checkbox_yaw=v.latent_space_yaw_,checkbox_pitch=v.latent_space_pitch_,checkbox_zoom=v.latent_space_zoom_;
        click(v.last_latent_space_realtime_rect_);
        TASSERT_TRUE(control->realtime_enabled.load() && control->view_active.load() && !v.dragging_latent_space_);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(realtime_center.x+40),int(realtime_center.y+8)}});
        v.processEvents();
        TASSERT_NEAR(v.latent_space_yaw_,checkbox_yaw,1e-6f);TASSERT_NEAR(v.latent_space_pitch_,checkbox_pitch,1e-6f);
        TASSERT_NEAR(v.latent_space_zoom_,checkbox_zoom,1e-6f);
        click(v.last_latent_space_realtime_rect_);
        TASSERT_TRUE(!control->realtime_enabled.load() && !v.dragging_latent_space_);
        click(v.last_latent_space_realtime_rect_);
        TASSERT_TRUE(control->realtime_enabled.load());
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.show_latent_space_ && !control->view_active.load() && control->realtime_enabled.load());
        draw();click(v.last_latent_space_button_rect_);v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.show_latent_space_ && control->view_active.load() && control->realtime_enabled.load());
        click(v.last_latent_space_close_rect_);
        TASSERT_TRUE(!v.show_latent_space_ && !control->view_active.load() && control->realtime_enabled.load());
        draw();click(v.last_latent_space_button_rect_);v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.show_latent_space_ && control->view_active.load() && control->realtime_enabled.load());
        auto verify_marker_bounds=[&] {
            const auto canvas=v.last_latent_space_canvas_rect_;
            for(const auto& hit:v.last_latent_space_point_hits_) {
                const auto bounds=hit.first;
                TASSERT_TRUE(bounds.size.x>0 && bounds.size.y>0 && hit.second<4);
                TASSERT_TRUE(canvas.contains(bounds.position));
                TASSERT_TRUE(canvas.contains({bounds.position.x+bounds.size.x-.1f,bounds.position.y+bounds.size.y-.1f}));
            }
        };
        verify_marker_bounds();
        const auto canvas=v.last_latent_space_canvas_rect_;
        const int cx=int(canvas.position.x+canvas.size.x/2),cy=int(canvas.position.y+canvas.size.y/2);
        const float old_yaw=v.latent_space_yaw_,old_zoom=v.latent_space_zoom_;
        click(canvas);TASSERT_TRUE(v.dragging_latent_space_);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{cx+60,cy+20}});v.processEvents();
        TASSERT_TRUE(v.latent_space_yaw_!=old_yaw);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{cx+60,cy+20}});v.processEvents();
        TASSERT_TRUE(!v.dragging_latent_space_);
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,1.f,{cx,cy}});v.processEvents();
        TASSERT_TRUE(v.latent_space_zoom_>old_zoom);
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,20.f,{cx,cy}});v.processEvents();
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()<4);
        verify_marker_bounds(); // Zoomed-outside points cannot own footer hitboxes.
        key(vizgfx::Keyboard::Key::R);
        TASSERT_NEAR(v.latent_space_yaw_,-.55f,1e-6f);TASSERT_NEAR(v.latent_space_zoom_,1.f,1e-6f);
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()==4);verify_marker_bounds();
        const auto point=v.last_latent_space_point_hits_.front().first;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(point.position.x+7),int(point.position.y+7)}});v.processEvents();
        draw();v.renderLatentSpaceOverlay();events->display();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_projection_3d_generic.png"));

        // Corpus correlations are a separate, opt-in view of measured
        // activations. Switching modes cannot relabel the preceding weight PCA.
        TASSERT_TRUE(!control->correlations_enabled.load());
        TASSERT_TRUE(v.last_latent_space_correlations_rect_ && v.last_latent_space_correlation_clear_rect_ &&
            v.last_latent_space_correlation_minus_rect_ && v.last_latent_space_correlation_plus_rect_ &&
            v.last_latent_space_correlation_all_rect_ && !v.latent_space_correlation_show_all_candidates_);
        click(v.last_latent_space_realtime_rect_);
        TASSERT_TRUE(!control->realtime_enabled.load());
        const float correlation_yaw=v.latent_space_yaw_,correlation_pitch=v.latent_space_pitch_;
        click(v.last_latent_space_correlations_rect_);
        TASSERT_TRUE(control->correlations_enabled.load() && control->requested.load() && !v.dragging_latent_space_);
        TASSERT_NEAR(v.latent_space_yaw_,correlation_yaw,1e-6f);
        TASSERT_NEAR(v.latent_space_pitch_,correlation_pitch,1e-6f);
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty());
        mimir::viz::LatentSpaceProjection correlations;
        correlations.generation=control->generation.load();
        correlations.correlation_epoch=control->correlation_epoch.load();
        TASSERT_TRUE(control->beginCorrelationProjection(correlations.generation,correlations.correlation_epoch,
            16,true,correlations.revision));
        correlations.model_name="volume_test";correlations.architecture="custom_conv3d";
        correlations.correlation_mode=true;correlations.correlation_ready=true;
        correlations.correlation_observations=16;correlations.correlation_training_observations=16;
        correlations.correlation_first_step=12;correlations.correlation_last_step=27;
        correlations.correlation_window=128;correlations.correlation_edge_candidates=6;
        correlations.explained_variance_percent={{70.f,20.f,10.f}};
        const std::array<std::array<float,3>,4> coordinates{{{{-1.f,0.f,0.f}},{{1.f,.8f,0.f}},
            {{0.f,-1.f,0.f}},{{100.f,0.f,0.f}}}};
        for(size_t index=0;index<coordinates.size();++index) {
            mimir::viz::LatentSpacePoint p;p.position=coordinates[index];
            p.label="L2 volume/latent/output canal "+std::to_string(index);
            p.group=mimir::viz::LatentVectorGroup::LatentActivation;p.layer=2;p.feature_index=index;
            p.activation_mean=.2f*index;p.activation_stddev=.5f;p.pooling="moyenne spatiale par contexte";
            correlations.points.push_back(std::move(p));
        }
        correlations.correlation_edges={{0,1,.9f},{1,2,-.8f},{0,2,.4f},{0,3,-.9f}};
        TASSERT_TRUE(control->updateCorrelationCount(correlations.generation,correlations.correlation_epoch,20));
        TASSERT_TRUE(control->publishProjection(std::move(correlations)));
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()==3);
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.size()==3);
        const auto correlation_canvas=v.last_latent_space_canvas_rect_;
        for(const auto& edge:v.last_latent_space_correlation_edge_hits_) {
            // A clipped segment may end exactly on the right or bottom edge.
            for(const auto endpoint:{edge.source,edge.target}) {
                TASSERT_TRUE(endpoint.x>=correlation_canvas.position.x && endpoint.y>=correlation_canvas.position.y);
                TASSERT_TRUE(endpoint.x<=correlation_canvas.position.x+correlation_canvas.size.x);
                TASSERT_TRUE(endpoint.y<=correlation_canvas.position.y+correlation_canvas.size.y);
            }
        }
        TASSERT_TRUE(control->projectionSnapshot()->correlation_observations==16 && control->correlation_observations.load()==20);
        for(int index=0;index<20;++index) click(v.last_latent_space_correlation_plus_rect_);
        TASSERT_NEAR(v.latent_space_correlation_threshold_,.95f,1e-6f);
        TASSERT_TRUE(!v.dragging_latent_space_ && !control->realtime_enabled.load());
        draw();v.renderLatentSpaceOverlay();TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.empty());
        for(int index=0;index<20;++index) click(v.last_latent_space_correlation_minus_rect_);
        TASSERT_NEAR(v.latent_space_correlation_threshold_,.30f,1e-6f);
        draw();v.renderLatentSpaceOverlay();TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.size()==4);
        const auto edge=v.last_latent_space_correlation_edge_hits_.front();
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int((edge.source.x+edge.target.x)*.5f),
            int((edge.source.y+edge.target.y)*.5f)}});v.processEvents();
        draw();v.renderLatentSpaceOverlay();events->display();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_projection_3d_correlations.png"));
        // The default view keeps its strongest-link cap and threshold. An
        // explicit toggle exposes every retained valid candidate immediately,
        // without requesting another model observation or changing the camera.
        auto dense=*control->projectionSnapshot();
        control->requestSnapshot();
        TASSERT_TRUE(control->beginCorrelationProjection(dense.generation,dense.correlation_epoch,
            17,true,dense.revision));
        dense.points.clear();dense.correlation_edges.clear();dense.correlation_candidate_edges.clear();
        constexpr size_t dense_points=33;
        constexpr double tau=6.2831853071795864769;
        for(size_t index=0;index<dense_points;++index) {
            mimir::viz::LatentSpacePoint p;
            p.position={{float(.7*std::cos(tau*index/dense_points)),
                float(.7*std::sin(tau*index/dense_points)),0.f}};
            p.label="dense correlation feature "+std::to_string(index);
            p.group=mimir::viz::LatentVectorGroup::LatentActivation;
            p.activation_stddev=1.f;dense.points.push_back(std::move(p));
        }
        for(size_t source=0;source<dense_points;++source) for(size_t target=source+1;target<dense_points;++target) {
            if(source==0 && target==16) continue; // Last candidate crosses the centre, clear of markers.
            const float r=dense.correlation_candidate_edges.size()<mimir::viz::LatentCorrelationWindow::max_edges?.8f:.4f;
            dense.correlation_candidate_edges.push_back({source,target,r});
        }
        dense.correlation_candidate_edges.push_back({0,16,.4f});
        dense.correlation_edge_candidates=dense.correlation_candidate_edges.size();
        dense.correlation_edges.assign(dense.correlation_candidate_edges.begin(),
            dense.correlation_candidate_edges.begin()+mimir::viz::LatentCorrelationWindow::max_edges);
        TASSERT_TRUE(dense.correlation_edge_candidates==528);
        TASSERT_TRUE(control->publishProjection(std::move(dense)));
        v.latent_space_correlation_threshold_=.65f;draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.size()==512);
        const auto candidate_yaw=v.latent_space_yaw_,candidate_pitch=v.latent_space_pitch_,candidate_zoom=v.latent_space_zoom_;
        const auto observation_count=control->correlation_observations.load();
        click(v.last_latent_space_correlation_all_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.latent_space_correlation_show_all_candidates_ && !v.dragging_latent_space_);
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.size()==528);
        events->display();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_projection_3d_all_candidates.png"));
        TASSERT_TRUE(control->correlation_observations.load()==observation_count && !control->requested.load());
        TASSERT_NEAR(v.latent_space_yaw_,candidate_yaw,1e-6f);
        TASSERT_NEAR(v.latent_space_pitch_,candidate_pitch,1e-6f);
        TASSERT_NEAR(v.latent_space_zoom_,candidate_zoom,1e-6f);
        click(v.last_latent_space_correlation_plus_rect_);click(v.last_latent_space_correlation_minus_rect_);
        TASSERT_NEAR(v.latent_space_correlation_threshold_,.65f,1e-6f);
        const auto last_candidate=v.last_latent_space_correlation_edge_hits_.back();
        TASSERT_TRUE(last_candidate.edge_index==527);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int((last_candidate.source.x+last_candidate.target.x)*.5f),
            int((last_candidate.source.y+last_candidate.target.y)*.5f)}});v.processEvents();
        draw();v.renderLatentSpaceOverlay(); // Tooltip indexes the selected candidate vector, beyond top512.
        click(v.last_latent_space_correlation_all_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(!v.latent_space_correlation_show_all_candidates_ &&
            v.last_latent_space_correlation_edge_hits_.size()==512);
        TASSERT_NEAR(v.latent_space_correlation_threshold_,.65f,1e-6f);
        // All controls stay above the plot when their row wraps on smaller views.
        for(int width:{420,740,1400}) {
            v.window_width=width;events->setSize({static_cast<unsigned>(width),1000});v.renderLatentSpaceOverlay();
            const std::optional<vizgfx::FloatRect>* controls[]={&v.last_latent_space_correlations_rect_,&v.last_latent_space_neurons_rect_,
                &v.last_latent_space_live_weights_rect_,
                &v.last_latent_space_correlation_all_rect_,
                &v.last_latent_space_correlation_minus_rect_,&v.last_latent_space_correlation_plus_rect_,
                &v.last_latent_space_correlation_clear_rect_};
            constexpr size_t control_count=sizeof(controls)/sizeof(*controls);
            for(size_t first=0;first<control_count;++first) {
                TASSERT_TRUE(*controls[first]);const auto bounds=**controls[first];
                TASSERT_TRUE(bounds.position.x>=16.f && bounds.position.x+bounds.size.x<=width-16.f);
                TASSERT_TRUE(bounds.position.y+bounds.size.y<v.last_latent_space_canvas_rect_.position.y);
                for(size_t second=first+1;second<control_count;++second)
                    TASSERT_TRUE(!bounds.findIntersection(**controls[second]).has_value());
            }
        }
        const auto epoch_before_clear=control->correlation_epoch.load();
        click(v.last_latent_space_correlation_clear_rect_);
        TASSERT_TRUE(control->correlations_enabled.load() && !v.dragging_latent_space_);
        TASSERT_TRUE(control->correlation_epoch.load()>epoch_before_clear && control->correlation_observations.load()==0);
        TASSERT_TRUE(!control->projectionSnapshot());
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.empty() && v.last_latent_space_point_hits_.empty());
        TASSERT_TRUE(control->updateCorrelationCount(control->generation.load(),control->correlation_epoch.load(),7));
        control->resetSession("correlation_new_model","custom_conv4d");v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->correlation_observations.load()==0 && !control->projectionSnapshot());
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.empty());
        click(v.last_latent_space_correlations_rect_);
        TASSERT_TRUE(!control->correlations_enabled.load() && control->requested.load() && !v.dragging_latent_space_);
        uint64_t weight_generation=0;TASSERT_TRUE(control->beginCapture(weight_generation));
        mimir::viz::LatentSpaceRawSnapshot weights_after_correlation;weights_after_correlation.generation=weight_generation;
        for(size_t index=0;index<values.size();++index) {
            mimir::viz::LatentVectorSample p;p.values=values[index];p.group=groups[index];
            p.label="returned weight "+std::to_string(index);weights_after_correlation.vectors.push_back(std::move(p));
        }
        TASSERT_TRUE(control->submitRawSnapshot(std::move(weights_after_correlation)));
        const auto weight_snapshot=control->takeRawSnapshot();TASSERT_TRUE(weight_snapshot);
        TASSERT_TRUE(control->publishProjection(mimir::viz::projectLatentSpace3D(*weight_snapshot)));
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(!control->projectionSnapshot()->correlation_mode && v.last_latent_space_point_hits_.size()==4);
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.empty());

        // Neuron distribution is an explicit topology mode, not a relabelled
        // PCA or correlation cloud. Unit and operation nodes remain distinct.
        TASSERT_TRUE(v.last_latent_space_neurons_rect_ && !control->neurons_enabled.load());
        const float neurons_yaw=v.latent_space_yaw_,neurons_pitch=v.latent_space_pitch_;
        click(v.last_latent_space_neurons_rect_);
        TASSERT_TRUE(control->neurons_enabled.load() && !control->correlations_enabled.load());
        TASSERT_TRUE(control->requested.load() && !v.dragging_latent_space_ && !control->projectionSnapshot());
        TASSERT_NEAR(v.latent_space_yaw_,neurons_yaw,1e-6f);TASSERT_NEAR(v.latent_space_pitch_,neurons_pitch,1e-6f);
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty() && v.last_latent_space_correlation_edge_hits_.empty());
        auto publish_neurons=[&] {
            uint64_t token=0,revision=0;bool neuron_capture=false;
            TASSERT_TRUE(control->beginCapture(token,false,control->generation.load(),
                mimir::viz::LatentSpaceControl::Clock::now(),&revision,&neuron_capture));
            TASSERT_TRUE(neuron_capture);
            mimir::viz::LatentSpaceRawSnapshot captured_network;
            captured_network.generation=token;captured_network.revision=revision;captured_network.neuron_mode=true;
            TASSERT_TRUE(control->submitRawSnapshot(std::move(captured_network)));
            const auto captured=control->takeRawSnapshot();TASSERT_TRUE(captured);
            TASSERT_TRUE(control->beginProjection(token,revision,true));
            mimir::viz::LatentSpaceProjection network;
            network.generation=token;network.revision=revision;network.neuron_mode=true;
            network.model_name="neuron_fixture";network.architecture="branched_conv3d";network.optimizer_step=31;
            network.neuron_total_units=42;network.neuron_total_layers=3;network.neuron_unknown_layers=1;
            network.neuron_sampling=true;
            for(size_t index=0;index<coordinates.size();++index) {
                mimir::viz::LatentSpacePoint unit;unit.position=coordinates[index];
                unit.label="L"+std::to_string(index)+" network/"+(index==1?"conv3d [Conv3d] canal ":
                    index==2?"unknown [CustomOp] opération ":"dense [Linear] neurone ")+std::to_string(index);
                unit.layer=int(index);unit.neuron_unit_index=index;unit.neuron_layer_units=index==2?0:14;
                unit.neuron_is_operation=index==2;
                unit.neuron_kind=index==1?"canal":index==2?"opération":"neurone";
                network.points.push_back(std::move(unit));
            }
            network.neuron_links={{0,1,0.f,false,true},{1,2,0.f,false,true},{0,3,0.f,false,true},
                {99,0,0.f,false,true}};
            TASSERT_TRUE(control->publishProjection(std::move(network)));
        };
        publish_neurons();draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()==3 && v.last_latent_space_neuron_edge_hits_.size()==3);
        TASSERT_TRUE(v.last_latent_space_correlation_edge_hits_.empty());
        TASSERT_TRUE(control->projectionSnapshot()->neuron_total_units==42 && control->projectionSnapshot()->neuron_unknown_layers==1);
        for(const auto& neuron_edge:v.last_latent_space_neuron_edge_hits_) {
            const auto network_canvas=v.last_latent_space_canvas_rect_;
            for(const auto endpoint:{neuron_edge.source,neuron_edge.target}) {
                TASSERT_TRUE(endpoint.x>=network_canvas.position.x && endpoint.y>=network_canvas.position.y);
                TASSERT_TRUE(endpoint.x<=network_canvas.position.x+network_canvas.size.x);
                TASSERT_TRUE(endpoint.y<=network_canvas.position.y+network_canvas.size.y);
            }
        }
        const auto neuron_edge=v.last_latent_space_neuron_edge_hits_.front();
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int((neuron_edge.source.x+neuron_edge.target.x)*.5f),
            int((neuron_edge.source.y+neuron_edge.target.y)*.5f)}});v.processEvents();
        draw();v.renderLatentSpaceOverlay();events->display();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_projection_3d_neurons.png"));
        for(int width:{420,740,1400}) {
            v.window_width=width;events->setSize({static_cast<unsigned>(width),1000});v.renderLatentSpaceOverlay();
            const auto bounds=*v.last_latent_space_neurons_rect_;
            TASSERT_TRUE(bounds.position.x>=16.f && bounds.position.x+bounds.size.x<=width-16.f);
            TASSERT_TRUE(bounds.position.y+bounds.size.y<v.last_latent_space_canvas_rect_.position.y);
            TASSERT_TRUE(!bounds.findIntersection(*v.last_latent_space_correlations_rect_));
        }
        // Correlations and neurons are mutually exclusive and purge old hits.
        click(v.last_latent_space_correlations_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->correlations_enabled.load() && !control->neurons_enabled.load());
        TASSERT_TRUE(v.last_latent_space_neuron_edge_hits_.empty() && v.last_latent_space_point_hits_.empty());
        click(v.last_latent_space_neurons_rect_);
        TASSERT_TRUE(control->neurons_enabled.load() && !control->correlations_enabled.load());
        publish_neurons();draw();v.renderLatentSpaceOverlay();
        control->resetSession("neuron_new_model","custom_conv4d");v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->neurons_enabled.load() && !control->projectionSnapshot());
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty() && v.last_latent_space_neuron_edge_hits_.empty());
        click(v.last_latent_space_neurons_rect_);
        TASSERT_TRUE(!control->neurons_enabled.load() && !control->correlations_enabled.load() && control->requested.load());
        TASSERT_TRUE(!v.dragging_latent_space_);

        // Live learned weights have their own mode and a stable topology.
        // Color/width show exact values; a halo requires a measured change.
        TASSERT_TRUE(v.last_latent_space_live_weights_rect_ && !control->live_weights_enabled.load());
        const float live_yaw=v.latent_space_yaw_,live_pitch=v.latent_space_pitch_;
        click(v.last_latent_space_live_weights_rect_);
        TASSERT_TRUE(control->live_weights_enabled.load() && control->realtime_enabled.load());
        TASSERT_TRUE(!control->neurons_enabled.load() && !control->correlations_enabled.load());
        TASSERT_TRUE(!v.dragging_latent_space_ && !control->projectionSnapshot());
        TASSERT_NEAR(v.latent_space_yaw_,live_yaw,1e-6f);TASSERT_NEAR(v.latent_space_pitch_,live_pitch,1e-6f);
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty() && v.last_latent_space_neuron_edge_hits_.empty());
        auto publish_live_weights=[&](bool compared) {
            control->requestSnapshot();
            uint64_t token=0,revision=0;bool neuron_capture=false,live_capture=false;
            TASSERT_TRUE(control->beginCapture(token,false,control->generation.load(),
                mimir::viz::LatentSpaceControl::Clock::now(),&revision,&neuron_capture,&live_capture));
            TASSERT_TRUE(live_capture && !neuron_capture);
            mimir::viz::LatentSpaceRawSnapshot raw_live;
            raw_live.generation=token;raw_live.revision=revision;raw_live.live_weight_mode=true;
            TASSERT_TRUE(control->submitRawSnapshot(std::move(raw_live)));
            const auto owned=control->takeRawSnapshot();TASSERT_TRUE(owned);
            TASSERT_TRUE(control->beginProjection(token,revision,false,true));
            mimir::viz::LatentSpaceProjection live;
            live.generation=token;live.revision=revision;live.live_weight_mode=true;
            live.live_weight_epoch=owned->live_weight_epoch;
            live.model_name="live_weight_fixture";live.architecture="exact_dense";
            live.optimizer_step=compared?41:40;live.live_weight_previous_step=40;
            live.live_weight_baseline_available=compared;live.live_weight_total_parameters=112;
            live.live_weight_nonfinite=1;live.live_weight_shared_layers=1;live.live_weight_skipped_layers=1;
            live.live_weight_sampled=true;live.neuron_total_layers=4;live.neuron_total_units=48;
            const std::array<float,3> live_positions[]={{{-1.f,-.5f,0.f}},{{0.f,1.3f,0.f}},
                {{1.f,-.6f,0.f}},{{0.f,-1.2f,.4f}},{{100.f,0.f,0.f}},{{1.f,1.4f,.2f}}};
            for(size_t index=0;index<6;++index) {
                mimir::viz::LatentSpacePoint unit;unit.position=live_positions[index];
                unit.label="L"+std::to_string(index)+" exact_dense/unit "+std::to_string(index);
                unit.layer=int(index);unit.neuron_kind="neurone";unit.neuron_layer_units=12;
                unit.neuron_unit_index=index;
                if(index==3) {unit.live_weight_coefficient=3;unit.neuron_kind="biais";}
                live.points.push_back(unit);
            }
            const float coefficient_values[]={compared?1.2f:1.f,-.25f,0.f,compared?.07f:.05f};
            for(size_t index=0;index<4;++index) {
                mimir::viz::LiveWeightCoefficient coefficient;
                coefficient.key="exact_dense:"+std::to_string(index);
                coefficient.label="exact_dense/"+std::string(index==3?"bias":"weight")+"["+std::to_string(index)+"]";
                coefficient.parameter_index=index;coefficient.value=coefficient_values[index];
                coefficient.has_delta=compared;
                coefficient.delta=compared?(index==0?static_cast<double>(coefficient_values[index])-1.f:
                    index==3?static_cast<double>(coefficient_values[index])-.05f:0.):0.;
                coefficient.mapped=index<3;coefficient.is_bias=index==3;
                if(index==3)coefficient.point_index=3;
                live.live_weight_coefficients.push_back(coefficient);
            }
            live.neuron_links={{0,2,0.f,false,true}};
            live.live_weight_links={{0,1,0},{2,5,1},{0,4,2},{99,1,0},{0,1,99}};
            TASSERT_TRUE(control->publishProjection(std::move(live)));
        };
        publish_live_weights(false);v.picker_mouse_={0.f,0.f};draw();v.renderLatentSpaceOverlay();events->display();
        TASSERT_TRUE(v.last_latent_space_point_hits_.size()==5 && v.last_latent_space_live_weight_edge_hits_.size()==3);
        TASSERT_TRUE(v.last_latent_space_neuron_edge_hits_.size()==1 && v.last_latent_space_correlation_edge_hits_.empty());
        TASSERT_TRUE(!control->projectionSnapshot()->live_weight_baseline_available);
        const auto first_live=control->projectionSnapshot();
        for(const auto& edge:v.last_latent_space_live_weight_edge_hits_) {
            const auto canvas=v.last_latent_space_canvas_rect_;
            for(const auto endpoint:{edge.source,edge.target}) {
                TASSERT_TRUE(endpoint.x>=canvas.position.x && endpoint.y>=canvas.position.y);
                TASSERT_TRUE(endpoint.x<=canvas.position.x+canvas.size.x && endpoint.y<=canvas.position.y+canvas.size.y);
            }
        }
        const auto before_live_image=events->captureImage();
        const auto positive_link=v.last_latent_space_live_weight_edge_hits_[0];
        const auto negative_link=v.last_latent_space_live_weight_edge_hits_[1];
        auto midpoint=[](const auto& edge) {return vizgfx::Vector2u{
            unsigned((edge.source.x+edge.target.x)*.5f),unsigned((edge.source.y+edge.target.y)*.5f)};};
        const auto positive_pixel=before_live_image.getPixel(midpoint(positive_link));
        const auto negative_pixel=before_live_image.getPixel(midpoint(negative_link));
        TASSERT_TRUE(positive_pixel.b>positive_pixel.r && negative_pixel.r>negative_pixel.b);
        // Separate graph edges must not acquire a visible batching connector.
        const auto gap_pixel=before_live_image.getPixel({unsigned((positive_link.target.x+negative_link.source.x)*.5f),
            unsigned((positive_link.target.y+negative_link.source.y)*.5f)});
        TASSERT_TRUE(gap_pixel.r<30 && gap_pixel.g<30 && gap_pixel.b<35);
        click(v.last_latent_space_realtime_rect_);
        TASSERT_TRUE(!control->realtime_enabled.load() && control->live_weights_enabled.load());
        draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->projectionSnapshot()->revision==first_live->revision);
        publish_live_weights(true);draw();v.renderLatentSpaceOverlay();events->display();
        const auto second_live=control->projectionSnapshot();
        TASSERT_TRUE(second_live->live_weight_baseline_available && second_live->live_weight_previous_step==40 &&
            second_live->optimizer_step==41 && second_live->live_weight_coefficients[0].has_delta);
        for(size_t index=0;index<first_live->points.size();++index)
            TASSERT_TRUE(first_live->points[index].position==second_live->points[index].position);
        auto yellow_pixels=[&](const auto& image) {
            const auto center=midpoint(positive_link);size_t count=0;
            for(int dy=-5;dy<=5;++dy)for(int dx=-5;dx<=5;++dx) {
                const auto pixel=image.getPixel({unsigned(int(center.x)+dx),unsigned(int(center.y)+dy)});
                if(pixel.r>100 && pixel.g>100 && pixel.b<pixel.r && pixel.b<pixel.g)++count;
            }
            return count;
        };
        TASSERT_TRUE(yellow_pixels(events->captureImage())>yellow_pixels(before_live_image));
        const auto exact_edge=v.last_latent_space_live_weight_edge_hits_.front();
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int((exact_edge.source.x+exact_edge.target.x)*.5f),
            int((exact_edge.source.y+exact_edge.target.y)*.5f)}});v.processEvents();
        draw();v.renderLatentSpaceOverlay();events->display();
        TASSERT_TRUE(events->captureImage().saveToFile("/tmp/mimir_projection_3d_live_weights.png"));
        for(int width:{420,740,1400}) {
            v.window_width=width;events->setSize({static_cast<unsigned>(width),1000});v.renderLatentSpaceOverlay();
            const auto bounds=*v.last_latent_space_live_weights_rect_;
            TASSERT_TRUE(bounds.position.x>=16.f && bounds.position.x+bounds.size.x<=width-16.f);
            TASSERT_TRUE(bounds.position.y+bounds.size.y<v.last_latent_space_canvas_rect_.position.y);
            TASSERT_TRUE(!bounds.findIntersection(*v.last_latent_space_neurons_rect_));
            TASSERT_TRUE(!bounds.findIntersection(*v.last_latent_space_correlations_rect_));
        }
        click(v.last_latent_space_correlations_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->correlations_enabled.load() && !control->live_weights_enabled.load());
        TASSERT_TRUE(v.last_latent_space_live_weight_edge_hits_.empty() && v.last_latent_space_point_hits_.empty());
        TASSERT_TRUE(!control->publishProjection(*second_live));
        click(v.last_latent_space_neurons_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->neurons_enabled.load() && !control->live_weights_enabled.load());
        click(v.last_latent_space_live_weights_rect_);draw();v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->live_weights_enabled.load() && !control->neurons_enabled.load() && !control->correlations_enabled.load());
        publish_live_weights(false);draw();v.renderLatentSpaceOverlay();
        control->resetSession("live_weights_new_model","custom_conv4d");v.renderLatentSpaceOverlay();
        TASSERT_TRUE(control->live_weights_enabled.load() && !control->projectionSnapshot());
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty() && v.last_latent_space_live_weight_edge_hits_.empty());
        click(v.last_latent_space_live_weights_rect_);
        TASSERT_TRUE(!control->live_weights_enabled.load() && !control->neurons_enabled.load() &&
            !control->correlations_enabled.load() && !v.dragging_latent_space_);
        control->setRealtimeEnabled(false);
        click(v.last_latent_space_realtime_rect_);TASSERT_TRUE(control->realtime_enabled.load());
        control->resetSession("second_volume","custom_conv4d");v.renderLatentSpaceOverlay();
        TASSERT_TRUE(v.last_latent_space_point_hits_.empty() && control->realtime_enabled.load());
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.show_latent_space_ && !control->view_active.load());
        draw();click(v.last_latent_space_button_rect_);
        TASSERT_TRUE(v.show_latent_space_ && control->view_active.load());
        v.shutdown();
        TASSERT_TRUE(!control->view_active.load());
    }
    static void runScrollPadding() {
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1800}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>();auto* events=window.get();
        events->setSize({1400,1800});v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        checkCommandCardLabels(v.font);
        v.initDefaultPanelsIfNeeded();using P=Visualizer::PanelId;
        for(auto& panel:v.panels_) panel.visible=false;
        auto center=[](const vizgfx::FloatRect& rect) {
            return vizgfx::Vector2i{int(rect.position.x+rect.size.x*.5f),int(rect.position.y+rect.size.y*.5f)};
        };
        auto press=[&](const vizgfx::FloatRect& rect) {
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,center(rect)});
            v.processEvents();
        };
        auto release=[&](vizgfx::Vector2i position) {
            events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,position});
            v.processEvents();
        };
        auto wheel=[&](const vizgfx::FloatRect& rect,float delta) {
            events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,delta,center(rect)});
            v.processEvents();
        };
        auto disjoint=[](const vizgfx::FloatRect& content,const vizgfx::FloatRect& track) {
            TASSERT_TRUE(content.size.x>0.f && content.size.y>0.f && track.size.x>0.f && track.size.y>0.f);
            TASSERT_TRUE(!content.findIntersection(track));
            TASSERT_TRUE(content.position.x+content.size.x+2.f<=track.position.x);
        };
        // Every scrolling panel reserves the same dedicated lane, including
        // before overflow. The layout must not jump when its scrollbar appears.
        for(const auto id:{P::Context,P::Output,P::Generated,P::Training,P::Metrics}) {
            auto& panel=v.panels_[size_t(id)];panel.visible=true;panel.pos={10,10};
            for(float width:{240.f,440.f,760.f}) {
                panel.size={width,360.f};
                v.setPanelContentHeight(id,0.f);v.drawPanelScrollbar(id);
                const auto before=v.panelContentRect(id);
                TASSERT_TRUE(v.panel_scroll_track_rects_[size_t(id)].size.x==0.f);
                v.setPanelContentHeight(id,before.size.y+500.f);v.drawPanelScrollbar(id);
                const auto after=v.panelContentRect(id),track=v.panel_scroll_track_rects_[size_t(id)];
                TASSERT_TRUE(before.position==after.position && before.size==after.size);
                disjoint(after,track);
                const auto viewport=v.panelViewportRect(id);
                TASSERT_TRUE(viewport.contains(track.position));
                TASSERT_TRUE(viewport.contains({track.position.x+track.size.x-.1f,track.position.y+track.size.y-.1f}));
            }
            panel.visible=false;
        }

        // An oversized preview hitbox must not steal clicks from its owner's
        // reserved scrollbar lane, even when the image covers that lane.
        auto& context=v.panels_[size_t(P::Context)];context.visible=true;
        context.pos={10,10};context.size={440,180};
        v.last_dataset_rect_=v.panelViewportRect(P::Context);
        v.focus_target_=Visualizer::FocusTarget::Generated;
        v.setPanelContentHeight(P::Context,600.f);v.drawPanelScrollbar(P::Context);
        const auto context_thumb=v.panel_scroll_thumb_rects_[size_t(P::Context)];
        press(context_thumb);
        TASSERT_TRUE(v.dragging_panel_scrollbar_ && v.scrolled_panel_==P::Context);
        TASSERT_TRUE(v.focus_target_==Visualizer::FocusTarget::Generated);
        release(center(context_thumb));v.last_dataset_rect_.reset();context.visible=false;

        auto& metrics=v.panels_[size_t(P::Metrics)];metrics.visible=true;metrics.pos={10,10};
        auto paused=std::make_shared<std::atomic<bool>>(false);v.setTrainingPauseControl(paused);
        auto draw_metrics=[&] {
            v.syncUIView();v.drawPanelChrome(P::Metrics);v.renderMetrics();v.drawPanelScrollbar(P::Metrics);
        };
        for(float width:{290.f,440.f,760.f}) {
            metrics.size={width,420.f};v.panel_scroll_y_[size_t(P::Metrics)]=0.f;draw_metrics();
            const auto content=v.panelContentRect(P::Metrics),track=v.panel_scroll_track_rects_[size_t(P::Metrics)];
            disjoint(content,track);
            TASSERT_TRUE(v.last_commands_rect_ && v.last_pause_button_rect_);
            const std::optional<vizgfx::FloatRect>* elements[]={&v.last_commands_rect_,&v.last_pause_button_rect_,
                &v.last_stop_button_rect_,&v.last_validation_button_rect_,&v.last_live_lr_track_,
                &v.last_live_lr_value_box_,&v.last_live_klb_track_,&v.last_live_klb_value_box_};
            for(const auto element:elements) {
                TASSERT_TRUE(*element);
                TASSERT_TRUE((**element).position.x+(**element).size.x<=content.position.x+content.size.x+.1f);
                TASSERT_TRUE(!(**element).findIntersection(track));
            }
            press(*v.last_pause_button_rect_);TASSERT_TRUE(paused->load());release({0,0});
            press(*v.last_pause_button_rect_);TASSERT_TRUE(!paused->load());release({0,0});
            wheel(track,-1.f);
            TASSERT_TRUE(v.panel_scroll_y_[size_t(P::Metrics)]>0.f && !paused->load());
            draw_metrics();
            const float before=v.panel_scroll_y_[size_t(P::Metrics)];
            const auto thumb=v.panel_scroll_thumb_rects_[size_t(P::Metrics)];
            press(thumb);TASSERT_TRUE(v.dragging_panel_scrollbar_ && !paused->load());
            const auto next=vizgfx::Vector2i{center(thumb).x,int(track.position.y+track.size.y-12.f)};
            events->events.emplace_back(vizgfx::Event::MouseMoved{next});v.processEvents();release(next);
            TASSERT_TRUE(!v.dragging_panel_scrollbar_ && v.panel_scroll_y_[size_t(P::Metrics)]>before);
            TASSERT_TRUE(v.panel_scroll_y_[size_t(P::Metrics)]<=v.panel_scroll_max_[size_t(P::Metrics)]);
        }
        metrics.visible=false;

        std::vector<Visualizer::BlockFrame> frames;
        for(int index=0;index<24;++index) {
            Visualizer::BlockFrame frame;frame.w=frame.h=2;frame.channels=1;frame.pixels={0,80,160,255};
            frame.label="model/blocks/layer_"+std::to_string(index)+"/Linear";frames.push_back(std::move(frame));
        }
        v.blocks_tips_visible_limit_=0;v.setLayerBlockImages(frames);
        auto& blocks=v.panels_[size_t(P::Blocks)];blocks.visible=true;blocks.pos={10,10};
        for(float width:{220.f,320.f,760.f}) {
            blocks.size={width,380.f};v.blocks_scroll_y_=0.f;v.syncUIView();
            v.drawPanelChrome(P::Blocks);v.renderLayerBlocks();
            const auto content=v.panelContentRect(P::Blocks),track=v.last_blocks_scroll_track_rect_;
            disjoint(content,track);
            bool visible_tile=false;
            for(const auto& tile:v.last_block_rects_) if(tile.size.x>0.f && tile.findIntersection(content)) {
                visible_tile=true;TASSERT_TRUE(!tile.findIntersection(track));
                TASSERT_TRUE(tile.position.x+tile.size.x<=content.position.x+content.size.x+.1f);
            }
            TASSERT_TRUE(visible_tile);
            wheel(track,-1.f);TASSERT_TRUE(v.blocks_scroll_y_>0.f);
            v.renderLayerBlocks();
            const float before=v.blocks_scroll_y_;const auto thumb=v.last_blocks_scroll_thumb_rect_;
            press(thumb);TASSERT_TRUE(v.dragging_blocks_scrollbar_);
            const auto next=vizgfx::Vector2i{center(thumb).x,int(track.position.y+track.size.y-12.f)};
            events->events.emplace_back(vizgfx::Event::MouseMoved{next});v.processEvents();release(next);
            TASSERT_TRUE(!v.dragging_blocks_scrollbar_ && v.blocks_scroll_y_>before);
        }
        // At minimum supported height, the wrapped Blocks toolbar can leave
        // less room than the normal minimum thumb height.
        blocks.size={Visualizer::kPanelMinW,Visualizer::kPanelMinH};
        v.drawPanelChrome(P::Blocks);v.renderLayerBlocks();
        const auto short_track=v.last_blocks_scroll_track_rect_,short_thumb=v.last_blocks_scroll_thumb_rect_;
        TASSERT_TRUE(short_thumb.size.y<=short_track.size.y);
        if(short_track.size.y>0.f) {
            TASSERT_TRUE(short_thumb.position.y>=short_track.position.y);
            TASSERT_TRUE(short_thumb.position.y+short_thumb.size.y<=short_track.position.y+short_track.size.y+.1f);
        }
        blocks.visible=false;

        // Dataset subsections keep their own text and scrollbar lanes inside
        // the card. Scrolling a card must not move the outer Metrics panel.
        metrics.visible=true;metrics.pos={10,10};metrics.size={760,1700};
        std::string prose;
        for(int row=0;row<80;++row) prose+="Texte complet avec accents, explications et contexte pour chaque section. ";
        v.setDatasetText(prose,prose,prose,prose);v.show_prompt_text_=true;
        v.panel_scroll_y_[size_t(P::Metrics)]=0.f;draw_metrics();
        for(size_t index=0;index<v.dataset_text_section_rects_.size();++index) {
            const auto section=v.dataset_text_section_rects_[index];
            const auto content=v.dataset_text_content_rects_[index],track=v.dataset_text_scroll_track_rects_[index];
            TASSERT_TRUE(v.dataset_text_scroll_max_[index]>0.f);disjoint(content,track);
            TASSERT_TRUE(section.contains(content.position));
            TASSERT_TRUE(section.contains({content.position.x+content.size.x-.1f,content.position.y+content.size.y-.1f}));
            TASSERT_TRUE(section.contains(track.position));
        }
        const size_t prompt=size_t(Visualizer::DatasetTextSection::Prompt);
        const float outer_before=v.panel_scroll_y_[size_t(P::Metrics)];
        wheel(v.dataset_text_scroll_track_rects_[prompt],-1.f);
        TASSERT_TRUE(v.dataset_text_scroll_y_[prompt]>0.f);
        TASSERT_TRUE(v.panel_scroll_y_[size_t(P::Metrics)]==outer_before);

        blocks.visible=true;blocks.pos={800,10};blocks.size={580,480};v.blocks_scroll_y_=0.f;
        v.window->clear(vizgfx::Color(12,14,18));draw_metrics();
        v.drawPanelChrome(P::Blocks);v.renderLayerBlocks();v.window->display();
        TASSERT_TRUE(v.window->captureImage().saveToFile("/tmp/mimir_viz_scroll_padding.png"));

        // Reproduce the supplied 529 px Metrics panel with the longest ready
        // action labels, including a completed 3D cloud and an available report.
        blocks.visible=false;
        events->setSize({529,866});v.window_width=529;v.window_height=866;
        metrics.pos={0,0};metrics.size={529,866};v.panel_scroll_y_[size_t(P::Metrics)]=0.f;
        auto report=std::make_shared<GraphReportControl>();report->state.store(GraphReportControl::State::Ready);
        v.setGraphReportControl(report);
        auto cloud=std::make_shared<mimir::viz::LatentSpaceControl>();
        const auto generation=cloud->resetSession("Layout test","basic_mlp",false);
        mimir::viz::LatentSpaceRawSnapshot raw;raw.generation=generation;
        mimir::viz::LatentVectorSample sample;sample.label="actual cloud sample";sample.values={1.f,2.f};
        raw.vectors.push_back(sample);
        TASSERT_TRUE(cloud->submitRawSnapshot(std::move(raw)));
        const auto copied=cloud->takeRawSnapshot();TASSERT_TRUE(copied);
        TASSERT_TRUE(cloud->publishProjection(mimir::viz::projectLatentSpace3D(*copied)));
        v.setLatentSpaceControl(cloud);v.validation_enabled_.store(true);
        auto split_kl=std::make_shared<LiveModelConfig>();
        split_kl->publish({
            {"kl_mu_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt},
            {"kl_logvar_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
        v.setRuntimeConfiguration(split_kl);
        v.window->clear(vizgfx::Color(12,14,18));draw_metrics();v.window->display();
        TASSERT_TRUE(v.last_graph_report_button_rect_ && v.last_latent_space_button_rect_);
        TASSERT_TRUE(v.last_commands_rect_->contains(v.last_graph_report_button_rect_->position));
        TASSERT_TRUE(v.last_commands_rect_->contains({v.last_latent_space_button_rect_->position.x+
            v.last_latent_space_button_rect_->size.x-.1f,v.last_latent_space_button_rect_->position.y+
            v.last_latent_space_button_rect_->size.y-.1f}));
        TASSERT_TRUE(v.window->captureImage().saveToFile("/tmp/mimir_viz_metrics_529.png"));
        press(*v.last_graph_report_button_rect_);TASSERT_TRUE(report->requested.load());release({0,0});
        press(*v.last_latent_space_button_rect_);TASSERT_TRUE(v.show_latent_space_ && cloud->view_active.load());
        release({0,0});v.setLatentSpaceVisible(false);
        v.shutdown();
    }

    static void runKlControls() {
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1800}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>(); auto* events=window.get();v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        v.initDefaultPanelsIfNeeded();
        using P=Visualizer::PanelId;
        for (auto& panel:v.panels_) panel.visible=false;
        v.panels_[size_t(P::Metrics)].visible=true;
        auto key=[&](vizgfx::Keyboard::Key k) {events->events.emplace_back(vizgfx::Event::KeyPressed{k});v.processEvents();};
        auto click=[&](const std::optional<vizgfx::FloatRect>& rect) {
            TASSERT_TRUE(rect);const auto b=*rect;
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,
                {int(b.position.x+b.size.x/2),int(b.position.y+b.size.y/2)}});
            v.processEvents();
        };
        // Component controls submit independent requests without switching LR/KL
        // scheduling to manual mode or overwriting the other component.
        auto kl_config=std::make_shared<LiveModelConfig>();
        kl_config->publish({
            {"kl_mu_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt},
            {"kl_logvar_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
        auto pending_kl = [&](const std::string& key) -> std::optional<json> {
            for (const auto& entry : kl_config->snapshot()) if (entry.key == key) return entry.pending;
            throw std::runtime_error("missing KL control");
        };
        v.setRuntimeConfiguration(kl_config);
        v.panels_[size_t(P::Metrics)].pos={10,10};
        v.panels_[size_t(P::Metrics)].size={700,1400};
        v.panel_scroll_y_[size_t(P::Metrics)]=0;
        v.live_overrides_enabled_.store(false);
        v.renderMetrics();
        TASSERT_TRUE(v.last_live_mu_value_box_ && v.last_live_logvar_value_box_);
        click(v.last_live_mu_value_box_);
        TASSERT_TRUE(v.live_input_target_==Visualizer::LiveInputTarget::KLMu);
        v.live_input_buffer_="0";key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(pending_kl("kl_mu_weight")==0.0);
        TASSERT_TRUE(!pending_kl("kl_logvar_weight"));
        TASSERT_TRUE(!v.live_overrides_enabled_.load());
        click(v.last_live_logvar_value_box_);
        v.live_input_buffer_="2";key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(pending_kl("kl_logvar_weight")==2.0);
        TASSERT_TRUE(pending_kl("kl_mu_weight")==0.0);
        TASSERT_TRUE(!v.live_overrides_enabled_.load());
    }

    static void run() {
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1800}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>(); auto* events=window.get();v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        v.initDefaultPanelsIfNeeded();
        using P=Visualizer::PanelId;
        v.panels_[size_t(P::Blocks)].pos={10,400};v.panels_[size_t(P::Blocks)].size={760,400};
        v.panels_[size_t(P::Context)].size={420,280};
        auto redraw=[&] {v.syncUIView();v.drawPanelChrome(P::Blocks);v.drawPanelChrome(P::Context);v.drawPanelChrome(P::Output);v.renderMetrics();};
        redraw();
        const auto blocks_size=v.panels_[size_t(P::Blocks)].size;
        for(float width:{220.f,400.f,760.f}) {
            v.panels_[size_t(P::Blocks)].size.x=width;redraw();
            const std::optional<vizgfx::FloatRect>* controls[]={&v.last_blocks_mode_box_,&v.last_blocks_smooth_box_,&v.last_blocks_palette_box_,&v.last_blocks_hide_act_box_,&v.last_blocks_hide_norm_box_,&v.last_blocks_tips_limit_box_};
            const auto header=v.panelTitleRect(P::Blocks),content=v.panelContentRect(P::Blocks);
            for(size_t i=0;i<6;++i) {
                TASSERT_TRUE(controls[i]->has_value());const auto r=**controls[i];
                TASSERT_TRUE(header.contains(r.position));
                TASSERT_TRUE(header.contains({r.position.x+r.size.x-.1f,r.position.y+r.size.y-.1f}));
                TASSERT_TRUE(r.position.y+r.size.y<content.position.y);
                for(size_t j=i+1;j<6;++j)TASSERT_TRUE(!r.findIntersection(**controls[j]).has_value());
            }
        }
        v.panels_[size_t(P::Blocks)].size=blocks_size;redraw();
        auto check_commands=[&] {
            TASSERT_TRUE(v.last_commands_rect_.has_value());
            const std::optional<vizgfx::FloatRect>* buttons[]={&v.last_debug_json_button_rect_,&v.last_pause_button_rect_,&v.last_stop_button_rect_,
                &v.last_validation_button_rect_,&v.last_skip_button_rect_,&v.last_live_overrides_box_,
                &v.last_live_kl_enable_box_,&v.last_live_recon_loss_box_};
            for (auto button:buttons) {
                TASSERT_TRUE(button->has_value());
                const auto r=**button;
                TASSERT_TRUE(v.last_commands_rect_->contains(r.position));
                TASSERT_TRUE(v.last_commands_rect_->contains({r.position.x+r.size.x-0.1f,r.position.y+r.size.y-0.1f}));
            }
            for (size_t i=0;i<8;++i) for (size_t j=i+1;j<8;++j)
                TASSERT_TRUE(!( **buttons[i]).findIntersection(**buttons[j]).has_value());
            TASSERT_TRUE(v.last_pause_button_rect_->size.y>=40.f);
            TASSERT_TRUE(v.last_pause_button_rect_->position.y<v.last_stop_button_rect_->position.y);
        };
        check_commands();
        const auto metrics_panel=v.panels_[size_t(P::Metrics)];
        v.panels_[size_t(P::Metrics)].size.x=290.f;v.renderMetrics();check_commands();
        v.panels_[size_t(P::Metrics)]=metrics_panel;redraw();
        if (const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear(vizgfx::Color(12,14,18));redraw();
            v.window->display();TASSERT_TRUE(v.window->captureImage().saveToFile(capture));
        }
        auto key=[&](vizgfx::Keyboard::Key k) {events->events.emplace_back(vizgfx::Event::KeyPressed{k});v.processEvents();};
        auto click=[&](const std::optional<vizgfx::FloatRect>& r) {
            TASSERT_TRUE(r.has_value()); const auto& b=*r;
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,
                {int(b.position.x+b.size.x/2),int(b.position.y+b.size.y/2)}});
            v.processEvents();
        };
        auto debug_control = std::atomic_load(&v.runtime_config_);
        key(vizgfx::Keyboard::Key::D);
        TASSERT_TRUE(v.debug_json_dialog_.visible);
        TASSERT_TRUE(!debug_control->consumeDebugJson());
        v.debug_json_dialog_.directory = "/tmp/mimir-debugjson-dialog-test";
        v.debug_json_dialog_.filename = "chosen.json";
        key(vizgfx::Keyboard::Key::Tab);
        TASSERT_TRUE(v.debug_json_dialog_.selected == 1);
        key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(!v.debug_json_dialog_.visible);
        std::string chosen_path;
        TASSERT_TRUE(debug_control->consumeDebugJson(&chosen_path));
        TASSERT_TRUE(chosen_path == "/tmp/mimir-debugjson-dialog-test/chosen.json");
        TASSERT_TRUE(!debug_control->requestDebugJson());
        debug_control->finishDebugJson("DebugJSON test");
        click(v.last_debug_json_button_rect_);
        TASSERT_TRUE(v.debug_json_dialog_.visible);
        v.renderDebugJsonDialog();
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.debug_json_dialog_.visible);
        TASSERT_TRUE(!debug_control->consumeDebugJson());
        TASSERT_TRUE(!v.stopTrainingRequested());
        TASSERT_TRUE(!std::atomic_load(&v.training_paused_)->load());
        const bool heat=v.heatmap_mode_;click(v.last_blocks_mode_box_);TASSERT_TRUE(v.heatmap_mode_!=heat);
        const bool smooth=v.smooth_layer_block_previews_;click(v.last_blocks_smooth_box_);TASSERT_TRUE(v.smooth_layer_block_previews_!=smooth);
        const auto palette=v.heatmap_palette_;click(v.last_blocks_palette_box_);TASSERT_TRUE(v.heatmap_palette_!=palette);
        const bool act=v.hide_activation_blocks;click(v.last_blocks_hide_act_box_);TASSERT_TRUE(v.hide_activation_blocks!=act && v.consumeResyncRequested());
        const bool norm=v.hide_normalisation_blocks;click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(v.hide_normalisation_blocks!=norm && v.consumeResyncRequested());
        // Native layer captures and legacy previews obey both filters immediately.
        std::vector<Visualizer::BlockFrame> filter_frames;
        auto frame_for=[&](const std::string& label) {
            Visualizer::BlockFrame frame;frame.w=frame.h=1;frame.channels=1;frame.pixels={120};frame.label=label;
            filter_frames.push_back(std::move(frame));
        };
        const std::vector<std::string> activation_types={"ReLU","LeakyReLU","GELU","GEGLU","SiLU","Tanh","Sigmoid", "Softmax","LogSoftmax","Softplus","Mish","HardSigmoid","HardSwish","Swish"};
        const std::vector<std::string> norm_types={"BatchNorm1d","BatchNorm2d","LayerNorm","GroupNorm","InstanceNorm2d","RMSNorm"};
        for(const auto& type:activation_types) frame_for("model/blocks/model/opaque/"+type+"/vec");
        for(const auto& type:norm_types) frame_for("model/blocks/model/opaque/"+type+"/vec");
        frame_for("model/component/activation");frame_for("model/component/norm1");
        frame_for("model/input/dataset/rgb");
        frame_for("model/blocks/model/norm_projection/Conv2d");
        frame_for("model/blocks/model/activation_features/Linear");
        const size_t remaining=3;
        v.setLayerBlockImages(filter_frames);TASSERT_TRUE(v.layer_block_images.size()==remaining);
        TASSERT_TRUE(v.last_block_frames_.size()==filter_frames.size());
        click(v.last_blocks_hide_act_box_);TASSERT_TRUE(!v.hide_activation_blocks);
        TASSERT_TRUE(v.layer_block_images.size()==remaining+activation_types.size()+1);
        click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(!v.hide_normalisation_blocks);
        TASSERT_TRUE(v.layer_block_images.size()==filter_frames.size());
        click(v.last_blocks_hide_act_box_);TASSERT_TRUE(v.layer_block_images.size()==remaining+norm_types.size()+1);
        v.setLayerBlockImages(filter_frames);TASSERT_TRUE(v.layer_block_images.size()==remaining+norm_types.size()+1);
        click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(v.layer_block_images.size()==remaining);
        v.setLayerBlockImages({});
        Visualizer::BlockFrame palette_encoder;palette_encoder.w=palette_encoder.h=2;palette_encoder.channels=1;
        palette_encoder.pixels={0,64,128,255};palette_encoder.label="model/blocks/encoder/Conv2d";
        auto palette_diff=palette_encoder;palette_diff.label="model/blocks/resdiff_abs/Conv2d";
        auto palette_identity=palette_encoder;palette_identity.label="model/blocks/identity/Identity";
        const bool previous_heatmap_mode=v.heatmap_mode_;
        const auto previous_palette=v.heatmap_palette_;
        v.heatmap_mode_=true;v.heatmap_palette_=Visualizer::HeatmapPalette::Classic;
        v.setLayerBlockImages({palette_encoder,palette_diff,palette_identity});
        const auto encoder_classic=v.layer_block_images[0].pixels;
        const auto diff_classic=v.layer_block_images[1].pixels;
        const auto identity_classic=v.layer_block_images[2].pixels;
        v.heatmap_palette_=Visualizer::HeatmapPalette::Turbo;
        v.setLayerBlockImages({palette_encoder,palette_diff,palette_identity});
        TASSERT_TRUE(v.layer_block_images[0].pixels!=encoder_classic);
        TASSERT_TRUE(v.layer_block_images[1].pixels!=diff_classic);
        TASSERT_TRUE(v.layer_block_images[2].pixels==identity_classic);
        v.heatmap_mode_=previous_heatmap_mode;v.heatmap_palette_=previous_palette;
        v.setLayerBlockImages({});
        click(v.last_blocks_tips_limit_box_);TASSERT_TRUE(v.blocks_tips_limit_editing_);
        events->events.emplace_back(vizgfx::Event::TextEntered{U'2'});v.processEvents();key(vizgfx::Keyboard::Key::Enter);TASSERT_TRUE(v.blocks_tips_visible_limit_==2);
        const bool detection=v.show_detection_overlay_;click(v.last_detection_toggle_box_);TASSERT_TRUE(v.show_detection_overlay_!=detection);
        click(v.last_detection_threshold_track_);
        TASSERT_NEAR(v.detection_score_threshold_,0.5f,0.01f);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        auto pause=std::make_shared<std::atomic<bool>>(false); v.setTrainingPauseControl(pause);
        click(v.last_pause_button_rect_);TASSERT_TRUE(pause->load());
        click(v.last_pause_button_rect_);TASSERT_TRUE(!pause->load());
        const bool validation=v.validationEnabledSnapshot();click(v.last_validation_button_rect_);TASSERT_TRUE(v.validationEnabledSnapshot()!=validation);
        v.skip_control_->available=true;const bool skips=v.skip_control_->enabled;click(v.last_skip_button_rect_);TASSERT_TRUE(v.skip_control_->enabled!=skips);
        click(v.last_live_overrides_box_);TASSERT_TRUE(v.liveTrainParamsSnapshot().overrides_enabled);
        const bool kl=v.live_ui_kl_enabled_;click(v.last_live_kl_enable_box_);TASSERT_TRUE(v.liveTrainParamsSnapshot().kl_enabled!=kl);
        const int recon=v.live_ui_recon_loss_index_;click(v.last_live_recon_loss_box_);TASSERT_TRUE(v.live_ui_recon_loss_index_==(recon+1)%6);
        const std::optional<vizgfx::FloatRect>* inputs[]={&v.last_live_lr_value_box_,&v.last_live_lrwu_value_box_,&v.last_live_klb_value_box_,&v.last_live_klwu_value_box_};
        for(auto input:inputs) {click(*input);TASSERT_TRUE(v.live_input_target_!=Visualizer::LiveInputTarget::None);key(vizgfx::Keyboard::Key::Escape);}
        const std::optional<vizgfx::FloatRect>* tracks[]={&v.last_live_lr_track_,&v.last_live_lrwu_track_,&v.last_live_klb_track_,&v.last_live_klwu_track_};
        for(auto track:tracks) {const auto version=v.liveTrainParamsVersion();click(*track);TASSERT_TRUE(v.liveTrainParamsVersion()>version);events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();}
        // The correspondence is shown beside the residual and retains raw feature inspection.
        json annotation={{"entities",json::array({
            {{"id","left"},{"role","primary_subject"},{"bbox",{.05,.1,.45,.9}},
             {"skeleton",{{"name","custom"},{"points",json::array({{{"name","hand"},{"position",{.3,.5}},{"parent",nullptr},{"entities",{"right"}}}})}}}},
            {{"id","right"},{"role","object"},{"bbox",{.6,.15,.95,.85}}}
        })},{"relations",json::array({{{"subject","left"},{"predicate","holds"},{"object","right"}}})}};
        Visualizer::BlockFrame feature;feature.w=32;feature.h=16;feature.channels=1;feature.label="vae_conv/recon";
        feature.pixels.assign(512,90);feature.tensor_sample_channels=2;
        for(int y=0;y<16;++y)for(int x=0;x<32;++x){feature.tensor_values.push_back(x<16?1.f:0.f);feature.tensor_values.push_back(x<16?0.f:1.f);}
        auto residual=feature;residual.label="vae_conv/err/resdiff_abs";residual.tensor_values.clear();residual.tensor_sample_channels=0;
        std::vector<Visualizer::BlockFrame> annotation_frames={feature,residual};
        AnnotationFeatureOverlay::decorate(annotation_frames,annotation,json{{"bbox_loss_weight",1.f}},32,16);
        v.blocks_tips_visible_limit_=0;v.setLayerBlockImages(annotation_frames);v.renderLayerBlocks();
        TASSERT_TRUE(v.layer_block_images.size()==4 && v.last_block_rects_.size()==4);
        TASSERT_NEAR(v.last_block_rects_[1].position.y,v.last_block_rects_[2].position.y,1.f);
        TASSERT_TRUE(v.last_block_rects_[2].position.x>v.last_block_rects_[1].position.x);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.drawPanelChrome(P::Blocks);v.renderLayerBlocks();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-annotations.png"));
        }
        // F enables the probe on the actual hovered spatial tap, preserving raw data.
        v.focus_target_=Visualizer::FocusTarget::LayerBlock;v.focus_block_index_=0;
        v.zoom_active_=true;v.renderZoomOverlay();
        const auto feature_rect=*v.picker_image_rect_;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(feature_rect.position.x+feature_rect.size.x*.25f),int(feature_rect.position.y+feature_rect.size.y*.5f)}});
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(v.show_feature_probe_);
        TASSERT_TRUE(v.last_picker_text_.find("Section similaire : 256 cellules")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_text_.find("Canal 0 : 1")!=std::string::npos);
        TASSERT_TRUE(v.show_picker_tooltip_ && !v.last_picker_text_.empty());
        key(vizgfx::Keyboard::Key::I);v.renderZoomOverlay();
        TASSERT_TRUE(!v.show_picker_tooltip_ && v.last_picker_text_.empty());
        key(vizgfx::Keyboard::Key::I);v.renderZoomOverlay();
        TASSERT_TRUE(v.show_picker_tooltip_ && !v.last_picker_text_.empty());
        TASSERT_TRUE(v.last_block_frames_[0].tensor_values==feature.tensor_values);
        const auto section=FeatureProbe::analyze(feature.tensor_values,32,16,2,8,8);
        TASSERT_TRUE(section.cells.size()==256 && section.channels[0].first==0);
        TASSERT_TRUE(FeatureProbe::analyze({0,0,0,0},2,1,2,0,0).cells.empty());
        TASSERT_TRUE(FeatureProbe::analyze({1,0,-1,0,1,0},3,1,2,0,0).cells.size()==1);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-features.png"));
        }
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(!v.show_feature_probe_ && v.last_picker_text_.find("Section similaire")==std::string::npos);
        v.blocks_tips_visible_limit_=2;
        Visualizer::BlockFrame sampled;
        sampled.w=sampled.h=2;sampled.channels=1;sampled.label="lumen/blocks/lumen/text/token_embedding/Embedding/vec";
        sampled.pixels={0,0,0,0};sampled.tensor_values={-4.f,12.5f,0.f,8.f};sampled.tensor_indices={10,20,30,40};sampled.tensor_sample_channels=1;
        v.setLayerBlockImages({sampled});
        TASSERT_TRUE(v.has_projection_image && v.has_understanding_image);
        v.focus_target_=Visualizer::FocusTarget::LayerBlock;v.focus_block_index_=0;
        const float sidebar=std::clamp(float(v.window_width)*0.40f,360.f,520.f);
        const float iw=std::max(120.f,float(v.window_width)-56.f-22.f-sidebar);
        const float ih=std::max(120.f,float(v.window_height)-56.f);
        const float box=std::min(iw,ih);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(28.f+(iw-box)*0.5f+box*0.75f),int(28.f+(ih-box)*0.5f+box*0.25f)}});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_text_.find("[20] = 12.5")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_rect_.has_value());
        if (const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-picker.png"));
        }
        TASSERT_NEAR(v.last_picker_rect_->size.x,box/2.f,0.01f);
        TASSERT_NEAR(v.last_picker_rect_->position.x,28.f+(iw-box)*0.5f+box/2.f,0.01f);
        v.zoom_actual_size_=true;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(28.f+(iw-2.f)*0.5f+1.f),int(28.f+(ih-2.f)*0.5f)}});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_rect_.has_value());TASSERT_NEAR(v.last_picker_rect_->size.x,1.f,0.01f);
        v.zoom_actual_size_=false;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{0,0}});v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(!v.last_picker_rect_.has_value());
        // Drag in reverse: inspect all four raw values and keep the capture stable.
        v.zoom_active_=true;
        const auto image_rect=*v.picker_image_rect_;
        auto point=[&](float nx,float ny) {return vizgfx::Vector2i{
            int(image_rect.position.x+image_rect.size.x*nx),int(image_rect.position.y+image_rect.size.y*ny)};};
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,point(0.75f,0.75f)});
        events->events.emplace_back(vizgfx::Event::MouseMoved{point(0.25f,0.25f)});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.picker_selection_active_ && !v.picker_selecting_ && v.picker_selection_rect_);
        const auto& selected=v.picker_selection_summary_;
        TASSERT_TRUE(selected.values.size()==4 && selected.finite==4 && selected.zeros==1);
        TASSERT_NEAR(selected.min,-4.f,1e-6f);TASSERT_NEAR(selected.max,12.5f,1e-6f);
        TASSERT_NEAR(selected.mean,4.125,1e-6f);TASSERT_NEAR(selected.l2,std::sqrt(236.25),1e-6f);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-selection.png"));
        }
        sampled.tensor_values={100.f,100.f,100.f,100.f};v.setLayerBlockImages({sampled});v.renderZoomOverlay();
        TASSERT_NEAR(v.picker_selection_summary_.mean,4.125,1e-6f);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.picker_selection_active_ && v.zoom_active_);
        // New selection samples the latest capture; right click clears it.
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        v.processEvents();TASSERT_NEAR(v.picker_selection_summary_.mean,100.f,1e-6f);
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Right,point(0.25f,0.25f)});
        v.processEvents();TASSERT_TRUE(!v.picker_selection_active_);
        // A small viewport still permits scrolling through all selected values.
        const int saved_height=v.window_height;v.window_height=360;v.renderZoomOverlay();
        const auto small=*v.picker_image_clip_;
        const vizgfx::Vector2i center{int(small.position.x+small.size.x/2),int(small.position.y+small.size.y/2)};
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,center});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,center});
        v.processEvents();v.renderZoomOverlay();TASSERT_TRUE(v.zoom_info_scroll_max_>0.f);
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,center});
        v.processEvents();TASSERT_TRUE(v.zoom_info_scroll_>0.f);
        key(vizgfx::Keyboard::Key::T);v.renderZoomOverlay();
        TASSERT_TRUE(v.picker_selection_rect_ && v.picker_selection_rect_->size.x==1.f);
        TASSERT_NEAR(v.picker_selection_summary_.mean,100.f,1e-6f);
        key(vizgfx::Keyboard::Key::T);
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Right,center});
        v.processEvents();v.window_height=saved_height;v.renderZoomOverlay();
        v.zoom_active_=false;
        // Statistics ignore repeated source indices and count non-finite samples separately.
        const auto exceptional=TensorSelection::analyze({0.f,0.f,std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),-2.f,4.f},{0,0,1,2,3,4},3,2,1,-20,-20,20,20);
        TASSERT_TRUE(exceptional.values.size()==5 && exceptional.finite==3 && exceptional.zeros==1);
        TASSERT_TRUE(exceptional.nan==1 && exceptional.inf==1);
        TASSERT_TRUE(exceptional.positive==1 && exceptional.negative==1);
        TASSERT_NEAR(exceptional.mean,2.0/3.0,1e-6f);
        const auto strongest=TensorSelection::strongest(exceptional,2);
        TASSERT_TRUE(strongest.size()==2 && strongest[0].first==4 && strongest[0].second==4.f);
        TASSERT_TRUE(strongest[1].first==3 && strongest[1].second==-2.f);
        TASSERT_TRUE(TensorSelection::analyze({}, {},0,0,0,0,0,1,1).values.empty());
        MetricsRecord old{};old.loss=3.5f;MetricsRecord next{};next.loss=0.25f;
        v.setMetricsHistory({old,next});TASSERT_TRUE(v.full_loss_history.size()==2 && v.loss_history.size()==2);
        TASSERT_NEAR(v.loss_min_,0.25f,1e-6f);TASSERT_NEAR(v.loss_max_,3.5f,1e-6f);
        MetricsRecord tail{};tail.loss=0.125f;
        v.full_loss_history.push_back(next); // Provisional updateMetrics row.
        v.setMetricsHistory({tail},2);
        TASSERT_TRUE(v.full_loss_history.size()==3 && v.loss_history.size()==3);
        TASSERT_NEAR(v.full_loss_history.back().loss,0.125f,1e-6f);
        v.setMetricsHistory({},3);
        TASSERT_TRUE(v.full_loss_history.size()==3 && v.loss_history.size()==3);
        TASSERT_NEAR(v.loss_min_,0.125f,1e-6f);
        auto gradient_config = std::make_shared<LiveModelConfig>();
        v.setRuntimeConfiguration(gradient_config);
        TASSERT_TRUE(gradient_config->gradient_diagnostics.requested());
        gradient_config->gradient_diagnostics.record(1, false);
        gradient_config->gradient_diagnostics.record(9, true);
        v.updateMetrics(1, 2, .1f, .001f);
        TASSERT_TRUE(v.current_gradient_diagnostics_.samples == 2);
        TASSERT_NEAR(v.current_gradient_diagnostics_.clip_rate, 50, 1e-6f);
        TASSERT_NEAR(v.current_gradient_diagnostics_.grad_norm_median, 5, 1e-6f);
        TASSERT_NEAR(v.current_gradient_diagnostics_.grad_norm_p95, 9, 1e-6f);
        TASSERT_NEAR(v.current_gradient_diagnostics_.grad_norm_unclipped_max, 9, 1e-6f);
        v.renderMetrics();
        if (const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear(vizgfx::Color(12,14,18));redraw();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture) + ".gradients.png"));
        }
        v.setLayerBlockImages({});TASSERT_TRUE(!v.has_projection_image && !v.has_understanding_image);
        // Input preview selection must choose the image subsequently enlarged by Z.
        v.setDatasetImage(std::vector<uint8_t>(8*4*3,128),8,4,3,"input");
        v.renderContextImages();click(v.last_dataset_rect_);
        TASSERT_TRUE(v.focus_target_==Visualizer::FocusTarget::Dataset);
        v.setDetections({{.1f,.1f,.9f,.9f,.9f,2,"main"},{0,0,1,1,.1f,3,"rejected"}},Visualizer::DetectionTarget::Dataset);
        v.zoom_active_=true;v.renderZoomOverlay();
        const auto detected_rect=*v.picker_image_rect_;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(detected_rect.position.x+detected_rect.size.x*.5f),int(detected_rect.position.y+detected_rect.size.y*.5f)}});
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_text_.find("Detecte : main 90.0%")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_text_.find("rejected")==std::string::npos);
        v.setDetections({{.1f,.1f,.9f,.9f,.9f,2,"wrong target"}},Visualizer::DetectionTarget::Output);
        v.renderZoomOverlay();TASSERT_TRUE(v.last_picker_text_.find("wrong target")==std::string::npos);
        key(vizgfx::Keyboard::Key::F);v.zoom_active_=false;
        key(vizgfx::Keyboard::Key::Z);click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());key(vizgfx::Keyboard::Key::Escape);
        key(vizgfx::Keyboard::Key::H);click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());key(vizgfx::Keyboard::Key::H);
        auto configuration=std::make_shared<LiveModelConfig>();
        configuration->publish({{"learning_rate",0.001,LiveModelConfig::Owner::Training,0,1,{},"",std::nullopt}});
        v.setRuntimeConfiguration(configuration);
        key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
        TASSERT_TRUE(v.config_editor_.visible && !v.config_row_rects_.empty());
        click(v.config_row_rects_[0].first);TASSERT_TRUE(v.config_editor_.editing);
        v.config_editor_.buffer.clear();
        for(char32_t c:std::u32string(U"0.02")) events->events.emplace_back(vizgfx::Event::TextEntered{c});
        v.processEvents();key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(configuration->snapshot()[0].pending.has_value());
        TASSERT_NEAR(configuration->snapshot()[0].pending->get<float>(),0.02f,1e-6f);
        key(vizgfx::Keyboard::Key::C);
        configuration->publish({{"bbox_loss_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
        key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
        const auto rows=configuration->snapshot();
        for(const auto& row:v.config_row_rects_) if(rows[row.second].key=="bbox_loss_weight") click(row.first);
        TASSERT_TRUE(v.config_editor_.editing_key=="bbox_loss_weight");
        v.config_editor_.buffer.clear();events->events.emplace_back(vizgfx::Event::TextEntered{U'2'});
        v.processEvents();key(vizgfx::Keyboard::Key::Enter);
        bool bbox_requested=false;
        for(const auto& row:configuration->snapshot()) if(row.key=="bbox_loss_weight") bbox_requested=row.pending && *row.pending==2;
        TASSERT_TRUE(bbox_requested);key(vizgfx::Keyboard::Key::C);
        for(const auto* role_key : {"relation_primary_weight","relation_secondary_weight","relation_other_weight"}) {
            configuration->publish({{role_key,1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
            key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
            const auto role_rows=configuration->snapshot();
            for(const auto& row:v.config_row_rects_) if(role_rows[row.second].key==role_key) click(row.first);
            TASSERT_TRUE(v.config_editor_.editing_key==role_key);
            v.config_editor_.buffer.clear();events->events.emplace_back(vizgfx::Event::TextEntered{U'5'});
            v.processEvents();key(vizgfx::Keyboard::Key::Enter);
            bool requested=false;
            for(const auto& row:configuration->snapshot()) if(row.key==role_key) requested=row.pending && *row.pending==5;
            TASSERT_TRUE(requested);key(vizgfx::Keyboard::Key::C);
        }
        // Panel title and resize handle initiate and release their intended drag.
        click(vizgfx::FloatRect{v.panels_[size_t(P::Blocks)].pos,{40,20}});
        TASSERT_TRUE(v.dragging_panel_);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        TASSERT_TRUE(!v.dragging_panel_);
        v.panels_[size_t(P::Generated)].visible=false;
        click(v.panelResizeHandleRect(P::Blocks));TASSERT_TRUE(v.resizing_panel_);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        TASSERT_TRUE(!v.resizing_panel_);v.panels_[size_t(P::Generated)].visible=true;
        // A covered button cannot change the underlying panel.
        auto& graph=v.panels_[size_t(P::Graph)];const auto old_graph=graph;
        graph.pos=v.last_stop_button_rect_->position;graph.size={200,100};
        click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());graph=old_graph;
        click(v.last_stop_button_rect_);TASSERT_TRUE(v.consumeStopTrainingRequested());TASSERT_TRUE(!v.consumeStopTrainingRequested());
        for(P p:{P::Context,P::Output,P::Blocks,P::Generated,P::Training,P::Metrics,P::Graph}) {
            click(v.panelCloseButtonRect(p));TASSERT_TRUE(!v.isPanelVisible(p));
        }
        // Semantic sections ignore the model prefix and the word "output" in time projections.
        using Mimir::Viz::lumenTipSection;
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/text/token_embedding/Embedding") == "Text / Cond");
        TASSERT_TRUE(lumenTipSection("lumen/blocks/lumen/time/output/Linear") == "Diffusion");
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/dit/block1/cross_attention/CrossAttention") == "Diffusion");
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/output/z/Identity | reconstructed image") == "Outputs");
        TASSERT_TRUE(lumenTipSection("diffusion/compare/C_model_decode_z0_pred | timestep=20") == "Validation");
        TASSERT_TRUE(lumenTipSection("resdiff_abs") == "Validation");
        TASSERT_TRUE(lumenTipSection("diffusion/reference") == "Validation");
        TASSERT_TRUE(lumenTipSection("diffusion_out") == "Validation");
        TASSERT_TRUE(lumenTipSection("vae_conv/err/resdiff_abs").empty());
        const auto blocks_before_sections = v.panels_[size_t(P::Blocks)];
        const bool hide_norm_before_sections = v.hide_normalisation_blocks;
        v.hide_normalisation_blocks = false;
        v.panels_[size_t(P::Blocks)].size = {760, 1100};
        v.blocks_tips_visible_limit_ = 0;
        v.blocks_scroll_y_ = v.blocks_scroll_max_ = 0;
        std::vector<Visualizer::BlockFrame> section_frames;
        for (const auto* label : {
            "lumen_diffusion/blocks/lumen/text/token_embedding/Embedding",
            "lumen_diffusion/blocks/lumen/time/output/Linear",
            "lumen_diffusion/blocks/lumen/output/z/Identity | role=model_output tensor=z",
            "diffusion/compare/C_model_decode_z0_pred",
            "resdiff_abs",
            "lumen_diffusion/blocks/lumen/text/final_norm/LayerNorm",
            "lumen_diffusion/blocks/lumen/dit/block1/cross_attention/CrossAttention",
            "diffusion/reference"}) {
            Visualizer::BlockFrame frame;
            frame.w = frame.h = 2;
            frame.channels = 3;
            frame.pixels.assign(12, 42);
            frame.label = label;
            section_frames.push_back(std::move(frame));
        }
        v.setLayerBlockImages(section_frames);
        v.renderLayerBlocks();
        TASSERT_TRUE(v.last_block_rects_.size() == section_frames.size());
        const auto y_of = [&](size_t i) { return v.last_block_rects_[i].position.y; };
        TASSERT_NEAR(y_of(0), y_of(5), 1.f);
        TASSERT_NEAR(y_of(1), y_of(6), 1.f);
        TASSERT_NEAR(y_of(3), y_of(4), 1.f);
        TASSERT_NEAR(y_of(3), y_of(7), 1.f);
        TASSERT_TRUE(y_of(0) < y_of(1) && y_of(1) < y_of(2) && y_of(2) < y_of(3));
        v.panels_[size_t(P::Blocks)] = blocks_before_sections;
        v.hide_normalisation_blocks = hide_norm_before_sections;
        // Only the explicit reconstructed z may own [ST], even when a time
        // projection or an older decoded comparison comes later in the list.
        Visualizer::BlockFrame reconstructed_z;
        reconstructed_z.w = reconstructed_z.h = 2;
        reconstructed_z.channels = 3;
        reconstructed_z.pixels.assign(12, 42);
        reconstructed_z.label = "lumen_diffusion/blocks/lumen/output/z/Identity | role=model_output tensor=z";
        auto time_output = reconstructed_z;
        time_output.pixels.assign(12, 99);
        time_output.label = "lumen_diffusion/blocks/lumen/time/output/Linear";
        auto comparison_output = time_output;
        comparison_output.label = "diffusion_out";
        v.setLayerBlockImages({reconstructed_z, time_output, comparison_output});
        TASSERT_TRUE(v.output_image_.pixels == reconstructed_z.pixels);
        TASSERT_TRUE(v.output_thumb_label_ == reconstructed_z.label);
        v.setLayerBlockImages({comparison_output});
        TASSERT_TRUE(v.output_image_.pixels == reconstructed_z.pixels);
        TASSERT_TRUE(v.output_thumb_label_ == reconstructed_z.label);
        // Validation pairs outlive graph/training updates and the ordinary image cap.
        v.clearImages();
        v.image_grid_cols=1;v.image_grid_rows=1;
        v.addGeneratedImage(std::vector<uint8_t>(12,7),2,2,3,"completed training step");
        const auto completed_output=v.output_image_.pixels;
        v.beginValidationImages();
        for (int sample=0;sample<3;++sample) {
            v.addGeneratedImage(std::vector<uint8_t>(12,10+sample),2,2,3,"VAL target " + std::to_string(sample),true);
            v.addGeneratedImage(std::vector<uint8_t>(12,20+sample),2,2,3,"VAL recon " + std::to_string(sample),true);
        }
        TASSERT_TRUE(v.output_image_.pixels==completed_output);
        for (int step=0;step<8;++step) {
            v.addGeneratedImage(std::vector<uint8_t>(12,90),2,2,3,"training/graph " + std::to_string(step));
            v.setLayerBlockImages({sampled});
        }
        TASSERT_TRUE(v.validation_image_count_==6 && v.generated_images.size()==7);
        for (int sample=0;sample<3;++sample) {
            TASSERT_TRUE(v.generated_images[2*sample].pixels.front()==10+sample);
            TASSERT_TRUE(v.generated_images[2*sample+1].pixels.front()==20+sample);
        }
        v.rebuildAllTextures();
        TASSERT_TRUE(v.validation_image_count_==6 && v.generated_images[0].pixels.front()==10);
        v.renderGeneratedImages();TASSERT_TRUE(v.last_generated_indices_.size()==7);
        v.beginValidationImages();TASSERT_TRUE(v.validation_image_count_==0 && v.generated_images.size()==1);
        v.addGeneratedImage(std::vector<uint8_t>(12,30),2,2,3,"VAL target next",true);
        v.addGeneratedImage(std::vector<uint8_t>(12,40),2,2,3,"VAL recon next",true);
        TASSERT_TRUE(v.generated_images[0].pixels.front()==30 && v.generated_images[1].pixels.front()==40);
        v.clearImages();TASSERT_TRUE(v.generated_images.empty() && v.validation_image_count_==0);

        // Last drawn script button wins when controls overlap.
        v.configureScript({{"controls",{{{"id","under"},{"x",10},{"y",10}},{{"id","above"},{"x",10},{"y",10}}}}});
        click(vizgfx::FloatRect{{10,10},{150,36}});
        const auto log=v.pollScriptEvents();TASSERT_TRUE(!log.empty() && log.back().value("id",std::string())=="above");
        // Typing a prompt must not open the global graph/config/help views.
        v.configureScript({{"events",true},{"capture_keys",true}});
        key(vizgfx::Keyboard::Key::N);
        key(vizgfx::Keyboard::Key::C);
        key(vizgfx::Keyboard::Key::H);
        events->events.emplace_back(vizgfx::Event::TextEntered{233});
        v.processEvents();
        TASSERT_TRUE(!v.show_model_graph_ && !v.config_editor_.visible && !v.show_help_overlay_);
        int captured_keys=0,captured_text=0;
        for(const auto& event:v.pollScriptEvents()) {
            captured_keys+=event.value("type",std::string())=="key";
            captured_text+=event.value("type",std::string())=="text" && event.value("unicode",0)==233;
        }
        TASSERT_TRUE(captured_keys==3 && captured_text==1);
        v.configureScript({{"events",false},{"capture_keys",false}});
        v.processEvents();v.pollScriptEvents();
        // Help keeps fixed readable text, adapts columns and captures its input.
        key(vizgfx::Keyboard::Key::H);TASSERT_TRUE(v.show_help_overlay_);
        v.renderHelpOverlay();TASSERT_TRUE(v.help_columns_==2 && !v.help_row_rects_.empty());
        auto verify_help=[&] {
            TASSERT_TRUE(v.help_body_rect_->position.x>=0.f && v.help_body_rect_->position.y>=0.f);
            TASSERT_TRUE(v.help_body_rect_->position.x+v.help_body_rect_->size.x<=v.window_width);
            TASSERT_TRUE(v.help_body_rect_->position.y+v.help_body_rect_->size.y<=v.window_height);
            for(size_t i=0;i<v.help_row_rects_.size();++i) for(size_t j=i+1;j<v.help_row_rects_.size();++j)
                TASSERT_TRUE(!v.help_row_rects_[i].findIntersection(v.help_row_rects_[j]));
        };
        verify_help();
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderHelpOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-help-wide.png"));
        }
        const auto focus=v.focus_target_;
        key(vizgfx::Keyboard::Key::F1);key(vizgfx::Keyboard::Key::C);
        TASSERT_TRUE(v.focus_target_==focus && !v.config_editor_.visible);
        events->events.emplace_back(vizgfx::Event::Resized{{520,400}});v.processEvents();v.renderHelpOverlay();
        TASSERT_TRUE(v.help_columns_==1 && v.help_scroll_max_>0.f);verify_help();
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderHelpOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-help-narrow.png"));
        }
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,{260,200}});
        v.processEvents();TASSERT_TRUE(v.help_scroll_y_>0.f);
        const float scroll=v.help_scroll_y_;key(vizgfx::Keyboard::Key::Down);TASSERT_TRUE(v.help_scroll_y_>scroll);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.show_help_overlay_);
        key(vizgfx::Keyboard::Key::H);TASSERT_TRUE(v.show_help_overlay_ && v.help_scroll_y_==0.f);
        v.renderHelpOverlay();click(v.help_close_rect_);TASSERT_TRUE(!v.show_help_overlay_);
        events->events.emplace_back(vizgfx::Event::Resized{{1000,800}});v.processEvents();
        // The graph is modal and receives immutable metadata from the model.
        auto graph_config=std::atomic_load(&v.runtime_config_);
        graph_config->publishModelGraph(json{{"mode","desactive"},{"planned",false},
            {"nodes",json::array({json{{"index",0},{"name","test"},{"type","Linear"},
                {"lane",0},{"depth",0},{"runtime","non planifie"},{"parameters",4},
                {"inputs",json::array({"x"})},{"output","y"},{"capability","inconnue"}}})},
            {"edges",json::array()},{"external_inputs",json::array()}});
        auto graph_sample=*graph_config->modelGraph();
        auto second=graph_sample["nodes"][0];
        second["index"]=1;second["name"]="activation_with_a_long_descriptive_name";second["type"]="ReLU";
        second["depth"]=1;second["inputs"]=json::array({"y"});second["output"]="z";
        graph_sample["nodes"].push_back(second);
        graph_sample["edges"].push_back(json{{"from",0},{"to",1},{"tensor","y"}});
        graph_sample["fusion_groups"]=json::array({json{{"id",0},{"producer",0},
            {"layers",json::array({0,1})},{"children",json::array()},{"parent",nullptr}}});
        graph_config->publishModelGraph(std::move(graph_sample));
        key(vizgfx::Keyboard::Key::N);
        TASSERT_TRUE(v.show_model_graph_ && graph_config->modelGraphRequested());
        v.setRuntimeConfiguration(graph_config);
        TASSERT_TRUE(graph_config->modelGraphRequested());
        v.renderModelGraphOverlay();
        TASSERT_TRUE(v.model_graph_hits_.size()==2);
        click(v.model_graph_hits_[1].first);
        TASSERT_TRUE(v.model_graph_selected_==1);
        v.renderModelGraphOverlay();
        key(vizgfx::Keyboard::Key::X);TASSERT_TRUE(v.model_graph_png_requested_);
        v.model_graph_png_requested_=false;events->display();
        TASSERT_TRUE(v.saveModelGraphPng());
        const auto graph_layout=Mimir::Viz::modelGraphTree(*graph_config->modelGraph());
        TASSERT_TRUE(graph_layout.fusion_blocks.size()==1);
        vizgfx::Image graph_image;
        TASSERT_TRUE(graph_image.loadFromFile(v.last_model_graph_path_));
        TASSERT_TRUE(graph_image.getSize().x>=graph_layout.width()+32.f);
        TASSERT_TRUE(graph_image.getSize().y>=graph_layout.height()+112.f);
        TASSERT_TRUE(graph_image.getSize().x>events->getSize().x);
        const auto fusion_block=graph_layout.fusion_blocks.front();
        const auto fusion_pixel=graph_image.getPixel({unsigned(fusion_block.x+4.f),
            unsigned(fusion_block.y+104.f)});
        TASSERT_TRUE(fusion_pixel.r!=16 || fusion_pixel.g!=23 || fusion_pixel.b!=32);
        json large_graph=*graph_config->modelGraph();
        large_graph["nodes"]=json::array();
        large_graph["edges"]=json::array();
        for(size_t i=0;i<55;++i) {
            large_graph["nodes"].push_back({{"index",i},{"name","linear layer "+std::to_string(i)},
                {"type","Linear"},{"depth",i},{"lane",0},{"runtime","CPU"},{"parameters",4},
                {"inputs",json::array()},{"output","tensor_"+std::to_string(i)},
                {"capability","connue"},{"graph_fused",false},{"kernel_fused",false},{"skip",false}});
            if(i>0) large_graph["edges"].push_back({{"from",i-1},{"to",i},
                {"tensor","tensor_"+std::to_string(i-1)}});
        }
        graph_config->publishModelGraph(std::move(large_graph));
        v.renderModelGraphOverlay();
        const auto large_layout=Mimir::Viz::modelGraphTree(*graph_config->modelGraph());
        TASSERT_TRUE(large_layout.width()+32.f>16384.f);
        TASSERT_TRUE(v.saveModelGraphPng());
        TASSERT_TRUE(graph_image.loadFromFile(v.last_model_graph_path_));
        TASSERT_TRUE(graph_image.getSize().x>=large_layout.width()+32.f);
        TASSERT_TRUE(graph_image.getSize().x>16384);
        if(const char* capture=std::getenv("MIMIR_GRAPH_TEST_CAPTURE")) {
            events->display();TASSERT_TRUE(events->captureImage().saveToFile(capture));
        }
        key(vizgfx::Keyboard::Key::Right);
        TASSERT_TRUE(v.model_graph_x_>0.f);
        key(vizgfx::Keyboard::Key::R);TASSERT_TRUE(v.model_graph_x_==0.f && v.model_graph_y_==0.f);
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.show_model_graph_ && !graph_config->modelGraphRequested());
        key(vizgfx::Keyboard::Key::E);TASSERT_TRUE(v.snapshot_dialog_visible_);
        key(vizgfx::Keyboard::Key::Space);TASSERT_TRUE(!v.snapshot_include_window_);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.snapshot_dialog_visible_);
        v.snapshot_include_window_=true;v.snapshot_include_layer_tips_=false;
        v.snapshot_include_projection_=v.snapshot_include_output_=false;
        TASSERT_TRUE(v.saveSnapshotPng());
        vizgfx::Image snapshot;
        TASSERT_TRUE(snapshot.loadFromFile(v.last_snapshot_path_));
        TASSERT_TRUE(snapshot.getSize().x==events->getSize().x && snapshot.getSize().y==events->getSize().y);
        v.shutdown();
    }
};
#endif
int main(int argc, char** argv) {
#ifndef ENABLE_VIZ
    return 77;
#else
#ifdef MIMIR_VIZ_BACKEND_SFML
    if(!std::getenv("DISPLAY")) return 77;
#endif
    const auto dir=fs::temp_directory_path() / ("mimir_viz_events_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto previous=fs::current_path();fs::create_directories(dir);fs::current_path(dir);
    struct Cleanup {fs::path previous,dir;~Cleanup(){fs::current_path(previous);fs::remove_all(dir);}} cleanup{previous,dir};
    try {
        if (argc>1 && std::string(argv[1])=="--kl-only") VisualizerEventTestAccess::runKlControls();
        else if (argc>1 && std::string(argv[1])=="--scroll-padding-only") VisualizerEventTestAccess::runScrollPadding();
        else if (argc>1 && std::string(argv[1])=="--projection-only") VisualizerEventTestAccess::runGenericProjection();
        else if (argc>1 && std::string(argv[1])=="--export-3d-only") VisualizerEventTestAccess::runExport3D();
        else if (argc>1 && std::string(argv[1])=="--panel-manager-only") VisualizerEventTestAccess::runPanelManager();
        else VisualizerEventTestAccess::run();
    }
    catch(const std::exception& error) {std::cerr << error.what() << "\n"; return 1;}
    std::cout << "PASS VIZ button events, modal isolation and overlap routing\n";
#endif
}
