// Explicit graphical smoke target, never part of headless CTest runs.
#include "VizWindow.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        auto window = createVizWindow({320,240}, "Mimir Viz backend smoke", 30);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(argc>1?std::stoi(argv[1]):5);
        bool checked=false;
        while(window->isOpen() && std::chrono::steady_clock::now()<deadline) {
            while(auto e=window->pollEvent()) {
                if(e->is<vizgfx::Event::Closed>()) {std::cout<<"CLOSE"<<std::endl;window->close();}
                if(auto* k=e->getIf<vizgfx::Event::KeyPressed>()) {
                    std::cout<<"KEY "<<int(k->code)<<" "<<k->control<<" "<<k->shift<<std::endl;
                    if(k->code==vizgfx::Keyboard::Key::R)window->setSize({400,280});
                }
                if(auto* t=e->getIf<vizgfx::Event::TextEntered>())std::cout<<"TEXT "<<uint32_t(t->unicode)<<std::endl;
                if(auto* m=e->getIf<vizgfx::Event::MouseButtonPressed>())std::cout<<"DOWN "<<m->position.x<<" "<<m->position.y<<std::endl;
                if(auto* m=e->getIf<vizgfx::Event::MouseButtonReleased>())std::cout<<"UP "<<m->position.x<<" "<<m->position.y<<std::endl;
                if(auto* w=e->getIf<vizgfx::Event::MouseWheelScrolled>())std::cout<<"WHEEL "<<w->delta<<std::endl;
                if(auto* r=e->getIf<vizgfx::Event::Resized>())window->setView(vizgfx::View(vizgfx::FloatRect({0,0},{float(r->size.x),float(r->size.y)})));
            }
            if(!window->isOpen())break;
            window->clear(vizgfx::Color(17,34,51));
            vizgfx::RectangleShape rect({70,50});rect.setPosition({10,20});rect.setFillColor(vizgfx::Color(231,42,63));window->draw(rect);
            window->display();
            if(!checked) {
                auto image=window->captureImage();
                if(image.getPixel({0,0})!=vizgfx::Color(17,34,51) || image.getPixel({20,30})!=vizgfx::Color(231,42,63))throw std::runtime_error("Framebuffer pixel mismatch");
                if(argc>2 && !image.saveToFile(argv[2]))throw std::runtime_error("Cannot save smoke PNG");
                std::cout<<"PIXELS_OK"<<std::endl;checked=true;
            }
        }
        return checked?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
