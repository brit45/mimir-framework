#include "Ipc.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <unistd.h>
using namespace mimir::viz;
namespace {
std::mutex frameMutex;
Frame latest;
bool updated=false;
std::atomic<bool> ended{false};
void emitEvent(Event e) { if(!transfer(STDOUT_FILENO,&e,sizeof(e),true)) ended=true; }
void readFrames() {
    while(!ended) {
        FrameHeader h;
        if(!transfer(STDIN_FILENO,&h,sizeof(h),false) || !validSize(h.width,h.height)) break;
        Frame f; f.width=h.width; f.height=h.height; f.cursor=h.cursor; f.rgba.resize(size_t(h.width)*h.height*4);
        if(!transfer(STDIN_FILENO,f.rgba.data(),f.rgba.size(),false)) break;
        std::lock_guard<std::mutex> lock(frameMutex); latest=std::move(f); updated=true;
    }
    ended=true;
}
bool takeFrame(Frame& f) { std::lock_guard<std::mutex> lock(frameMutex); if(!updated) return false; f=std::move(latest); updated=false; return true; }
}
#ifdef MIMIR_VIZ_BACKEND_QT
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QImage>
#include <QTimer>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QCloseEvent>
#include <QFocusEvent>
namespace {
int keyCode(int k) {
    if(k>=Qt::Key_A && k<=Qt::Key_Z) return k;
    if(k>=Qt::Key_0 && k<=Qt::Key_9) return k;
    if(k>=Qt::Key_F1 && k<=Qt::Key_F12) return 281+k-Qt::Key_F1;
    switch(k) { case Qt::Key_Escape:return 256; case Qt::Key_Return:case Qt::Key_Enter:return 257;
      case Qt::Key_Tab:return 258; case Qt::Key_Backspace:return 259;
      case Qt::Key_Left:return 260; case Qt::Key_Right:return 261; case Qt::Key_Up:return 262; case Qt::Key_Down:return 263; default:return 0; }
}
int mods(Qt::KeyboardModifiers m) { return (m.testFlag(Qt::ControlModifier)?1:0)|(m.testFlag(Qt::ShiftModifier)?2:0)|(m.testFlag(Qt::AltModifier)?4:0)|(m.testFlag(Qt::MetaModifier)?8:0); }
class Canvas final : public QWidget {
    Frame frame_; QTimer timer_;
    void mouse(QMouseEvent* e,int type) {
        emitEvent(Event{type,int(e->position().x()),int(e->position().y()),e->button()==Qt::RightButton?1:e->button()==Qt::MiddleButton?2:0});
    }
public:
    Canvas(const char* title) {
        setWindowTitle(QString::fromUtf8(title)); setMouseTracking(true); setFocusPolicy(Qt::StrongFocus);
        resize(800,600);
        connect(&timer_,&QTimer::timeout,this,[this] {
            if(ended) { QApplication::quit(); return; }
            if(takeFrame(frame_)) {
                setFixedSize(int(frame_.width),int(frame_.height));
                const Qt::CursorShape cursors[]={Qt::ArrowCursor,Qt::PointingHandCursor,Qt::CrossCursor,Qt::SizeFDiagCursor};
                setCursor(cursors[frame_.cursor<4?frame_.cursor:0]); update();
            }
        }); timer_.start(16);
    }
protected:
    bool focusNextPrevChild(bool) override { return false; }
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        if(!frame_.rgba.empty()) p.drawImage(0,0,QImage(frame_.rgba.data(),frame_.width,frame_.height,QImage::Format_RGBA8888));
    }
    void mouseMoveEvent(QMouseEvent* e) override { mouse(e,Move); }
    void mousePressEvent(QMouseEvent* e) override { mouse(e,Down); }
    void mouseReleaseEvent(QMouseEvent* e) override { mouse(e,Up); }
    void wheelEvent(QWheelEvent* e) override { emitEvent(Event{Wheel,int(e->position().x()),int(e->position().y()),0,0,float(e->angleDelta().y())/120.f}); }
    void keyPressEvent(QKeyEvent* e) override {
        emitEvent(Event{KeyDown,0,0,keyCode(e->key()),mods(e->modifiers())});
        for(auto c:e->text().toUcs4()) emitEvent(Event{Text,0,0,int(c)});
    }
    void keyReleaseEvent(QKeyEvent* e) override { if(!e->isAutoRepeat()) emitEvent(Event{KeyUp,0,0,keyCode(e->key()),mods(e->modifiers())}); }
    void focusOutEvent(QFocusEvent*) override { emitEvent(Event{Blur}); }
    void closeEvent(QCloseEvent* e) override { emitEvent(Event{Close}); e->accept(); }
};
}
int main(int argc,char** argv) {
    QApplication app(argc,argv); Canvas canvas(argc>1?argv[1]:"Mímir Viz"); canvas.show();
    emitEvent(Event{100}); std::thread reader(readFrames);
    int result=app.exec(); ended=true; shutdown(STDIN_FILENO,SHUT_RDWR); reader.join(); return result;
}
#else
#include <gtk/gtk.h>
namespace {
Frame frame;
GtkWidget* window=nullptr;
GtkWidget* canvas=nullptr;
int keyCode(guint k) {
    k=gdk_keyval_to_upper(k);
    if((k>='A'&&k<='Z')||(k>='0'&&k<='9')) return int(k);
    if(k>=GDK_KEY_KP_0 && k<=GDK_KEY_KP_9) return '0'+k-GDK_KEY_KP_0;
    if(k>=GDK_KEY_F1&&k<=GDK_KEY_F12) return 281+k-GDK_KEY_F1;
    switch(k) {case GDK_KEY_Escape:return 256;case GDK_KEY_Return:case GDK_KEY_KP_Enter:return 257;case GDK_KEY_Tab:case GDK_KEY_ISO_Left_Tab:return 258;
    case GDK_KEY_BackSpace:return 259;case GDK_KEY_Left:return 260;case GDK_KEY_Right:return 261;case GDK_KEY_Up:return 262;case GDK_KEY_Down:return 263;default:return 0;}
}
gboolean draw(GtkWidget*,cairo_t* cr,gpointer) {
    if(frame.rgba.empty()) return FALSE;
    auto* pix=gdk_pixbuf_new_from_data(frame.rgba.data(),GDK_COLORSPACE_RGB,TRUE,8,frame.width,frame.height,frame.width*4,nullptr,nullptr);
    gdk_cairo_set_source_pixbuf(cr,pix,0,0); cairo_paint(cr); g_object_unref(pix); return FALSE;
}
gboolean event(GtkWidget*,GdkEvent* e,gpointer) {
    switch(e->type) {
    case GDK_MOTION_NOTIFY:emitEvent(Event{Move,int(e->motion.x),int(e->motion.y)});return TRUE;
    case GDK_BUTTON_PRESS:case GDK_BUTTON_RELEASE:
        if(e->button.button<=3) emitEvent(Event{e->type==GDK_BUTTON_PRESS?Down:Up,int(e->button.x),int(e->button.y),e->button.button==3?1:e->button.button==2?2:0});
        return TRUE;
    case GDK_SCROLL: {
        float delta=e->scroll.direction==GDK_SCROLL_UP?1:e->scroll.direction==GDK_SCROLL_DOWN?-1:-e->scroll.delta_y;
        emitEvent(Event{Wheel,int(e->scroll.x),int(e->scroll.y),0,0,delta});return TRUE;
    }
    case GDK_KEY_PRESS:case GDK_KEY_RELEASE: {
        int m=(e->key.state&GDK_CONTROL_MASK?1:0)|(e->key.state&GDK_SHIFT_MASK?2:0)|(e->key.state&GDK_MOD1_MASK?4:0)|(e->key.state&GDK_SUPER_MASK?8:0);
        emitEvent(Event{e->type==GDK_KEY_PRESS?KeyDown:KeyUp,0,0,keyCode(e->key.keyval),m});
        if(e->type==GDK_KEY_PRESS) { auto c=gdk_keyval_to_unicode(e->key.keyval); if(e->key.keyval==GDK_KEY_BackSpace)c=8; if(c)emitEvent(Event{Text,0,0,int(c)}); } return TRUE;
    }
    case GDK_FOCUS_CHANGE:if(!e->focus_change.in)emitEvent(Event{Blur});return FALSE;
    default:return FALSE;
    }
}
gboolean tick(gpointer) {
    if(ended) { gtk_main_quit(); return G_SOURCE_REMOVE; }
    if(takeFrame(frame)) {
        gtk_widget_set_size_request(canvas,frame.width,frame.height); gtk_window_resize(GTK_WINDOW(window),frame.width,frame.height);
        const char* cursors[]={"default","pointer","crosshair","nwse-resize"};
        auto* cursor=gdk_cursor_new_from_name(gtk_widget_get_display(canvas),cursors[frame.cursor<4?frame.cursor:0]);
        gdk_window_set_cursor(gtk_widget_get_window(canvas),cursor); if(cursor)g_object_unref(cursor);
        gtk_widget_queue_draw(canvas);
    }
    return G_SOURCE_CONTINUE;
}
}
int main(int argc,char** argv) {
    if(!gtk_init_check(&argc,&argv)) return 1;
    window=gtk_window_new(GTK_WINDOW_TOPLEVEL); gtk_window_set_title(GTK_WINDOW(window),argc>1?argv[1]:"Mímir Viz");
    gtk_window_set_resizable(GTK_WINDOW(window),FALSE);
    canvas=gtk_drawing_area_new(); gtk_container_add(GTK_CONTAINER(window),canvas);
    gtk_widget_add_events(canvas,GDK_POINTER_MOTION_MASK|GDK_BUTTON_PRESS_MASK|GDK_BUTTON_RELEASE_MASK|GDK_SCROLL_MASK|GDK_SMOOTH_SCROLL_MASK);
    g_signal_connect(canvas,"draw",G_CALLBACK(draw),nullptr); g_signal_connect(canvas,"event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"key-press-event",G_CALLBACK(event),nullptr); g_signal_connect(window,"key-release-event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"focus-out-event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"destroy",G_CALLBACK(+[](GtkWidget*,gpointer){emitEvent(Event{Close});gtk_main_quit();}),nullptr);
    gtk_widget_show_all(window); g_timeout_add(16,tick,nullptr);
    emitEvent(Event{100}); std::thread reader(readFrames); gtk_main(); ended=true;
    shutdown(STDIN_FILENO,SHUT_RDWR); reader.join(); return 0;
}
#endif
