#include "Host.hpp"
#include "Ipc.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <atomic>
#include <cmath>
#include <charconv>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace mimir::viz {
namespace {
const char* page=R"HTML(<!doctype html><meta charset="utf-8"><title>Mímir Viz</title>
<style>body{margin:0;background:#151515;color:white;font:14px sans-serif}canvas{display:block;outline:none}button{margin:4px}</style>
<button id="stop">Fermer la Viz</button><span id="status">Connexion…</span><canvas tabindex="0"></canvas>
<script>
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),status=document.querySelector('#status');
const base=location.pathname.replace(/\/$/,'');let active=true,chain=Promise.resolve(),x=0,y=0;
function send(t,v=0,m=0,d=0){chain=chain.then(async()=>{const r=await fetch(`${base}/event?t=${t}&x=${x}&y=${y}&v=${v}&m=${m}&d=${d}`,{method:'POST'});if(!r.ok)throw Error('événement refusé');}).catch(()=>{status.textContent='Connexion perdue';active=false;});}
function point(e){const r=canvas.getBoundingClientRect();x=Math.round((e.clientX-r.left)*canvas.width/r.width);y=Math.round((e.clientY-r.top)*canvas.height/r.height);}
canvas.onpointermove=e=>{point(e);send(2)};
canvas.onpointerdown=e=>{if(e.button>2)return;canvas.focus();canvas.setPointerCapture(e.pointerId);point(e);send(3,e.button===2?1:e.button===1?2:0);e.preventDefault()};
canvas.onpointerup=e=>{point(e);send(4,e.button===2?1:e.button===1?2:0);if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId)};
canvas.onpointercancel=()=>send(4);canvas.oncontextmenu=e=>e.preventDefault();
canvas.addEventListener('wheel',e=>{point(e);send(5,0,0,-e.deltaY/(e.deltaMode===0?100:3));e.preventDefault()},{passive:false});
function key(e){const k=e.key;const map={Escape:256,Enter:257,Tab:258,Backspace:259,ArrowLeft:260,ArrowRight:261,ArrowUp:262,ArrowDown:263};if(map[k])return map[k];if(/^F([1-9]|1[012])$/.test(k))return 280+Number(k.slice(1));if(/^[a-z0-9]$/i.test(k))return k.toUpperCase().charCodeAt(0);return 0;}
canvas.onkeydown=e=>{send(6,key(e),(e.ctrlKey?1:0)|(e.shiftKey?2:0)|(e.altKey?4:0)|(e.metaKey?8:0));if(e.key==='Backspace')send(8,8);else if(!e.ctrlKey&&!e.metaKey&&!e.isComposing&&[...e.key].length===1)send(8,e.key.codePointAt(0));e.preventDefault()};
canvas.onkeyup=e=>{send(7,key(e));e.preventDefault()};canvas.oncompositionend=e=>{for(const c of e.data)send(8,c.codePointAt(0))};
window.onblur=()=>{send(4);send(9)};
document.querySelector('#stop').onclick=()=>{send(1);active=false;status.textContent='Viz fermée'};
async function refresh(){if(!active)return;try{const r=await fetch(base+'/frame');if(r.status===204){setTimeout(refresh,50);return}if(!r.ok)throw Error();const url=URL.createObjectURL(await r.blob());const im=new Image();try{im.src=url;await im.decode();if(canvas.width!==im.width||canvas.height!==im.height){canvas.width=im.width;canvas.height=im.height}ctx.drawImage(im,0,0);}finally{URL.revokeObjectURL(url)}canvas.style.cursor=['default','pointer','crosshair','nwse-resize'][Number(r.headers.get('X-Viz-Cursor'))]||'default';status.textContent='Connecté';setTimeout(refresh,33)}catch(e){status.textContent='Connexion perdue';setTimeout(refresh,1000)}}
canvas.focus();refresh();
</script>)HTML";
bool parseEvent(const std::string& query, Event& e) {
    if(query.size()>256) return false;
    std::istringstream input(query);
    const char* names[]={"t=","x=","y=","v=","m="};
    int32_t* values[]={&e.type,&e.x,&e.y,&e.value,&e.modifiers};
    std::string part;
    for(int i=0;i<5;++i) {
        if(!std::getline(input,part,'&') || part.rfind(names[i],0)!=0) return false;
        auto result=std::from_chars(part.data()+2,part.data()+part.size(),*values[i]);
        if(result.ec!=std::errc{} || result.ptr!=part.data()+part.size()) return false;
    }
    if(!std::getline(input,part) || part.rfind("d=",0)!=0) return false;
    std::istringstream delta(part.substr(2));
    delta.imbue(std::locale::classic());
    delta >> std::noskipws >> e.delta;
    return !delta.fail() && delta.eof();
}
void little(std::vector<uint8_t>& b,size_t offset,uint32_t v) { for(int i=0;i<4;++i)b[offset+i]=uint8_t(v>>(i*8)); }
}
class WebHost final : public Host {
    int server_=-1; std::atomic<bool> stopped_{false}; std::thread worker_;
    std::mutex mutex_; Frame frame_; std::deque<Event> events_; std::string token_;
    void reply(int fd,const std::string& code,const std::string& type,const void* bytes,size_t count,unsigned cursor=0) {
        std::string h="HTTP/1.1 "+code+"\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(count)+
            "\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nX-Frame-Options: DENY\r\nConnection: close\r\nX-Viz-Cursor: "+std::to_string(cursor)+"\r\n\r\n";
        if(transfer(fd,h.data(),h.size(),true) && count) transfer(fd,const_cast<void*>(bytes),count,true);
    }
    void serve(int fd) {
        timeval timeout{0,250000}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)); setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        std::string request; char buf[2048];
        // Bound both bytes and wall time, including a client trickling bytes.
        auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(500);
        while(request.find("\r\n\r\n")==std::string::npos) {
            auto n=recv(fd,buf,sizeof(buf),0); if(n<=0)return; request.append(buf,n);
            if(request.size()>8192 || std::chrono::steady_clock::now()>deadline)return;
        }
        std::istringstream first(request); std::string method,path,version; first>>method>>path>>version;
        const std::string root="/"+token_;
        if(method=="GET" && (path==root || path==root+"/")) { reply(fd,"200 OK","text/html; charset=utf-8",page,std::char_traits<char>::length(page));return; }
        if(method=="GET" && path==root+"/frame") {
            Frame f; {std::lock_guard<std::mutex> lock(mutex_);f=frame_;}
            if(f.rgba.empty()){reply(fd,"204 No Content","image/bmp",nullptr,0);return;}
            std::vector<uint8_t> bmp(54+f.rgba.size(),0); bmp[0]='B';bmp[1]='M';little(bmp,2,bmp.size());little(bmp,10,54);little(bmp,14,40);
            little(bmp,18,f.width);little(bmp,22,uint32_t(-int32_t(f.height)));bmp[26]=1;bmp[28]=32;
            for(size_t i=0;i<f.rgba.size();i+=4){bmp[54+i]=f.rgba[i+2];bmp[55+i]=f.rgba[i+1];bmp[56+i]=f.rgba[i];bmp[57+i]=255;}
            reply(fd,"200 OK","image/bmp",bmp.data(),bmp.size(),f.cursor);return;
        }
        if(method=="POST" && path.rfind(root+"/event?",0)==0) {
            Event e;
            const auto query=path.substr(root.size()+7);
            if(!parseEvent(query,e) ||
               e.type<Close || e.type>Blur || e.x< -32768 || e.x>32768 || e.y< -32768 || e.y>32768 || !std::isfinite(e.delta) || std::abs(e.delta)>10000 ||
               e.value<0 || e.value>0x10ffff || e.modifiers<0 || e.modifiers>15) {reply(fd,"400 Bad Request","text/plain",nullptr,0);return;}
            bool full=false;
            {std::lock_guard<std::mutex> lock(mutex_);
             if(e.type==Move && !events_.empty() && events_.back().type==Move) events_.back()=e;
             else if(events_.size()<1024)events_.push_back(e);else full=true;}
            reply(fd,full?"429 Too Many Requests":"204 No Content","text/plain",nullptr,0);return;
        }
        reply(fd,"404 Not Found","text/plain",nullptr,0);
    }
public:
    explicit WebHost(const std::string&) {
        unsigned long port=0;
        if(const char* s=std::getenv("MIMIR_VIZ_WEB_PORT")) { char* end=nullptr;port=strtoul(s,&end,10);if(!*s||*end||port>65535)throw std::runtime_error("Invalid MIMIR_VIZ_WEB_PORT"); }
        std::random_device rng;std::ostringstream token;token<<std::hex<<std::setfill('0');for(int i=0;i<4;++i)token<<std::setw(8)<<rng();token_=token.str();
        server_=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);if(server_<0)throw std::runtime_error("Viz Web socket failed");
        sockaddr_in address{}; address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(port);
        if(bind(server_,reinterpret_cast<sockaddr*>(&address),sizeof(address)) || listen(server_,8)) {::close(server_);throw std::runtime_error("Cannot bind Viz Web loopback port");}
        socklen_t len=sizeof(address);getsockname(server_,reinterpret_cast<sockaddr*>(&address),&len);
        std::cerr<<"[Viz WEB] http://127.0.0.1:"<<ntohs(address.sin_port)<<"/"<<token_<<"/\n";
        worker_=std::thread([this]{while(!stopped_){pollfd p{server_,POLLIN,0};if(::poll(&p,1,100)<=0)continue;int fd=accept4(server_,nullptr,nullptr,SOCK_CLOEXEC);if(fd<0)continue;serve(fd);::close(fd);}});
    }
    ~WebHost() override { stopped_=true;shutdown(server_,SHUT_RDWR);if(worker_.joinable())worker_.join();::close(server_); }
    void present(Frame f) override {std::lock_guard<std::mutex> lock(mutex_);frame_=std::move(f);}
    std::optional<Event> poll() override {std::lock_guard<std::mutex> lock(mutex_);if(events_.empty())return {};auto e=events_.front();events_.pop_front();return e;}
};
std::unique_ptr<Host> makeHost(const std::string& title) {return std::make_unique<WebHost>(title);}
}
