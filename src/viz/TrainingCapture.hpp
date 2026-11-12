#pragma once
#include "LatentSpaceExport.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace mimir::viz {
// Runs exclusively on the projection worker. No tensor access or training lock
// during serialization/I/O. Each complete line is independently recoverable.
class TrainingCapture {
    std::filesystem::path path_;
    std::ofstream stream_;
    std::string csv_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
public:
    static std::string replayHtml() {
        const auto viewer = latentSpaceExportHtml(nlohmann::json::object());
        std::string html = R"CAP(<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Mímir · Relecture des captures</title>
<style>body{background:#10141d;color:#dde3ef;font:14px system-ui}iframe{width:100%;height:78vh;border:0}pre{white-space:pre-wrap}#info{line-height:1.6}details{margin:8px 0}input[type=range]{width:40%}</style>
<p>Ouvrir le fichier annexe .jsonl. La relecture restitue les captures échantillonnées, pas l'exécution du modèle.</p>
<input id="file" type="file" accept=".jsonl"><button id="play">Lecture / pause</button>
<input id="frame" type="range" min="0" max="0" value="0"><label>Vitesse <select id="speed"><option>1</option><option>2</option><option>4</option><option>0.5</option></select>×</label>
<p id="info"></p><details><summary>Toutes les mesures et provenance</summary><pre id="metadata"></pre></details><iframe id="view" sandbox="allow-scripts"></iframe>
<script>
const template=)CAP";
        html += latent_export_detail::scriptJson(nlohmann::json(viewer));
        html += R"CAP(;
const el=id=>document.getElementById(id);let frames=[],timer=null,playing=false,rejected=0;
function stop(){playing=false;clearTimeout(timer);}
function show(){const r=frames[+el('frame').value];if(!r)return;
const parsed=new DOMParser().parseFromString(template,'text/html');
parsed.getElementById('graph-data').textContent=JSON.stringify(r.graph).replace(/</g,'\\u003c');
el('view').srcdoc='<!doctype html>'+parsed.documentElement.outerHTML;
const m=r.metrics||{},value=v=>v===null||v===undefined?'indisponible':String(v);
el('info').textContent='Capture '+(Number(el('frame').value)+1)+' / '+frames.length+' · '+(r.capture.elapsed_ms/1000).toFixed(3)+' s · Pas modèle '+value(r.graph.model?.optimizer_step_decimal??r.graph.model?.optimizer_step)+' · Pas métriques '+value(m.optimizer_step)+' · Loss '+value(m.loss)+' · LR '+value(m.learning_rate)+' · Norme gradient '+value(m.gradient_norm)+' · Batch '+value(m.batch_time_ms)+' ms · Mémoire '+value(m.memory_mb)+' Mo'+(rejected?' · Lignes invalides ou incomplètes ignorées : '+rejected:'');
el('metadata').textContent=JSON.stringify({capture:r.capture,metrics:m},null,2);}

function next(){if(!playing)return;const i=+el('frame').value;if(i>=frames.length-1){stop();return;}
const delay=Math.max(1,frames[i+1].capture.elapsed_ms-frames[i].capture.elapsed_ms)/Number(el('speed').value);
timer=setTimeout(()=>{el('frame').value=String(i+1);show();next();},delay);}
el('file').onchange=async()=>{if(!el('file').files.length)return;stop();frames=[];rejected=0;el('view').srcdoc='';el('metadata').textContent='';for(const line of (await el('file').files[0].text()).split('\n')){if(!line.trim())continue;try{const r=JSON.parse(line);if(r.format!=='mimir.training-capture'||r.version!==1||!r.graph||!r.capture||!Number.isFinite(r.capture.elapsed_ms))throw Error();frames.push(r);}catch{rejected++;}}
el('frame').max=String(Math.max(0,frames.length-1));el('frame').value='0';show();if(!frames.length)el('info').textContent='Aucune capture valide. Lignes invalides ou incomplètes ignorées : '+rejected;};
el('frame').oninput=()=>{stop();show();};el('play').onclick=()=>{if(playing){stop();return;}playing=true;next();};el('speed').onchange=()=>{clearTimeout(timer);next();};
</script></html>)CAP";
        return html;
    }
    const std::filesystem::path& path() const { return path_; }
    void append(const LatentSpaceProjection& projection, const std::string& csv,
                const nlohmann::json& metrics) {
        if (!stream_.is_open() || csv_ != csv) {
            stream_.close(); stream_.clear(); csv_=csv;
            const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const std::filesystem::path base(csv);
            std::filesystem::create_directories(base.has_parent_path()?base.parent_path():std::filesystem::path("."));
            path_=base.parent_path()/(base.stem().string()+".viz3d_"+std::to_string(stamp)+".jsonl");
            std::ofstream viewer(path_.string()+".html",std::ios::binary);
            viewer << replayHtml(); viewer.close();
            if(!viewer) throw std::runtime_error("écriture du lecteur 3D impossible");
            stream_.open(path_,std::ios::binary|std::ios::app);
            if(!stream_) throw std::runtime_error("ouverture du fichier de captures impossible");
        }
        const auto now=std::chrono::system_clock::now();
        auto graph=latentSpaceExportDocument(projection,LatentSpaceExportView{});
        nlohmann::json record={{"format","mimir.training-capture"},{"version",1},
            {"capture",{{"unix_us",std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count())},
                {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start_).count()},
                {"metrics_csv",csv},{"metrics_alignment","latest_monitor_observation; compare optimizer_step"},
                {"scope","sampled 3D projections; not model checkpoints"}}},
            {"metrics",metrics},{"graph",std::move(graph)}};
        stream_ << record.dump() << '\n'; stream_.flush();
        if(!stream_) throw std::runtime_error("écriture du fichier de captures impossible");
    }
};
}
