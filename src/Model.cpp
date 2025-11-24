#include "Model.hpp"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <cmath>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <algorithm>

using json = nlohmann::json;
namespace fs = std::filesystem;

// === constructeurs / destructeurs (déjà présents) ===
Model::Model()
    : tokenizer(20000), encoder(64, 20000), hasTokenizer(true), hasEncoder(true)
{
    tw = 64; th = 64;
}
Model::~Model() = default;

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    layers.push_back({name, type, params_count});
}

size_t Model::totalParamCount() const {
    size_t s = 0;
    for (const auto &L : layers) s += L.paramsCount;
    return s;
}

void Model::allocateParams() {
    size_t tot = totalParamCount();
    params.clear();
    params.resize(tot);
    for (size_t i = 0; i < tot; ++i) {
        params[i].Weight = 0;
        params[i].Value = 0;
    }
}

std::vector<uint16_t> Model::getWeights() const {
    std::vector<uint16_t> out;
    out.reserve(params.size());
    for (const auto &p : params) out.push_back(p.Weight);
    return out;
}

void Model::setTokenizer(const Tokenizer &t) { tokenizer = t; hasTokenizer = true; }
void Model::setEncoder(const Encoder &e) { encoder = e; hasEncoder = true; }

void Model::forward(std::vector<uint8_t> &out_uint8) const {
    const size_t N = static_cast<size_t>(tw) * static_cast<size_t>(th);
    out_uint8.assign(N, 0);
    if (params.empty()) return;
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i % params.size();
        out_uint8[i] = params[idx].Value;
    }
}

void Model::setOutputTarget(const std::vector<uint8_t> &target) {
    // simple mapping: write target into tail of params[].Value if sizes allow
    size_t needed = target.size();
    if (needed == 0 || params.empty()) return;
    for (size_t i = 0; i < needed; ++i) {
        params[i % params.size()].Value = target[i];
    }
}

void Model::applyParamUpdate(float learning_rate) {
    // naive linear interpolation from Weight->Value
    for (auto &p : params) {
        float cur = static_cast<float>(p.Weight) / 65535.0f;
        float tgt = static_cast<float>(p.Value) / 255.0f;
        float upd = cur + learning_rate * (tgt - cur);
        upd = std::clamp(upd, 0.0f, 1.0f);
        p.Weight = static_cast<uint16_t>(std::lround(upd * 65535.0f));
    }
}

// Multi-optimizer step (SGD, Adam, AdamW)
void Model::optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients) {
    size_t n = params.size();
    if (n == 0) return;
    
    // Utiliser le LR decay si configuré, sinon utiliser le learning_rate fourni
    float effective_lr = learning_rate;
    if (opt.decay_strategy != LRDecayStrategy::NONE) {
        effective_lr = opt.getCurrentLR();
    }
    
    switch (opt.type) {
        case OptimizerType::SGD: {
            // Stochastic Gradient Descent (simple)
            for (size_t i = 0; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = static_cast<float>(params[i].Weight) / 65535.0f;
                    grad = target - current;
                }
                
                float current = static_cast<float>(params[i].Weight) / 65535.0f;
                float updated = current - effective_lr * grad;
                updated = std::clamp(updated, 0.0f, 1.0f);
                params[i].Weight = static_cast<uint16_t>(std::lround(updated * 65535.0f));
            }
            break;
        }
        
        case OptimizerType::ADAM: {
            // Adam optimizer
            opt.ensure(n);
            opt.step += 1;
            
            const float b1 = opt.beta1, b2 = opt.beta2;
            float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
            float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
            if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
            if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
            
            for (size_t i = 0; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = static_cast<float>(params[i].Weight) / 65535.0f;
                    grad = target - current;
                }
                
                opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * (grad * grad);
                float m_hat = opt.m[i] / bias_correction1;
                float v_hat = opt.v[i] / bias_correction2;
                float denom = std::sqrt(v_hat) + opt.eps;
                float delta = effective_lr * (m_hat / denom);
                float current = static_cast<float>(params[i].Weight) / 65535.0f;
                float updated = current - delta;
                updated = std::clamp(updated, 0.0f, 1.0f);
                params[i].Weight = static_cast<uint16_t>(std::lround(updated * 65535.0f));
            }
            break;
        }
        
        case OptimizerType::ADAMW: {
            // AdamW optimizer (Adam with decoupled weight decay)
            opt.ensure(n);
            opt.step += 1;
            
            const float b1 = opt.beta1, b2 = opt.beta2;
            float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
            float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
            if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
            if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
            
            for (size_t i = 0; i < n; ++i) {
                float grad;
                if (gradients) {
                    grad = gradients->get(i);
                } else {
                    float target = static_cast<float>(params[i].Value) / 255.0f;
                    float current = static_cast<float>(params[i].Weight) / 65535.0f;
                    grad = target - current;
                }
                
                opt.m[i] = b1 * opt.m[i] + (1.0f - b1) * grad;
                opt.v[i] = b2 * opt.v[i] + (1.0f - b2) * (grad * grad);
                float m_hat = opt.m[i] / bias_correction1;
                float v_hat = opt.v[i] / bias_correction2;
                float denom = std::sqrt(v_hat) + opt.eps;
                
                float current = static_cast<float>(params[i].Weight) / 65535.0f;
                
                // AdamW: Weight decay appliqué directement aux poids (découplé du gradient)
                float weight_decay_term = opt.weight_decay * current;
                float adam_update = effective_lr * (m_hat / denom);
                
                float updated = current - adam_update - effective_lr * weight_decay_term;
                updated = std::clamp(updated, 0.0f, 1.0f);
                params[i].Weight = static_cast<uint16_t>(std::lround(updated * 65535.0f));
            }
            break;
        }
    }
}

Model::DecoderOutput Model::eval(const std::vector<uint8_t> &target) const {
    DecoderOutput out;
    std::vector<uint8_t> gen;
    forward(gen);
    if (gen.size() != target.size() || gen.empty()) { out.mse = -1.0; return out; }
    double s = 0.0;
    for (size_t i = 0; i < gen.size(); ++i) {
        double d = double(gen[i]) - double(target[i]);
        s += d * d;
    }
    out.mse = s / double(gen.size());

    if (!hasTokenizer) return out;
    size_t vs = tokenizer.getVocabSize();
    if (vs == 0) return out;
    // produce trivial logits from generated image
    out.logits.assign(vs, 0.0f);
    for (size_t i = 0; i < out.logits.size(); ++i) out.logits[i] = 1.0f / float(out.logits.size());
    // top-k tokens
    for (size_t i = 0; i < std::min<size_t>(8, out.logits.size()); ++i) out.tokens.push_back(int(i));
    return out;
}

void Model::setLastEncoding(const std::vector<float> &e) { lastEncoding = e; }

// ---------------- file helpers ----------------
// convert MagicToken vector to JSON
static json magic_tokens_to_json(const std::vector<MagicToken> &mvec) {
    json a = json::array();
    for (const auto &m : mvec) {
        json mj;
        mj["modality_mask"] = m.modality_mask;
        mj["seed"] = m.seed;
        mj["embed"] = json::array();
        for (int i = 0; i < 8; ++i) mj["embed"].push_back(m.embed[i]);
        a.push_back(mj);
    }
    return a;
}

// read magic tokens from JSON
static void json_to_magic_tokens(const json &j, std::vector<MagicToken> &outMagic) {
    if (!j.is_array()) return;
    for (const auto &m : j) {
        MagicToken mt{};
        mt.modality_mask = m.value("modality_mask", 0u);
        mt.seed = m.value("seed", 0u);
        if (m.contains("embed") && m["embed"].is_array()) {
            for (size_t i = 0; i < 8 && i < m["embed"].size(); ++i) mt.embed[i] = m["embed"][i].get<float>();
        }
        outMagic.push_back(mt);
    }
}

// ---------------- static persistence helpers ----------------
// helper: sanitize strings in id2token array (replace control chars by '<NL>' or space)
static void sanitize_id2token_json(json &tokj) {
    if (!tokj.is_object() || !tokj.contains("id2token")) return;
    try {
        auto &arr = tokj["id2token"];
        if (!arr.is_array()) return;
        for (auto &el : arr) {
            if (!el.is_string()) continue;
            std::string s = el.get<std::string>();
            bool changed = false;
            for (char &c : s) {
                if (static_cast<unsigned char>(c) <= 0x1F) { // control chars
                    changed = true;
                    c = ' '; // replace with space to avoid embedded newlines
                }
            }
            if (changed) el = s;
        }
    } catch (...) { /* best-effort */ }
}

bool Model::saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch) {
    try {
        std::string epoch_name = (epoch >= 0) ? ("epoch_" + std::to_string(epoch)) : std::string("epoch_latest");
        fs::path outdir = dir / epoch_name;
        fs::path tmpdir = outdir.string() + ".tmp";

        if (fs::exists(tmpdir)) fs::remove_all(tmpdir);
        fs::create_directories(tmpdir);

        json tokj;
        // id2Token

        tokj = tokenizer.to_json();

            // sanitize id2token entries to avoid embedded control chars that break JSON files when edited
            sanitize_id2token_json(tokj);

        // write tokenizer.json atomically (write tmp file then rename)
        {
            fs::path tf_tmp = tmpdir / "tokenizer.json.tmp";
            std::ofstream tf(tf_tmp.string(), std::ios::binary);
            if (!tf) { fs::remove_all(tmpdir); return false; }
            tf << std::setw(2) << tokj;
            tf.close();
            fs::rename(tf_tmp, tmpdir / "tokenizer.json");
        }

        // write metadata with epoch and timestamp
        json meta;
        meta["created_by"] = "Model::saveCheckpoint";
        meta["timestamp"] = static_cast<long long>(std::time(nullptr));
        meta["epoch"] = epoch;
        meta["magic_tokens"] = magic_tokens_to_json(magic_tokens);
        {
            fs::path mf_tmp = tmpdir / "metadata.json.tmp";
            std::ofstream mf(mf_tmp.string(), std::ios::binary);
            if (!mf) { fs::remove_all(tmpdir); return false; }
            mf << std::setw(2) << meta;
            mf.close();
            fs::rename(mf_tmp, tmpdir / "metadata.json");
        }

        // placeholder for weights
        {
            
            fs::path wf_tmp = tmpdir / "weights.u16.tmp";
            std::ofstream wf(wf_tmp.string(), std::ios::binary);
            if (!wf) { fs::remove_all(tmpdir); return false; }

            // write params[].Weight as little-endian uint16 array
            for (size_t i = 0; i < params.size(); ++i) {
                uint16_t w = params[i].Weight;
                uint8_t bytes[2];
                bytes[0] = static_cast<uint8_t>(w & 0xFF);
                bytes[1] = static_cast<uint8_t>((w >> 8) & 0xFF);
                wf.write(reinterpret_cast<const char*>(bytes), 2);
                if (!wf) { fs::remove_all(tmpdir); return false; }
            }

            wf.close();
            fs::rename(wf_tmp, tmpdir / "weights.u16");
            
        }

        if (fs::exists(outdir)) fs::remove_all(outdir);
        fs::rename(tmpdir, outdir);

        return true;
    } catch (...) {
        try { /* best-effort cleanup */ } catch(...) {}
        return false;
    }
}

static void write_u64_le(std::ofstream &f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    f.write(reinterpret_cast<char*>(b), 8);
}

// writer for a set of float32 tensors into a safetensors-like file.
// Format written:
// [8 bytes little-endian u64] header_length
// [header_length bytes UTF-8 JSON header]
// [binary blob of tensors concatenated as raw little-endian float32]
//
// Header format (JSON object) follows safetensors style:
// { "metadata": {}, "tensors": { "name": { "dtype":"f32", "shape":[N], "data":[offset, length] }, ... } }
static bool write_safetensors_file(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors, std::string *err = nullptr) {
    try {
        // prepare metadata and compute offsets
        json header;
        header["metadata"] = json::object();
        json tensors_meta = json::object();

        uint64_t offset = 0; // data offset after header
        std::vector<std::pair<const std::string*, const std::vector<float>*>> order;
        order.reserve(tensors.size());
        for (const auto &kv : tensors) order.emplace_back(&kv.first, &kv.second);

        // compute total data size to help building header offsets (we don't need it here)
        for (const auto &p : order) {
            const std::string &name = *p.first;
            const std::vector<float> &buf = *p.second;
            uint64_t byte_len = static_cast<uint64_t>(buf.size()) * sizeof(float);
            // record meta: data = [offset, length]
            json m;
            m["dtype"] = "f32";
            m["shape"] = json::array({ static_cast<uint64_t>(buf.size()) });
            m["data"] = json::array({ offset, byte_len });
            tensors_meta[name] = m;
            offset += byte_len;
        }
        header["tensors"] = tensors_meta;

        std::string header_str = header.dump();
        uint64_t header_len = static_cast<uint64_t>(header_str.size());

        // open file and write header length + header
        std::ofstream ofs(outpath.string(), std::ios::binary);
        if (!ofs) {
            if (err) *err = "failed to open output file";
            return false;
        }

        write_u64_le(ofs, header_len);
        ofs.write(header_str.data(), static_cast<std::streamsize>(header_len));

        // now write tensor data in the same order as header (order vector)
        for (const auto &p : order) {
            const std::vector<float> &buf = *p.second;
            if (!buf.empty()) {
                // write raw floats (assume host is little-endian; if not, convert)
                ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
            }
        }

        ofs.close();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    } catch (...) {
        if (err) *err = "unknown error";
        return false;
    }
}

// Model::packToSafetensor implementation that delegates to writer above.
// Utilise une map fournie par l'appelant (nom -> float buffer).
bool Model::packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const {
    // create parent dir
    try {
        if (outpath.has_parent_path()) fs::create_directories(outpath.parent_path());
    } catch (...) { /* ignore */ }

    std::string err;
    if (!write_safetensors_file(outpath, tensors, &err)) {
        std::cerr << "packToSafetensor: failed to write " << outpath << " : " << err << "\n";
        return false;
    }
    return true;
}

bool Model::tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, Encoder &outEnc, std::vector<MagicToken> &outMagic) {
    bool loaded_any = false;
    try {
        fs::path sjson = safep; sjson += ".json";
        if (fs::exists(sjson) && fs::is_regular_file(sjson)) {
            try {
                std::ifstream f(sjson);
                if (f) {
                    json full; f >> full;
                    if (full.contains("tokenizer")) { try { outTok.from_json(full["tokenizer"]); loaded_any = true; } catch(...) {} }
                    else if (full.contains("id2token")) { json tj; tj["id2token"] = full["id2token"]; try { outTok.from_json(tj); loaded_any = true; } catch(...) {} }
                    if (full.contains("magic_tokens")) { try { json_to_magic_tokens(full["magic_tokens"], outMagic); loaded_any = true; } catch(...) {} }
                    if (full.contains("encoder")) {
                        try {
                            auto ej = full["encoder"];
                            outEnc.dim = ej.value("dim", outEnc.dim);
                            if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                                auto &rows = ej["embeddings"];
                                outEnc.vocab_size = (int)rows.size();
                                outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                                for (size_t r = 0; r < rows.size(); ++r)
                                    for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                        outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                                loaded_any = true;
                            }
                        } catch (...) {}
                    }
                    if (loaded_any) return true;
                }
            } catch (...) {
                // fallback: safep json is invalid/corrupted, ignore and continue to checkpoint folders
            }
        }
    } catch (...) {}

    try {
        if (fs::exists(ckdir) && fs::is_directory(ckdir)) {
            int best_epoch = -1; fs::path best_dir;
            for (auto &p : fs::directory_iterator(ckdir)) {
                if (!p.is_directory()) continue;
                std::string n = p.path().filename().string();
                if (n.rfind("epoch_", 0) == 0) {
                    try { int e = std::stoi(n.substr(6)); if (e > best_epoch) { best_epoch = e; best_dir = p.path(); } } catch(...) {}
                }
            }
            if (best_epoch >= 0 && !best_dir.empty()) {
                fs::path tokp = best_dir / "tokenizer.json";
                fs::path encp = best_dir / "encoder.json";
                fs::path mp = best_dir / "metadata.json";
                if (fs::exists(tokp)) {
                    try {
                        std::ifstream tf(tokp);
                        json tj;
                        tf >> tj;
                        outTok.from_json(tj);
                        loaded_any = true;
                    } catch(...) {
                        // tokenizer.json is invalid -> fallback to minimal tokenizer
                        try {
                            json minimal;
                            minimal["id2token"] = json::array({ "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<NL>" });
                            outTok.from_json(minimal);
                            loaded_any = true;
                        } catch(...) {}
                    }
                }
                if (fs::exists(encp)) {
                    try { std::ifstream ef(encp); json ej; ef >> ej;
                        outEnc.dim = ej.value("dim", outEnc.dim);
                        if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                            auto &rows = ej["embeddings"];
                            outEnc.vocab_size = (int)rows.size();
                            outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                            for (size_t r = 0; r < rows.size(); ++r)
                                for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                    outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                            loaded_any = true;
                        }
                    } catch(...) {}
                }
                if (fs::exists(mp)) {
                    try { std::ifstream mf(mp); json mj; mf >> mj; if (mj.contains("magic_tokens")) { json_to_magic_tokens(mj["magic_tokens"], outMagic); loaded_any = true; } } catch(...) {}
                }
                if (loaded_any) return true;
            }
        }
    } catch (...) {}

    return loaded_any;
}

void Model::conv2d_same(const std::vector<float> &in, std::vector<float> &out, int W, int H, const std::vector<float> &kernel, int ksize)
{

    out.assign(W * H, 0.0f);
    const int khalf = ksize / 2;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky)
            {
                const int iy = y + ky - khalf;
                if (iy < 0 || iy >= H)
                    continue;
                for (int kx = 0; kx < ksize; ++kx)
                {
                    const int ix = x + kx - khalf;
                    if (ix < 0 || ix >= W)
                        continue;
                    sum += in[iy * W + ix] * kernel[ky * ksize + kx];
                }
            }
            out[y * W + x] = sum;
        }
    }
}

// --- Définitions vides pour méthodes virtuelles afin de fournir la vtable ---
void Model::buildBackboneUNet(int /*stages*/, int /*blocks_per_stage*/, int /*bottleneck_depth*/) { /* noop */ }
void Model::injectMagicToken(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildTextBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildAudioBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildImageBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildVideoBranch(const MagicToken & /*tok*/) { /* noop */ }

// === build & autoBuildFromDataset ===

void Model::build()
{
    // Construction générique du modèle
    // Peut être surchargée pour définir une architecture spécifique
    
    // Exemple: backbone U-Net simple
    buildBackboneUNet(4, 2, 3);  // 4 stages, 2 blocs par stage, 3 blocs bottleneck
    
    std::cout << "Model::build() - Architecture construite" << std::endl;
    std::cout << "  Couches: " << layers.size() << std::endl;
    std::cout << "  Paramètres totaux: " << totalParamCount() << std::endl;
}

void Model::autoBuildFromDataset(const std::string &dataset_dir)
{
    // Analyse automatique du dataset pour construire l'architecture appropriée
    
    std::cout << "Model::autoBuildFromDataset(" << dataset_dir << ")" << std::endl;
    
    // Charger le dataset avec cache et validation flexible (min 1 modalité)
    std::vector<DatasetItem> items;
    try {
        items = loadDatasetCached(dataset_dir, 64, 64, 1);  // min_modalities = 1
    } catch (const std::exception &e) {
        std::cerr << "Erreur chargement dataset: " << e.what() << std::endl;
        // Fallback: construction par défaut
        build();
        return;
    }
    
    if (items.empty()) {
        std::cerr << "Dataset vide, construction par défaut" << std::endl;
        build();
        return;
    }
    
    std::cout << "  Items trouvés: " << items.size() << std::endl;
    
    // Analyser les modalités présentes et les linkables
    bool has_text = false;
    bool has_image = false;
    bool has_audio = false;
    bool has_video = false;
    size_t linkable_count = 0;
    
    for (const auto &item : items) {
        if (!item.text_file.empty()) has_text = true;
        if (!item.image_file.empty()) has_image = true;
        if (!item.audio_file.empty()) has_audio = true;
        if (!item.video_file.empty()) has_video = true;
        if (item.is_linked && item.countModalities() >= 2) linkable_count++;
    }
    
    std::cout << "  Modalités détectées:" << std::endl;
    std::cout << "    - Texte:  " << (has_text ? "✓" : "✗") << std::endl;
    std::cout << "    - Image:  " << (has_image ? "✓" : "✗") << std::endl;
    std::cout << "    - Audio:  " << (has_audio ? "✓" : "✗") << std::endl;
    std::cout << "    - Vidéo:  " << (has_video ? "✓" : "✗") << std::endl;
    std::cout << "    - Linkables validés: " << linkable_count << std::endl;
    
    // Construire le backbone de base
    buildBackboneUNet(4, 2, 3);
    
    // Créer les magic tokens pour chaque modalité détectée
    std::vector<MagicToken> magic_tokens;
    
    if (has_text) {
        MagicToken tok;
        tok.modality_mask = 0x01;  // bit 0 = text
        tok.seed = 42;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.1f * (i + 1);
        magic_tokens.push_back(tok);
        buildTextBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche texte ajoutée" << std::endl;
    }
    
    if (has_image) {
        MagicToken tok;
        tok.modality_mask = 0x02;  // bit 1 = image
        tok.seed = 43;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.2f * (i + 1);
        magic_tokens.push_back(tok);
        buildImageBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche image ajoutée" << std::endl;
    }
    
    if (has_audio) {
        MagicToken tok;
        tok.modality_mask = 0x04;  // bit 2 = audio
        tok.seed = 44;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.3f * (i + 1);
        magic_tokens.push_back(tok);
        buildAudioBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche audio ajoutée" << std::endl;
    }
    
    if (has_video) {
        MagicToken tok;
        tok.modality_mask = 0x08;  // bit 3 = video
        tok.seed = 45;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.4f * (i + 1);
        magic_tokens.push_back(tok);
        buildVideoBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche vidéo ajoutée" << std::endl;
    }
    
    std::cout << "  Architecture auto-construite:" << std::endl;
    std::cout << "    - Couches: " << layers.size() << std::endl;
    std::cout << "    - Paramètres: " << totalParamCount() << std::endl;
    std::cout << "    - Magic tokens: " << magic_tokens.size() << std::endl;
}

// --- Fin ---