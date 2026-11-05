#!/usr/bin/env python3
"""Train a tiny natural-text VAE, then condition and resume native Lumen."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(os.environ, OMP_NUM_THREADS="1", MIMIR_RUNTIME="cpu")


def run(script, *args):
    result = subprocess.run([str(ROOT / "run_mimir.sh"), "--lua", str(script), "--", *map(str, args)],
                            cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace")[-7000:])
    return result.stdout.decode(errors="replace")


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


with tempfile.TemporaryDirectory(prefix="mimir-text-vae-") as temporary:
    root = Path(temporary)
    texts = root / "contexts"
    texts.mkdir()
    sentence = "a blue sky above a green tree near a quiet river"
    (texts / "context.txt").write_text((sentence + ". ") * 4)
    data = root / "images"
    data.mkdir()
    pixels = b"".join(b"\0" + bytes((x * 13 + y * 17) % 256 for x in range(24)) for y in range(8))
    (data / "scene.png").write_bytes(b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    (data / "scene.txt").write_text(sentence)
    fixture = root / "fixture.lua"
    fixture.write_text('''
local A = dofile(ROOTWORK.."/scripts/modules/args.lua")
local out = A.get_str(A.parse(arg), "out", "")
assert(Mimir.Tokenizer.create(128))
assert(Mimir.Tokenizer.ensure_vocab_from_text("a blue sky above a green tree near a quiet river"))
assert(Mimir.Tokenizer.save(out.."/tokenizer.json"))
local cfg = assert(Mimir.Architectures.default_config("vae_conv"))
for k,v in pairs({image_w=8,image_h=8,image_c=3,latent_w=4,latent_h=4,latent_c=4,
 base_channels=8,resnet=false,attention=false,enc_norm="none",dec_norm="none",
 use_skip_connections=false,use_encoder_prior=false,text_cond=false}) do cfg[k]=v end
assert(Mimir.Model.create("vae_conv",cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier",42))
assert(Mimir.Serialization.save(out.."/image_vae", "raw_folder",
 {save_optimizer=false,save_tokenizer=false,save_encoder=false}))
''')
    run(fixture, "--out", root)
    run(ROOT / "scripts/training/train_vae_texte.lua", "--context-dir", texts,
        "--base-tokenizer", root / "tokenizer.json", "--out-dir", root / "text_vae",
        "--seq-len", 8, "--d-model", 8, "--layers", 1, "--heads", 2,
        "--mlp-hidden", 16, "--latent-tokens", 2, "--proj-dim", 8,
        "--epochs", 1, "--validate-items", 0, "--autosave-every-epochs", 0, "--cpu-only")
    text_architecture = json.loads((root / "text_vae/final/model/architecture.json").read_text())
    assert text_architecture["model_config"]["text_context_version"] == 2
    run(ROOT / "scripts/training/train_vae_texte.lua", "--context-dir", texts,
        "--base-tokenizer", root / "tokenizer.json", "--out-dir", root / "text_vae",
        "--seq-len", 8, "--d-model", 8, "--layers", 1, "--heads", 2,
        "--mlp-hidden", 16, "--latent-tokens", 2, "--proj-dim", 8,
        "--epochs", 1, "--validate-items", 0, "--autosave-every-epochs", 0,
        "--cpu-only", "--resume", "true")
    config = {"lumen_diffusion": {"image_w": 8, "image_h": 8, "image_c": 3,
        "latent_w": 4, "latent_h": 4, "latent_c": 4, "patch_size": 2,
        "hidden_size": 8, "depth": 1, "mlp_ratio": 2, "num_heads": 2, "diffusion_steps": 4}}
    (root / "config.json").write_text(json.dumps(config))
    common = ["--config", root / "config.json", "--dataset", data,
        "--vae-checkpoint", root / "image_vae", "--text-vae-checkpoint", root / "text_vae",
        "--validation-items", 0, "--vae-calibration-items", 1, "--epochs", 1]
    run(ROOT / "scripts/training/train_lumen_diffusion.lua", *common, "--out", root / "lumen")
    saved = root / "lumen/final"
    architecture = json.loads((saved / "model/architecture.json").read_text())
    assert architecture["model_config"]["dynamic_text_seq_len"] is False
    assert architecture["model_config"]["text_seq_len"] == 8
    assert any(layer["name"] == "lumen/text_vae/vae_text/mu" for layer in architecture["layers"])
    assert json.loads((saved / "tokenizer/tokenizer.json").read_text()) == json.loads(
        (root / "text_vae/final/tokenizer/tokenizer.json").read_text())
    run(ROOT / "scripts/training/train_lumen_diffusion.lua", *common,
        "--resume", saved, "--out", root / "resumed")
    imported_weights = list((saved / "tensors/lumen/text_vae").rglob("*_weights.bin"))
    assert imported_weights, "No imported encoder weights were saved"
    for tensor in imported_weights:
        source_relative = tensor.relative_to(saved / "tensors/lumen/text_vae")
        source = root / "text_vae/final/tensors" / source_relative
        assert tensor.read_bytes() == source.read_bytes(), source_relative
        assert tensor.read_bytes() == (root / "resumed/final" / tensor.relative_to(saved)).read_bytes()
    run(ROOT / "scripts/inferences/lumen_text2img.lua", "--checkpoint", saved,
        "--headless", "--prompt", sentence, "--steps", 2, "--out", root / "sample.ppm")
    assert (root / "sample.ppm").read_bytes().startswith(b"P6\n8 8\n255\n")
    print("LUMEN_NATURAL_TEXT_VAE_OK")
