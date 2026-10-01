#!/usr/bin/env python3
"""Bounded native Lumen resume regression; only generated 8x8 images and checkpoints."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "bin/mimir"
ENV = dict(os.environ, OMP_NUM_THREADS="1")


def run(script, *args):
    result = subprocess.run([str(BINARY), "--lua", str(script), "--", *map(str, args)],
                            cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace")[-6000:])


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


with tempfile.TemporaryDirectory(prefix="mimir-lumen-resume-") as temporary:
    root = Path(temporary)
    dataset = root / "data"
    dataset.mkdir()
    for index in range(3):
        pixels = b"".join(b"\0" + bytes((x * 17 + y * 13 + index * 31) % 256
                                       for x in range(24)) for y in range(8))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
        (dataset / f"{index}.png").write_bytes(png)
        (dataset / f"{index}.txt").write_text("un paysage bleu")
    fixture = root / "fixture.lua"
    fixture.write_text('''
local cfg = assert(Mimir.Architectures.default_config("vae_conv"))
for k,v in pairs({image_w=8,image_h=8,image_c=3,latent_w=4,latent_h=4,latent_c=4,
  base_channels=8,resnet=false,attention=false,enc_norm="none",dec_norm="none",
  use_skip_connections=false,use_encoder_prior=false,text_cond=false,
  decoder_upsample="nearest_conv"}) do cfg[k]=v end
assert(Mimir.Model.create("vae_conv",cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier",42))
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
assert(Mimir.Serialization.save(Args.get_str(Args.parse(arg),"out",""), "raw_folder",
  {save_optimizer=false,save_tokenizer=false,save_encoder=false}))
''')
    run(fixture, "--out", root / "vae")
    config = {"lumen_diffusion": {"vae_checkpoint": str(root / "vae"),
              "image_w": 8, "image_h": 8, "image_c": 3,
              "latent_w": 4, "latent_h": 4, "latent_c": 4, "patch_size": 2,
              "hidden_size": 8, "depth": 1, "mlp_ratio": 2, "vocab_size": 32,
              "text_seq_len": 4, "text_layers": 1, "num_heads": 2, "diffusion_steps": 4},
              "training": {"save_every": 1, "warmup_steps": 2}, "tokenizer": {"max_vocab": 32}}
    (root / "config.json").write_text(json.dumps(config))
    common = ["--config", root / "config.json", "--dataset", dataset,
              "--tokenizer", root / "tokenizer.json", "--validation-items", "0",
              "--vae-calibration-items", "1", "--seed", "42"]
    script = ROOT / "scripts/training/train_lumen_diffusion.lua"
    run(script, *common, "--out", root / "full", "--epochs", "2")
    run(script, *common, "--out", root / "split", "--epochs", "1")
    run(script, *common, "--out", root / "resumed", "--epochs", "1",
        "--resume", root / "split/final")
    run(script, *common, "--out", root / "mid", "--epochs", "2",
        "--resume", root / "full/step_00000001")
    expected = root / "full/final"
    for name in ("resumed", "mid"):
        actual = root / name / "final"
        for source in (expected / "tensors").rglob("*.bin"):
            relative = source.relative_to(expected)
            assert source.read_bytes() == (actual / relative).read_bytes(), (name, relative)
        assert json.loads((expected / "model/training.json").read_text()) == json.loads(
            (actual / "model/training.json").read_text()), name
        assert not (actual / "encoder").exists()
    print("LUMEN_RESUME_OK: epoch and mid-epoch resumes match continuous training exactly")
