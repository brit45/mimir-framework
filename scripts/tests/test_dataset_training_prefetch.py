#!/usr/bin/env python3
"""Compare complete native training with synchronous and prefetched datasets."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
MODELS = ["vae_conv", "vae_text", "vgg16_feat", "vgg16", "vgg19", "medical_detector"]
SCRIPT = r'''
local data, output, kind = arg[1], arg[2], arg[3]
local expected_items=6
if kind=="video_vae_conv" then kind="vae_conv";expected_items=1 end
local cfg = assert(Mimir.Architectures.default_config(kind))
cfg.seed=123; cfg.max_items=6; cfg.log_every=1000
cfg.autosave_every_epochs=0; cfg.checkpoint_dir=""
cfg.optimizer="adamw"; cfg.dropout=0; cfg.stochastic_latent=false
cfg.validate_every_steps=2; cfg.validate_items=1; cfg.validate_every_epochs=1
cfg.validate_holdout=false; cfg.validation_fraction=0.2
if kind=="vae_text" then
  assert(Mimir.Tokenizer.create(256))
  for i=1,6 do
    local f=assert(io.open(data.."/sample_"..i..".txt","rb"))
    assert(Mimir.Tokenizer.ensure_vocab_from_text(f:read("*a")));f:close()
  end
  cfg.seq_len=8;cfg.d_model=8;cfg.num_layers=1;cfg.num_heads=2;cfg.mlp_hidden=16
  cfg.latent_tokens=2;cfg.proj_dim=8;cfg.vocab_size=Mimir.Tokenizer.vocab_size()
  cfg.natural_context_windows=true
else
  cfg.image_w=32;cfg.image_h=32;cfg.image_c=3;cfg.base_channels=1
  cfg.num_classes=2;cfg.fc_hidden=4;cfg.proj_dim=4
  cfg.enc_norm="none";cfg.dec_norm="none"
  cfg.resnet=false;cfg.attention=false;cfg.text_cond=false
  cfg.latent_c=2;cfg.latent_h=8;cfg.latent_w=8;cfg.latent_dim=128;cfg.kl_beta=0
  cfg.tags_vocab={"mass","calcification"};cfg.use_dataset_annotation_labels=true
  cfg.class_names={"mass","calcification"}
end
if kind=="vgg16_feat" then cfg.enc_norm="groupnorm" end
assert(Mimir.Model.create(kind,cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier",123))
local loaded,count=Mimir.Dataset.load(data,32,32,1,false,"",64,true)
assert(loaded,count);assert(count==expected_items,count)
local ok,steps=Mimir.Model.train(2,0.0001)
assert(ok,steps)
if type(Mimir.Dataset.prefetch_stats)=="function" then
  write_json(output..".loader.json",Mimir.Dataset.prefetch_stats())
end
assert(Mimir.Serialization.save(output,"raw_folder",{
  save_optimizer=true,save_tokenizer=true,save_encoder=false,include_gradients=true
}))
print("DATASET_TRAIN_PARITY_OK",kind,steps)
'''


def fixture(root):
    images, texts = root / "images", root / "texts"
    images.mkdir(parents=True, exist_ok=True)
    texts.mkdir(parents=True, exist_ok=True)
    for index in range(1, 7):
        name = f"sample_{index}"
        pixels = bytes((index * 13 + pixel * 7) % 256 for pixel in range(64 * 64 * 3))
        (images / (name + ".ppm")).write_bytes(b"P6\n64 64\n255\n" + pixels)
        mask = bytes(255 if (pixel + index) % 3 else 0 for pixel in range(64 * 64))
        (images / (name + "_mask.pgm")).write_bytes(b"P5\n64 64\n255\n" + mask)
        boxes = [] if index == 6 else [{"x1": .1, "y1": .2, "x2": .8, "y2": .9,
                                       "class_id": index % 2, "label": ["Mass", "Calcification"][index % 2]}]
        (images / (name + ".json")).write_text(json.dumps({"normalized": True, "boxes": boxes}))
        content = "a blue sky" if index % 2 else "a blue sky above a green tree and a red house. " * 8
        (texts / (name + ".txt")).write_text(content)
    return images, texts


def video_fixture(root):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("The optional video_vae_conv case requires ffmpeg and an FFmpeg-enabled Mimir build")
    video, frames = root / "video", root / "video-frames"
    video.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    header = b"P6\n32 32\n255\n"
    for index, color in enumerate([bytes([0, 0, 255]), bytes([0, 255, 0])]):
        (frames / f"frame_{index:02d}.ppm").write_bytes(header + color * (32 * 32))
    (video / "sample.ppm").write_bytes(header + bytes([255, 0, 0]) * (32 * 32))
    (video / "sample.txt").write_text("a video with a distinct cover image")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", str(frames / "frame_%02d.ppm"),
                    "-frames:v", "2", "-c:v", "ffv1", str(video / "sample.mkv")], check=True)
    return video


def run(binary, directory, script, model, data, ahead, label):
    output = directory / label
    env = dict(os.environ, MIMIR_EXECUTABLE=str(binary), MIMIR_DATASET_PREFETCH=str(ahead),
               MIMIR_DATASET_PREFETCH_BYTES=str(64 * 1024 * 1024),
               MIMIR_DISABLE_ROCM="1", MIMIR_DISABLE_CUDA="1", MIMIR_DISABLE_OPENCL="1",
               MIMIR_DISABLE_VULKAN="1", MIMIR_RUNTIME_AUTOTUNE="0", MIMIR_PLANNER_DUMP="0",
               OMP_NUM_THREADS="1")
    log = directory / (label + ".log")
    start = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run([str(REPO / "run_mimir.sh"), "--lua", str(script), str(data), str(output), model],
                                cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{model} {label}: exit={result.returncode}\n" + log.read_text()[-7000:])
    payloads = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((output / "tensors").rglob("*.bin"))}
    if not payloads or "DATASET_TRAIN_PARITY_OK" not in log.read_text():
        raise RuntimeError(f"Missing native training output: {log}")
    stats_path=directory/(label+".loader.json")
    stats=json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is not None and ahead and (not stats["enabled"] or not stats["requests"]):
        raise RuntimeError("Native trainer did not use its dataset reader: " + model)
    return {"elapsed_seconds": time.perf_counter() - start, "tensor_sha256": payloads, "loader":stats,"log": str(log)}


def verify(work, args):
    images, texts = fixture(work / "data")
    video = video_fixture(work / "data") if "video_vae_conv" in args.models else None
    script = work / "train.lua"
    script.write_text(SCRIPT)
    report = {}
    for model in args.models:
        data = video if model == "video_vae_conv" else texts if model == "vae_text" else images
        before = run(args.binary, work, script, model, data, 0, model + "-sync")
        after = run(args.binary, work, script, model, data, 3, model + "-prefetch")
        if before["tensor_sha256"] != after["tensor_sha256"]:
            raise RuntimeError("Prefetch changed trained tensors: " + model)
        row = {"synchronous": before, "prefetch": after, "bitwise_equal": True}
        if args.reference:
            original = run(args.reference, work, script, model, data, 0, model + "-reference")
            if original["tensor_sha256"] != before["tensor_sha256"]:
                raise RuntimeError("Synchronous training changed from reference engine: " + model)
            row["reference"] = original
        report[model] = row
        print("DATASET_NATIVE_TRAINING_OK", model, len(before["tensor_sha256"]), "exact tensor payloads", flush=True)
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=REPO / "bin/mimir")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS + ["video_vae_conv"], default=MODELS)
    parser.add_argument("--out", type=Path, help="Keep fixtures, tensor dumps and per-model report in this new directory")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if args.reference:
        args.reference = args.reference.resolve()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        verify(args.out.resolve(), args)
    else:
        with tempfile.TemporaryDirectory(prefix="mimir-dataset-training-") as directory:
            verify(Path(directory), args)


if __name__ == "__main__":
    main()
