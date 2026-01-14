-- Smoke test: vérifie que l'Encoder (token_embeddings) est bien sérialisé
-- Usage: ./bin/mimir --lua scripts/tests/smoke_save_ponyxl_encoder.lua

local function log(x)
  print(x)
end

local cfg = {
  d_model = 64,
  seq_len = 16,
  max_vocab = 128,
  hidden_dim = 128,
  latent_dim = 32,
  blur_levels = 4,
  image_w = 64,
  image_h = 64,
  image_c = 3,
  cfg_dropout_prob = 0.0,
  optimizer = "adamw",
  warmup_steps = 0,
}

local out_dir = "checkpoint/_smoke_ponyxl_encoder"

local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then error("create failed: " .. tostring(err_create)) end

local ok_build, err_build = Mimir.Model.build()
if not ok_build then error("build failed: " .. tostring(err_build)) end

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("alloc failed: " .. tostring(err_alloc)) end

local ok_init, err_init = Mimir.Model.init_weights("xavier", 1337)
if not ok_init then error("init failed: " .. tostring(err_init)) end

log("params=" .. tostring(Mimir.Model.total_params()))

local ok_raw, err_raw = Mimir.Serialization.save(out_dir, "raw_folder", {
  save_optimizer = false,
  save_tokenizer = true,
  save_encoder = true,
  include_gradients = false,
  include_checksums = true,
})
if not ok_raw then error("save raw failed: " .. tostring(err_raw)) end

local ok_st, err_st = Mimir.Serialization.save(out_dir .. "/model.safetensors", "safetensors", {
  save_optimizer = false,
  save_tokenizer = true,
  save_encoder = true,
  include_gradients = false,
})
if not ok_st then error("save safetensors failed: " .. tostring(err_st)) end

log("✓ ok: " .. out_dir)
