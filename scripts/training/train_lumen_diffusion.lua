#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile("scripts/modules/args.lua")
local FS = dofile("scripts/modules/fs.lua")
local BaseTok = dofile("scripts/modules/base_tokenizer.lua")
local Checkpoint = dofile("scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}

local function logf(format, ...)
  log(string.format(format, ...))
end

local function die(message)
  error(tostring(message), 0)
end

local function check(ok, err, operation)
  if not ok then die(operation .. ": " .. tostring(err or "échec")) end
end

local function is_finite(value)
  return type(value) == "number" and value == value and
    value > -math.huge and value < math.huge
end

local config_path = Args.get_str(opts, "config", "configs/lumen_diffusion.json")
local document, config_err = read_json(config_path)
if type(document) ~= "table" then die("configuration: " .. tostring(config_err)) end

local cfg = document.lumen_diffusion or {}
local training = document.training or {}
local tokenizer_cfg = document.tokenizer or {}
local viz_enabled = Args.get_bool(opts, "viz", false)

if viz_enabled then
  local ok_viz, viz_err = Mimir.Viz.create(document)
  check(ok_viz, viz_err, "Viz.create")
  check(Mimir.Viz.set_validation({ enabled = true }), nil,
    "Viz.set_validation(enabled)")
  log("[train_lumen] interface VIZ activée")
end

local dataset_root = Args.get_str(opts, "dataset", "")
if dataset_root == "" then
  die("--dataset est requis (images avec captions .txt de même nom)")
end

local tokenizer_path = Args.get_str(opts, "tokenizer", tokenizer_cfg.path or
  "checkpoint/base_tokenizer/tokenizer.json")
local require_tokenizer = Args.get_bool(opts, "require-tokenizer", false)
local tokenizer_max_vocab = math.max(1, Args.get_int(opts, "max-vocab",
  tokenizer_cfg.max_vocab or cfg.vocab_size or 32000))
local output_dir = Args.get_str(opts, "out", "checkpoints/lumen_diffusion")
local resume_path = Args.get_str(opts, "resume", "")
local prepare_tokenizer_only = Args.get_bool(opts, "prepare-tokenizer-only", false)
if resume_path == "0" or resume_path == "false" or resume_path == "off" or
    resume_path == "none" then
  resume_path = ""
end
local epochs = math.max(1, Args.get_int(opts, "epochs", training.epochs or 10))
local learning_rate = Args.get_num(opts, "lr", training.learning_rate or 1e-4)
local optimizer = Args.get_str(opts, "optimizer", training.optimizer or "adamw")
local weight_decay = Args.get_num(opts, "weight-decay", training.weight_decay or 0.01)
local cfg_dropout = Args.get_num(opts, "cfg-dropout", training.cfg_dropout or 0.1)
local warmup_steps = math.max(0, Args.get_int(opts, "warmup-steps",
  Args.get_int(opts, "lr-warmup_steps", training.warmup_steps or 100)))
local requested_kl_beta = math.max(0.0, Args.get_num(opts, "kl-beta",
  training.kl_beta or cfg.kl_beta or 0.0))
local requested_kl_warmup_steps = math.max(0, Args.get_int(opts, "kl-warmup-steps",
  Args.get_int(opts, "kl-warmup_steps",
    training.kl_warmup_steps or cfg.kl_warmup_steps or 0)))
local kl_beta = 0.0
local kl_warmup_steps = 0
if requested_kl_beta > 0.0 then
  logf("[train_lumen] --kl-beta=%.8g ignoré: Lumen entraîne epsilon avec MSE pure",
    requested_kl_beta)
end
local log_every = math.max(1, Args.get_int(opts, "log-every", training.log_every or 10))
local save_every = math.max(0, Args.get_int(opts, "save-every", training.save_every or 500))
local max_items = math.max(0, Args.get_int(opts, "max-items", training.max_items or 0))
local vae_calibration_items = math.max(1, Args.get_int(opts,
  "vae-calibration-items", training.vae_calibration_items or 100))
local seed = Args.get_int(opts, "seed", cfg.seed or 1337)
local validation_every_steps = math.max(0, Args.get_int(opts, "validation-every-steps",
  training.validation_every_steps or 100))
local validation_every_epochs = math.max(0, Args.get_int(opts, "validation-every-epochs",
  training.validation_every_epochs or 1))
local validation_items = math.max(0, Args.get_int(opts, "validation-items",
  training.validation_items or 4))
local validation_holdout = Args.get_bool(opts, "validation-holdout",
  training.validation_holdout ~= false)
local validation_feedback_enabled = Args.get_bool(opts, "validation-feedback",
  training.validation_feedback_enabled ~= false)
local validation_reward_factor = Args.get_num(opts, "validation-reward-factor",
  training.validation_reward_factor or 1.05)
local validation_penalty_factor = Args.get_num(opts, "validation-penalty-factor",
  training.validation_penalty_factor or 0.70)
local validation_lr_scale_min = Args.get_num(opts, "validation-lr-scale-min",
  training.validation_lr_scale_min or 0.10)
local validation_lr_scale_max = Args.get_num(opts, "validation-lr-scale-max",
  training.validation_lr_scale_max or 1.50)
local validation_improve_threshold = Args.get_num(opts, "validation-improve-threshold",
  training.validation_improve_threshold or 0.001)
local validation_feedback_min_steps = math.max(0, Args.get_int(opts,
  "validation-feedback-min-steps", training.validation_feedback_min_steps or 0))
local validation_epsilon_weight = math.max(0.0, Args.get_num(opts,
  "validation-epsilon-weight", training.validation_epsilon_weight or 0.5))
local validation_reconstruction_weight = math.max(0.0, Args.get_num(opts,
  "validation-reconstruction-weight", training.validation_reconstruction_weight or 0.5))
local validation_ema_alpha = Args.get_num(opts, "validation-ema-alpha",
  training.validation_ema_alpha or 0.25)
local validation_penalty_patience = math.max(1, Args.get_int(opts,
  "validation-penalty-patience", training.validation_penalty_patience or 2))
local validation_max_penalties = math.max(0, Args.get_int(opts,
  "validation-max-penalties", training.validation_max_penalties or 0))
local validation_save_best = Args.get_bool(opts, "validation-save-best",
  training.validation_save_best ~= false)

if learning_rate <= 0 then die("--lr doit être strictement positif") end
if cfg_dropout < 0 or cfg_dropout > 1 then die("--cfg-dropout doit être dans [0,1]") end
if validation_reward_factor < 1 then die("--validation-reward-factor doit être >= 1") end
if validation_penalty_factor <= 0 or validation_penalty_factor > 1 then
  die("--validation-penalty-factor doit être dans ]0,1]")
end
if validation_lr_scale_min <= 0 or validation_lr_scale_max < validation_lr_scale_min then
  die("bornes validation-lr-scale invalides")
end
if validation_improve_threshold < 0 then
  die("--validation-improve-threshold doit être positif")
end
if validation_epsilon_weight + validation_reconstruction_weight <= 0 then
  die("au moins un poids de validation doit être strictement positif")
end
if validation_ema_alpha <= 0 or validation_ema_alpha > 1 then
  die("--validation-ema-alpha doit être dans ]0,1]")
end

cfg.image_w = Args.get_int(opts, "image-w", cfg.image_w or 512)
cfg.image_h = Args.get_int(opts, "image-h", cfg.image_h or 512)
cfg.image_c = Args.get_int(opts, "image-c", cfg.image_c or 3)
cfg.latent_w = Args.get_int(opts, "latent-w", cfg.latent_w or 64)
cfg.latent_h = Args.get_int(opts, "latent-h", cfg.latent_h or 64)
cfg.latent_c = Args.get_int(opts, "latent-c", cfg.latent_c or 4)
cfg.vae_checkpoint = Args.get_str(opts, "vae-checkpoint", cfg.vae_checkpoint or "")
cfg.vae_base_channels = Args.get_int(opts, "vae-base-channels", cfg.vae_base_channels or 16)
cfg.vae_stochastic_latent = Args.get_bool(opts, "vae-stochastic-latent",
  cfg.vae_stochastic_latent ~= false)
cfg.vae_use_resnet = Args.get_bool(opts, "vae-resnet", cfg.vae_use_resnet ~= false)
cfg.vae_use_attn = Args.get_bool(opts, "vae-attn", cfg.vae_use_attn ~= false)
cfg.vae_use_skip_connections = Args.get_bool(opts, "vae-use-skip-connections",
  cfg.vae_use_skip_connections ~= false)
cfg.vae_use_encoder_prior = Args.get_bool(opts, "vae-encoder-prior",
  cfg.vae_use_encoder_prior ~= false)
cfg.vae_enc_norm = Args.get_str(opts, "vae-enc-norm", cfg.vae_enc_norm or "none")
cfg.vae_dec_norm = Args.get_str(opts, "vae-dec-norm", cfg.vae_dec_norm or "groupnorm")
cfg.vae_decoder_upsample = Args.get_str(opts, "vae-decoder-upsample",
  cfg.vae_decoder_upsample or "nearest_conv")
cfg.vae_enc_gn_groups = Args.get_int(opts, "vae-enc-gn-groups", cfg.vae_enc_gn_groups or 16)
cfg.vae_dec_gn_groups = Args.get_int(opts, "vae-dec-gn-groups", cfg.vae_dec_gn_groups or 32)
cfg.vae_attn_heads = Args.get_int(opts, "vae-attn-heads", cfg.vae_attn_heads or 4)
cfg.vae_attn_max_tokens = Args.get_int(opts, "vae-attn-max-tokens",
  cfg.vae_attn_max_tokens or 4096)
cfg.vae_resnet_max_tokens = Args.get_int(opts, "vae-resnet-max-tokens",
  cfg.vae_resnet_max_tokens or 4096)
cfg.vae_scale = Args.get_num(opts, "vae-scale", cfg.vae_scale or 0.0)
cfg.vae_shift = Args.get_num(opts, "vae-shift", cfg.vae_shift or 0.0)
cfg.preview_timestep = Args.get_int(opts, "preview-timestep", cfg.preview_timestep or 50)
cfg.patch_size = Args.get_int(opts, "patch-size", cfg.patch_size or 4)
cfg.hidden_size = Args.get_int(opts, "hidden-size", cfg.hidden_size or 384)
cfg.depth = Args.get_int(opts, "depth", cfg.depth or 8)
cfg.mlp_ratio = Args.get_num(opts, "mlp-ratio", cfg.mlp_ratio or 4.0)
if cfg.image_w <= 0 or cfg.image_h <= 0 then die("--image-w et --image-h doivent être positifs") end
if cfg.latent_w <= 0 or cfg.latent_h <= 0 or cfg.latent_c <= 0 then
  die("--latent-w, --latent-h et --latent-c doivent être positifs")
end
if cfg.vae_checkpoint == "" and not prepare_tokenizer_only then
  die("--vae-checkpoint est requis (dossier epoch_* du VAEConv entraîné)")
end
if not prepare_tokenizer_only then
  local resolved_vae_checkpoint = Checkpoint.resolve_dir(cfg.vae_checkpoint)
  if not resolved_vae_checkpoint then
    die("--vae-checkpoint ne contient aucun checkpoint VAEConv exploitable: " ..
      cfg.vae_checkpoint)
  end
  if resolved_vae_checkpoint ~= cfg.vae_checkpoint then
    log("[train_lumen] checkpoint VAE résolu: " .. resolved_vae_checkpoint)
    cfg.vae_checkpoint = resolved_vae_checkpoint
  end
end
if cfg.patch_size < 1 or cfg.latent_w % cfg.patch_size ~= 0 or
    cfg.latent_h % cfg.patch_size ~= 0 then
  die("--patch-size doit diviser exactement --latent-w et --latent-h")
end
if cfg.hidden_size < 1 or cfg.depth < 1 or cfg.mlp_ratio < 1 then
  die("--hidden-size, --depth et --mlp-ratio doivent être positifs")
end
local patch_tokens = (cfg.latent_w / cfg.patch_size) * (cfg.latent_h / cfg.patch_size)
local max_patch_tokens = math.max(1, Args.get_int(opts, "max-patch-tokens", 1024))
if patch_tokens > max_patch_tokens then
  die(string.format(
    "grille DiT latente trop grande: %d tokens (%dx%d, patch=%d); " ..
    "le backward d'attention est quadratique. Utilisez --patch-size 64 " ..
    "pour 1024x1024, ou augmentez explicitement --max-patch-tokens.",
    patch_tokens, cfg.latent_w, cfg.latent_h, cfg.patch_size))
end
if cfg.image_c ~= 1 and cfg.image_c ~= 3 and cfg.image_c ~= 4 then
  die("--image-c doit valoir 1, 3 ou 4")
end

cfg.weight_decay = weight_decay
cfg.beta1 = training.beta1 or 0.9
cfg.beta2 = training.beta2 or 0.999
cfg.epsilon = training.epsilon or 1e-8
cfg.kl_beta = kl_beta
cfg.kl_warmup_steps = kl_warmup_steps
cfg.seed = seed
cfg.vocab_size = tokenizer_max_vocab

local function convert_image_channels(rgb)
  if cfg.image_c == 3 then return rgb end
  local converted = {}
  for index = 1, #rgb, 3 do
    local red = tonumber(rgb[index]) or 0
    local green = tonumber(rgb[index + 1]) or 0
    local blue = tonumber(rgb[index + 2]) or 0
    if cfg.image_c == 1 then
      converted[#converted + 1] = math.floor(
        0.299 * red + 0.587 * green + 0.114 * blue + 0.5)
    else
      converted[#converted + 1] = red
      converted[#converted + 1] = green
      converted[#converted + 1] = blue
      converted[#converted + 1] = 255
    end
  end
  return converted
end

local tokenizer_source_path = tokenizer_path
local resume_tokenizer_path = FS.join(resume_path, "tokenizer", "tokenizer.json")
if resume_path ~= "" and FS.file_exists(resume_tokenizer_path) then
  tokenizer_source_path = resume_tokenizer_path
end
local ok_tokenizer, tokenizer_err = BaseTok.load_base({
  path = tokenizer_source_path,
  max_vocab = tokenizer_max_vocab,
  require = require_tokenizer,
})
check(ok_tokenizer, tokenizer_err, "Tokenizer.load_or_create")
local ok_max_vocab, max_vocab_err = Mimir.Tokenizer.set_max_vocab(tokenizer_max_vocab)
check(ok_max_vocab, max_vocab_err, "Tokenizer.set_max_vocab")

local cache_path = Args.get_str(opts, "cache", dataset_root .. "/dataset_cache.json")
local ok_dataset, dataset_count_or_error = Mimir.Dataset.load(
  dataset_root, cfg.image_w, cfg.image_h, 2, true, cache_path, 10240, true)
check(ok_dataset, dataset_count_or_error, "Dataset.load")
local dataset_total = tonumber(dataset_count_or_error) or 0
local dataset_count = dataset_total
if max_items > 0 then dataset_count = math.min(dataset_count, max_items) end
if dataset_count < 1 then
  die("dataset vide; format attendu: <nom>.png|jpg accompagné de <nom>.txt")
end
local validation_capacity = validation_holdout and
  math.max(0, dataset_count - 1) or dataset_count
local validation_count = math.min(validation_items, validation_capacity)
local training_count = validation_holdout and
  (dataset_count - validation_count) or dataset_count
if validation_items > 0 and validation_count < validation_items then
  logf(
    "[train_lumen] validation réduite à %d item(s) pour conserver un item d'entraînement",
    validation_count)
end
logf(
  "[train_lumen] partition: train=%d validation=%d holdout=%s",
  training_count, validation_count, tostring(validation_holdout))

local function hydrate_tokenizer(reason)
  local stats, hydrate_err = Mimir.Dataset.hydrate_tokenizer()
  if type(stats) ~= "table" then
    die("Dataset.hydrate_tokenizer: " .. tostring(hydrate_err or "échec"))
  end
  logf(
    "[train_lumen] tokenizer hydraté (%s): captions=%d/%d vocab=%d->%d max=%d erreurs=%d",
    reason, stats.captions or 0, stats.items or 0, stats.vocab_before or 0,
    stats.vocab_after or 0, tokenizer_max_vocab, stats.errors or 0)
end

hydrate_tokenizer("dataset")
local tokenizer_vocab_size = math.floor(tonumber(Mimir.Tokenizer.vocab_size()) or 0)
if tokenizer_vocab_size < 7 then
  die("tokenizer incomplet après hydratation: vocab=" .. tokenizer_vocab_size)
end
if tokenizer_vocab_size > tokenizer_max_vocab then
  die(string.format("tokenizer trop grand: vocab=%d max=%d",
    tokenizer_vocab_size, tokenizer_max_vocab))
end
local ok_freeze, freeze_err = Mimir.Tokenizer.set_max_vocab(tokenizer_vocab_size)
check(ok_freeze, freeze_err, "Tokenizer.freeze_vocab")

if type(Mimir.Dataset.prepare_sequences) == "function" then
  local ok_sequences, sequences_err =
    Mimir.Dataset.prepare_sequences(cfg.text_seq_len or 77)
  check(ok_sequences, sequences_err, "Dataset.prepare_sequences")
end

FS.mkdir_p(output_dir)
local ok_source_save, source_save_err = Mimir.Tokenizer.save(tokenizer_path)
check(ok_source_save, source_save_err, "Tokenizer.save(source)")
local output_tokenizer_path = FS.join(output_dir, "tokenizer.json")
if FS.normalize(output_tokenizer_path) ~= FS.normalize(tokenizer_path) then
  local ok_output_save, output_save_err = Mimir.Tokenizer.save(output_tokenizer_path)
  check(ok_output_save, output_save_err, "Tokenizer.save(output)")
end
logf(
  "[train_lumen] tokenizer construit et sérialisé avant modèle: vocab=%d path=%s",
  tokenizer_vocab_size, output_tokenizer_path)
if prepare_tokenizer_only then
  log("[train_lumen] préparation tokenizer terminée; modèle non créé")
  return
end

local ok_create, create_err = Mimir.Model.create("lumen_diffusion", cfg)
check(ok_create, create_err, "Model.create")
local ok_allocate, allocate_err = Mimir.Model.allocate_params()
check(ok_allocate, allocate_err, "Model.allocate_params")

if resume_path ~= "" then
  local ok_resume, resume_err = Mimir.Serialization.load(resume_path, "raw_folder", {
    load_tokenizer = false,
    load_encoder = false,
    load_optimizer = true,
    strict_mode = false,
    validate_checksums = true,
  })
  check(ok_resume, resume_err, "Serialization.load")
  log("[train_lumen] reprise: " .. resume_path)
else
  local ok_init, init_err =
    Mimir.Model.init_weights(Args.get_str(opts, "init", "xavier"), seed)
  check(ok_init, init_err, "Model.init_weights")
end

local ok_build, build_err = Mimir.Model.build()
check(ok_build, build_err, "Model.build(tokenizer hydraté)")
logf(
  "[train_lumen] loss: epsilon_mse pure (KL/Wasserstein diagnostiques), " ..
  "kl_beta=%.8g kl_warmup_steps=%d",
  cfg.kl_beta, cfg.kl_warmup_steps)

local calibration_count = math.min(vae_calibration_items, training_count)
local ok_calibration, calibration_err = Mimir.Model.lumen_begin_vae_calibration()
check(ok_calibration, calibration_err, "Model.lumen_begin_vae_calibration")
for index = 1, calibration_count do
  local item, item_err = Mimir.Dataset.get(index)
  if type(item) ~= "table" or type(item.image) ~= "table" then
    die("calibration VAE item " .. index .. ": " ..
      tostring(item_err or "image absente"))
  end
  local stats, add_err = Mimir.Model.lumen_add_vae_calibration_image(
    convert_image_channels(item.image))
  if type(stats) ~= "table" then
    die(add_err or "ajout calibration VAE impossible")
  end
end
local calibration, finish_err = Mimir.Model.lumen_finish_vae_calibration()
if type(calibration) ~= "table" then
  die(finish_err or "finalisation calibration VAE impossible")
end
logf(
  "[train_lumen] VAE auto-calibré avant entraînement: items=%d/%d values=%d " ..
  "shift=%.9g scale=%.9g",
  calibration.items or 0, vae_calibration_items, calibration.values or 0,
  calibration.shift or 0.0, calibration.scale or 0.0)

check(Mimir.Serialization.save(output_dir .. "/starttrain.json", "debug_json", {
  save_tokenizer = true,
  save_encoder = false,
  include_git_info = true,
}), nil, "Serialization.save(starttrain)")

math.randomseed(seed)
local order = {}
for index = 1, training_count do order[index] = index end
local global_step = 0
local running_loss = 0.0
local total_loss = 0.0
local stop_requested = false
local validation_lr_scale = 1.0
local best_validation_score = math.huge
local validation_score_ema = nil
local validation_bad_rounds = 0
local validation_penalties = 0

local function save_checkpoint(label)
  local path = output_dir .. "/" .. label
  FS.mkdir_p(path)
  check(Mimir.Serialization.save(path, "raw_folder", {
    save_optimizer = true,
    save_tokenizer = true,
    save_encoder = false,
    include_checksums = true,
    include_git_info = true,
    include_optimizer_state = true,
  }), nil, "Serialization.save(" .. label .. ")")
  log("[train_lumen] checkpoint: " .. path)
end

local function run_validation(reason)
  if validation_count < 1 then return nil end
  if type(Mimir.Viz.validation_enabled) == "function" and
      not Mimir.Viz.validation_enabled() then
    return nil
  end
  local total = validation_count
  local loss_sum = 0.0
  local reconstruction_mae_sum = 0.0
  local reconstruction_mse_sum = 0.0
  local kl_sum = 0.0
  local wasserstein_sum = 0.0
  local entropy_sum = 0.0
  local moment_sum = 0.0
  local spatial_sum = 0.0
  local temporal_sum = 0.0
  Mimir.Viz.set_validation({
    in_progress = true, step = global_step, done = 0, total = total,
    has = false, ok = true, recon = 0.0, kl = 0.0, align = 0.0,
  })
  for offset = 0, total - 1 do
    if type(Mimir.Viz.validation_enabled) == "function" and
        not Mimir.Viz.validation_enabled() then
      Mimir.Viz.set_validation({
        in_progress = false, step = global_step, done = offset, total = total,
        has = false, ok = false,
      })
      logf(
        "[train_lumen] validation interrompue step=%d items=%d/%d",
        global_step, offset, total)
      return nil
    end
    local dataset_index = math.floor(dataset_count - offset)
    local item, item_err = Mimir.Dataset.get(dataset_index)
    if type(item) ~= "table" or type(item.image) ~= "table" then
      die("validation item " .. dataset_index .. ": " ..
        tostring(item_err or "image absente"))
    end
    local stats, validation_err = Mimir.Model.lumen_validate_step(
      convert_image_channels(item.image), tostring(item.text or ""),
      seed + dataset_index * 1000003)
    if type(stats) ~= "table" then
      die(validation_err or "validation diffusion impossible")
    end
    loss_sum = loss_sum + stats.loss
    reconstruction_mae_sum = reconstruction_mae_sum + stats.reconstruction_mae
    reconstruction_mse_sum = reconstruction_mse_sum + stats.reconstruction_mse
    kl_sum = kl_sum + stats.kl
    wasserstein_sum = wasserstein_sum + stats.wasserstein
    entropy_sum = entropy_sum + stats.entropy_diff
    moment_sum = moment_sum + stats.moment_mismatch
    spatial_sum = spatial_sum + stats.spatial_coherence
    temporal_sum = temporal_sum + stats.temporal_consistency
    Mimir.Viz.set_validation({
      in_progress = true, step = global_step, done = offset + 1, total = total,
      has = true, ok = true, recon = reconstruction_mae_sum / (offset + 1),
      kl = reconstruction_mse_sum / (offset + 1), align = loss_sum / (offset + 1),
    })
  end
  local epsilon_mse = loss_sum / total
  local reconstruction_mae = reconstruction_mae_sum / total
  local reconstruction_mse = reconstruction_mse_sum / total
  local kl = kl_sum / total
  local wasserstein = wasserstein_sum / total
  local entropy_diff = entropy_sum / total
  local moment_mismatch = moment_sum / total
  local spatial_coherence = spatial_sum / total
  local temporal_consistency = temporal_sum / total
  local validation_weight_sum = validation_epsilon_weight +
    validation_reconstruction_weight
  local validation_score =
    (validation_epsilon_weight * epsilon_mse +
     validation_reconstruction_weight * reconstruction_mae) /
    validation_weight_sum
  if not is_finite(validation_score) then
    validation_score_ema = validation_score
  elseif validation_score_ema == nil or not is_finite(validation_score_ema) then
    validation_score_ema = validation_score
  else
    validation_score_ema = validation_ema_alpha * validation_score +
      (1.0 - validation_ema_alpha) * validation_score_ema
  end
  local feedback = "NEUTRAL"
  local reward_penalty = 0.0
  if validation_feedback_enabled and global_step >= validation_feedback_min_steps then
    if not is_finite(validation_score_ema) then
      validation_bad_rounds = validation_penalty_patience
      feedback = "INVALID"
    elseif best_validation_score == math.huge then
      best_validation_score = validation_score_ema
      validation_bad_rounds = 0
      feedback = "BASELINE"
      if validation_save_best then save_checkpoint("validation_best") end
    else
      local denominator = math.max(math.abs(best_validation_score), 1e-12)
      reward_penalty = (best_validation_score - validation_score_ema) / denominator
      if reward_penalty > validation_improve_threshold then
        validation_lr_scale = math.min(validation_lr_scale_max,
          validation_lr_scale * validation_reward_factor)
        best_validation_score = validation_score_ema
        validation_bad_rounds = 0
        feedback = "REWARD"
        if validation_save_best then save_checkpoint("validation_best") end
      elseif reward_penalty < -validation_improve_threshold then
        validation_bad_rounds = validation_bad_rounds + 1
        feedback = "WATCH"
      else
        validation_bad_rounds = math.max(0, validation_bad_rounds - 1)
      end
    end
    if validation_bad_rounds >= validation_penalty_patience then
      validation_lr_scale = math.max(validation_lr_scale_min,
        validation_lr_scale * validation_penalty_factor)
      validation_bad_rounds = 0
      validation_penalties = validation_penalties + 1
      feedback = "PENALTY"
      if validation_max_penalties > 0 and
          validation_penalties >= validation_max_penalties then
        stop_requested = true
        feedback = "EARLY_STOP"
      end
    end
  end
  Mimir.Viz.set_validation({
    in_progress = false, step = global_step, done = total, total = total,
    has = true, ok = is_finite(validation_score),
    recon = reconstruction_mae, kl = reconstruction_mse, align = epsilon_mse,
    feedback = feedback == "REWARD" and "reward" or
      feedback == "PENALTY" and "penalty" or
      feedback == "WATCH" and "plateau" or
      feedback == "EARLY_STOP" and "penalty" or "none",
  })
  logf(
    "[train_lumen] validation=%s step=%d items=%d epsilon_mse=%.7g recon_mae=%.7g " ..
    "recon_mse=%.7g kl=%.7g wa=%.7g entropy=%.7g moment_miss=%.7g spatial_co=%.7g " ..
    "temporal=%.7g score=%.7g ema=%.7g feedback=%s signal=%+.6g lr_scale=%.6g " ..
    "bad=%d penalties=%d",
    reason, global_step, total, epsilon_mse, reconstruction_mae,
    reconstruction_mse, kl, wasserstein, entropy_diff, moment_mismatch,
    spatial_coherence, temporal_consistency, validation_score,
    validation_score_ema, feedback, reward_penalty, validation_lr_scale,
    validation_bad_rounds, validation_penalties)
  return validation_score
end

for epoch = 1, epochs do
  for index = training_count, 2, -1 do
    local other = math.random(index)
    order[index], order[other] = order[other], order[index]
  end

  for position = 1, training_count do
    local item, item_err = Mimir.Dataset.get(order[position])
    if type(item) ~= "table" then die(item_err or "Dataset.get impossible") end
    if type(item.image) ~= "table" then die("image absente à l'item " .. order[position]) end

    global_step = global_step + 1
    local prompt = tostring(item.text or "")
    if math.random() < cfg_dropout then prompt = "" end
    local warmup = warmup_steps > 0 and math.min(1.0, global_step / warmup_steps) or 1.0
    local step_lr = learning_rate * warmup * validation_lr_scale
    local step_seed = seed + global_step * 1000003

    local stats, step_err = Mimir.Model.lumen_train_step(
      convert_image_channels(item.image), prompt, step_seed, step_lr, optimizer, {
        epoch = epoch,
        total_epochs = epochs,
        batch = position,
        total_batches = training_count,
        step = global_step,
        completed_steps = global_step - 1,
        loss_sum_before = total_loss,
      })
    if type(stats) ~= "table" then die(step_err or "lumen_train_step impossible") end
    running_loss = running_loss + stats.loss
    total_loss = total_loss + stats.loss

    if stats.stop_requested then
      logf("[train_lumen] arrêt demandé depuis Viz au step=%d", global_step)
      stop_requested = true
      break
    end

    if global_step == 1 or global_step % log_every == 0 then
      local divisor = global_step == 1 and 1 or log_every
      logf(
        "[train_lumen] epoch=%d/%d step=%d item=%d/%d loss=%.7g avg=%.7g grad=%.6g " ..
        "mse=%.6g kl=%.6g kl_beta_eff=%.6g wa=%.6g ent=%.6g mom=%.6g " ..
        "spat=%.6g temp=%.6g t=%d lr=%.6g",
        epoch, epochs, global_step, position, training_count, stats.loss,
        running_loss / divisor, stats.grad_norm, stats.mse, stats.kl,
        stats.kl_beta_effective, stats.wasserstein,
        stats.entropy_diff, stats.moment_mismatch, stats.spatial_coherence,
        stats.temporal_consistency, stats.timestep, step_lr)
      running_loss = 0.0
    end

    if save_every > 0 and global_step % save_every == 0 then
      save_checkpoint(string.format("step_%08d", global_step))
    end
    if validation_every_steps > 0 and global_step % validation_every_steps == 0 then
      run_validation("step")
      if stop_requested then break end
    end
  end
  if stop_requested then break end
  if validation_every_epochs > 0 and epoch % validation_every_epochs == 0 then
    run_validation("epoch")
    if stop_requested then break end
  end
end

save_checkpoint("final")
check(Mimir.Serialization.save(output_dir .. "/endtrain.json", "debug_json", {
  save_tokenizer = true,
  save_encoder = false,
  include_git_info = true,
  include_optimizer_state = true,
}), nil, "Serialization.save(endtrain)")

logf("[train_lumen] terminé: epochs=%d steps=%d sortie=%s",
  epochs, global_step, output_dir)
