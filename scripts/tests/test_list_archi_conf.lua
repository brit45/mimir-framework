Mimir.Allocator.configure({
  enable_compression = true,
  swap_strategy = "lru",
  max_ram_gb = 10,
  offload_threshold_mb = 1000
})

local name , err = Mimir.Architectures.available()

if err then

  log("[ERROR] ".. err)

end


log("\n\n* Liste des architectures supporté par le Framework actuellement : ")


for i, n in pairs(name) do

  local arch = n
  local conf_arch = Mimir.Architectures.default_config(arch)

  log("\t → " .. conf_arch)

end

local cfg = Mimir.Architectures.default_config("vae_conv")


local m_ok, m_err = Mimir.Model.create("vae_conv", cfg)

if not m_ok then

  log("[ERROR] " .. tostring(m_err))

end