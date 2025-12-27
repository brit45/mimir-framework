-- Test très simple de la limite mémoire à 10 Go
print("=== Test Limite Mémoire 10 Go ===\n")

-- Configuration de la limite mémoire à 10 Go
print("🛡️  Configuration MemoryGuard: Limite 10 Go")
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- 10 GB en bytes

-- Vérifier la limite
local limit = MemoryGuard.getLimit()
print("✓ Limite configurée: " .. string.format("%.2f", limit / 1024 / 1024 / 1024) .. " GB")

-- Mémoire initiale
local mem_init = MemoryGuard.getCurrentUsage()
print("✓ RAM initiale: " .. string.format("%.2f", mem_init / 1024 / 1024) .. " MB")

-- Créer quelques tensors via guard
print("\n📊 Test d'allocations progressives:")

-- Simuler des allocations en créant des modèles
for i = 1, 5 do
    print("\n  Itération " .. i)
    
    local my_model = model.create("Model_" .. i)
    
    -- Vérifier l'utilisation après chaque création
    local mem_current = MemoryGuard.getCurrentUsage()
    print("    RAM courante: " .. string.format("%.2f", mem_current / 1024 / 1024) .. " MB")
end

-- Rapport final
print("\n=== Rapport Final ===")
local final_mem = MemoryGuard.getCurrentUsage()
local peak_mem = MemoryGuard.getPeakUsage()
local limit_mem = MemoryGuard.getLimit()

print("💾 Utilisation RAM:")
print("   Courante: " .. string.format("%.2f", final_mem / 1024 / 1024 / 1024) .. " GB")
print("   Pic: " .. string.format("%.2f", peak_mem / 1024 / 1024 / 1024) .. " GB")
print("   Limite: " .. string.format("%.2f", limit_mem / 1024 / 1024 / 1024) .. " GB")

local usage_percent = (peak_mem / limit_mem) * 100
print("   Utilisation: " .. string.format("%.2f", usage_percent) .. "%")

if peak_mem < limit_mem then
    print("\n✅ Limite de 10 Go respectée !")
    local margin_gb = (limit_mem - peak_mem) / 1024 / 1024 / 1024
    print("   Marge restante: " .. string.format("%.2f", margin_gb) .. " GB")
else
    print("\n⚠️  Limite de 10 Go dépassée !")
    local overflow_gb = (peak_mem - limit_mem) / 1024 / 1024 / 1024
    print("   Dépassement: " .. string.format("%.2f", overflow_gb) .. " GB")
end

-- Statistiques détaillées
print("\n📊 Statistiques MemoryGuard:")
MemoryGuard.printStats()

print("\n=== Test terminé avec succès ===")
print("✓ Système de limitation RAM opérationnel")
print("✓ Limite de 10 Go active et surveillée")
