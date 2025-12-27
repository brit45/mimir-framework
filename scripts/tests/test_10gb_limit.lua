-- Test simple de la limite mémoire à 10 Go
print("=== Test Limite Mémoire 10 Go ===\n")

-- Configuration de la limite mémoire à 10 Go
print("🛡️  Configuration MemoryGuard: Limite 10 Go")
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- 10 GB en bytes
print("")

-- Vérifier la limite
local limit = MemoryGuard.getLimit()
print("✓ Limite configurée: " .. string.format("%.2f", limit / 1024 / 1024 / 1024) .. " GB\n")

-- Test 1: Petite allocation
print("📊 Test 1: Petite allocation (100 MB)")
local mem_before = MemoryGuard.getCurrentUsage()
print("   RAM avant: " .. string.format("%.2f", mem_before / 1024 / 1024) .. " MB")

-- Créer un modèle simple
local my_model = model.create("TestModel")
model.push_layer("linear", {input_size=1024, output_size=1024})
model.allocate_params()

local mem_after = MemoryGuard.getCurrentUsage()
print("   RAM après: " .. string.format("%.2f", mem_after / 1024 / 1024) .. " MB")
print("   Allocation: " .. string.format("%.2f", (mem_after - mem_before) / 1024 / 1024) .. " MB")

-- Test 2: Allocation moyenne
print("\n📊 Test 2: Allocation moyenne (~500 MB)")
local my_model2 = model.create("TestModel2")
for i = 1, 10 do
    model.push_layer("linear", {input_size=4096, output_size=4096})
end
model.allocate_params()

local mem_after2 = MemoryGuard.getCurrentUsage()
print("   RAM totale: " .. string.format("%.2f", mem_after2 / 1024 / 1024 / 1024) .. " GB")

-- Test 3: Grande allocation (mais < 10 GB)
print("\n📊 Test 3: Grande allocation (~2 GB)")
local my_model3 = model.create("TestModel3")
for i = 1, 50 do
    model.push_layer("linear", {input_size=8192, output_size=8192})
end

mem_before = MemoryGuard.getCurrentUsage()
model.allocate_params()
mem_after = MemoryGuard.getCurrentUsage()

print("   RAM avant: " .. string.format("%.2f", mem_before / 1024 / 1024 / 1024) .. " GB")
print("   RAM après: " .. string.format("%.2f", mem_after / 1024 / 1024 / 1024) .. " GB")
print("   Allocation: " .. string.format("%.2f", (mem_after - mem_before) / 1024 / 1024 / 1024) .. " GB")

-- Rapport final
print("\n=== Rapport Mémoire Final ===")
local final_mem = MemoryGuard.getCurrentUsage()
local peak_mem = MemoryGuard.getPeakUsage()
local limit_mem = MemoryGuard.getLimit()

print("💾 Utilisation RAM:")
print("   Courante: " .. string.format("%.2f", final_mem / 1024 / 1024 / 1024) .. " GB")
print("   Pic: " .. string.format("%.2f", peak_mem / 1024 / 1024 / 1024) .. " GB")
print("   Limite: " .. string.format("%.2f", limit_mem / 1024 / 1024 / 1024) .. " GB")
print("   Utilisation: " .. string.format("%.1f", (peak_mem / limit_mem) * 100) .. "%")

if peak_mem < limit_mem then
    print("\n✅ Limite de 10 Go respectée !")
    print("   Marge restante: " .. string.format("%.2f", (limit_mem - peak_mem) / 1024 / 1024 / 1024) .. " GB")
else
    print("\n⚠️  Limite de 10 Go dépassée !")
    print("   Dépassement: " .. string.format("%.2f", (peak_mem - limit_mem) / 1024 / 1024 / 1024) .. " GB")
end

-- Statistiques détaillées
print("\n📊 Statistiques MemoryGuard:")
MemoryGuard.printStats()

print("\n=== Test terminé ===")
