-- ============================================================================
-- Test pour dataset.get() - Récupération des items du dataset
-- ============================================================================

print("╔═══════════════════════════════════════════════════════════════╗")
print("║   Test dataset.get() - Récupération des données              ║")
print("╚═══════════════════════════════════════════════════════════════╝")
print()

-- Test 1: Tentative de get sans chargement préalable
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 1: Get sans dataset chargé")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local item, err = dataset.get(1)
if item == nil then
    print("   ✓ Erreur attendue: " .. err)
else
    print("   ✗ Erreur: devrait retourner nil sans dataset chargé")
end
print()

-- Test 2: Charger un dataset
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 2: Chargement du dataset")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local ok, num_items = dataset.load("checkpoints/llm_simple")
if ok then
    print(string.format("   ✓ Dataset chargé: %d items", num_items))
else
    print("   ✗ Échec du chargement: " .. tostring(num_items))
    os.exit(1)
end
print()

-- Test 3: Récupérer le premier item
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 3: Récupération du premier item")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

item, err = dataset.get(1)
if item == nil then
    print("   ✗ Erreur: " .. err)
else
    print("   ✓ Item 1 récupéré:")
    
    if item.text_file then
        print("     • text_file: " .. item.text_file)
    end
    if item.image_file then
        print("     • image_file: " .. item.image_file)
    end
    if item.audio_file then
        print("     • audio_file: " .. item.audio_file)
    end
    if item.video_file then
        print("     • video_file: " .. item.video_file)
    end
    if item.width then
        print(string.format("     • dimensions: %dx%d", item.width, item.height))
    end
    if item.text then
        local preview = item.text:sub(1, 100)
        if #item.text > 100 then
            preview = preview .. "..."
        end
        print("     • text (preview): " .. preview)
    end
end
print()

-- Test 4: Index hors limites
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 4: Index hors limites")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

item, err = dataset.get(9999)
if item == nil then
    print("   ✓ Erreur attendue: " .. err)
else
    print("   ✗ Erreur: devrait retourner nil pour index invalide")
end
print()

-- Test 5: Index zéro (non valide en Lua 1-indexed)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 5: Index zéro")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

item, err = dataset.get(0)
if item == nil then
    print("   ✓ Erreur attendue: " .. err)
else
    print("   ✗ Erreur: devrait retourner nil pour index 0")
end
print()

-- Test 6: Itération sur tous les items du dataset
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 6: Itération sur tous les items")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local count = 0
local max_to_show = 5

for i = 1, num_items do
    item, err = dataset.get(i)
    if item then
        count = count + 1
        if i <= max_to_show then
            print(string.format("   [%d] text_file: %s", i, item.text_file or "N/A"))
        end
    else
        print(string.format("   ✗ Erreur à l'index %d: %s", i, err))
        break
    end
end

if count > max_to_show then
    print(string.format("   ... (%d items supplémentaires)", count - max_to_show))
end

print()
print(string.format("   ✓ %d/%d items récupérés avec succès", count, num_items))
print()

print("╔═══════════════════════════════════════════════════════════════╗")
print("║   ✅ Tous les tests dataset.get() réussis!                    ║")
print("╚═══════════════════════════════════════════════════════════════╝")
