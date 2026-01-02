#!/bin/bash
# ============================================================================
# Script de migration API Lua vers la nouvelle syntaxe moderne
# Migre: guard.* → MemoryGuard.*, memory.* → Memory.*, allocator.* → Allocator.*
# ============================================================================

set -e

echo "🔄 Migration des scripts Lua vers la nouvelle API..."
echo ""

# Compteurs
total_files=0
migrated_files=0

# Trouver tous les fichiers .lua
while IFS= read -r file; do
    ((total_files++))
    modified=false
    
    # Sauvegarder le fichier original
    cp "$file" "$file.bak"
    
    # Appliquer les migrations avec sed
    sed -i \
        -e 's/\bguard\.set_limit\b/MemoryGuard.setLimit/g' \
        -e 's/\bguard\.get_stats\b/MemoryGuard.getStats/g' \
        -e 's/\bguard\.print_stats\b/MemoryGuard.printStats/g' \
        -e 's/\bguard\.reset\b/MemoryGuard.reset/g' \
        -e 's/\bmemory\.set_limit\b/Memory.setLimit/g' \
        -e 's/\bmemory\.get_stats\b/Memory.getStats/g' \
        -e 's/\bmemory\.print_stats\b/Memory.printStats/g' \
        -e 's/\bmemory\.clear\b/Memory.clear/g' \
        -e 's/\bmemory\.get_usage\b/Memory.getUsage/g' \
        -e 's/\bmemory\.config\b/Memory.config/g' \
        -e 's/\ballocator\.configure\b/Allocator.configure/g' \
        -e 's/\ballocator\.print_stats\b/Allocator.printStats/g' \
        -e 's/\ballocator\.get_stats\b/Allocator.getStats/g' \
        "$file"
    
    # Vérifier si le fichier a été modifié
    if ! diff -q "$file" "$file.bak" > /dev/null 2>&1; then
        echo "✓ Migré: $file"
        ((migrated_files++))
        modified=true
    fi
    
    # Supprimer le backup si aucune modification
    if [ "$modified" = false ]; then
        rm "$file.bak"
    fi
    
done < <(find scripts -name "*.lua" -type f)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Résumé:"
echo "  • Fichiers analysés: $total_files"
echo "  • Fichiers migrés:   $migrated_files"
echo "  • Fichiers inchangés: $((total_files - migrated_files))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Migration terminée!"
echo "   Les backups (.bak) des fichiers modifiés ont été conservés."
echo ""
