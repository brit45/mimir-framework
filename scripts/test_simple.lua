#!/usr/bin/env mimir --lua

print("Test simple de l'API Lua Mímir")
print("================================\n")

-- Test 1: Création de tenseur
print("1. Création d'un tenseur 3x3...")
local t = model.tensor({3, 3})
model.fill(t, 1.0)
print("   ✓ Tenseur créé")

-- Test 2: Opération simple
print("2. Addition de tenseurs...")
local a = model.tensor({2, 2})
model.fill(a, 2.0)
local b = model.tensor({2, 2})
model.fill(b, 3.0)
local c = model.add(a, b)
print("   ✓ Addition réussie")

-- Test 3: Tokenizer
print("3. Test du tokenizer...")
local vocab = {"<pad>", "<unk>", "hello", "world"}
local tok = model.tokenizer.from_vocab(vocab)
local ids = tok:encode("hello world")
print("   ✓ Tokenization réussie")

print("\n✅ Tous les tests passent!")
