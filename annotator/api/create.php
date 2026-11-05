<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) {
    $parent = realpath($d['parent'] ?? '');
    if (!$parent || !is_dir($parent) || !is_writable($parent) || $parent === DIRECTORY_SEPARATOR) {
        fail('Choisissez un dossier parent existant, accessible en écriture et différent de la racine.');
    }
    $name = $d['name'] ?? null;
    if (!is_string($name) || $name !== trim($name) || $name === '.' || $name === '..' ||
        !preg_match('/^[\pL\pN][\pL\pN ._-]{0,99}$/u', $name) || str_ends_with($name, '.') || str_ends_with($name, ' ')) {
        fail('Nom de dataset invalide. Utilisez un nom simple sans séparateur de chemin.');
    }
    $path = $parent . DIRECTORY_SEPARATOR . $name;
    if (file_exists($path) || is_link($path)) fail('Un fichier ou dossier porte déjà ce nom.', 409);
    if (!mkdir($path, 0755)) fail('Impossible de créer le dossier du dataset.', 500);
    $root = realpath($path);
    if (!$root || dirname($root) !== $parent) fail('Le dossier créé ne se trouve pas dans le parent choisi.', 500);
    $_SESSION['dataset_root'] = $root;
    $_SESSION['images'] = [];
    return ['root'=>$root];
});
