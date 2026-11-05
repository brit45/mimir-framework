<?php
require dirname(__DIR__) . '/config.php';
try {
    respond((function () {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail('POST requis.', 405);
    if (!hash_equals($_SESSION['csrf'], $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '')) fail('Jeton de session invalide. Rechargez la page.', 403);
    if (!isset($_FILES['file']) || !is_array($_FILES['file'])) {
        fail('Aucun fichier reçu. Vérifiez la limite post_max_size de PHP, puis réessayez.');
    }
    $root = root();
    if (($_POST['dataset_root'] ?? null) !== $root) fail('Le dataset actif a changé dans un autre onglet. Rouvrez votre dossier.', 409);
    $file = $_FILES['file'];
    if (($file['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK) {
        $reason = match ($file['error'] ?? UPLOAD_ERR_NO_FILE) {
            UPLOAD_ERR_INI_SIZE, UPLOAD_ERR_FORM_SIZE => 'Image trop volumineuse pour la configuration PHP.',
            UPLOAD_ERR_PARTIAL => 'Transfert incomplet. Réessayez cette image.',
            UPLOAD_ERR_NO_FILE => 'Aucune image reçue.',
            default => 'Échec du transfert de l’image.',
        };
        fail($reason, 413);
    }
    $relative = $_POST['relative_path'] ?? null;
    if (!is_string($relative) || $relative === '' || str_contains($relative, '\\') ||
        str_starts_with($relative, '/') || str_contains($relative, "\0")) fail('Chemin d’image invalide.');
    $parts = explode('/', $relative);
    foreach ($parts as $part) {
        if ($part === '' || $part === '.' || $part === '..' || strlen($part) > 255 ||
            preg_match('/[\x00-\x1f\x7f]/', $part)) fail('Chemin d’image interdit.', 403);
    }
    if (count($parts) > 1 && count($parts) > 32) fail('Arborescence d’image trop profonde.');
    $filename = end($parts);
    $extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
    $isImage = in_array($extension, ['jpg','jpeg','png','webp','bmp'], true);
    $isSidecar = $extension === 'json';
    $isSummary = $filename === 'composition.mimir-meta';
    if (!$isImage && !$isSidecar && !$isSummary) fail('Format non pris en charge. Importez des images, annotations JSON ou composition.mimir-meta.');
    if (!is_uploaded_file($file['tmp_name'] ?? '')) fail('Fichier transféré invalide.');
    if ($isImage && @getimagesize($file['tmp_name']) === false) fail('Le fichier transféré ne contient pas une image décodable.');
    if ($isSidecar || $isSummary) {
        if ((int)$file['size'] > 4 * 1024 * 1024) fail('Le fichier de métadonnées dépasse 4 Mio.');
        $metadata = json_decode(file_get_contents($file['tmp_name']));
        if (!is_object($metadata)) fail('Le fichier de métadonnées doit contenir un objet JSON valide.');
    }
    $directory = $root;
    foreach (array_slice($parts, 0, -1) as $part) {
        $directory .= DIRECTORY_SEPARATOR . $part;
        if (is_link($directory)) fail('Les liens symboliques sont interdits dans le dataset.', 409);
        if (!is_dir($directory) && !mkdir($directory, 0755)) fail('Impossible de créer un sous-dossier du dataset.', 500);
        $resolved = realpath($directory);
        if (!$resolved || !inside($resolved, $root)) fail('Dossier d’import hors du dataset.', 403);
        $directory = $resolved;
    }
    $destination = $directory . DIRECTORY_SEPARATOR . end($parts);
    if (is_link($destination) || (file_exists($destination) && !is_file($destination))) fail('La destination n’est pas un fichier ordinaire.', 409);
    if (is_file($destination)) {
        if (filesize($destination) === (int)$file['size'] && hash_file('sha256', $destination) === hash_file('sha256', $file['tmp_name'])) {
            return ['relative_path'=>$relative,'skipped'=>true];
        }
        fail('Une image différente existe déjà à ce chemin. Aucun fichier n’a été remplacé.', 409);
    }
    $target = @fopen($destination, 'xb');
    if (!$target) fail('Impossible de créer le fichier sans écraser un contenu existant.', 409);
    $source = @fopen($file['tmp_name'], 'rb');
    if (!$source) {
        fclose($target); unlink($destination);
        fail('Impossible de lire l’image transférée.', 500);
    }
    $written = stream_copy_to_stream($source, $target);
    fclose($source); fflush($target); fclose($target);
    if ($written !== (int)$file['size']) {
        unlink($destination);
        fail('Écriture incomplète de l’image importée.', 500);
    }
    return ['relative_path'=>$relative,'skipped'=>false];
    })());
} catch (Throwable $e) {
    if (http_response_code() < 400) http_response_code(400);
    respond(['error'=>$e->getMessage()]);
}
