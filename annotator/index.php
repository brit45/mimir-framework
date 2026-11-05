<?php require __DIR__ . '/config.php'; ?>
<!doctype html>
<html lang="fr">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta name="csrf-token" content="<?= htmlspecialchars($_SESSION['csrf']) ?>">
    <title>Lumen · Atelier d’annotation</title>
    <link rel="stylesheet" href="assets/style.css">
    <script defer src="assets/skeleton-links.js"></script>
    <script defer src="assets/app.js"></script>
</head>

<body>
    <header>
        <div><span class="eyebrow">MIMIR / DATASET</span>
            <h1>Atelier d’annotation</h1>
        </div><span id="status" class="badge">Ouvrez un dataset</span>
    </header>
    <details id="create-dataset">
        <summary>Créer un dataset et importer des images</summary>
        <form id="create-form"><label for="dataset-parent">Dossier parent (existant et accessible)</label>
            <div class="row"><input id="dataset-parent" placeholder="/chemin/vers/mes-datasets" required><button id="browse-parent" type="button">Parcourir…</button></div><label for="dataset-name">Nom du dataset</label><input id="dataset-name" placeholder="mon-dataset" required pattern="[\p{L}\p{N}][\p{L}\p{N} ._-]{0,99}" title="Nom simple sans séparateur de chemin">
            <div class="import-pickers"><label>Choisir des images ou métadonnées<input id="import-files" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.json,.mimir-meta,image/jpeg,image/png,image/webp,image/bmp,application/json" multiple></label><label>Choisir un dossier de dataset<input id="import-folder" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.json,.mimir-meta,image/jpeg,image/png,image/webp,image/bmp,application/json" multiple webkitdirectory directory></label></div>
            <p class="hint">Les images, annotations JSON et le résumé Mímir du dossier sont copiés avec leurs sous-dossiers. Les images déjà composées pré-remplissent les boîtes et classes dans Annoter. Les fichiers identiques sont ignorés; rien n’est écrasé.</p>
            <div class="row"><button id="create-submit" class="primary">Créer et importer</button><progress id="import-progress" max="1" value="0" hidden></progress><span id="import-status" aria-live="polite"></span></div>
        </form>
    </details>
    <dialog id="directory-dialog" aria-labelledby="directory-title">
        <h2 id="directory-title">Sélectionner un dossier</h2>
        <div class="directory-header"><strong id="directory-current"></strong><button id="directory-up" type="button">↑ Parent</button></div>
        <div id="directory-list" class="directory-list"></div>
        <div class="row"><button id="directory-select" class="primary" type="button">Choisir ce dossier</button><button id="directory-cancel" type="button">Annuler</button></div>
    </dialog>
    <form id="open"><label for="root">Dossier racine du dataset à ouvrir ou modifier</label>
        <div class="row"><input id="root" placeholder="/chemin/vers/dataset" required><button id="browse-root" type="button">Choisir un dossier…</button><button>Ouvrir / rescanner</button></div>
    </form>
    <div id="message" role="status" aria-live="polite"></div>
    <details id="warnings" hidden>
        <summary>Fichiers à vérifier</summary>
        <pre></pre>
    </details>
    <section id="workspace" hidden>
        <div class="progress"><span id="counter"></span><span id="counts"></span><progress id="progress" max="100" value="0"></progress></div>
        <nav aria-label="Navigation"><button id="prev">← Précédente</button><button id="save" class="primary">Sauvegarder</button><button id="save-next" class="primary">Sauvegarder et suivante →</button><button id="next">Suivante →</button><button id="unannotated">Image non annotée suivante</button><button id="last">Dernière annotation</button></nav>
        <main>
            <section class="viewer">
                <div class="viewer-title"><strong id="filename"></strong><span id="dimensions"></span></div>
                <div id="stage"><img id="image" alt="Image du dataset à annoter" draggable="false"><canvas id="canvas" tabindex="0" aria-label="Dessiner, sélectionner, déplacer et redimensionner les boîtes, masques et points de squelette des entités"></canvas></div>
                <p class="hint">Glisser sur l’image : créer une boîte · Cliquer : sélectionner · Glisser une boîte : déplacer · Poignées : redimensionner · Suppr : supprimer la boîte.</p>
                <div class="row"><button id="draw">Dessiner une nouvelle boîte</button><button id="draw-mask">Tracer un masque de décor</button><button id="draw-ocr">Annoter une zone OCR</button><button id="remove-box">Supprimer la boîte sélectionnée</button></div>
                <div class="row track-tools"><label for="track-identity">Identité connue</label><select id="track-identity" aria-label="Identité connue">
                        <option value="">Choisir une identité suivie</option>
                    </select><button id="draw-track-occurrence">Tracer une occurrence</button></div><datalist id="track-identities"></datalist>
                <p class="hint">Les identités suivies sont partagées entre les images de ce dataset et retrouvées lors d’un rescannage.</p>
            </section>
            <section class="editor">
                <div class="panel">
                    <h2>Simulation des attaches</h2><label class="check"><input id="simulate-links" type="checkbox"> Vérifier les liens squelette → entité</label>
                    <p class="hint">Estimation géométrique à partir des points, boîtes et masques : elle aide à vérifier le placement, sans reconnaître le contenu de l’image. Les racines vérifient aussi l’attache à l’entité propriétaire.</p><label for="link-tolerance">Tolérance (% de la diagonale de l’image)</label><input id="link-tolerance" type="number" min="0" max="10" step="0.5" value="3">
                    <div id="link-results" aria-live="polite"></div>
                </div>
                <div class="panel">
                    <h2>Description naturelle</h2><label for="caption">Description principale</label><textarea id="caption" rows="4" placeholder="Une femme aux cheveux noirs se tient près d’une fenêtre, éclairée par une lumière chaude…"></textarea><label for="alternatives">Descriptions alternatives (une par ligne)</label><textarea id="alternatives" rows="2"></textarea>
                </div>
                <div class="panel">
                    <div class="row between">
                        <h2>Entités</h2><button id="add-entity">+ Entité</button>
                    </div>
                    <div id="entities"></div>
                </div>
                <div class="panel">
                    <div class="row between">
                        <h2>Attributs</h2><button id="add-attribute">+ Attribut</button>
                    </div>
                    <div id="attributes"></div>
                </div>
                <div class="panel">
                    <div class="row between">
                        <h2>Relations</h2><button id="add-relation">+ Relation</button>
                    </div>
                    <div id="relations"></div>
                </div>
                <div class="panel">
                    <h2>Éclairage et composition</h2><label for="light-description">Éclairage</label><textarea id="light-description" rows="2"></textarea><label for="light-source">Source</label><input id="light-source"><label for="light-target">Cible</label><input id="light-target"><label for="composition">Composition</label><textarea id="composition" rows="2"></textarea>
                </div>
                <div class="panel">
                    <h2>Vérification</h2><label class="check"><input id="caption-verified" type="checkbox"> Description vérifiée</label><label class="check"><input id="relations-verified" type="checkbox"> Relations vérifiées</label>
                </div>
            </section>
        </main>
    </section>
</body>

</html>