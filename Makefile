
.PHONY: all build build-static clean mk-video check-docs build-docs

FLAGS= -ffp-contract=fast \
	-funroll-loops \
	-fno-trapping-math \
	-fno-math-errno \
	-mf16c \
	-mbmi2 \
	-DUSE_HUGEPAGES

DOCS_OUTPUT_DIR := docs/output
DOCS_PDF := $(DOCS_OUTPUT_DIR)/Documentation_Mimir-FRAMEWORK.pdf
DOCS_HTML := $(DOCS_OUTPUT_DIR)/Documentation_Mimir-FRAMEWORK.html
DOCS_RESOURCE_PATH := .:docs:docs/01-Getting-Started:docs/02-User-Guide:docs/03-API-Reference:docs/04-Architecture-Internals:docs/05-Advanced:docs/06-Contributing:docs/07-Devs:docs/08-Tuto
PANDOC_DOCS_FLAGS := --from=markdown+tex_math_single_backslash --resource-path=$(DOCS_RESOURCE_PATH) --lua-filter=scripts/tools/pandoc_project_logo.lua

# Sources de la documentation publique, dans l'ordre de lecture du site.
# Les rapports générés de docs/graphs restent exclus du livre; seul leur index
# est inclus. Ajouter une nouvelle section numérotée implique de l'ajouter ici.
DOCS_SOURCES := \
	docs/00-INDEX.md \
	docs/00-PROJECT-STATUS.md \
	docs/00-Framework-Philosophy.md \
	docs/00-STYLE.md \
	$(sort $(wildcard docs/01-Getting-Started/*.md)) \
	$(sort $(wildcard docs/02-User-Guide/*.md)) \
	$(sort $(wildcard docs/03-API-Reference/*.md)) \
	$(sort $(wildcard docs/04-Architecture-Internals/*.md)) \
	$(sort $(wildcard docs/05-Advanced/*.md)) \
	$(sort $(wildcard docs/06-Contributing/*.md)) \
	$(sort $(wildcard docs/07-Devs/*.md)) \
	$(sort $(wildcard docs/08-Tuto/*.md)) \
	docs/graphs/README.md

all: build

build:
	@echo "🏗️  Build via CMake (recommandé)"
	@cmake -S . -B build
	@cmake --build build --parallel 12

build-static:
	@echo "🏗️  Build statique: mimir_static (sortie dans ./bin)"
	@cmake -S . -B build_static -DBUILD_MIMIR_STATIC=ON
	@cmake --build build_static --parallel 12 --target mimir_static



mk-video:
	@ffmpeg -framerate 8 -f image2 -pattern_type glob -i "generated_epoch_*.pgm" \
       -vf "format=yuv420p" -c:v libx264 -preset veryslow -crf 0 \
       output_raw.avi

clean:
	@echo "🧹 Nettoyage des fichiers de compilation..."
	@rm -f bin/*
	@rm -f src/*.o
	@rm -f *.o
	@echo "✓ Nettoyage terminé"

check-docs:
	@echo "🔎 Vérification de la documentation et de l'API Lua..."
	@./scripts/tools/verify_api_sync.sh
	@./scripts/tools/verify_docs.py

build-docs: check-docs
	@echo "📚 Génération de la documentation PDF..."
	@if ! command -v pandoc >/dev/null 2>&1; then \
		echo "❌ Erreur: pandoc n'est pas installé"; \
		echo "   Installer avec: sudo apt install pandoc texlive-xetex texlive-latex-extra"; \
		exit 1; \
	fi
	@mkdir -p $(DOCS_OUTPUT_DIR)
	@set -e; \
	if pandoc $(PANDOC_DOCS_FLAGS) $(DOCS_SOURCES) \
		-o $(DOCS_PDF) \
		--pdf-engine=xelatex \
		--toc \
		--toc-depth=3 \
		--number-sections \
		-V geometry:margin=2.5cm \
		-V fontsize=11pt \
		-V documentclass=report \
		--highlight-style=tango \
		--title-prefix "Mimir Framework"; then \
		echo "✓ Documentation PDF générée: $(DOCS_PDF)"; \
		ls -lh $(DOCS_PDF) | awk '{print "  Taille:", $$5}'; \
		exit 0; \
	fi; \
	echo "⚠️  Échec xelatex, tentative wkhtmltopdf..."; \
	if command -v wkhtmltopdf >/dev/null 2>&1; then \
		pandoc $(PANDOC_DOCS_FLAGS) $(DOCS_SOURCES) \
			-o $(DOCS_PDF) \
			--pdf-engine=wkhtmltopdf \
			--toc \
			--toc-depth=3; \
		echo "✓ Documentation PDF générée: $(DOCS_PDF)"; \
		ls -lh $(DOCS_PDF) | awk '{print "  Taille:", $$5}'; \
		exit 0; \
	fi; \
	echo "❌ Aucun moteur PDF disponible (xelatex/wkhtmltopdf)"; \
	echo "   Génération HTML à la place..."; \
	pandoc $(PANDOC_DOCS_FLAGS) $(DOCS_SOURCES) \
		-o $(DOCS_HTML) \
		--toc \
		--toc-depth=3 \
		--standalone \
		--embed-resources \
		--css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css; \
	echo "✓ Documentation HTML générée: $(DOCS_HTML)"; \
	ls -lh $(DOCS_HTML) | awk '{print "  Taille:", $$5}'
