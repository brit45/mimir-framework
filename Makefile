
.PHONY: all build clean mk-video build-docs

FLAGS= -ffp-contract=fast \
	-funroll-loops \
	-fno-trapping-math \
	-fno-math-errno \
	-mf16c \
	-mbmi2 \
	-DUSE_HUGEPAGES \

all: build

build:
	@echo "🏗️  Build via CMake (recommandé)"
	@cmake -S . -B build
	@cmake --build build -j



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

build-docs:
	@echo "📚 Génération de la documentation PDF..."
	@if ! command -v pandoc >/dev/null 2>&1; then \
		echo "❌ Erreur: pandoc n'est pas installé"; \
		echo "   Installer avec: sudo apt install pandoc texlive-latex-base texlive-latex-extra"; \
		exit 1; \
	fi
	@mkdir -p docs/output
	@pandoc docs/index.md docs/README.md docs/Model.md docs/Tokenizer.md docs/Encoder.md docs/HtopDisplay.md docs/Sha256.md docs/Helpers.md docs/Tensor.md docs/Visualizer.md \
		-o docs/output/Documentation_Tensor.pdf \
		--pdf-engine=xelatex \
		--toc \
		--toc-depth=3 \
		--number-sections \
		-V geometry:margin=2.5cm \
		-V fontsize=11pt \
		-V documentclass=report \
		--highlight-style=tango \
		--title-prefix "openTensor" \
		2>&1 || { \
			echo "⚠️  pdflatex non disponible, tentative avec wkhtmltopdf..."; \
			if command -v wkhtmltopdf >/dev/null 2>&1; then \
				pandoc docs/index.md docs/README.md docs/Model.md docs/Tokenizer.md docs/Encoder.md docs/HtopDisplay.md docs/Sha256.md docs/Helpers.md docs/Tensor.md docs/Visualizer.md \
					-o docs/output/Documentation_Tensor.pdf \
					--pdf-engine=wkhtmltopdf \
					--toc \
					--toc-depth=3; \
			else \
				echo "❌ Aucun moteur PDF disponible"; \
				echo "   Génération HTML à la place..."; \
				pandoc docs/index.md docs/README.md docs/Model.md docs/Tokenizer.md docs/Encoder.md docs/HtopDisplay.md docs/Sha256.md docs/Helpers.md docs/Tensor.md docs/Visualizer.md \
					-o docs/output/Documentation_Tensor.html \
					--toc \
					--toc-depth=3 \
					--standalone \
					--css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css; \
				echo "✓ Documentation HTML générée: docs/output/Documentation_Tensor.pdf"; \
				exit 0; \
			fi; \
		}
	@if [ -f docs/output/Documentation_Tensor.html ]; then \
		echo "✓ Documentation PDF générée: docs/output/Documentation_Tensor.pdf"; \
		ls -lh docs/output/Documentation_Tensor.*| awk '{print "  Taille:", $$5}'; \
	fi
