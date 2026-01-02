
.PHONY: mk-video clean build-docs

FLAGS= -ffp-contract=fast \
	-funroll-loops \
	-fno-trapping-math \
	-fno-math-errno \
	-mf16c \
	-mbmi2 \
	-DUSE_HUGEPAGES \

all: bin/mimir bin/model_architectures_demo

bin/mimir: src/*.cpp src/*.hpp src/LuaScripting.cpp src/LuaScripting.hpp src/Models/*.cpp src/Models/*.hpp src/Serialization/*.cpp src/Serialization/*.hpp
	@echo "🏭️  Compilation de Mímir Framework avec optimisations avancées..."
	@echo "   • FMA saturé (3 ops/cycle)"
	@echo "   • FP16 storage + F16C"
	@echo "   • BMI2 pour quantification"
	@echo "   • HugePages (2MB) + madvise"
	@echo "   • Architectures modernes prêtes à l'emploi"
	@echo "   • Compression LZ4 pour gestion mémoire"
	@echo "   • Module de sérialisation (SafeTensors, RawFolder, DebugJson)"
	g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp src/Encoder.cpp src/main.cpp src/Model.cpp src/Sha256.cpp src/stb_image_impl.cpp src/tensors.cpp src/Tokenizer.cpp src/Visualizer.cpp src/LuaScripting.cpp src/Models/FluxModel.cpp src/Models/VAEModel.cpp src/Serialization/Serialization.cpp src/Serialization/SafeTensorsWriter.cpp src/Serialization/SafeTensorsReader.cpp src/Serialization/RawCheckpointWriter.cpp src/Serialization/RawCheckpointReader.cpp src/Serialization/DebugJsonDump.cpp -I./src -I/usr/include/lua5.3 -o bin/mimir -lOpenCL -lsfml-graphics -lsfml-window -lsfml-system -llua5.3 -llz4 -lvulkan -fopenmp $(FLAGS)
	@echo "✓ Mímir Framework compilé avec hardware opt: bin/mimir"
	@ls -lh bin/mimir | awk '{print "  Taille:", $$5}'

bin/model_architectures_demo: examples/model_architectures_demo.cpp src/Model.cpp src/Encoder.cpp src/Sha256.cpp src/tensors.cpp src/Tokenizer.cpp src/Visualizer.cpp
	@echo "🔧 Compilation de l'exemple d'architectures..."
	g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp examples/model_architectures_demo.cpp src/Model.cpp src/Encoder.cpp src/Sha256.cpp src/tensors.cpp src/Tokenizer.cpp src/Visualizer.cpp -I./src -o bin/model_architectures_demo -lOpenCL -lsfml-graphics -lsfml-window -lsfml-system -llz4 -lvulkan -fopenmp $(FLAGS)
	@echo "✓ Exemple compilé: bin/model_architectures_demo"



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
