"""Keep self-contained Colab snapshots limited to live source inputs."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('colab_embed_sources', ROOT / 'scripts/colab/embed_sources.py')
embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed)


class SourceSnapshot(unittest.TestCase):
    def test_inputs_after_in_source_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            files = {
                '.gitignore': 'CMakeFiles/\nTests/PortableNativeOps\n',
                'CMakeLists.txt': 'project(fixture)\n',
                'src/tracked.hpp': 'old\n',
                'src/new.hpp': 'new\n',
                'ide/rust/src/lib.rs': '// SDK build input\n',
                'Tests/test_case.cpp': '// test source\n',
                'Tests/PortableNativeOps': '\x7fELFbinary',
                'Tests/unignored_probe': '\x7fELFbinary',
                'Tests/CMakeFiles/generated.cpp': '// build output\n',
                'Tests/CTestTestfile.cmake': '# generated\n',
                'Tests/Makefile': '# generated\n',
                'shaders/conv2d.comp': '// GLSL\n',
                'shaders/conv2d.comp.spv': 'binary',
                'outputs/result.json': '{}',
            }
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            subprocess.run(['git', '-C', root, 'add', 'src/tracked.hpp'], check=True)
            (root / 'src/tracked.hpp').write_text('local change\n')
            with patch.object(embed, 'ROOT', root):
                names = {str(path.relative_to(root)) for path in embed.source_files()}
            self.assertEqual(names, {'CMakeLists.txt', 'src/tracked.hpp', 'src/new.hpp',
                                    'ide/rust/src/lib.rs', 'Tests/test_case.cpp', 'shaders/conv2d.comp'})

    def test_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            (root / 'src').mkdir()
            (root / 'src/alias.hpp').symlink_to('/tmp/mimir-unpackaged-input.hpp')
            with patch.object(embed, 'ROOT', root), self.assertRaisesRegex(RuntimeError, 'Source symlink'):
                embed.source_files()


if __name__ == '__main__':
    unittest.main()
