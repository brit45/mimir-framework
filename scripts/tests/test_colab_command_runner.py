"""Exercise the notebook runner against actual quiet/failing child processes."""
import ast
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class CommandRunner(unittest.TestCase):
    def runner(self, directory):
        notebook = json.loads((ROOT / 'Colab-bench-multi-framework.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if 'def run(' in ''.join(c['source']))
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        module = ast.Module(body=[node], type_ignores=[])
        scope = dict(LOG_DIR=Path(directory), datetime=datetime, timezone=timezone,
                     subprocess=subprocess, os=os, signal=signal, time=time)
        exec(compile(module, 'notebook-command-runner', 'exec'), scope)
        return scope['run']

    def test_quiet_child_has_progress_and_finishes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as captured:
            output = self.runner(directory)(
                [sys.executable, '-c', 'import time; time.sleep(.5); print("completed")'],
                timeout=5, heartbeat_interval=.1)
            self.assertEqual(output.strip(), 'completed')
            self.assertIn('actif depuis', captured.getvalue())

    def test_timeout_reports_last_stage_and_terminates_child(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            pid_file = Path(directory) / 'pid'
            code = ('import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
                    'print("GPU_NATIVE_STAGE runtime-shutdown",flush=True); time.sleep(60)')
            with self.assertRaisesRegex(TimeoutError, 'GPU_NATIVE_STAGE runtime-shutdown'):
                self.runner(directory)([sys.executable, '-c', code, str(pid_file)], timeout=.5)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_fatal_error_and_skip_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = self.runner(directory)
            with self.assertRaisesRegex(RuntimeError, 'GPU forward failed'):
                run([sys.executable, '-c', 'print("GPU forward failed"); raise SystemExit(1)'],
                    allowed_returncodes=(77,))
            output = run([sys.executable, '-c', 'print("SKIP: no device"); raise SystemExit(77)'],
                         allowed_returncodes=(77,))
            self.assertIn('SKIP:', output)

    def test_driver_ignoring_kill_does_not_block_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as captured:
            run = self.runner(directory)
            process = Mock(pid=424242)
            process.poll.return_value = None
            process.wait.side_effect = subprocess.TimeoutExpired('stuck-driver', 5)
            def start(*args, **kwargs):
                kwargs['stdout'].write('GPU_NATIVE_STAGE runtime-shutdown\n')
                kwargs['stdout'].flush()
                return process
            with patch.object(subprocess, 'Popen', side_effect=start), patch.object(os, 'killpg') as kill:
                with self.assertRaisesRegex(TimeoutError, 'GPU_NATIVE_STAGE runtime-shutdown'):
                    run(['stuck-driver'], timeout=-1)
            kill.assert_called_once_with(424242, signal.SIGKILL)
            process.wait.assert_called_once_with(timeout=5)
            self.assertIn('redémarrer cet environnement Colab', captured.getvalue())


if __name__ == '__main__':
    unittest.main()
