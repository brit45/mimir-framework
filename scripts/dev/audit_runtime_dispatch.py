#!/usr/bin/env python3
"""Inventory runtime sources and reject numerical/device dispatch leaks.

This is a source boundary check, not a numerical or hardware execution test.
Run from any directory; --output optionally saves the inventory as JSON.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = re.compile(
    r"\b(?:clEnqueueNDRangeKernel|vkCmdDispatch|vkCmdDispatchIndirect|vkQueueSubmit|"
    r"cudaLaunchKernel|hipLaunchKernelGGL|hipLaunchKernel|"
    r"cublas\w*[Gg]emm\w*|rocblas_\w*gemm\w*)\s*\(|<<<")
DIRECT_CPU = re.compile(
    r"\b(?:LayerOps|LayerOpsExt|CPUConv|NormalizationKernels|"
    r"CausalAttentionOps|CPUPooling)::\w+\s*\(")


def uncomment(text):
    # Preserve line numbers; comment markers inside C++ strings are retained.
    pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/')
    return pattern.sub(lambda m: re.sub(r"[^\n]", " ", m[0])
                       if m[0].startswith(("//", "/*")) else m[0], text)


def audit():
    violations, launches, backends = [], [], {}
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix not in {".cpp", ".hpp", ".h", ".cu", ".cuh", ".cl", ".comp", ".py"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        runtime = relative.startswith("src/runtimes/")
        source = path.read_text(encoding="utf-8")
        code = uncomment(source) if path.suffix not in {".py", ".cl", ".comp"} else source
        if runtime:
            backend = path.relative_to(ROOT / "src/runtimes").parts[0]
            if "/" not in str(path.relative_to(ROOT / "src/runtimes")):
                backend = "shared"
            backends.setdefault(backend, []).append(relative)
        for match in LAUNCH.finditer(code):
            item = {"file": relative, "line": code.count("\n", 0, match.start()) + 1,
                    "call": match[0]}
            launches.append(item)
            if not runtime:
                violations.append({**item, "reason": "device launch outside runtime"})
        if not runtime:
            for match in DIRECT_CPU.finditer(code):
                violations.append({"file": relative,
                    "line": code.count("\n", 0, match.start()) + 1,
                    "reason": "direct CPU calculation outside runtime", "call": match[0]})
    layer_source = uncomment((ROOT / "src/Layers.hpp").read_text())
    if re.search(r"namespace\s+(?:Activation|Conv|Pooling|Normalization|Operations)\s*\{", layer_source):
        violations.append({"file": "src/Layers.hpp", "reason": "numerical namespace in metadata header"})
    return {"scope": "source boundary; execution requires backend parity tests",
            "runtime_files": backends, "device_launches": launches, "violations": violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for violation in report["violations"]:
        print(json.dumps(violation, ensure_ascii=False))
    print(f"runtime audit: {sum(map(len, report['runtime_files'].values()))} source files, "
          f"{len(report['device_launches'])} device launch sites, "
          f"{len(report['violations'])} boundary violations")
    return bool(report["violations"])


if __name__ == "__main__":
    raise SystemExit(main())
