#!/usr/bin/env python3
"""
Automated CPU-backend patcher for Chatterbox.
Applies memory-efficient zero-copy safetensors streaming and device-agnostic
(CPU-safe FP32) model placement for CPU-only inference.
"""

import sys
import subprocess
from pathlib import Path

def apply_patch(chatterbox_dir: Path):
    chatterbox_dir = chatterbox_dir.resolve()
    patch_file = Path(__file__).resolve().parent.parent / "resources" / "chatterbox_cpu.patch"

    if not patch_file.exists():
        print(f"Error: Patch file not found at {patch_file}")
        sys.exit(1)

    print(f"Applying Chatterbox CPU backend patch to {chatterbox_dir}...")
    try:
        # First check if patch applies cleanly
        res = subprocess.run(
            ["git", "-C", str(chatterbox_dir), "apply", "--check", str(patch_file)],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            subprocess.run(
                ["git", "-C", str(chatterbox_dir), "apply", str(patch_file)],
                check=True
            )
            print("✓ Patch successfully applied via git apply!")
            return
        else:
            print("Notice: 'git apply --check' reported conflicts (possibly already applied or dirty tree).")
            # Check if already applied
            check_already = subprocess.run(
                ["git", "-C", str(chatterbox_dir), "apply", "--reverse", "--check", str(patch_file)],
                capture_output=True,
                text=True
            )
            if check_already.returncode == 0:
                print("✓ Optimizations are ALREADY applied to this repository!")
                return
            
            # Fallback to patch utility
            print("Attempting with patch utility...")
            subprocess.run(
                ["patch", "-p1", "-d", str(chatterbox_dir), "-i", str(patch_file)],
                check=True
            )
            print("✓ Patch successfully applied via patch utility!")
    except Exception as e:
        print(f"Error applying patch: {e}")
        print("Please check resources/chatterbox_cpu.patch manually.")
        sys.exit(1)

def _find_chatterbox_dir(arg_target: str | None = None) -> Path:
    if arg_target:
        return Path(arg_target)
    if Path("chatterbox").exists():
        return Path("chatterbox")
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (curr / "chatterbox").exists():
            return curr / "chatterbox"
        curr = curr.parent
    return Path("chatterbox")

if __name__ == "__main__":
    target = _find_chatterbox_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    apply_patch(target)
