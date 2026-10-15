#!/usr/bin/env python3
"""
Chatterbox Voice Clone CLI for NVIDIA Quadro T2000 & Turing Architecture GPUs (4GB VRAM)
Synthesizes speech in your cloned voice offline using Chatterbox / Chatterbox-Nano on CUDA.
Optimized for Turing Architecture (Compute Capability 7.5, sm_75) with FP16 Tensor Core math.
"""

import os
import sys
from pathlib import Path

# 1. Automatically re-exec into project's .venv if running with system python
def _find_venv() -> Path | None:
    cwd_venv = Path.cwd() / ".venv"
    if (cwd_venv / "bin" / "python3").exists():
        return cwd_venv
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = curr / ".venv"
        if (candidate / "bin" / "python3").exists():
            return candidate
        curr = curr.parent
    return None

_venv_dir = _find_venv()
if _venv_dir:
    _venv_python = _venv_dir / "bin" / "python3"
    if _venv_python.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

# 2. Configure PyTorch caching allocator before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import re
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

# 3. Enable Ampere+ TF32 if supported on the host device
if torch.cuda.is_available():
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    ChatterboxTurboTTS = None


def split_into_sentences(text: str) -> list[str]:
    """Split text into manageable sentence chunks for speech synthesis."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > 250:
                sub_parts = re.split(r'(?<=[,;])\s+', s)
                accum = ""
                for part in sub_parts:
                    if len(accum) + len(part) < 220:
                        accum = f"{accum} {part}".strip()
                    else:
                        if accum:
                            chunks.append(accum)
                        accum = part
                if accum:
                    chunks.append(accum)
            else:
                chunks.append(s)

    return chunks


def play_audio(audio_path: Path):
    """Play generated audio via available Linux audio players or Windows host in WSL2."""
    audio_path = audio_path.resolve()
    # Check for WSL environment with Windows powershell
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists() and shutil.which("powershell.exe"):
        try:
            wslpath_cmd = ["wslpath", "-w", str(audio_path)]
            win_path = subprocess.check_output(wslpath_cmd, text=True).strip()
            ps_script = f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_script], check=True)
            return
        except Exception:
            pass

    # Native Linux audio playback
    for player in ["paplay", "aplay", "ffplay", "mpv", "pw-play"]:
        if shutil.which(player):
            try:
                cmd = [player]
                if player == "ffplay":
                    cmd += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
                cmd.append(str(audio_path))
                subprocess.run(cmd, check=True)
                return
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Clone voice and synthesize speech on NVIDIA Quadro T2000 & Turing Architecture GPUs (4GB VRAM)."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--text", "-t", type=str, help="Text string to synthesize"
    )
    group.add_argument(
        "--script", "-s", type=str, help="Path to a text file (.txt) containing the script to synthesize"
    )

    parser.add_argument(
        "--reference", "-r",
        type=str,
        default="sample_reference.wav",
        help="Path to clean 5-15s WAV reference voice sample (default: sample_reference.wav)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Path where synthesized WAV file will be saved (default: output.wav)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to run inference on: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--nano",
        action="store_true",
        default=True,
        help="Use Chatterbox-Nano checkpoint (default: True, optimal for 4GB VRAM)",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        default=False,
        help="Use full Chatterbox-Turbo model instead of Nano",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=False,
        help="Enable sequential stage offloading to CPU for memory-constrained multitasking (<1GB peak VRAM)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for speech generation (default: 0.8)",
    )
    parser.add_argument(
        "--pause-duration",
        type=float,
        default=0.2,
        help="Silence duration between sentences in seconds (default: 0.2s)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        default=False,
        help="Automatically play the generated audio upon completion",
    )

    args = parser.parse_args()

    use_nano = not args.turbo

    if ChatterboxTurboTTS is None:
        print("Error: chatterbox package is not installed.")
        print("Please run:")
        print("  git clone https://github.com/resemble-ai/chatterbox.git")
        print("  python3 .agents/skills/quadro-turing-voice-clone/scripts/patch_chatterbox_turing.py chatterbox")
        print("  uv pip install --python .venv/bin/python -e chatterbox")
        sys.exit(1)

    # 1. Validate inputs
    ref_path = Path(args.reference)
    if not ref_path.exists():
        # Check standard user recording locations
        candidates = [
            Path("/home/mboyar/Downloads/my_voice.wav"),
            Path("/home/mboyar/Downloads/my_voice2.wav"),
            Path.home() / "Downloads" / "my_voice.wav",
            Path.home() / "Downloads" / "my_voice2.wav",
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                ref_path = cand
                found = True
                print(f"ℹ️ Reference voice auto-detected at: {ref_path}")
                break
        if not found:
            print(f"Error: Reference voice file not found: {ref_path}")
            print("Please provide a valid 5-15s WAV file with --reference <path>.")
            sys.exit(1)

    ref_info = sf.info(str(ref_path))
    if ref_info.duration < 5.0:
        print(f"Error: Reference audio duration is {ref_info.duration:.2f}s, but Chatterbox requires > 5.0 seconds.")
        sys.exit(1)

    # 2. Extract text to synthesize
    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}")
            sys.exit(1)
        text_content = script_path.read_text(encoding="utf-8")
    elif args.text:
        text_content = args.text
    else:
        # Default fallback to input.txt if present
        default_script = Path("input.txt")
        if default_script.exists():
            print("ℹ️ No text argument supplied; auto-detected input.txt")
            text_content = default_script.read_text(encoding="utf-8")
        else:
            print("Error: No text provided. Please specify --text <string> or --script <file.txt>.")
            sys.exit(1)

    sentences = split_into_sentences(text_content)
    if not sentences:
        print("Error: No text provided or input is empty.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
    compute_cap = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}" if torch.cuda.is_available() else "N/A"

    print("\n" + "=" * 65)
    print("🚀  Chatterbox Quadro Turing Voice Clone Synthesizer")
    print("=" * 65)
    print(f"• Target Device     : {args.device.upper()} ({gpu_name})")
    print(f"• Compute Capability: {compute_cap} (sm_{compute_cap.replace('.', '')}) | VRAM: {gpu_mem}")
    print(f"• Model Variant     : {'Chatterbox-Nano (Fast, Low VRAM)' if use_nano else 'Chatterbox-Turbo (Full Model)'}")
    print(f"• Pipeline Mode     : {'Low VRAM Offloading (<950MB)' if args.low_vram else 'Resident VRAM (~1.2GB peak)'}")
    print(f"• Reference voice   : {ref_path} ({ref_info.duration:.2f}s, {ref_info.samplerate}Hz)")
    print(f"• Output destination: {args.output}")
    print(f"• Sentences count   : {len(sentences)}")
    print("=" * 65 + "\n")

    # 3. Load Model
    print(f"⏳ [1/3] Loading Chatterbox onto {args.device.upper()}...")
    t_start = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=use_nano)
    if hasattr(model, "low_vram"):
        model.low_vram = args.low_vram
    load_time = time.time() - t_start
    print(f"✓ Model loaded in {load_time:.2f}s!\n")

    # 4. Prepare Voice Conditionals (Embedding cache)
    print("⏳ [2/3] Extracting voice conditionals & timbre...")
    t_embed = time.time()
    model.prepare_conditionals(str(ref_path))
    print(f"✓ Voice timbre embedded in {time.time() - t_embed:.2f}s!\n")

    # 5. Synthesize sentences
    print(f"⏳ [3/3] Synthesizing {len(sentences)} sentence(s)...")
    audio_chunks = []
    sr = model.sr
    pause_samples = int(args.pause_duration * sr)
    silence_gap = np.zeros(pause_samples, dtype=np.float32)

    total_gen_time = 0.0
    for idx, sentence in enumerate(sentences, start=1):
        print(f"  [{idx}/{len(sentences)}] \"{sentence}\"")
        t_chunk = time.time()
        wav = model.generate(
            text=sentence,
            temperature=args.temperature,
        )
        chunk_time = time.time() - t_chunk
        total_gen_time += chunk_time
        if isinstance(wav, torch.Tensor):
            wav_np = wav.squeeze().detach().cpu().numpy()
        else:
            wav_np = np.asarray(wav).squeeze()
        audio_chunks.append(wav_np)
        if idx < len(sentences):
            audio_chunks.append(silence_gap)
        print(f"      ↳ Generated {len(wav_np)/sr:.2f}s audio in {chunk_time:.2f}s (Speed: {(len(wav_np)/sr)/max(chunk_time, 0.001):.2f}x real-time)")

    # Concatenate all audio chunks
    final_audio = np.concatenate(audio_chunks)
    total_audio_duration = len(final_audio) / sr

    # Save to output file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final_audio, sr)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    curr_vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 65)
    print("🎉 Synthesis Complete!")
    print("=" * 65)
    print(f"• Output file       : {out_path.resolve()}")
    print(f"• Total audio       : {total_audio_duration:.2f} seconds")
    print(f"• Generation time   : {total_gen_time:.2f} seconds")
    print(f"• Real-time factor  : {total_gen_time / total_audio_duration:.2f}x (Lower is faster)")
    print(f"• Playback speed    : {total_audio_duration / total_gen_time:.2f}x real-time")
    if torch.cuda.is_available():
        print(f"• Peak GPU Memory   : {peak_vram_mb:.1f} MB (Active: {curr_vram_mb:.1f} MB)")
    print("=" * 65 + "\n")

    if args.play:
        print("🔊 Playing synthesized audio...")
        play_audio(out_path)


if __name__ == "__main__":
    main()
