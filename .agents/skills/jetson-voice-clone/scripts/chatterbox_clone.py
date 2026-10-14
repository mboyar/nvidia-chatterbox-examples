#!/usr/bin/env python3
"""
Chatterbox Voice Clone CLI for NVIDIA Jetson Orin Nano
Synthesizes speech in your cloned voice offline using Chatterbox-Nano on CUDA.
"""

import os
import sys
from pathlib import Path

# 1. Automatically re-exec into project's .venv if running with system python
_script_dir = Path(__file__).resolve().parent
_venv_dir = _script_dir / ".venv"
_venv_python = _venv_dir / "bin" / "python3"
if _venv_python.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

# 2. Configure PyTorch unified memory allocator before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 3. Filter out noisy Tegra nvmap driver allocation fallback notices from stderr
import atexit
import threading

class _StderrFilter:
    def __init__(self, patterns=("NvMapMemAllocInternalTagged", "NvMapMemHandleAlloc")):
        self.patterns = patterns
        self.pipe_r, self.pipe_w = os.pipe()
        sys.stderr.flush()
        self.orig_fd2 = os.dup(2)
        os.dup2(self.pipe_w, 2)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        buf = b""
        while self.running:
            try:
                data = os.read(self.pipe_r, 1024)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace")
                    if not any(p in line_str for p in self.patterns):
                        os.write(self.orig_fd2, line + b"\n")
            except Exception:
                break
        if buf:
            line_str = buf.decode("utf-8", errors="replace")
            if not any(p in line_str for p in self.patterns):
                os.write(self.orig_fd2, buf)

    def close(self):
        if not self.running:
            return
        self.running = False
        sys.stderr.flush()
        try:
            os.dup2(self.orig_fd2, 2)
            os.close(self.orig_fd2)
            os.close(self.pipe_w)
            os.close(self.pipe_r)
        except Exception:
            pass

_stderr_filter = _StderrFilter()
atexit.register(_stderr_filter.close)

import argparse
import re
import time

import numpy as np
import soundfile as sf
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS


def split_into_sentences(text: str) -> list[str]:
    """Split text into manageable sentence chunks for speech synthesis."""
    text = text.strip()
    if not text:
        return []

    # Normalize whitespace and newlines
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        # Split on sentence boundaries (., !, ?) keeping end punctuation
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # If a single chunk is excessively long (>250 chars), further subdivide on semicolons or commas
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


def main():
    parser = argparse.ArgumentParser(
        description="Clone voice and synthesize speech on NVIDIA Jetson Orin Nano."
    )
    group = parser.add_mutually_exclusive_group(required=True)
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
        help="Path to your clean 5-15s WAV reference voice sample (default: sample_reference.wav)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Path where the synthesized WAV file will be saved (default: output.wav)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to run inference on: 'cuda' or 'cpu' (default: cuda)",
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

    args = parser.parse_args()

    # 1. Validate inputs
    ref_path = Path(args.reference)
    if not ref_path.exists():
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
    else:
        text_content = args.text

    sentences = split_into_sentences(text_content)
    if not sentences:
        print("Error: No text provided or input is empty.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"🎙️  Chatterbox Jetson Voice Clone Synthesizer")
    print(f"=======================================================")
    print(f"• Reference voice : {ref_path} ({ref_info.duration:.2f}s, {ref_info.samplerate}Hz)")
    print(f"• Device          : {args.device.upper()}")
    print(f"• Output destination: {args.output}")
    print(f"• Sentences to synthesize: {len(sentences)}")
    print(f"=======================================================\n")

    # 3. Load Model
    print(f"⏳ [1/3] Loading Chatterbox-Nano model onto {args.device.upper()}...")
    t_start = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=True)
    load_time = time.time() - t_start
    print(f"✓ Model loaded in {load_time:.2f}s!\n")

    # 4. Prepare Voice Conditionals (Embedding cache)
    print(f"⏳ [2/3] Embedding reference voice timbre...")
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
        wav_np = wav.squeeze(0).numpy()
        audio_chunks.append(wav_np)
        if idx < len(sentences):
            audio_chunks.append(silence_gap)
        print(f"      ↳ Generated {len(wav_np)/sr:.2f}s of audio in {chunk_time:.2f}s")

    # Concatenate all audio chunks
    final_audio = np.concatenate(audio_chunks)
    total_audio_duration = len(final_audio) / sr

    # Save to output file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final_audio, sr)

    print(f"\n=======================================================")
    print(f"🎉 Synthesis Complete!")
    print(f"=======================================================")
    print(f"• Output file     : {out_path.resolve()}")
    print(f"• Total duration  : {total_audio_duration:.2f} seconds")
    print(f"• Generation time : {total_gen_time:.2f} seconds")
    print(f"• Real-time factor: {total_gen_time / total_audio_duration:.2f}x (lower is faster)")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()

