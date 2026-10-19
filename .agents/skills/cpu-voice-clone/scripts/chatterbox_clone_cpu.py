#!/usr/bin/env python3
"""
Chatterbox Voice Clone CLI for CPU-only hosts (x86_64, no GPU required).
Synthesizes speech in your cloned voice offline using Chatterbox / Chatterbox-Nano on CPU.
Optimized for FP32 MKL/oneDNN kernels with physical-core thread pinning and optional
int8 dynamic quantization of the T3 GPT-2 backbone.
"""

import os
import sys
from pathlib import Path


# 1. Automatically re-exec into project's venv if running with system python
#    Preference: .venv-cpu (CPU-only torch wheels) -> .venv (may be a CUDA build)
def _find_venv() -> Path | None:
    search_roots = [Path.cwd()]
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        search_roots.append(curr)
        curr = curr.parent
    for name in (".venv-cpu", ".venv"):
        for root in search_roots:
            candidate = root / name
            if (candidate / "bin" / "python3").exists():
                return candidate
    return None


_venv_dir = _find_venv()
if _venv_dir:
    _venv_python = _venv_dir / "bin" / "python3"
    if _venv_python.exists() and Path(sys.prefix).resolve() != _venv_dir.resolve():
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)


def _physical_cores() -> int:
    """Best-effort physical core count (hyperthreads excluded)."""
    try:
        import psutil  # optional dependency
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    # Parse /proc/cpuinfo for unique (physical id, core id) pairs
    try:
        cores, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                cores.add((phys, core))
        if cores:
            return len(cores)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _early_threads() -> int:
    """Read --threads before torch is imported (thread env must be set first)."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--threads", "-j") and i + 1 < len(argv):
            try:
                return max(1, int(argv[i + 1]))
            except ValueError:
                break
        if a.startswith("--threads="):
            try:
                return max(1, int(a.split("=", 1)[1]))
            except ValueError:
                break
    return _physical_cores()


# 2. Force CPU and configure thread pools BEFORE importing torch
NUM_THREADS = _early_threads()
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide any GPU so CUDA wheels stay on CPU
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(NUM_THREADS))

import argparse
import re
import resource
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # interop pool already started

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
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists() and shutil.which("powershell.exe"):
        try:
            win_path = subprocess.check_output(["wslpath", "-w", str(audio_path)], text=True).strip()
            ps_script = f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_script], check=True)
            return
        except Exception:
            pass

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


def cpu_info() -> tuple[str, str]:
    """Return (CPU model name, notable ISA extensions)."""
    model, flags = "Unknown CPU", set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model == "Unknown CPU":
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
    except Exception:
        pass
    wanted = ["avx2", "fma", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "amx_bf16"]
    isa = ", ".join(f.upper() for f in wanted if f in flags) or "baseline"
    return model, isa


def _conv1d_to_linear(module: torch.nn.Module) -> int:
    """Recursively replace HF GPT-2 Conv1D layers with equivalent nn.Linear layers.

    HF Conv1D stores weight as (in_features, out_features) and computes x @ W + b,
    so the equivalent Linear uses W.T. Returns the number of converted layers.
    """
    from transformers.pytorch_utils import Conv1D

    converted = 0
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            in_f, out_f = child.weight.shape
            lin = torch.nn.Linear(in_f, out_f, bias=True)
            lin.weight.data = child.weight.data.t().contiguous()
            lin.bias.data = child.bias.data.clone()
            setattr(module, name, lin)
            converted += 1
        else:
            converted += _conv1d_to_linear(child)
    return converted


def quantize_t3_int8(model) -> int:
    """Apply int8 dynamic quantization to the T3 GPT-2 transformer backbone only.

    The T3 heads (speech_head etc.) are left in FP32 because T3.dtype is derived
    from speech_head.weight.dtype and is used for conditioning tensors.
    """
    converted = _conv1d_to_linear(model.t3.tfmr)
    model.t3.tfmr = torch.ao.quantization.quantize_dynamic(
        model.t3.tfmr, {torch.nn.Linear}, dtype=torch.qint8
    )
    return converted


def peak_rss_mb() -> float:
    # ru_maxrss is reported in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    parser = argparse.ArgumentParser(
        description="Clone voice and synthesize speech on CPU-only hosts (no GPU required)."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", "-t", type=str, help="Text string to synthesize")
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
        "--threads", "-j",
        type=int,
        default=NUM_THREADS,
        help=f"Intra-op CPU threads (default: physical cores = {_physical_cores()})",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "int8"],
        default="none",
        help="Experimental: int8 dynamic quantization of the T3 GPT-2 backbone (default: none / FP32)",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        default=False,
        help="Use full Chatterbox-Turbo model instead of Nano (slower on CPU)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for speech generation (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random)",
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if ChatterboxTurboTTS is None:
        print("Error: chatterbox package is not installed.")
        print("Please run:")
        print("  git clone https://github.com/resemble-ai/chatterbox.git")
        print("  python3 .agents/skills/cpu-voice-clone/scripts/patch_chatterbox_cpu.py chatterbox")
        print("  uv pip install --python .venv-cpu/bin/python -e chatterbox")
        sys.exit(1)

    # 1. Validate inputs
    ref_path = Path(args.reference)
    if not ref_path.exists():
        candidates = [
            Path.home() / "Downloads" / "my_voice.wav",
            Path.home() / "Downloads" / "my_voice2.wav",
        ]
        for cand in candidates:
            if cand.exists():
                ref_path = cand
                print(f"ℹ️ Reference voice auto-detected at: {ref_path}")
                break
        else:
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

    model_name, isa = cpu_info()
    torch_build = f"CUDA {torch.version.cuda} wheels (GPU hidden)" if torch.version.cuda else "CPU-only wheels"

    print("\n" + "=" * 65)
    print("🖥️  Chatterbox CPU Voice Clone Synthesizer")
    print("=" * 65)
    print(f"• Target Device     : CPU ({model_name})")
    print(f"• ISA Extensions    : {isa}")
    print(f"• Threads           : {torch.get_num_threads()} intra-op / {torch.get_num_interop_threads()} inter-op")
    print(f"• PyTorch           : {torch.__version__} ({torch_build}, MKL={'yes' if torch.backends.mkl.is_available() else 'no'}, oneDNN={'yes' if torch.backends.mkldnn.is_available() else 'no'})")
    print(f"• Model Variant     : {'Chatterbox-Nano (Fast)' if use_nano else 'Chatterbox-Turbo (Full Model)'}")
    print(f"• Precision         : {'FP32 + int8 dynamic T3 backbone (experimental)' if args.quantize == 'int8' else 'FP32'}")
    print(f"• Reference voice   : {ref_path} ({ref_info.duration:.2f}s, {ref_info.samplerate}Hz)")
    print(f"• Output destination: {args.output}")
    print(f"• Sentences count   : {len(sentences)}")
    print("=" * 65 + "\n")

    with torch.inference_mode():
        # 3. Load Model
        print("⏳ [1/3] Loading Chatterbox onto CPU...")
        t_start = time.time()
        model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=use_nano)
        if hasattr(model, "low_vram"):
            model.low_vram = False
        if args.quantize == "int8":
            n = quantize_t3_int8(model)
            print(f"  ↳ Quantized T3 backbone to int8 ({n} Conv1D layers converted)")
        assert next(model.s3gen.parameters()).device.type == "cpu", "S3Gen is not on CPU!"
        assert model.t3.speech_head.weight.device.type == "cpu", "T3 is not on CPU!"
        load_time = time.time() - t_start
        print(f"✓ Model loaded in {load_time:.2f}s! (Peak RSS: {peak_rss_mb():.0f} MB)\n")

        # 4. Prepare Voice Conditionals (Embedding cache)
        print("⏳ [2/3] Extracting voice conditionals & timbre...")
        t_embed = time.time()
        model.prepare_conditionals(str(ref_path))
        print(f"✓ Voice timbre embedded in {time.time() - t_embed:.2f}s!\n")

        # 5. Synthesize sentences
        print(f"⏳ [3/3] Synthesizing {len(sentences)} sentence(s)...")
        audio_chunks = []
        sr = model.sr
        silence_gap = np.zeros(int(args.pause_duration * sr), dtype=np.float32)

        total_gen_time = 0.0
        for idx, sentence in enumerate(sentences, start=1):
            print(f"  [{idx}/{len(sentences)}] \"{sentence}\"")
            t_chunk = time.time()
            wav = model.generate(text=sentence, temperature=args.temperature)
            chunk_time = time.time() - t_chunk
            total_gen_time += chunk_time
            if isinstance(wav, torch.Tensor):
                wav_np = wav.squeeze().detach().cpu().numpy()
            else:
                wav_np = np.asarray(wav).squeeze()
            audio_chunks.append(wav_np.astype(np.float32))
            if idx < len(sentences):
                audio_chunks.append(silence_gap)
            print(f"      ↳ Generated {len(wav_np)/sr:.2f}s audio in {chunk_time:.2f}s (Speed: {(len(wav_np)/sr)/max(chunk_time, 0.001):.2f}x real-time)")

    final_audio = np.concatenate(audio_chunks)
    total_audio_duration = len(final_audio) / sr

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final_audio, sr)

    print("\n" + "=" * 65)
    print("🎉 Synthesis Complete!")
    print("=" * 65)
    print(f"• Output file       : {out_path.resolve()}")
    print(f"• Total audio       : {total_audio_duration:.2f} seconds")
    print(f"• Generation time   : {total_gen_time:.2f} seconds")
    print(f"• Real-time factor  : {total_gen_time / max(total_audio_duration, 1e-6):.2f}x (Lower is faster)")
    print(f"• Playback speed    : {total_audio_duration / max(total_gen_time, 1e-6):.2f}x real-time")
    print(f"• Peak RSS (RAM)    : {peak_rss_mb():.0f} MB")
    print("=" * 65 + "\n")

    if args.play:
        print("🔊 Playing synthesized audio...")
        play_audio(out_path)


if __name__ == "__main__":
    main()

