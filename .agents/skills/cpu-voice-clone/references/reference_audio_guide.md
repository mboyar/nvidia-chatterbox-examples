# Reference Audio Recording & Selection Guide

Zero-shot voice cloning quality depends heavily on the reference audio provided. This guide covers how to prepare, clean, and verify your reference audio file for the best cloning fidelity.

---

## 1. Quick Requirements

| Parameter | Specification | Reason |
| :--- | :--- | :--- |
| **Duration** | **8 to 12 seconds** | Minimum 5.0s required by model; >15s increases memory without timbre gains. |
| **Channels** | **Mono (1 Channel)** | Stereo causes phase cancellation during feature extraction. |
| **Format** | **WAV (PCM 16-bit or 24-bit)** | Avoid MP3/AAC compression artifacts in speaker embeddings. |
| **Sample Rate** | **24,000 Hz or 44,100 / 48,000 Hz** | Resampled automatically to 16kHz & 24kHz internally. |
| **Acoustics** | Dry, clean, minimal room echo | Background noise degrades the flow-matching vocoder conditioning. |

---

## 2. Recommended Reading Passages

Read naturally at a moderate speaking pace:

### Passage A (Conversational):
> *"The quick brown fox jumps over the lazy dog. Voice synthesis technology has advanced dramatically with modern deep learning and neural vocoders running completely offline on GPU hardware."*

### Passage B (Technical / Narrative):
> *"Artificial intelligence and speech synthesis allow real-time voice conversion and offline text-to-speech. By conditioning the acoustic models on reference audio, we capture pitch, timbre, and cadence."*

---

## 3. Recording & Converting Audio

### On Linux (Native / ALSA):
```bash
# Record 10 seconds of mono 48kHz audio using arecord:
arecord -D default -f S16_LE -r 48000 -c 1 -d 10 reference.wav
```

### On Windows / Mobile:
Convert to clean WAV using FFmpeg:
```bash
ffmpeg -i input_recording.m4a -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

### Verifying File Properties:
```bash
ffprobe reference.wav
# or with python:
python3 -c "import soundfile as sf; info = sf.info('reference.wav'); print(f'Duration: {info.duration}s, SR: {info.samplerate}Hz, Channels: {info.channels}')"
```
