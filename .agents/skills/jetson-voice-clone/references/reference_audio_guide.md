# Reference Audio Recording Guide for Zero-Shot Voice Cloning

This guide explains how to record high-quality reference audio to get optimal voice similarity with Chatterbox on Jetson.

---

## 1. Golden Rules for Reference Audio

| Parameter | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Duration** | **8 to 12 seconds** | Chatterbox strictly enforces `> 5.0 seconds`. Audio longer than 15s increases CPU embedding time without improving quality. |
| **Channels** | **Mono (1 Channel)** | Stereo adds phase issues and unnecessary memory overhead. |
| **Format** | **WAV (PCM 16-bit or 32-bit float)** | Avoid MP3, AAC, or lossy formats. |
| **Sample Rate** | **24 kHz or 44.1 kHz or 48 kHz** | S3Gen natively decodes to 24 kHz; reference is internally resampled to 16 kHz and 24 kHz. |
| **Noise Floor** | Clean, quiet room | Avoid background fan noise, echo, and PC hum. |
| **Delivery** | Natural conversational tone | Speak clearly with varied pitch and normal cadence. |

---

## 2. Recommended Reference Reading Passages

Read one of the following passages naturally (takes ~10 seconds):

### English:
> *"The quick brown fox jumps over the lazy dog. Artificial intelligence and edge computing are transforming robotics and offline voice synthesis across the world."*

### Turkish:
> *"Yapay zeka sistemleri artık uç cihazlarda tamamen yerel olarak çalışabiliyor. Kendi ses tonumu doğrudan Jetson Orin Nano üzerinde başarıyla klonluyorum."*

---

## 3. Recording Commands via Terminal

### Using `arecord` (ALSA):
```bash
# Record 10 seconds of clean mono audio at 44.1kHz:
arecord -D hw:1,0 -f S16_LE -r 44100 -c 1 -d 10 /tmp/my_voice.wav
```
*(Replace `hw:1,0` with your USB microphone device ID from `arecord -l`)*

### Converting Existing Audio to Clean Mono WAV using `sox` or `ffmpeg`:
```bash
# Convert any file (m4a, mp3, etc.) to 24kHz mono WAV:
ffmpeg -i input_recording.m4a -ac 1 -ar 24000 -af "silenceremove=1:0:-45dB" my_voice.wav
```

