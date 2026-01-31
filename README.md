🎙️ AI-Generated Voice Detection – API Service

A lightweight API-based voice analysis system designed to detect whether a given voice sample is AI-generated or human-generated using acoustic feature analysis.

✅ Features

REST API for AI vs Human voice classification

Accepts Base64-encoded MP3 audio

Supports multiple languages (Tamil, English, Hindi, Malayalam, Telugu)

Confidence-scored and explainable results

Stable, deterministic, and evaluation-ready

📄 API Overview
Endpoint	Description
/api/voice-detection	Detects whether a voice sample is AI-generated or human
🧠 Detection Approach

Heuristic-based analysis using acoustic features

librosa used only for feature extraction (not detection)

MFCC variance used to measure spectral consistency

Spectral flatness used to detect uniform frequency patterns

Rule-based logic combines features to estimate AI likelihood

🔐 Authentication

API key–based authentication via request headers

Invalid or missing keys are rejected

📥 Input

Audio format: MP3

Encoding: Base64

Language metadata included in request

📤 Output

Classification: AI or Human

Confidence score (0.0–1.0) indicating likelihood of AI-generated voice

Human-readable explanation for the decision

🎯 Learning Outcomes

Designed a heuristic-based AI detection pipeline

Applied speech signal processing concepts for voice analysis

Built a stable FastAPI service with authentication and validation

Learned to generate explainable AI outputs

🧰 Tech Stack

Language: Python

Framework: FastAPI

Audio Processing: librosa, NumPy

Authentication: API key via headers

