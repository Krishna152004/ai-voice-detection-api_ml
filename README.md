
---

# AI-Generated Voice Detection API

## Overview

The **AI-Generated Voice Detection API** is a RESTful service that determines whether a given voice sample is **AI-generated** or **human-generated**.
The service supports **Tamil, English, Hindi, Malayalam, and Telugu** voice samples.

The API is designed to be **stable, low-latency, and evaluation-ready**, following GUVI hackathon submission requirements.

---

## Base URL

```
https://<your-railway-app-url>
```

---

## Authentication

All requests require an API key.

### Header

```
x-api-key: sk_test_123456789
```

> **Note:** This is a test-only API key.

---

## Endpoint

### POST `/api/voice-detection`

Classifies a voice sample as **AI-generated** or **Human-generated**.

---

## Request

### Headers

```
Content-Type: application/json
x-api-key: sk_test_123456789
```

### Body Parameters

| Field         | Type   | Required | Description                                  |
| ------------- | ------ | -------- | -------------------------------------------- |
| `language`    | string | Yes      | Tamil / English / Hindi / Malayalam / Telugu |
| `audioFormat` | string | Yes      | Must be `mp3`                                |
| `audioBase64` | string | Yes      | Base64-encoded MP3 audio                     |

### Example Request

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_AUDIO_STRING>"
}
```

---

## Response

### Success Response

**Status Code:** `200 OK`

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "HUMAN",
  "confidence": 0.94,
  "confidenceScore": 0.94,
  "explanation": "Natural voice variations observed"
}
```

### Response Fields

| Field             | Type   | Description                  |
| ----------------- | ------ | ---------------------------- |
| `status`          | string | Request status               |
| `language`        | string | Language provided in request |
| `classification`  | string | `AI_GENERATED` or `HUMAN`    |
| `confidence`      | number | Confidence score (0.0 – 1.0) |
| `confidenceScore` | number | Same as confidence           |
| `explanation`     | string | Reason for classification    |

---

## Error Responses

### Unauthorized

**Status Code:** `401`

```json
{
  "detail": "Invalid API key"
}
```

### Invalid Input

**Status Code:** `400`

```json
{
  "detail": "Only mp3 format is supported"
}
```

---

## Detection Methodology

* Audio features extracted using **librosa**
* Features include:

  * MFCC statistics
  * Spectral contrast
  * Pitch variability
  * Zero-crossing rate
* Language-aware normalization applied
* Classification performed using a **pre-trained RandomForest model**
* Model loaded at runtime for fast inference

> Training audio data is intentionally excluded from this repository.

---

## Supported Languages

* Tamil
* English
* Hindi
* Malayalam
* Telugu

---

## Local Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## Project Structure

```
ai-voice-detection-api/
├── app.py
├── model/
│   ├── detector.py
│   ├── feature_extractor.py
│   └── artifacts/
│       └── voice_classifier.pkl
├── utils/
│   └── audio_utils.py
├── requirements.txt
├── Procfile
└── README.md
```

---

## Notes

* Only **MP3** audio is supported
* Audio must be **Base64-encoded**
* Designed for **automated evaluation**
* Single endpoint as required by GUVI

---
