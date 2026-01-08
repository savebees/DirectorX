# DirectorX

DirectorX is a Director-led multi-agent system for long-form video editing. The Director Agent owns the brief, delegates bounded tasks, reviews specialist results, requests revisions, and approves project decisions. Specialists may consult one another when the communication policy allows it, but they cannot assign work or change project-level state.

DirectorX is not modeled as a fixed pipeline. The order of work may change as the Director receives evidence, asks questions, or sends an artifact back for revision.

## Agent roles

| Role | Ownership |
| --- | --- |
| Director Agent | Creative brief, delegation, revisions, decisions, and final approval |
| Footage Analyst Agent | Source facts, scene understanding, and footage evidence |
| Screenwriter Agent | Narrative structure and narration text |
| Narration Agent | Voice delivery, timing, pronunciation, and subtitles |
| Grounding Agent | Exact source clips that satisfy approved visual intent |
| Sound Agent | Music selection, sound design, and mix intent |
| Review Agent | Independent review of approved artifacts and issue reporting |

The Director is the only formal authority. Direct specialist communication is limited to one scoped question and one response:

| Sender | May consult |
| --- | --- |
| Director | Every specialist |
| Footage Analyst | Director |
| Screenwriter | Footage Analyst, Director |
| Narration | Screenwriter, Director |
| Grounding | Footage Analyst, Screenwriter, Director |
| Sound | Narration, Director |
| Review | Director |

A consultation cannot assign a task, request a formal revision, mutate an artifact, or approve a decision. An unresolved question is returned to the Director.

## Context model

The coordination layer keeps three explicit context scopes. `ProjectMemory` contains only the approved brief, global constraints, and approved artifact references; only the Director can write it. `TaskContext` contains the minimum input required by one assignee and is readable only by that assignee and the Director. `ConsultationRequest` and `ConsultationResponse` carry only the question, reason, required answer, and relevant artifact references; conversation histories are never copied between agents.

Tasks, results, consultations, and decisions are immutable files. A repeated identifier fails instead of overwriting prior state. Only Director-approved information enters project memory; transient discussion stays outside it.

The first implemented specialist capability is `FootageAnalystAgent`: it understands a source video through scene boundaries, dialogue, keyframes, dense visual captions, normalized retrieval tags, and embeddings, then returns a searchable `VideoIndex`. The VLM writes plain-text dense captions; an LLM combines those captions with subtitles or ASR transcripts and writes the small searchable tag set back to each scene. It does not write creative recommendations or project decisions. The other agents remain independent capability code until their own first capability is defined.

## Setup

Requirements are Python 3.11 or newer and FFmpeg/ffprobe on `PATH`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

DirectorX reads API keys from the process environment and does not load `.env` automatically. Set `SILICONFLOW_API_KEY` for the Qwen3-VL captioner and `VYCE_API_KEY` for the GPT-5.6 Luna scene tagger before indexing. Install `requirements-asr.txt` when using Whisper transcription.

## Configuration

[`config.toml`](config.toml) is the single configuration entry point for paths, indexing, transcription, embedding, VLM, LLM, TTS, rendering, and edit defaults. Footage indexing uses PySceneDetect's AdaptiveDetector and duration-aware sharpness-based keyframes: candidates are sampled locally, divided across the shot timeline, and ranked by frame clarity before the scene sends at most eight images to the VLM. Transcription defaults to `auto`: an available sidecar subtitle is preferred, then an embedded text subtitle track, then faster-whisper ASR. Provider names are validated; adding a provider requires an explicit adapter.

The standalone index command remains available while the Footage Analyst role is developed:

```bash
SILICONFLOW_API_KEY='<secret>' \
  .venv/bin/python -m directorx.cli.index \
  --config config.toml \
  --video media/videos/Casino.Royale.movie/Casino.Royale.2006.PROPER.1080p.BluRay.H264.AAC-LAMA.mp4
```

## Structure

```text
directorx/
  agents/         Director and specialist implementations
  coordination/   roles, contracts, permissions, context storage, and runtime
  core/           video-editing domain models and provider protocols
  indexing/       scene detection, transcription, captions, tags, and search
  rendering/      deterministic FFmpeg execution
  services/       LLM, VLM, TTS, and media adapters
  cli/            standalone operational commands
media/
  music/          local music inputs
  videos/         local video and subtitle inputs
artifacts/        generated project state and media outputs
tests/            contract and capability tests
config.toml       unified runtime configuration
```

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff format --check .
.venv/bin/ruff check .
python3 -m pytest -q
```

The repository runs directly from its root. It is not a Python distribution and does not use a `dist/` build workflow.

## 💗 Acknowledgement

DirectorX's scene-indexing field organization was informed by the separated visual, speech, label, object, and topic metadata exposed by [Google Cloud Video Intelligence](https://cloud.google.com/video-intelligence) and [Azure AI Video Indexer](https://learn.microsoft.com/azure/azure-video-indexer/). These projects helped shape the distinction between dense visual evidence and normalized retrieval tags.
