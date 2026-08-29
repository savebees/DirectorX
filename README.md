<h1 align="center"><img src="assets/directorx-logo.png" alt="DirectorX logo" width="64" height="64" valign="middle"> DirectorX</h1>

<p align="center">
  English | <a href="README.zh-CN.md"><ins>简体中文</ins></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a> <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-workflow-1C3C3C?style=flat-square" alt="LangGraph workflow"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
</p>

DirectorX is a director-led, multi-agent video editing tool for turning local footage and a creative brief into a polished, reviewable cut. It understands the footage, shapes the story, writes a storyboard and narration, matches each beat to the right source moment, selects background music, renders the video, and checks the result.

## ⭐ Highlights

- **Brief to final video**: Turn a creative brief and local footage into a structured, narrated edit.
- **Evidence-backed footage understanding**: Detect shots, group scenes, read subtitles or speech, and make source moments searchable.
- **Story-first editing**: Organize the source into a story hierarchy, then turn the selected direction into beat-level narration and visual intent.
- **Matched visuals, voice, and music**: Connect every beat to a verified source interval, measured narration, and a semantically selected music track.
- **Inspectable results**: Keep the storyboard, narration, grounding, sound plan, rendered video, and review report as readable project outputs.
- **Easy to refine**: Use the intermediate results and review feedback to iterate toward a better cut.

## Agent roles

| Agent | Responsibility |
| --- | --- |
| Director Agent | Brief, delegation, decisions, and approval |
| Footage Analyst Agent | Shot/scene understanding and searchable evidence |
| Screenwriter Agent | Narrative structure and narration text |
| Narration Agent | Voice synthesis and timing measurement |
| Grounding Agent | Exact source intervals for each beat |
| Sound Agent | Whole-edit music selection and mix intent |
| Render Agent | Deterministic FFmpeg assembly |
| Review Agent | Independent review of the rendered video |

## Core capabilities

DirectorX helps turn raw footage into a story that is ready to share. Give it a creative direction and it helps shape the narrative, find the moments that matter, build the voice track, choose a fitting soundtrack, and bring everything together into a polished cut that can be reviewed and refined.

**Find the story in your footage.** Start with a clear brief instead of a timeline full of manual searching. DirectorX understands what is happening across your video, surfaces the scenes that matter, and gives the edit a coherent beginning, middle, and end.

**Create a script with purpose.** Turn your direction into a storyboard with clear beats, meaningful narration, and a pace that keeps the audience moving through the story.

**Put the right picture behind every word.** Each beat is matched with a source moment that supports its meaning, so the edit feels intentional rather than assembled from convenient clips.

**Finish with sound and a cut you can improve.** A fitting soundtrack, balanced voice and music, and a final review bring the first version together and give the next revision a concrete starting point.

Provider settings, model choices, and media paths are configured in [`config.toml`](config.toml).

## Quick start

### Requirements

Use Python 3.11 or newer and install `ffmpeg` and `ffprobe` so both commands are available on `PATH`. A real run also needs one OpenAI-compatible VLM endpoint, one OpenAI-compatible LLM endpoint, and at least one `.mp3`, `.wav`, `.m4a`, `.aac`, or `.flac` file in `media/music/`. The first run may download CLIP, sentence-transformers, and CLAP checkpoints; Whisper is optional.

### Install

```bash
git clone https://github.com/savebees/DirectorX.git
cd DirectorX

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Optional: install this when config.toml uses Whisper ASR.
.venv/bin/python -m pip install -r requirements-asr.txt
```

DirectorX reads credentials from environment variables and does not load `.env` by itself. Copy the example file, fill in both keys, and load it into your shell:

```bash
cp .env.example .env
set -a
source .env
set +a
```

`VLM_API_KEY` is used by the default Qwen3-VL captioning, grounding, and review calls. `LLM_API_KEY` is used by the default scene tagger, story-structure, and screenwriter calls. Change the provider URLs, model names, or environment-variable names in `config.toml` when using another OpenAI-compatible service.

### Prepare and run

Put a source video in `media/videos/` and music files in `media/music/`. Before the first run, adjust the transcription section in `config.toml`: the checked-in example points to a sample subtitle path, so set `subtitle_path` to an existing sidecar, choose `provider = "embedded"` or `"whisper"`, or use `provider = "none"` when no transcript is available.

Build the music index once, then run the workflow. The runner automatically discovers `artifacts/music-index.json` when `--music-index` is omitted:

```bash
.venv/bin/python -m directorx.cli.music_index --config config.toml

.venv/bin/python -m directorx.cli.check \
  --config config.toml \
  --video media/videos/example.mp4

.venv/bin/python -m directorx.cli.run \
  --config config.toml \
  --video media/videos/example.mp4 \
  --brief "Explain the key relationship and turning point in this footage." \
  --target-duration 60
```

Use `--brief-file` for a longer brief, `--constraint` more than once for editorial constraints, and `--project-id` to choose an explicit output directory. Project identifiers are immutable: use a new identifier for a new run instead of reusing a completed project.

## Outputs

For a project named `example`, the workflow writes the following durable artifacts:

```text
artifacts/
  music-index.json
  example/
    story-summary.json
    storyboard.json
    narration/
      narration.json
      *.wav
    grounding.json
    sound-plan.json
    final.mp4
    review.json
```

The searchable footage cache lives under `.video-index/` and contains `index.json`, `search.sqlite3`, keyframes, and model checkpoints for each source filename. Coordination records and LangGraph checkpoints are stored separately so the manifests remain easy to inspect or move.

## Configuration

`config.toml` is the single configuration entry point. The sections cover paths, shot detection and scene grouping, transcription, text and visual embeddings, VLM/LLM providers, Edge TTS, grounding sampling, music analysis, review frame limits, render dimensions, and target duration. The default render is 1920x1080 at 30 FPS; set `render.aspect` to `portrait` or `square` when needed.

Transcription with `provider = "auto"` tries the configured sidecar subtitle, then an embedded text track, then faster-whisper. Scene grouping uses local CLIP similarity; semantic search combines SQLite FTS5 with dense-vector reranking. Music indexing samples each track once, so refresh `music-index.json` after changing the library. The CLAP checkpoint is downloaded to the normal Hugging Face cache and is not copied into the repository.

## Standalone indexing

The footage indexer can be run without the full workflow. This is useful for inspecting scene extraction or warming the cache:

```bash
.venv/bin/python -m directorx.cli.index \
  --config config.toml \
  --video media/videos/example.mp4
```

It prints the scene count and paths to the generated `index.json` and `search.sqlite3`.

## Repository layout

```text
directorx/agents/         Director and specialist agents
directorx/coordination/   Contracts, permissions, and context storage
directorx/indexing/       Shot detection, captions, tags, and search
directorx/rendering/      FFmpeg execution
directorx/services/       LLM, VLM, TTS, and music adapters
directorx/cli/            Operational commands
tests/                    Contract and capability tests
```

## Development

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

The repository runs directly from its root; it is not packaged into a `dist/` directory. Tests use fakes for providers and media engines, so a full test run does not require API keys or a real video.

## 🩷 Acknowledgement

DirectorX's scene-indexing field organization was informed by the separated visual, speech, label, object, and topic metadata exposed by:

- [Google Cloud Video Intelligence](https://cloud.google.com/video-intelligence)
- [Azure AI Video Indexer](https://learn.microsoft.com/azure/azure-video-indexer/)

Shot detection is provided by [PySceneDetect](https://www.scenedetect.com/). Music embeddings use the [LAION larger_clap_music](https://huggingface.co/laion/larger_clap_music) model.

## License

DirectorX is released under the [Apache License 2.0](LICENSE).
