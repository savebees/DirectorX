<h1 align="center"><img src="assets/directorx-logo.png" alt="DirectorX logo" width="64" height="64" valign="middle"> DirectorX</h1>

<p align="center">
  <em>一个由导演 Agent 主导的视频剪辑多智能体系统。</em>
</p>

<p align="center">
  <a href="README.md"><ins>English</ins></a> | 简体中文
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"></a> <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-workflow-1C3C3C?style=flat-square" alt="LangGraph workflow"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
</p>

DirectorX 是一个“导演驱动”的多智能体视频剪辑工具：输入本地视频和一段创作 brief，系统会理解素材、梳理故事、生成 storyboard 和旁白、为每个 beat 匹配合适的源画面、选择配乐、渲染视频并检查成片。

## ⭐ Highlights

- **从 brief 到成片**：从一段创作 brief 和本地素材开始，生成结构清晰、带旁白的成片。
- **基于证据理解素材**：检测镜头、合并场景、读取字幕或语音，并让源视频中的关键时刻可检索。
- **以故事为中心剪辑**：先梳理素材的故事层级，再将选定方向转化为逐 beat 的旁白和画面意图。
- **画面、声音与配乐匹配**：为每个 beat 找到经过核验的源片段，匹配实际旁白时长，并选择语义相关的音乐。
- **结果清晰可查看**：保留 storyboard、旁白、画面定位、声音方案、成片和审核报告等项目产物。
- **支持持续完善**：结合中间结果和审核反馈，持续迭代和优化成片。

## Agent 分工

| Agent | 负责内容 |
| --- | --- |
| Director Agent | brief、任务分派、决策和批准 |
| Footage Analyst Agent | 镜头/场景理解与可检索素材证据 |
| Screenwriter Agent | 叙事结构与旁白文案 |
| Narration Agent | 语音合成与时长测量 |
| Grounding Agent | 为每个 beat 定位精确源片段 |
| Sound Agent | 整条成片的音乐选择与混音意图 |
| Render Agent | 确定性的 FFmpeg 组装 |
| Review Agent | 独立检查渲染后的视频 |

## 核心能力

DirectorX 帮你把原始素材整理成一条可以分享的故事。从一个创作方向出发，它会梳理叙事、找出真正重要的画面、生成旁白、选择合适的音乐，并将这些内容组合成一条可以审核和继续优化的成片。

**从素材中找到故事。** 不用先面对一条需要手动翻找的时间线。DirectorX 会理解视频中的画面和语音内容，找出值得保留的场景，并帮助故事形成清晰的起承转合。

**写出有目的的脚本。** 将你的创作方向整理成节奏清楚的 storyboard，为每个 beat 配上有意义的旁白，让观众顺着故事自然看下去。

**让每句话都有对应的画面。** 每个 beat 都会匹配能够支撑其含义的源视频时刻，让成片看起来是经过设计的表达，而不是把方便找到的片段简单拼在一起。

**用声音完成一条可以继续优化的成片。** 合适的配乐、平衡的人声与音乐，再加上最后的成片检查，让第一版有完整的观看体验，也为下一轮修改提供明确起点。

服务配置、模型选择和媒体路径统一在 [`config.toml`](config.toml) 中设置。

## 快速开始

### 环境要求

需要 Python 3.11 或更高版本，并确保 `ffmpeg`、`ffprobe` 已安装且位于 `PATH`。真实运行还需要一个 OpenAI 兼容的 VLM 服务、一个 OpenAI 兼容的 LLM 服务，以及 `media/music/` 中至少一份 `.mp3`、`.wav`、`.m4a`、`.aac` 或 `.flac` 音频文件。首次运行可能会下载 CLIP、sentence-transformers 和 CLAP 模型；只有使用 Whisper ASR 时才需要额外安装 Whisper 依赖。

### 安装

```bash
git clone https://github.com/savebees/DirectorX.git
cd DirectorX

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# config.toml 使用 Whisper ASR 时再安装。
.venv/bin/python -m pip install -r requirements-asr.txt
```

DirectorX 只从环境变量读取密钥，不会自动加载 `.env`。复制示例文件、填写两个密钥，再将它加载到当前 shell：

```bash
cp .env.example .env
set -a
source .env
set +a
```

默认配置中，`VLM_API_KEY` 用于 Qwen3-VL 的描述、画面定位和审核请求；`LLM_API_KEY` 用于场景标签、故事结构和脚本请求。使用其他 OpenAI 兼容服务时，直接修改 `config.toml` 中的 URL、模型名和环境变量名即可。

### 准备素材并运行

把源视频放入 `media/videos/`，把音乐放入 `media/music/`。第一次运行前请先调整 `config.toml` 的转录配置：仓库内的示例指向一份示例字幕路径。你可以将 `subtitle_path` 改成真实存在的外挂字幕，选择 `provider = "embedded"` 或 `"whisper"`，没有转录时则使用 `provider = "none"`。

先建立音乐索引，再执行工作流。未传入 `--music-index` 时，运行器会自动寻找 `artifacts/music-index.json`：

```bash
.venv/bin/python -m directorx.cli.music_index --config config.toml

.venv/bin/python -m directorx.cli.check \
  --config config.toml \
  --video media/videos/example.mp4

.venv/bin/python -m directorx.cli.run \
  --config config.toml \
  --video media/videos/example.mp4 \
  --brief "讲清楚这段素材中的人物关系和转折。" \
  --target-duration 60
```

较长的 brief 可以写入文件后使用 `--brief-file`；多个编辑要求可以重复传入 `--constraint`；用 `--project-id` 可以显式指定输出目录。项目 ID 对应不可覆盖的产物，因此重新生成时请使用新的 ID，不要复用已经完成的项目。

## 输出文件

以 `example` 为项目 ID 时，工作流会生成以下持久化产物：

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

素材搜索缓存位于 `.video-index/`，包含每个源文件对应的 `index.json`、`search.sqlite3`、关键帧和阶段检查点。协调记录与 LangGraph 检查点单独保存，因此生成的清单可以直接查看或迁移。

## 配置说明

`config.toml` 是唯一配置入口，覆盖路径、镜头检测与场景合并、字幕/ASR、文本和视觉向量、VLM/LLM 服务、Edge TTS、画面定位采样、音乐分析、审核抽帧、渲染尺寸和目标时长。默认输出为 1920x1080、30 FPS；需要竖屏或方形视频时，将 `render.aspect` 改为 `portrait` 或 `square`。

`transcription.provider = "auto"` 时，系统依次尝试配置的外挂字幕、视频内嵌文本轨道和 faster-whisper。场景合并使用本地 CLIP 相似度；语义检索结合 SQLite FTS5 和稠密向量重排。音乐库变化后需要重新生成 `music-index.json`。CLAP 模型会下载到 Hugging Face 默认缓存目录，不会复制进仓库。

## 单独建立素材索引

如果只想检查场景切分或预热缓存，可以不运行完整剪辑流程：

```bash
.venv/bin/python -m directorx.cli.index \
  --config config.toml \
  --video media/videos/example.mp4
```

命令会输出场景数量，以及生成的 `index.json` 和 `search.sqlite3` 路径。

## 仓库结构

```text
directorx/agents/         Director 与各专业 Agent
directorx/coordination/   合同、权限和上下文存储
directorx/indexing/       镜头检测、描述、标签和搜索
directorx/rendering/      FFmpeg 执行
directorx/services/       LLM、VLM、TTS 和音乐适配器
directorx/cli/            命令行入口
tests/                    合同与能力测试
```

## 开发与测试

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

仓库直接从根目录运行，不会构建 `dist/` 发布目录。测试对模型服务和媒体引擎使用 fake，因此完整测试不需要 API 密钥或真实视频文件。

## 🩷 Acknowledgement

DirectorX 的素材索引字段组织参考了以下项目所呈现的视觉、语音、标签、对象和主题等分类方式：

- [Google Cloud Video Intelligence](https://cloud.google.com/video-intelligence)
- [Azure AI Video Indexer](https://learn.microsoft.com/azure/azure-video-indexer/)

镜头检测使用 [PySceneDetect](https://www.scenedetect.com/)，音乐向量使用 [LAION larger_clap_music](https://huggingface.co/laion/larger_clap_music) 模型。

## 许可证

DirectorX 使用 [Apache License 2.0](LICENSE) 发布。
