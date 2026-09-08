<h1 align="center">Gander</h1>

<p align="center">
  <strong>Omni Interaction Agent</strong><br>
  <sub>Continuous perception, native realtime interaction, and long-horizon agentic execution in one system.</sub>
</p>

<!-- Replace the remaining #resources targets when the public pages are released. -->
<p align="center">
  <a href="docs/gander-technical-report.pdf"><img src="https://img.shields.io/badge/Paper-PDF-C62828?style=flat-square" alt="Paper PDF"></a>
  <a href="#resources"><img src="https://img.shields.io/badge/Demo-Coming_Soon-2563EB?style=flat-square" alt="Demo coming soon"></a>
  <a href="#resources"><img src="https://img.shields.io/badge/Dataset-Coming_Soon-059669?style=flat-square" alt="Dataset coming soon"></a>
  <a href="https://huggingface.co/Gander-Omni/Gander"><img src="https://img.shields.io/badge/Model-Hugging_Face-D97706?style=flat-square" alt="Gander model on Hugging Face"></a>
</p>

<p align="center">
  <img src="docs/assets/gander-capabilities.png" alt="Gander capabilities across live audio-visual interaction and agentic tasks" width="100%">
</p>

<p align="center">
  <strong>Talk while it works.</strong><br>
  Interrupt it, redirect it, or show it what you see without stopping the task.
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#training">Training</a> ·
  <a href="#offline-inference">Offline Inference</a>
</p>

---

## Overview

Most voice assistants still operate as request-response systems: they wait for an
utterance to end, produce an answer, and pause the interaction while a longer task
runs elsewhere. Gander is built for a different form of collaboration. It
continuously receives speech, video, and text; regulates when to listen, speak, or
yield; and remains available while a general-purpose agent reasons and acts in the
background.

<table>
  <tr>
    <td width="33%" valign="top">
      <p align="center"><strong>Omni Perception</strong></p>
      <p>Speech, camera, and screen streams share a causal timeline, allowing language to be grounded in what is happening now.</p>
    </td>
    <td width="33%" valign="top">
      <p align="center"><strong>Native Interaction</strong></p>
      <p>Listening, speaking, interruption, backchannels, and proactive responses are model decisions rather than a collection of external heuristics.</p>
    </td>
    <td width="34%" valign="top">
      <p align="center"><strong>Agentic Collaboration</strong></p>
      <p>Long-running work is delegated without freezing the conversation, and remains steerable through follow-ups, questions, permissions, and cancellation.</p>
    </td>
  </tr>
</table>

Gander follows two principles. First, **interactivity is part of the model**. The
Cerebellum learns interaction control directly from a continuous audio-visual
stream instead of relying on a voice activity detector to define every turn.
Second, **realtime presence and long-horizon reasoning have different latency
requirements**. A Brain-Cerebellum architecture lets a compact streaming model
stay responsive while a stronger, training-free worker handles complex tool use.

This repository implements the complete path around those ideas: multimodal data
loading and augmentation, Thinker and Talker training, turn-based and chunk-wise
offline inference, a realtime browser experience, and the agent orchestration
runtime that connects the Cerebellum to pluggable Brain providers.

## Architecture

<p align="center">
  <img src="docs/assets/brain-cerebellum-runtime.png" alt="Gander Brain-Cerebellum architecture" width="96%"><br>
  <sub>The Cerebellum owns the realtime loop; the Brain owns long-horizon work; the runtime keeps both on the same task timeline.</sub>
</p>

Gander is organized into three components with deliberately different
responsibilities:

| Component | Role |
| --- | --- |
| **Front Cerebellum** | Continuously consumes microphone, camera, and screen streams. It can answer locally, remain silent, interrupt its own speech, or emit a structured task operation. |
| **Agent Orchestration Runtime** | Binds model actions to trusted user turns, tracks task and execution state, selects multimodal context, routes worker events, and enforces delivery and permission semantics. |
| **Back Brain** | Uses a general-purpose agent to reason, retrieve information, operate tools, edit files, and complete workflows asynchronously. |

### From live input to completed work

1. The browser streams audio and optional camera or screen frames. Source
   timestamps keep visual evidence aligned with the audio unit being processed.
2. The Cerebellum consumes the latest causal unit and predicts an interaction
   decision. Simple requests stay local; complex work becomes `task_start`,
   `task_send`, or `task_resolve`.
3. The runtime binds the operation to the finalized user turn and starts a tracked
   Brain execution with bounded text and visual context.
4. The Brain works asynchronously while the Cerebellum remains available.
   Milestones, questions, permissions, and the final result return through the
   live interaction; superseded runs are fenced from delivery.

The Cerebellum consumes raw audio directly; managed ASR supplies the browser
transcript and the trusted text instruction passed to the Brain.

### Back-Brain tools and providers

With `worker.profile: full`, the Brain retains the complete tool surface of the
configured worker and receives three runtime-specific interfaces:

| Interface | Purpose |
| --- | --- |
| Provider tools | The worker's normal file, shell, retrieval, application, and other configured tools. |
| `context_fetch` | Retrieves selected trusted turns, task events, artifacts, and camera or screen frames from the current task lineage. |
| `memory_search` | Retrieves durable evidence from earlier sessions when an external memory backend is configured. |
| `share` | Sends a verified milestone, important finding, or correction back to the realtime Cerebellum while work continues. |
| Native questions and approvals | Presents missing-information questions or permission requests in the browser and resumes the same worker execution with the user's response. |

The included provider is Codex. Provider construction itself is registry-based:
each implementation declares typed settings and capabilities, so another Brain
can be integrated without changing the Cerebellum task protocol or gateway state
machine.

## Quick Start

This starts the complete browser system: managed ASR, streaming Thinker,
detached Talker, agent runtime, web UI, and Codex-backed Brain. For model-only
evaluation, see [Offline Inference](#offline-inference).

### 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| System | Linux, Conda, and an NVIDIA driver compatible with the CUDA 12.4 PyTorch stack. |
| GPUs | The example uses three physical GPUs, one each for Thinker, Talker, and ASR. |
| Models | [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5), a Gander Thinker checkpoint, its matching Talker checkpoint, and faster-whisper large-v3. |
| Brain | An authenticated Codex-compatible executable supporting `app-server --stdio`. |

### 2. Install and download

```bash
git clone https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent.git
cd Omni-Interaction-Agent

conda env create -f environment.yml
conda activate gander

mkdir -p checkpoints
hf download openbmb/MiniCPM-o-4_5 \
  --local-dir checkpoints/MiniCPM-o-4_5
hf download Gander-Omni/Gander \
  --local-dir checkpoints/Gander
hf download Systran/faster-whisper-large-v3 \
  --local-dir checkpoints/faster-whisper-large-v3
```

Existing local model copies work as well. Use the matching Thinker and Talker
pair identified on the [Gander model
page](https://huggingface.co/Gander-Omni/Gander).

### 3. Configure

```bash
cp gander_runtime/configs/serve.example.yaml \
  gander_runtime/configs/serve.local.yaml
mkdir -p workspace
```

Replace the path placeholders in `serve.local.yaml`:

| Asset | Configuration fields |
| --- | --- |
| MiniCPM-o 4.5 | `model.model_name_or_path`, `model.processor_name_or_path`, `model.token2wav_dir`, and `duplex.ref_audio_path`. |
| Gander | `duplex.checkpoint` and `duplex.talker_checkpoint`. |
| Managed ASR | `asr.model_path`. |
| Brain | `worker.settings.codex_bin`; set `codex_home` only for a compatible wrapper with a separate authenticated home. |

The template already contains the released 8-text/50-speech-token alignment,
`context_slate`, full Brain tools, and the three-GPU mapping: GPU 0 for Thinker,
GPU 1 for Talker, and GPU 2 for ASR.

### 4. Validate and launch

```bash
./scripts/serve.sh --check-config
./scripts/serve.sh
```

The preflight checks paths, dependencies, the worker executable, and GPU
assignment without loading weights. Open `http://127.0.0.1:8000` after the
service starts; `/health` and `/api/asr/health` report runtime and ASR readiness.

For remote access, keep Gander on loopback and proxy both HTTP and WebSocket
routes through HTTPS so browser microphone, camera, and screen permissions work
normally.

## Streaming Cerebellum

<p align="center">
  <img src="docs/assets/streaming-thinker-talker.png" alt="Streaming Thinker-Talker architecture" width="96%"><br>
  <sub>The Thinker controls interaction and content; the detached Talker renders speech without blocking the next perception step.</sub>
</p>

### Chunk-level interaction modeling

The Cerebellum starts from MiniCPM-o 4.5 and flattens continuous interaction into
one-second causal units. Each unit places newly available visual and acoustic
representations before the model output:

```text
[video] [audio] [optional task context] -> [control] [text or tool content]
```

The control token is predicted before content, separating the decision of
*whether to act* from the decision of *what to produce*.

| Control | Behavior |
| --- | --- |
| `listen` | Stay silent and continue observing the stream. |
| `speak` | Generate a bounded text segment and send it to the Talker. |
| `interrupt` | Stop an utterance whose content is no longer appropriate, including when the user takes the floor. |
| `tool` | Emit a structured task operation for the orchestration runtime. |

Because the model sees evolving semantics before making this decision, a pause
does not automatically end a turn and overlapping speech does not automatically
cancel the response. The same formulation supports backchannels, barge-in,
proactive visual responses, and silence under irrelevant or non-directed input.

### Thinker-Talker speech generation

On a `speak` unit, the Talker conditions on Thinker hidden states and aligned text
tokens to predict S3 speech tokens; a causal flow-matching decoder renders them
incrementally. The released alignment is eight text tokens to 50 S3 tokens per
unit. Serving the Talker on a separate GPU lets the Thinker process the next
audio-visual unit while speech is synthesized and played.

The Talker checkpoint contains only trainable TTS tensors. Deployment composes
the base model with the matching Thinker checkpoint before applying the Talker.

### Bounded context for long sessions

The realtime model keeps at most 128 recent units, roughly two minutes of direct
streaming context. Gander provides three policies for what happens beyond that
window:

| Mode | Behavior |
| --- | --- |
| `context_no_previous` | Evicts old units and continues with the recent causal stream only. |
| `context_slate` | Preserves a compact live task slate alongside the rolling stream; no memory model or service is required. |
| `context_memory` | Adds retrieved cross-session memory episodes through a configured memory provider. |

The supplied serving profile uses `context_slate`, which keeps active task state
visible to the Cerebellum without introducing a dependency on a memory model.

## Agent Task Lifecycle

<p align="center">
  <img src="docs/assets/agent-task-lifecycle.png" alt="A live task being delegated, revised, fenced, and delivered" width="96%"><br>
  <sub>Users can keep talking or revise a task while execution generations prevent stale work from becoming the final answer.</sub>
</p>

The Cerebellum exposes a small task vocabulary; the runtime turns it into a full
execution lifecycle.

| Operation | Meaning |
| --- | --- |
| `task_start` | Creates a background task from the current trusted user turn. |
| `task_send` with `main` | Adds information or a constraint to the active task and steers its primary execution. |
| `task_send` with `fork` | Runs a read-only side inquiry without blocking or modifying the primary task. |
| `task_resolve` | Cancels a task or answers a permission request with `allow_once`, `allow_session`, or `deny`. |

Projects, tasks, runs, worker events, and deliveries are persisted as separate
entities. When a user revises an objective, the runtime can create a new execution
generation and mark output from the older generation stale. Progress is not
inferred from arbitrary worker narration: meaningful updates use `share`, native
worker questions remain questions, and final responses pass through the normal
delivery validation path.

This is what lets the browser support interruption, follow-up constraints,
progress requests, multi-question clarification, permission decisions, stop,
reset, and clear without turning them into unrelated chat messages.

## Data and Training Design

<p align="center">
  <img src="docs/assets/agent-data-pipeline.png" alt="Gander audio-agent and omni-agent data construction pipelines" width="96%"><br>
  <sub>Training examples preserve the same causal unit timeline and task lifecycle used by the deployed system.</sub>
</p>

Conventional turn-based data do not teach a model when to remain silent, how to
interpret overlap, or how to coordinate with a worker that may still be running.
Gander is trained on a 2.7M-example mixture organized around those behaviors.

| Data family | Scale | Main supervision |
| --- | ---: | --- |
| Speech interaction | 1.01M | Spoken dialogue and QA, InteractionSpeech full-duplex behavior, interruption, backchannels, and simultaneous translation. |
| Audio-visual interaction | 1.10M | Streaming video QA, narration, temporal grounding, and proactive responses to evolving scenes. |
| Agentic interaction | 359.6K | Delegation, follow-up constraints, progress queries, clarification, permissions, cancellation, GUI trajectories, and tool-grounded reasoning. |
| Robustness and negatives | 229.7K | Irrelevant video, no-command environments, acoustic interference, overlapping distractors, and multi-party addressee tracking. |

Agentic samples are timestamped User-Cerebellum-Brain trajectories rather than
isolated tool calls. Audio-agent and omni-agent paths are filtered for task
validity, lifecycle completeness, conversational coherence, speech quality, and
temporal consistency.

### Train-serve alignment

The training path and realtime runtime share the same core contracts:

- Multimodal inputs and outputs use the same one-second units, with one control
  decision per unit and visual evidence selected by source time.
- Long-context training retains up to 128 prior units and a bounded previous-text
  context instead of silently truncating an oversized example.
- Training-only acoustic augmentation can add ambient noise, babble, transients,
  device coloration, room response, echo, and idle periods while preserving the
  original interaction timeline.

Training is separated into two reproducible stages:

| Stage | Trainable modules | Objective |
| --- | --- | --- |
| **Thinker** | LLM and audio projection | Text, interaction-control, and structured tool-call tokens. |
| **Talker** | TTS projection and decoder; Thinker frozen | Time-aligned S3 speech tokens conditioned on Thinker hidden states. |

A joint mode is included for experiments, but the released recipe and checkpoint
composition follow the two-stage path.

## Evaluation Snapshot

The technical report evaluates Gander as a complete interaction system rather
than only as a static multimodal model. A few results illustrate the behavior the
architecture is designed to produce:

| Evaluation | Result |
| --- | ---: |
| Full-Duplex-Bench v3, appropriate turn-taking | 100% across 100 scenarios |
| Full-Duplex-Bench v3, premature interruption | 6.0% |
| Delegated scenarios correctly bound to the final response | 40 / 40 |
| Audio-visual fusion gain on WorldSense | +5.01 points |
| Audio-visual fusion gain on Daily-Omni | +19.13 points |

These numbers are not presented as a single overall ranking. The report also
documents the current limitations: conservative delegation and limited
multi-tool supervision reduce task accuracy, speech synthesis and ASR errors
affect short spoken answers, and interaction tuning introduces a trade-off on
some static visual-understanding tasks. See the
[technical report](docs/gander-technical-report.pdf) for protocols, complete
tables, ablations, and failure analysis.

## Training

Training uses one entry point and layered YAML configuration. `scripts/train.sh`
loads `minicpm_ft/configs/train.yaml` first, applies each supplied dataset or
experiment YAML in order, and applies dotted command-line overrides last. The
base file owns coherent Thinker, Talker, and joint mode definitions; a release
overlay normally needs to describe only its data and the selected mode.

### Validate the bundled examples

The repository includes 12 Thinker and 6 Talker samples with local audio and
image media. They cover the released manifest format and provide a small
end-to-end data-path check.

```bash
./scripts/prepare_data.sh check \
  minicpm_ft/examples/release/thinker/train_config.yaml
./scripts/prepare_data.sh check \
  minicpm_ft/examples/release/talker/train_config.yaml
```

### Train the Thinker

```bash
BASE=/absolute/path/to/MiniCPM-o-4_5

./scripts/train.sh minicpm_ft/examples/release/thinker/train_config.yaml \
  --model.model_name_or_path "$BASE" \
  --model.processor_name_or_path "$BASE" \
  --launch.gpus_per_node 1 \
  --train.output_dir outputs/thinker
```

Thinker mode trains the LLM and audio projection while keeping the vision tower,
resampler, and TTS modules frozen. The default objective combines text loss with
an interaction-control loss weighted by 1.5.

### Train the Talker

Talker training starts from the completed Thinker and freezes its LLM and audio
projection. Only the TTS projection and decoder are optimized.

```bash
BASE=/absolute/path/to/MiniCPM-o-4_5
THINKER=/absolute/path/to/thinker-checkpoint

./scripts/train.sh minicpm_ft/examples/release/talker/train_config.yaml \
  --model.model_name_or_path "$BASE" \
  --model.processor_name_or_path "$BASE" \
  --runtime.init_checkpoint "$THINKER" \
  --launch.gpus_per_node 1 \
  --train.output_dir outputs/talker
```

The bundled Talker samples already contain S3 targets. For a full release that
does not store `turn.meta.s3_codes` inline, build the cache before training:

```bash
./scripts/prepare_data.sh s3 /path/to/release/train_config.yaml --device cuda:0
```

Both stages default to `save_trainable_only: true`. A Thinker checkpoint is
therefore loaded on top of MiniCPM-o 4.5, and a Talker checkpoint is loaded on top
of that composed Thinker. The smaller files are intentional; neither checkpoint
is a standalone copy of every frozen base-model parameter.

### Use a full dataset release

Create an overlay YAML with `data.release_root`, the ordered manifest lists, and
their row counts or sampling weights. Relative media paths resolve below the
release root. Set `audio_augment.enabled: true` and provide the noise/RIR indices
and profile rules when using the training-time acoustic augmentation pipeline.

Multi-node execution is configured through `launch.hosts` or
`launch.hostfile`. The launcher uses the same YAML on every node and places
transient launcher/cache data under the configured local paths. Inspect the fully
merged recipe before a large run with:

```bash
./scripts/train.sh /path/to/release/train_config.yaml --show-config
```

## Offline Inference

The offline entry point covers two distinct evaluation paths:

| Mode | Input and behavior |
| --- | --- |
| `turn` | Conventional text, audio, image, or combined turn input followed by one generated response. |
| `duplex` | An audio file is consumed in the same one-second units used online, producing per-unit control decisions, incremental text, tool traces, and optional speech. |

Start from the release profile:

```bash
cp minicpm_ft/configs/infer.yaml /tmp/gander-infer.yaml
```

Set `model.model_name_or_path`, `model.processor_name_or_path`,
`inference.checkpoint`, and the fields under `inference.input`, then run:

```bash
./scripts/infer.sh /tmp/gander-infer.yaml
```

For turn-based inference, keep `inference.mode: turn` and provide any supported
combination of `input.text`, `input.audio`, and `input.image`.

For chunk-wise duplex inference, use the streaming values of the released
checkpoint:

```yaml
inference:
  mode: duplex
  checkpoint: /path/to/gander-thinker-checkpoint
  input:
    audio: /path/to/input.wav
  duplex:
    chunk_ms: 1000
    speak_text_tokens_per_unit: 8
    talker_speech_tokens_per_unit: 50
    sliding_window_mode: context_slate
    context_previous_max_tokens: 1500
```

To generate speech, additionally set:

```yaml
model:
  init_tts: true
  token2wav_dir: /path/to/MiniCPM-o-4_5/assets/token2wav

inference:
  talker_checkpoint: /path/to/gander-talker-checkpoint
  generate_audio: true
  output_audio: /path/to/output.wav
  input:
    audio: /path/to/input.wav
    ref_audio: /path/to/reference.wav
```

Turn mode and duplex mode deliberately remain separate: the former measures
response quality for a bounded multimodal request, while the latter exercises the
same incremental control and context behavior used by the realtime Cerebellum.

## Deployment Options

The main deployment choices live in the same serving YAML:

| Setting | Options and effect |
| --- | --- |
| `server.mode` | `lean` directly executes trusted Cerebellum task actions with the shortest control path. `coordinator` adds a planning control layer for per-task reasoning, supervision, and delivery policy. |
| `duplex.media_mode` | `voice` starts with audio interaction; `omni` requires visual input; `auto` permits client selection. `allow_client_video` independently controls camera and screen attachment. |
| `worker.provider` / `worker.profile` | Selects a registered Brain backend and either its task-scoped or full tool surface. This release includes `codex`. |
| `asr.mode` | Runs managed local ASR, connects to an external ASR service, or disables ASR where the workflow does not require transcripts. |

## Repository Layout

| Path | Contents |
| --- | --- |
| `minicpm_ft/mcpmft/data/` | Manifest loading, media resolution, augmentation, temporal serialization, collation, and S3 targets. |
| `minicpm_ft/mcpmft/modeling/` | Model loading, trainable-module selection, omni forward path, and streaming acoustic features. |
| `minicpm_ft/mcpmft/infer/` | Turn-based and chunk-wise inference, context windows, detached Talker, ASR, and browser client. |
| `gander_runtime/gander_runtime/` | Realtime transport, media timeline, task gateway, ledger, context routing, worker events, and supervision. |
| `gander_runtime/gander_runtime/providers/` | Typed Brain provider registry, factory, capabilities, and Codex adapter. |
| `minicpm_ft/examples/release/` | Self-contained Thinker and Talker manifest examples with local media. |
| `scripts/` | Stable entry points for data validation, training, offline inference, and serving. |
| `docs/` | Technical report and selected system figures. |

## Resources

| Resource | Link |
| --- | --- |
| Paper | [Omni Interaction Agent Technical Report](docs/gander-technical-report.pdf) |
| Model | [Gander-Omni/Gander](https://huggingface.co/Gander-Omni/Gander) |
| Dataset | Coming soon |
| Demo | Coming soon |

## License

The code in this repository is released under the
[Apache License 2.0](LICENSE). 
