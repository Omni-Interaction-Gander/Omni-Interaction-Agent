<h1 align="center">Gander</h1>

<p align="center">
  <strong>Omni Interaction Agent</strong><br>
  <sub>Continuous audio-visual interaction and long-running agent work in one system.</sub>
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

Gander is a full-duplex audio-visual agent for conversations that continue while
real work is being done. A low-latency **Cerebellum** keeps listening, watching,
speaking, and reacting; an asynchronous, tool-using **Brain** handles longer
tasks. The runtime joins them into one interaction rather than making the user
choose between a voice assistant and an agent.

<table>
  <tr>
    <td width="33%" valign="top">
      <p align="center"><strong>Realtime Cerebellum</strong></p>
      <p>Streams live audio and video, handles overlap and interruption, and stays responsive to the scene as it changes.</p>
    </td>
    <td width="33%" valign="top">
      <p align="center"><strong>Tool-using Brain</strong></p>
      <p>Runs longer tasks asynchronously while conversation, follow-ups, and redirection remain available.</p>
    </td>
    <td width="34%" valign="top">
      <p align="center"><strong>Coordinated Runtime</strong></p>
      <p>Keeps context, milestones, questions, permissions, cancellation, and final delivery attached to the right task.</p>
    </td>
  </tr>
</table>

The repository contains the complete path from multimodal training and detached
Thinker-Talker inference to the browser runtime and provider-neutral Brain
integration.

## Architecture

<p align="center">
  <img src="docs/assets/brain-cerebellum-runtime.png" alt="Gander Brain-Cerebellum architecture" width="96%"><br>
  <sub>The Cerebellum owns the realtime loop; the Brain owns long-horizon work; the runtime keeps them synchronized.</sub>
</p>

Gander separates work by latency and responsibility:

| Component | Responsibility |
| --- | --- |
| **Front Cerebellum** | Continuously perceives microphone, camera, and screen streams; responds locally or emits a task tool call. |
| **Agent Orchestration Runtime** | Binds actions to trusted user turns, tracks task state and execution generations, routes context, handles questions and permissions, and schedules delivery. |
| **Back Brain** | Uses a general-purpose worker agent to reason, retrieve information, operate tools, edit files, and complete long-running workflows. |

A typical interaction follows four steps:

1. The Cerebellum consumes the current audio-visual chunk and chooses whether to
   listen, speak, interrupt, or invoke a task operation.
2. The runtime binds that operation to the finalized user turn and creates or
   updates a tracked task.
3. A selected worker receives the instruction plus bounded multimodal context and
   executes asynchronously while the Cerebellum remains available.
4. Worker milestones, questions, approvals, corrections, and final results return
   through the runtime. The Cerebellum decides when and how to present them in the
   ongoing conversation.

The bundled worker is Codex. Worker construction is provider-neutral: a provider
registers typed settings, a factory, and declared capabilities, so another Brain
can be integrated without changing the Cerebellum protocol.

### Back-Brain Tools and Interaction

With `worker.profile: full`, the Brain keeps the tool surface of the configured
Codex installation. Gander also injects two run-scoped MCP servers that connect
the worker to the live task runtime:

| Surface | Purpose |
| --- | --- |
| Provider tools | The Brain's normal file, shell, search, application, and other configured tools. |
| `context_fetch` | Pull the minimum required current-task context: trusted turns, the frozen task-start timeline, runtime events, artifacts, and selected camera or screen images. |
| `memory_search` | Retrieve durable evidence from earlier sessions when a memory backend is configured. |
| `share` | Deliver a verified milestone, important finding, or correction to the realtime Cerebellum while work continues. |
| Native questions and approvals | Route missing-information questions and permission requests to the browser, then resume the same worker run with the user's answer. |

The runtime deliberately does not copy the whole conversation into every worker
prompt. The Brain pulls bounded context when needed, reports intermediate progress
only through `share`, and returns its complete result through the normal final
message. A `task_send` update steers the active run; a `fork` query creates a
read-only side branch without blocking or modifying the main task. Cancellation,
superseded generations, permissions, and final delivery remain enforced by the
runtime rather than by prompt convention.

## Quick Start

Download the Thinker and matching Talker checkpoints from
[Gander-Omni/Gander](https://huggingface.co/Gander-Omni/Gander). You also need the
MiniCPM-o 4.5 base model, a local `faster-whisper-large-v3` model, and an
authenticated Codex-compatible command.

```bash
conda env create -f environment.yml
conda activate gander

cp gander_runtime/configs/serve.example.yaml \
  gander_runtime/configs/serve.local.yaml
mkdir -p workspace
```

In `serve.local.yaml`, set the base-model paths under `model`, the Thinker and
Talker paths under `duplex`, `asr.model_path`, and
`worker.settings.codex_bin`. The example assigns Thinker, Talker, and ASR to
physical GPUs 0, 1, and 2 respectively and enables video, streaming speech,
`context_slate`, and the full Brain tool profile.

```bash
./scripts/serve.sh gander_runtime/configs/serve.local.yaml --check-config
./scripts/serve.sh gander_runtime/configs/serve.local.yaml
```

Open `http://127.0.0.1:8000`. For a remote browser, proxy the HTTP and WebSocket
routes through HTTPS while keeping the service bound to loopback.

## Streaming Thinker-Talker

<p align="center">
  <img src="docs/assets/streaming-thinker-talker.png" alt="Streaming Thinker-Talker architecture" width="96%"><br>
  <sub>A streaming Thinker decides when and what to say; a detached Talker renders speech without blocking perception.</sub>
</p>

The Cerebellum starts from MiniCPM-o 4.5 and organizes continuous interaction into
fixed one-second units. Each unit places newly available video and audio tokens
before the model output. The Thinker first predicts a control decision, then emits
text or a structured tool call when required:

```text
[video] [audio] [optional input/tool result] -> [listen | speak | interrupt | tool] [content]
```

This formulation matters because a pause is not necessarily the end of a turn,
and overlapping speech is not necessarily an interruption. The control decision
uses the same semantic representation as the response, allowing the model to
learn backchannels, barge-in handling, proactive visual responses, and silence in
irrelevant or non-directed contexts.

On the speak path, the Talker conditions on Thinker hidden states and text tokens
to predict compact S3 speech tokens. A causal flow-matching decoder converts those
tokens into waveform chunks as they arrive. The deployed runtime can place the
Talker on a separate GPU so speech synthesis does not block the Thinker's next
interaction decision.

Long sessions use a bounded 128-unit sliding context. Gander provides three
serving policies: direct unit eviction with absolute positions, a pinned live task
slate, or a task slate plus summarized memory episodes.

## Agent Task Lifecycle

<p align="center">
  <img src="docs/assets/agent-task-lifecycle.png" alt="A live task being delegated, revised, fenced, and delivered" width="96%"><br>
  <sub>Tasks remain steerable while execution generations keep superseded work from leaking into the final answer.</sub>
</p>

The Cerebellum exposes only three task operations:

| Operation | Meaning |
| --- | --- |
| `task_start` | Start a background task from the current trusted user turn. |
| `task_send` | Update the active task on the `main` lane, or ask a read-only `fork` question without disturbing it. |
| `task_resolve` | Cancel a task or resolve a pending permission with `allow_once`, `allow_session`, or `deny`. |

The runtime turns these compact model actions into a reliable lifecycle. It keeps a
task ledger, isolates worker projects, associates follow-ups with the correct task,
and fences superseded executions. In the example above, adding a Python 3.12
constraint creates a new execution; output from the old execution is marked stale
and cannot be delivered as the final answer.

Intermediate worker narration is not forwarded blindly. Meaningful milestones and
corrections use the provider-neutral `share` channel, native worker questions and
approval requests become user interactions, and the final worker response is
validated before delivery. This preserves a responsive foreground conversation
without losing control of background work.

## Data Design

<p align="center">
  <img src="docs/assets/agent-data-pipeline.png" alt="Gander audio-agent and omni-agent data construction pipelines" width="96%"><br>
  <sub>Training examples preserve the same causal timeline and interaction lifecycle used by the deployed system.</sub>
</p>

Gander is trained on a 2.7M-example mixture organized around behavior rather than
only modality:

| Family | What it supervises |
| --- | --- |
| **Speech interaction** | Spoken dialogue and QA, InteractionSpeech full-duplex behavior, interruption, backchannels, and simultaneous translation. |
| **Audio-visual interaction** | Streaming video QA, narration, temporal grounding, and proactive responses to evolving scenes. |
| **Agentic interaction** | Delegation, follow-up constraints, progress queries, clarification, permissions, cancellation, GUI trajectories, and final delivery. |
| **Robustness and negatives** | Irrelevant video, no-command environments, noise, overlapping distractors, and multi-party addressee tracking. |

Agentic examples are synthesized as timestamped User-Cerebellum-Brain trajectories,
not isolated tool calls. Audio-agent data begin with hierarchical task and
interaction seeds; omni-agent data begin with action-aligned GUI or video
trajectories. Both paths are filtered for task validity, interaction coherence,
lifecycle completeness, speech quality, and timeline consistency.

All training views share the same causal unit contract used online. Visual evidence
is selected according to its source time, speech and interaction events retain
their global timing, and the model is supervised on one control action per unit.
The standard long-context recipe keeps at most 128 preceding units and 1,500
preceding-context tokens rather than silently truncating an example. Type-aware
augmentation covers ambient noise, babble, transient sounds, device coloration,
room acoustics, echo, and endpoint idle periods while preserving the media
timeline.

Training has two stages:

| Stage | Trainable modules | Target |
| --- | --- | --- |
| **Thinker** | LLM and audio projection | Text, interaction-control, and tool-call tokens |
| **Talker** | TTS projection and decoder; Thinker frozen | Time-aligned S3 speech tokens |

The code also supports a joint recipe for experiments that optimize both paths.

## Repository

| Path | Contents |
| --- | --- |
| `minicpm_ft/mcpmft/data/` | Manifest loading, multimodal feature extraction, augmentation, serialization, collation, and S3 targets |
| `minicpm_ft/mcpmft/modeling/` | Model loading, trainable-module selection, omni forward path, and streaming audio features |
| `minicpm_ft/mcpmft/infer/` | Turn-based and chunk-wise inference, detached Talker, sliding context, ASR, and browser client |
| `gander_runtime/gander_runtime/` | Realtime transport, task gateway, ledger, context routing, worker events, and supervision |
| `gander_runtime/gander_runtime/providers/` | Typed worker provider registry and Codex adapter |
| `minicpm_ft/examples/release/` | Small self-contained Thinker and Talker training views with local media |
| `scripts/` | Data, training, inference, and serving entry points |
| `docs/` | Technical report and selected architecture figures |

## Training

The repository includes real local media for 12 Thinker and 6 Talker examples.
Validate them first:

```bash
./scripts/prepare_data.sh check minicpm_ft/examples/release/thinker/train_config.yaml
./scripts/prepare_data.sh check minicpm_ft/examples/release/talker/train_config.yaml
```

Run a one-GPU Thinker job; increase `launch.gpus_per_node` for the GPUs on each
host:

```bash
BASE=/path/to/MiniCPM-o-4_5
./scripts/train.sh minicpm_ft/examples/release/thinker/train_config.yaml \
  --model.model_name_or_path "$BASE" \
  --model.processor_name_or_path "$BASE" \
  --launch.gpus_per_node 1 \
  --train.output_dir outputs/thinker
```

The Talker is trained from the finished Thinker. Its checkpoint intentionally
stores only trainable TTS tensors:

```bash
BASE=/path/to/MiniCPM-o-4_5
THINKER=/path/to/thinker-checkpoint
./scripts/train.sh minicpm_ft/examples/release/talker/train_config.yaml \
  --model.model_name_or_path "$BASE" \
  --model.processor_name_or_path "$BASE" \
  --runtime.init_checkpoint "$THINKER" \
  --launch.gpus_per_node 1 \
  --train.output_dir outputs/talker
```

The compact Talker examples already contain S3 targets. For a release without
inline `turn.meta.s3_codes`, build them before training:

```bash
./scripts/prepare_data.sh s3 /path/to/release/train_config.yaml --device cuda:0
```

For full data, keep the same overlay format and set `data.release_root` plus the
manifest lists. Relative audio, image, and video paths resolve below that root;
video files are read directly. Multi-node jobs use `launch.hosts` or
`launch.hostfile` in the YAML. Use `--show-config` to inspect the final merged
recipe, and set `MCPMFT_PYTHON` only when the target Python is not active.

## Offline Inference

Copy the inference profile, then set the model, checkpoint, and input paths:

```bash
cp minicpm_ft/configs/infer.yaml /tmp/gander-infer.yaml
```

Required fields are `model.model_name_or_path`,
`model.processor_name_or_path`, `inference.checkpoint`, and one or more fields
under `inference.input`.

Run turn-based text, audio, image, or multimodal inference:

```bash
./scripts/infer.sh /tmp/gander-infer.yaml
```

For chunk-wise offline duplex inference, set:

```yaml
inference:
  mode: duplex
  checkpoint: /path/to/thinker-checkpoint
  input:
    audio: /path/to/input.wav
  duplex:
    speak_text_tokens_per_unit: 8
    talker_speech_tokens_per_unit: 50
    sliding_window_mode: context_slate
    context_previous_max_tokens: 1500
```

For speech output, set `inference.generate_audio: true`, provide the Talker
checkpoint in `inference.talker_checkpoint`, set `model.init_tts: true` and
`model.token2wav_dir`, then provide `inference.input.ref_audio` and
`inference.output_audio`. The text and speech token budgets must match the
checkpoint.

## Runtime Options

The serving profile exposes the main runtime choices:

| Setting | Options |
| --- | --- |
| `server.mode` | `lean` directly executes Cerebellum task actions; `coordinator` adds a planning control layer. |
| `duplex.sliding_window_mode` | `context_no_previous`, `context_slate`, or `context_memory`. |
| `worker.provider` | Registered Brain backend; this release includes `codex`. |
| `asr.mode` | Managed local ASR, external ASR service, or disabled. |
| `duplex.media_mode` | `voice`, `omni`, or client-selected `auto`. |

`context_no_previous` directly evicts old units, `context_slate` pins the live
task slate without a memory model, and `context_memory` also accepts summarized
memory episodes. Set `GANDER_PYTHON` only when serving with a Python outside the
active environment. Proxy variables exported before `scripts/serve.sh` are
inherited by the worker.

## Resources

| Resource | Link |
| --- | --- |
| Paper | [Omni Interaction Agent Technical Report](docs/gander-technical-report.pdf) |
| Demo | Coming soon |
| Dataset | Coming soon |
| Model | [Gander-Omni/Gander](https://huggingface.co/Gander-Omni/Gander) |
