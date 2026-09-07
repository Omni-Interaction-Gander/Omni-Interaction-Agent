'use strict';

const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const BUFFER_SIZE = 4096;
const WS_CONNECT_TIMEOUT_MS = 120000;
// Keep a short queue so normal inference and network jitter is not audible at
// every one-second PCM packet boundary.
const PLAYBACK_PREROLL_SECONDS = 0.2;
const PLAYBACK_UNDERRUN_GUARD_SECONDS = 0.04;
const PLAYBACK_STALE_TURN_SECONDS = 1.5;
const STOP_ACK_TIMEOUT_MS = 10000;
const RESET_ACK_TIMEOUT_MS = 30000;
const NOISE_GATE_HOLD_MS = 500;
const ASR_END_SILENCE_SAMPLES = Math.round(INPUT_RATE * 0.9);
const ASR_START_VOICE_SAMPLES = Math.round(INPUT_RATE * 0.22);
const ASR_MIN_VOICE_SAMPLES = Math.round(INPUT_RATE * 0.32);
const ASR_ROLLING_SEGMENT_SAMPLES = INPUT_RATE * 25;
const ASR_MIN_GATE_DB = -44;
const ASR_NOISE_WINDOW_FRAMES = 64;
const ASR_NOISE_QUANTILE = 0.35;
const ASR_NOISE_SAMPLE_CEILING = 0.02;
const ASR_START_NOISE_MULTIPLIER = 1.8;
const ASR_ACTIVE_NOISE_MULTIPLIER = 1.35;
const HISTORY_INDEX_KEY = 'minicpm-session-archive-index-v2';
const HISTORY_ITEM_PREFIX = 'minicpm-session-archive-v2:';
const HISTORY_MAX_ARCHIVES = 100;
const HISTORY_PAGE_SIZE = 40;
const LIVE_MESSAGE_LIMIT = 100;
const GATE_SETTINGS_VERSION = '4';
const CLIENT_STORAGE_KEY = 'minicpm-browser-client-v1';
const DEFAULT_REQUEST_TIMEOUT_MS = 45000;

const statusEl = document.getElementById('status');
const connectionStateEl = document.getElementById('connectionState');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const clearBtn = document.getElementById('clearBtn');
const historyBtn = document.getElementById('historyBtn');
const historyCountEl = document.getElementById('historyCount');
const historyDialog = document.getElementById('historyDialog');
const historyListEl = document.getElementById('historyList');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const meterEl = document.getElementById('meter');
const inputLevelEl = document.getElementById('inputLevel');
const gateSlider = document.getElementById('gateSlider');
const gateValue = document.getElementById('gateValue');
const languageSelect = document.getElementById('languageSelect');
const waveformEl = document.getElementById('waveform');
const transcriptEl = document.getElementById('transcript');
const emptyStateEl = document.getElementById('emptyState');
const liveCaptionEl = document.getElementById('liveCaption');
const eventLogEl = document.getElementById('eventLog');
const sessionTimerEl = document.getElementById('sessionTimer');
const transportValueEl = document.getElementById('transportValue');
const asrStatusEl = document.getElementById('asrStatus');
const micTimelineStateEl = document.getElementById('micTimelineState');
const chunkCountEl = document.getElementById('chunkCount');
const modelLagStateEl = document.getElementById('modelLagState');
const modelStateEl = document.getElementById('modelState');
const asrTurnCountEl = document.getElementById('asrTurnCount');
const contextStateEl = document.getElementById('contextState');
const outputAudioStateEl = document.getElementById('outputAudioState');

let transport = null;
let ws = null;
let connectingWs = null;
let capturePaused = false;
let stopping = false;
let sessionState = 'idle';
let stopAckHandle = null;
let resetAckHandle = null;
let cleanupPromise = null;

let micStream = null;
let micContext = null;
let micSource = null;
let micProcessor = null;
let silenceGain = null;
let gateOpenUntil = 0;
let capturedSamples = 0;
let audioFrameSequence = 0;
let audioInputProtocol = null;

let playbackContext = null;
let nextPlayTime = 0;
const playbackSources = new Set();
let playbackGeneration = 0;
let pendingBinaryAudio = null;
let assistantAudioFloorUnit = 0;

let sessionStartedAt = null;
let sessionWallStartedAt = null;
let sessionTimerHandle = null;
let chunkCount = 0;
let asrTurnCount = 0;
let assistantMessage = null;
let contextMaxUnits = 128;
let modelChunkMs = 1000;

let asrAvailable = false;
let asrUtterance = null;
let asrSpeechCandidate = null;
let asrQueue = Promise.resolve();
let asrNoiseSamples = [];
let asrNoiseFloorRms = 0;
let taskProtocol = null;
let finalTurnCounter = 0;
const pendingFinalTurns = new Map();
let asrGeneration = 0;
let historyIndex = [];
let volatileHistoryArchives = [];
let detachedTranscriptMessages = [];
let transcriptGeneration = 0;
let archivedTranscriptGeneration = -1;
let transcriptScrollFrame = null;

const waveformLevels = Array(56).fill(0);
let browserClientId = loadBrowserClientId();
const helpTooltipEl = document.createElement('div');
helpTooltipEl.className = 'help-tooltip';
helpTooltipEl.id = 'buttonHelpTooltip';
helpTooltipEl.setAttribute('role', 'tooltip');
helpTooltipEl.hidden = true;
document.body.append(helpTooltipEl);
let activeHelpTrigger = null;

function newBrowserClientId() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function loadBrowserClientId() {
  try {
    const stored = sessionStorage.getItem(CLIENT_STORAGE_KEY);
    if (stored) return stored;
    const created = newBrowserClientId();
    sessionStorage.setItem(CLIENT_STORAGE_KEY, created);
    return created;
  } catch (_) {
    return newBrowserClientId();
  }
}

function hideButtonHelp(trigger = activeHelpTrigger) {
  if (!trigger || trigger !== activeHelpTrigger) return;
  activeHelpTrigger = null;
  helpTooltipEl.hidden = true;
  helpTooltipEl.textContent = '';
}

function showButtonHelp(trigger) {
  const description = trigger.dataset.help;
  if (!description) return;
  activeHelpTrigger = trigger;
  helpTooltipEl.textContent = description;
  helpTooltipEl.hidden = false;
  helpTooltipEl.dataset.placement = 'below';

  const triggerRect = trigger.getBoundingClientRect();
  const tooltipRect = helpTooltipEl.getBoundingClientRect();
  const viewportPadding = 10;
  const gap = 9;
  const centeredLeft = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
  const left = Math.min(
    window.innerWidth - tooltipRect.width - viewportPadding,
    Math.max(viewportPadding, centeredLeft)
  );
  const arrowLeft = Math.min(
    tooltipRect.width - 12,
    Math.max(12, triggerRect.left + triggerRect.width / 2 - left)
  );
  let top = triggerRect.bottom + gap;
  if (top + tooltipRect.height > window.innerHeight - viewportPadding) {
    top = Math.max(viewportPadding, triggerRect.top - tooltipRect.height - gap);
    helpTooltipEl.dataset.placement = 'above';
  }
  helpTooltipEl.style.left = `${Math.round(left)}px`;
  helpTooltipEl.style.top = `${Math.round(top)}px`;
  helpTooltipEl.style.setProperty('--help-arrow-left', `${Math.round(arrowLeft)}px`);
}

function enhanceButtonHelp(button) {
  if (!button?.dataset.help || button.parentElement?.classList.contains('button-help-wrap')) {
    return button.parentElement || button;
  }
  const wrapper = document.createElement('span');
  wrapper.className = 'button-help-wrap';
  const trigger = document.createElement('span');
  trigger.className = 'button-help-trigger';
  trigger.textContent = '?';
  trigger.tabIndex = 0;
  trigger.setAttribute('role', 'button');
  trigger.setAttribute(
    'aria-label',
    `${button.textContent.trim()} 功能说明：${button.dataset.help}`
  );
  trigger.dataset.help = button.dataset.help;
  trigger.addEventListener('mouseenter', () => showButtonHelp(trigger));
  trigger.addEventListener('mouseleave', () => hideButtonHelp(trigger));
  trigger.addEventListener('focus', () => showButtonHelp(trigger));
  trigger.addEventListener('blur', () => hideButtonHelp(trigger));

  if (button.parentNode) button.parentNode.insertBefore(wrapper, button);
  wrapper.append(button, trigger);
  return wrapper;
}

window.addEventListener('scroll', () => hideButtonHelp(), true);
window.addEventListener('resize', () => hideButtonHelp());

function serviceUrl(path) {
  const base = new URL(window.location.href);
  base.hash = '';
  base.search = '';
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return new URL(path.replace(/^\//, ''), base).toString();
}

function setStatus(text, state = 'idle') {
  statusEl.textContent = text;
  connectionStateEl.dataset.state = state;
}

function setModelState(text) {
  modelStateEl.textContent = text;
}

function setSessionState(state) {
  sessionState = state;
  startBtn.disabled = state !== 'idle';
  stopBtn.disabled = !['starting', 'active'].includes(state);
  resetBtn.disabled = state !== 'active';
  window.GanderVideo?.setSessionState(state);
}

function clearControlTimers() {
  if (stopAckHandle !== null) window.clearTimeout(stopAckHandle);
  if (resetAckHandle !== null) window.clearTimeout(resetAckHandle);
  stopAckHandle = null;
  resetAckHandle = null;
}

function updatePipelineTiming() {
  const capturedSeconds = capturedSamples / INPUT_RATE;
  const processedSeconds = chunkCount * modelChunkMs / 1000;
  micTimelineStateEl.textContent = `${capturedSeconds.toFixed(1)} s`;
  modelLagStateEl.textContent = `${Math.max(0, capturedSeconds - processedSeconds).toFixed(1)} s`;
}

function formatTime(seconds, decimals = false) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  if (decimals) {
    return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(1).padStart(4, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(remainder)).padStart(2, '0')}`;
}

function sessionSeconds() {
  return sessionStartedAt ? (performance.now() - sessionStartedAt) / 1000 : 0;
}

function startSessionTimer() {
  stopSessionTimer();
  sessionStartedAt = performance.now();
  sessionWallStartedAt = Date.now();
  sessionTimerEl.textContent = '00:00';
  sessionTimerHandle = window.setInterval(() => {
    sessionTimerEl.textContent = formatTime(sessionSeconds());
  }, 250);
}

function stopSessionTimer() {
  if (sessionTimerHandle) window.clearInterval(sessionTimerHandle);
  sessionTimerHandle = null;
}

function resetRuntimeSignals({ clearEvents = false, resetClock = false } = {}) {
  chunkCount = 0;
  asrTurnCount = 0;
  capturedSamples = 0;
  audioFrameSequence = 0;
  gateOpenUntil = 0;
  assistantMessage = null;
  asrUtterance = null;
  asrSpeechCandidate = null;
  pendingBinaryAudio = null;
  chunkCountEl.textContent = '0';
  asrTurnCountEl.textContent = '0';
  micTimelineStateEl.textContent = '0.0 s';
  modelLagStateEl.textContent = '0.0 s';
  contextStateEl.textContent = `0 / ${contextMaxUnits}`;
  contextStateEl.title = '';
  liveCaptionEl.textContent = '--';
  stopPlayback();
  meterEl.style.width = '0%';
  inputLevelEl.textContent = '-120 dB';
  waveformLevels.fill(0);
  if (clearEvents) eventLogEl.textContent = '';
  if (resetClock) {
    stopSessionTimer();
    sessionStartedAt = null;
    sessionWallStartedAt = null;
    sessionTimerEl.textContent = '00:00';
  }
}

function addEvent(message) {
  const item = document.createElement('div');
  item.className = 'event-item';
  const time = document.createElement('time');
  time.textContent = formatTime(sessionSeconds());
  const text = document.createElement('span');
  text.textContent = message;
  item.append(time, text);
  eventLogEl.append(item);
  while (eventLogEl.children.length > 80) eventLogEl.firstElementChild.remove();
  eventLogEl.scrollTop = eventLogEl.scrollHeight;
}

function revealTranscript() {
  if (emptyStateEl && emptyStateEl.isConnected) emptyStateEl.remove();
}

function scheduleTranscriptScroll() {
  if (transcriptScrollFrame !== null) return;
  transcriptScrollFrame = window.requestAnimationFrame(() => {
    transcriptScrollFrame = null;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  });
}

function snapshotMessageNode(root) {
  const text = root.querySelector('.message-text');
  return {
    role: root.classList.contains('user') ? 'user' : 'assistant',
    start: Number(root.dataset.start) || 0,
    timing: root.querySelector('time')?.textContent || '',
    text: text?.textContent || ''
  };
}

function pruneLiveTranscript() {
  const messages = Array.from(transcriptEl.querySelectorAll('.message'));
  let removeCount = messages.length - LIVE_MESSAGE_LIMIT;
  if (removeCount <= 0) return;

  for (const root of messages) {
    if (removeCount <= 0) break;
    if (assistantMessage && root === assistantMessage.root) continue;
    detachedTranscriptMessages.push(snapshotMessageNode(root));
    root.remove();
    removeCount -= 1;
  }
}

function clearTranscript() {
  transcriptGeneration += 1;
  detachedTranscriptMessages = [];
  if (transcriptScrollFrame !== null) {
    window.cancelAnimationFrame(transcriptScrollFrame);
    transcriptScrollFrame = null;
  }
  transcriptEl.textContent = '';
  transcriptEl.append(emptyStateEl);
  assistantMessage = null;
  liveCaptionEl.textContent = '--';
}

function historyItemKey(id) {
  return `${HISTORY_ITEM_PREFIX}${id}`;
}

function loadHistoryIndex() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORY_INDEX_KEY) || '[]');
    if (!Array.isArray(value)) return [];
    return value
      .filter((item) => item && typeof item.id === 'string')
      .slice(0, HISTORY_MAX_ARCHIVES);
  } catch (_) {
    return [];
  }
}

function summarizeArchive(archive) {
  return {
    id: archive.id,
    startedAt: archive.startedAt,
    endedAt: archive.endedAt,
    duration: archive.duration,
    messageCount: archive.messages.length
  };
}

function removeStoredArchive(id) {
  try {
    localStorage.removeItem(historyItemKey(id));
  } catch (_) {
    // Keep the in-memory index usable when storage is unavailable.
  }
}

function persistHistoryArchive(archive) {
  const summary = summarizeArchive(archive);
  const serialized = JSON.stringify(archive);
  const retained = historyIndex.filter((item) => item.id !== archive.id);

  while (retained.length >= HISTORY_MAX_ARCHIVES) {
    removeStoredArchive(retained.pop().id);
  }

  while (true) {
    try {
      localStorage.setItem(historyItemKey(archive.id), serialized);
      break;
    } catch (_) {
      const evicted = retained.pop();
      if (!evicted) return false;
      removeStoredArchive(evicted.id);
    }
  }

  const nextIndex = [summary, ...retained];
  try {
    localStorage.setItem(HISTORY_INDEX_KEY, JSON.stringify(nextIndex));
  } catch (_) {
    removeStoredArchive(archive.id);
    return false;
  }
  historyIndex = nextIndex;
  return true;
}

function clearStoredHistory() {
  try {
    for (let index = localStorage.length - 1; index >= 0; index -= 1) {
      const key = localStorage.key(index);
      if (
        key === HISTORY_INDEX_KEY ||
        key?.startsWith(HISTORY_ITEM_PREFIX)
      ) {
        localStorage.removeItem(key);
      }
    }
  } catch (_) {
    // The visible history is still cleared when storage is unavailable.
  }
  historyIndex = [];
  volatileHistoryArchives = [];
}

function historyEntries() {
  return [
    ...volatileHistoryArchives.map((archive) => summarizeArchive(archive)),
    ...historyIndex
  ];
}

function updateHistoryControls() {
  const knownCount = historyEntries().length;
  historyCountEl.textContent = String(knownCount);
  historyBtn.disabled = knownCount === 0;
  clearHistoryBtn.disabled = historyBtn.disabled;
}

function snapshotTranscript() {
  const visible = Array.from(transcriptEl.querySelectorAll('.message')).map(snapshotMessageNode);
  return [...detachedTranscriptMessages, ...visible]
    .sort((left, right) => left.start - right.start)
    .map(({ role, timing, text }) => ({ role, timing, text }));
}

function archiveCurrentConversation() {
  if (archivedTranscriptGeneration === transcriptGeneration) return false;
  const messages = snapshotTranscript();
  if (!messages.length) return false;
  const archive = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    startedAt: sessionWallStartedAt || Date.now(),
    endedAt: Date.now(),
    duration: sessionSeconds(),
    messages
  };
  if (!persistHistoryArchive(archive)) volatileHistoryArchives.unshift(archive);
  archivedTranscriptGeneration = transcriptGeneration;
  updateHistoryControls();
  if (historyDialog.open || historyDialog.hasAttribute('open')) renderHistoryArchives();
  return true;
}

function appendArchivedMessage(parent, message) {
  const root = document.createElement('article');
  root.className = `message ${message.role === 'user' ? 'user' : 'assistant'}`;
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  const label = document.createElement('span');
  label.textContent = message.role === 'user' ? 'User' : 'Omni-Interaction-Agent';
  const timing = document.createElement('time');
  timing.textContent = message.timing || '--';
  meta.append(label, timing);

  const text = document.createElement('p');
  text.className = 'message-text';
  text.textContent = message.text || (message.words || []).map((word) => word.text || '').join('');
  root.append(meta, text);
  parent.append(root);
}

function renderArchivePage(parent, archive) {
  const messages = Array.isArray(archive.messages) ? archive.messages : [];
  let pageStart = Math.max(0, messages.length - HISTORY_PAGE_SIZE);

  const drawPage = () => {
    parent.textContent = '';
    const pageEnd = Math.min(messages.length, pageStart + HISTORY_PAGE_SIZE);
    if (messages.length > HISTORY_PAGE_SIZE) {
      const navigation = document.createElement('div');
      navigation.className = 'history-page-nav';
      const earlier = document.createElement('button');
      earlier.type = 'button';
      earlier.className = 'quiet-command';
      earlier.textContent = 'Earlier';
      earlier.dataset.help = '查看当前历史会话中更早的一页消息。';
      earlier.disabled = pageStart === 0;
      earlier.addEventListener('click', () => {
        pageStart = Math.max(0, pageStart - HISTORY_PAGE_SIZE);
        drawPage();
      });
      const position = document.createElement('span');
      position.textContent = `${pageStart + 1}-${pageEnd} of ${messages.length}`;
      const newer = document.createElement('button');
      newer.type = 'button';
      newer.className = 'quiet-command';
      newer.textContent = 'Newer';
      newer.dataset.help = '查看当前历史会话中更新的一页消息。';
      newer.disabled = pageEnd === messages.length;
      newer.addEventListener('click', () => {
        pageStart = Math.min(
          Math.max(0, messages.length - HISTORY_PAGE_SIZE),
          pageStart + HISTORY_PAGE_SIZE
        );
        drawPage();
      });
      navigation.append(
        enhanceButtonHelp(earlier),
        position,
        enhanceButtonHelp(newer)
      );
      parent.append(navigation);
    }
    for (const message of messages.slice(pageStart, pageEnd)) {
      appendArchivedMessage(parent, message);
    }
  };

  drawPage();
}

function loadStoredArchive(id) {
  const volatile = volatileHistoryArchives.find((archive) => archive.id === id);
  if (volatile) return volatile;
  try {
    const archive = JSON.parse(localStorage.getItem(historyItemKey(id)) || 'null');
    return archive && Array.isArray(archive.messages) ? archive : null;
  } catch (_) {
    return null;
  }
}

function createHistorySession(entry, loadArchive) {
  const session = document.createElement('details');
  session.className = 'history-session';
  const summary = document.createElement('summary');
  const date = document.createElement('strong');
  date.textContent = new Date(entry.startedAt).toLocaleString();
  const stats = document.createElement('span');
  stats.textContent = `${formatTime(entry.duration)} / ${entry.messageCount} messages`;
  summary.append(date, stats);
  const messages = document.createElement('div');
  messages.className = 'history-messages';
  session.append(summary, messages);

  session.addEventListener('toggle', () => {
    if (!session.open) {
      messages.textContent = '';
      return;
    }
    for (const other of historyListEl.children) {
      if (other !== session && other instanceof HTMLDetailsElement && other.open) {
        other.open = false;
        const otherMessages = other.querySelector('.history-messages');
        if (otherMessages) otherMessages.textContent = '';
      }
    }
    const archive = loadArchive();
    if (archive) {
      renderArchivePage(messages, archive);
    } else {
      const empty = document.createElement('div');
      empty.className = 'history-empty compact';
      empty.textContent = 'This conversation is unavailable';
      messages.append(empty);
    }
  });
  return session;
}

function renderHistoryArchives() {
  historyListEl.textContent = '';
  updateHistoryControls();
  const entries = historyEntries();

  for (const entry of entries) {
    historyListEl.append(createHistorySession(entry, () => loadStoredArchive(entry.id)));
  }

  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.textContent = 'No archived conversations';
    historyListEl.append(empty);
  }
}

function releaseHistoryView() {
  historyListEl.textContent = '';
  updateHistoryControls();
}

function closeHistoryDialog() {
  if (typeof historyDialog.close === 'function') historyDialog.close();
  else {
    historyDialog.removeAttribute('open');
    releaseHistoryView();
  }
}

function resetConversationView() {
  clearTranscript();
  resetRuntimeSignals({ clearEvents: true, resetClock: true });
  playbackGeneration = 0;
  startSessionTimer();
}

function createMessage(role, start, end, pending = false) {
  revealTranscript();
  const root = document.createElement('article');
  root.className = `message ${role}${pending ? ' pending' : ''}`;
  root.dataset.start = String(start);

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  const label = document.createElement('span');
  label.textContent = role === 'user' ? 'User' : 'Omni-Interaction-Agent';
  const timing = document.createElement('time');
  timing.textContent = `${formatTime(start, true)} - ${formatTime(end, true)}`;
  meta.append(label, timing);

  const text = document.createElement('p');
  text.className = 'message-text';
  root.append(meta, text);
  const nextMessage = Array.from(transcriptEl.querySelectorAll('.message')).find(
    (item) => Number(item.dataset.start) > start
  );
  transcriptEl.insertBefore(root, nextMessage || null);
  return { root, text, timing, value: '', start };
}

function renderUserTranscript(result, defaultStart, defaultEnd) {
  const segments = result.segments || [];
  const start = segments.length ? segments[0].start : defaultStart;
  const end = segments.length ? segments[segments.length - 1].end : defaultEnd;
  const message = createMessage('user', start, end);
  const words = result.words || [];
  if (words.length) {
    for (const word of words) {
      const span = document.createElement('span');
      span.className = 'word-timing';
      span.textContent = word.text;
      span.title = `${formatTime(word.start, true)} - ${formatTime(word.end, true)}`;
      message.text.append(span);
    }
  } else {
    message.text.textContent = result.text || '';
  }
  liveCaptionEl.textContent = result.text || '--';
  pruneLiveTranscript();
  scheduleTranscriptScroll();
}

function updateAssistantMessage(event) {
  const current = Number(event.current_time) || sessionSeconds();
  let created = false;
  if (!assistantMessage) {
    assistantMessage = createMessage('assistant', Math.max(0, current - 1), current);
    created = true;
  }
  assistantMessage.value += event.text || '';
  assistantMessage.text.textContent = assistantMessage.value;
  assistantMessage.timing.textContent =
    `${formatTime(assistantMessage.start, true)} - ${formatTime(current, true)}`;
  if (created) pruneLiveTranscript();
  scheduleTranscriptScroll();
  if (event.end_of_turn || event.interrupted) assistantMessage = null;
}

function playbackQueueLead() {
  if (!playbackContext || !nextPlayTime) return 0;
  return Math.max(0, nextPlayTime - playbackContext.currentTime);
}

function floatToPcm16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function downsampleToPcm16(input, inputRate) {
  if (inputRate === INPUT_RATE) return floatToPcm16(input).buffer;
  const ratio = inputRate / INPUT_RATE;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  let inputOffset = 0;
  for (let i = 0; i < length; i += 1) {
    const nextOffset = Math.floor((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = inputOffset; j < nextOffset && j < input.length; j += 1) {
      sum += input[j];
      count += 1;
    }
    output[i] = count ? sum / count : 0;
    inputOffset = nextOffset;
  }
  return floatToPcm16(output).buffer;
}

function rms(input) {
  let sum = 0;
  for (let i = 0; i < input.length; i += 1) sum += input[i] * input[i];
  return Math.sqrt(sum / Math.max(1, input.length));
}

function updateWaveform(level) {
  waveformLevels.push(Math.min(1, level * 16));
  waveformLevels.shift();
}

function drawWaveform() {
  const rect = waveformEl.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * scale));
  const height = Math.max(1, Math.round(rect.height * scale));
  if (waveformEl.width !== width || waveformEl.height !== height) {
    waveformEl.width = width;
    waveformEl.height = height;
  }
  const context = waveformEl.getContext('2d');
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#242a2c';
  context.lineWidth = scale;
  context.beginPath();
  context.moveTo(0, height / 2);
  context.lineTo(width, height / 2);
  context.stroke();

  const gap = width / waveformLevels.length;
  context.strokeStyle = '#55c2a4';
  context.lineWidth = Math.max(1, gap * 0.42);
  context.beginPath();
  waveformLevels.forEach((level, index) => {
    const x = gap * index + gap / 2;
    const amplitude = Math.max(scale, level * height * 0.44);
    context.moveTo(x, height / 2 - amplitude);
    context.lineTo(x, height / 2 + amplitude);
  });
  context.stroke();
  window.requestAnimationFrame(drawWaveform);
}

function ensurePlaybackContext() {
  if (!playbackContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    playbackContext = new AudioContextClass();
  }
  if (playbackContext.state === 'suspended') void playbackContext.resume();
  return playbackContext;
}

function playPcm16(data, generationId = playbackGeneration) {
  if (Number(generationId) !== playbackGeneration) return;
  if (data instanceof Blob) {
    void data.arrayBuffer()
      .then((value) => playPcm16(value, generationId))
      .catch(handleFatalError);
    return;
  }
  if (!data || !data.byteLength) return;
  const sourceBytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  const alignedBytes = sourceBytes.slice(0, sourceBytes.byteLength - (sourceBytes.byteLength % 2));
  if (!alignedBytes.byteLength) return;

  const context = ensurePlaybackContext();
  const pcm = new Int16Array(alignedBytes.buffer);
  const buffer = context.createBuffer(1, pcm.length, OUTPUT_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 32768;

  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  const now = context.currentTime;
  const queueRanDry = !nextPlayTime
    || nextPlayTime <= now + PLAYBACK_UNDERRUN_GUARD_SECONDS;
  const startAt = queueRanDry
    ? now + PLAYBACK_PREROLL_SECONDS
    : nextPlayTime;
  nextPlayTime = startAt + buffer.duration;
  playbackSources.add(source);
  outputAudioStateEl.textContent = 'Playing';
  source.onended = () => {
    playbackSources.delete(source);
    if (!playbackSources.size) outputAudioStateEl.textContent = 'Ready';
  };
  source.start(startAt);
}

function stopPlayback() {
  for (const source of playbackSources) {
    source.onended = null;
    try {
      source.stop();
    } catch (_) {
      // A source may have completed between iteration and stop().
    }
  }
  playbackSources.clear();
  nextPlayTime = 0;
  outputAudioStateEl.textContent = 'Ready';
}

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function updateContextState(metrics) {
  const context = metrics?.context_window;
  if (!context) return;
  const units = Number(context.unit_count) || 0;
  contextMaxUnits = Number(context.max_units) || contextMaxUnits;
  contextStateEl.textContent = `${units} / ${contextMaxUnits}`;
  const shifts = Number(context.sliding_events) || 0;
  const previous = Number(context.previous_tokens) || 0;
  contextStateEl.title = `${shifts} shifts, ${previous} previous-context tokens`;
}

function sendControlEvent(payload) {
  if (transport !== 'ws' || !ws || ws.readyState !== WebSocket.OPEN) {
    throw new Error('Control events require an active WebSocket session');
  }
  ws.send(JSON.stringify(payload));
}

async function submitToolResponse(content) {
  if (transport === 'ws' && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'tool.response', content }));
    return;
  }
  throw new Error('No active duplex session for tool response');
}

window.GanderDuplexTools = Object.freeze({ respond: submitToolResponse });

function submitFinalTurn(text, start, end) {
  if (taskProtocol !== 'task_tools_v1') return Promise.resolve(null);
  if (transport !== 'ws' || !ws || ws.readyState !== WebSocket.OPEN) {
    return Promise.reject(new Error('Task runtime requires an active WebSocket session'));
  }
  finalTurnCounter += 1;
  const turnId = `browser-${transcriptGeneration}-${finalTurnCounter}-${newBrowserClientId()}`;
  const payload = {
    type: 'turn.final',
    turn_id: turnId,
    text,
    start_ms: Math.round(start * 1000),
    end_ms: Math.round(end * 1000),
    timestamp_ms: Date.now(),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  };
  const accepted = new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      pendingFinalTurns.delete(turnId);
      reject(new Error('Runtime did not acknowledge the final transcript'));
    }, 5000);
    pendingFinalTurns.set(turnId, { resolve, reject, timeout });
  });
  ws.send(JSON.stringify(payload));
  return accepted;
}

async function checkRuntime() {
  const runtime = await requestJson('health', { timeoutMs: 5000 });
  window.GanderVideo?.applyRuntimeCapabilities(runtime.client_video);
  contextMaxUnits = Number(runtime.context_max_units) || contextMaxUnits;
  contextStateEl.textContent = `0 / ${contextMaxUnits}`;
  return runtime;
}

function handleServerEvent(message) {
  // Video owns its own control replies; app.js must not treat them as unknown.
  if (window.GanderVideo?.handleServerEvent(message)) return;
  if (message.type === 'ready') {
    playbackGeneration = Number(message.generation_id) || 0;
    pendingBinaryAudio = null;
    assistantAudioFloorUnit = 0;
    contextMaxUnits = Number(message.context_max_units) || contextMaxUnits;
    modelChunkMs = Number(message.chunk_ms) || modelChunkMs;
    taskProtocol = message.task_protocol || null;
    audioInputProtocol = message.audio_input?.protocol || null;
    contextStateEl.textContent = `0 / ${contextMaxUnits}`;
    window.GanderVideo?.applyCapabilities(message);
    updatePipelineTiming();
    setStatus('Listening', 'active');
    setModelState('Listening');
    setSessionState('active');
    addEvent('WebSocket ready');
    if (Number(message.pool_size) > 1) {
      addEvent(`Model capacity ${message.busy_slots} / ${message.pool_size}`);
    }
    return;
  }
  if (message.type === 'turn.final.accepted') {
    const pending = pendingFinalTurns.get(message.turn_id);
    if (pending) {
      window.clearTimeout(pending.timeout);
      pendingFinalTurns.delete(message.turn_id);
      pending.resolve(message);
    }
    addEvent('Final transcript bound to Runtime');
    return;
  }
  if (message.type === 'tool.response.queued') {
    addEvent('Tool response queued for the next media unit');
    return;
  }
  if (message.type === 'playback.cancel') {
    const cancelledAtGeneration = Number(message.generation_id) || 0;
    // The Talker pump and Thinker response use separate producer tasks. A
    // delayed cancellation from an older generation must not roll the browser
    // back after a newer text/audio generation has already arrived.
    if (cancelledAtGeneration && cancelledAtGeneration < playbackGeneration) return;
    playbackGeneration = cancelledAtGeneration || playbackGeneration + 1;
    if (pendingBinaryAudio) pendingBinaryAudio.accepted = false;
    assistantAudioFloorUnit = 0;
    stopPlayback();
    if (message.reason === 'model_interrupt') assistantMessage = null;
    addEvent(`Output cancelled: ${message.reason || 'interrupted'}`);
    return;
  }
  if (message.type === 'audio.chunk') {
    const generationId = Number(message.generation_id) || 0;
    const unitId = Number(message.unit_id) || 0;
    const accepted = sessionState !== 'resetting'
      && generationId === playbackGeneration
      && (!assistantAudioFloorUnit || !unitId || unitId >= assistantAudioFloorUnit);
    if (message.audio_pcm16_b64) {
      if (accepted) {
        playPcm16(decodeBase64(message.audio_pcm16_b64), generationId);
      }
    } else if (message.audio) {
      pendingBinaryAudio = { generationId, accepted };
    }
    return;
  }
  if (message.type === 'audio.done') {
    return;
  }
  if (message.type === 'tool.call') {
    closeAsrTurnAtModelBoundary();
    chunkCount = Math.max(chunkCount, Number(message.index) || chunkCount + 1);
    chunkCountEl.textContent = String(chunkCount);
    updatePipelineTiming();
    updateContextState(message.metrics);
    const names = (message.tool_calls || []).map((call) => call.name).filter(Boolean);
    setStatus('Working', 'active');
    setModelState('Tool');
    addEvent(`Tool call: ${names.join(', ') || 'unknown'}`);
    window.dispatchEvent(new CustomEvent('gander:tool-call', { detail: message }));
    return;
  }
  if (message.type === 'tool.error') {
    chunkCount = Math.max(chunkCount, Number(message.index) || chunkCount + 1);
    chunkCountEl.textContent = String(chunkCount);
    updateContextState(message.metrics);
    setStatus('Listening', 'active');
    setModelState('Tool error');
    addEvent(`Rejected tool output: ${message.tool_error || message.message || 'invalid call'}`);
    return;
  }
  if (message.type === 'chunk') {
    const generationId = Number(message.generation_id) || 0;
    if (generationId > playbackGeneration) playbackGeneration = generationId;
    if (message.interrupted) {
      stopPlayback();
      assistantMessage = null;
      addEvent(`Model interrupted output at chunk ${message.index}`);
    }
    chunkCount = Math.max(chunkCount, Number(message.index) || chunkCount + 1);
    chunkCountEl.textContent = String(chunkCount);
    updatePipelineTiming();
    updateContextState(message.metrics);
    if (!message.is_listen && message.text) {
      const startsAssistantTurn = !assistantMessage;
      if (startsAssistantTurn) {
        closeAsrTurnAtModelBoundary();
        // Detached Talker audio is keyed by output unit_id, not the global step index.
        assistantAudioFloorUnit = Number(message.unit_id) || 0;
        if (playbackQueueLead() > PLAYBACK_STALE_TURN_SECONDS) stopPlayback();
      }
      updateAssistantMessage(message);
    }
    if (message.is_listen) {
      setStatus('Listening', 'active');
      setModelState('Listening');
    } else {
      setStatus('Speaking', 'speaking');
      setModelState('Speaking');
    }
    if (message.audio_pcm16_b64) {
      playPcm16(decodeBase64(message.audio_pcm16_b64), playbackGeneration);
    } else if (message.audio) {
      pendingBinaryAudio = {
        generationId: playbackGeneration,
        accepted: true
      };
    }
    if (message.end_of_turn) {
      addEvent(`Model turn ended at chunk ${message.index}`);
    }
    return;
  }
  if (message.type === 'reset.done') {
    if (resetAckHandle !== null) window.clearTimeout(resetAckHandle);
    resetAckHandle = null;
    archiveCurrentConversation();
    resetConversationView();
    playbackGeneration = Number(message.generation_id) || 0;
    pendingBinaryAudio = null;
    assistantAudioFloorUnit = 0;
    capturePaused = false;
    setSessionState('active');
    setStatus('Listening', 'active');
    setModelState('Listening');
    addEvent('Model session reset');
    return;
  }
  if (message.type === 'turn.final.required') {
    closeAsrTurnAtModelBoundary();
    addEvent('Runtime is waiting for the final transcript');
    return;
  }
  if (message.type === 'delivery.failed') {
    setStatus('Listening', 'active');
    setModelState('Task error');
    addEvent(`Background result failed: ${message.message || message.reason || 'unknown error'}`);
    return;
  }
  if (message.type === 'task_status') {
    const count = Array.isArray(message.task?.tasks) ? message.task.tasks.length : 0;
    addEvent(`Background tasks: ${count}`);
    window.dispatchEvent(new CustomEvent('gander:task-status', { detail: message.task }));
    return;
  }
  if (message.type === 'memory.summary_needed') {
    addEvent('Runtime requested a memory summary');
    window.dispatchEvent(new CustomEvent('gander:memory-summary-needed', { detail: message }));
    return;
  }
  if (['memory.episode.done', 'break.done', 'clear_break.done', 'pong'].includes(message.type)) {
    return;
  }
  if (message.type === 'error') throw new Error(message.message || 'Server error');
}

async function requestJson(path, options = {}) {
  const { timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(fetchOptions.headers || {});
  if (browserClientId) headers.set('X-MiniCPM-Client', browserClientId);
  let response;
  try {
    response = await fetch(serviceUrl(path), {
      cache: 'no-store',
      ...fetchOptions,
      headers,
      signal: controller.signal
    });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    // A proxy error page is represented by the HTTP status below.
  }
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

function connectWebSocket() {
  return new Promise((resolve, reject) => {
    const socketUrl = new URL(serviceUrl('ws/duplex'));
    socketUrl.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socketUrl.searchParams.set('client_id', browserClientId);
    const candidate = new WebSocket(socketUrl);
    connectingWs = candidate;
    candidate.binaryType = 'arraybuffer';
    let ready = false;
    let finished = false;

    const timer = window.setTimeout(() => {
      if (ready || finished) return;
      finished = true;
      candidate.onclose = null;
      candidate.close();
      if (connectingWs === candidate) connectingWs = null;
      if (ws === candidate) {
        ws = null;
        transport = null;
      }
      reject(new Error('WebSocket ready handshake timed out'));
    }, WS_CONNECT_TIMEOUT_MS);

    candidate.onopen = () => {
      if (finished) return;
      if (connectingWs === candidate) connectingWs = null;
      ws = candidate;
      transport = 'ws';
      transportValueEl.textContent = 'WebSocket';
      setStatus('Preparing model');
    };
    candidate.onmessage = (event) => {
      if (ws !== candidate || finished) return;
      try {
        if (typeof event.data !== 'string') {
          const pending = pendingBinaryAudio;
          pendingBinaryAudio = null;
          if (pending?.accepted) {
            playPcm16(
              event.data,
              pending.generationId
            );
          }
          return;
        }
        const message = JSON.parse(event.data);
        if (message.type === 'session.done') {
          if (stopAckHandle !== null) window.clearTimeout(stopAckHandle);
          stopAckHandle = null;
          finished = true;
          void cleanupAfterClose(false, candidate, 'Idle');
          return;
        }
        handleServerEvent(message);
        if (message.type === 'ready' && !ready) {
          ready = true;
          window.clearTimeout(timer);
          resolve(candidate);
        }
      } catch (error) {
        finished = true;
        if (!ready) {
          window.clearTimeout(timer);
          candidate.onclose = null;
          candidate.close();
          if (connectingWs === candidate) connectingWs = null;
          if (ws === candidate) {
            ws = null;
            transport = null;
          }
          reject(error);
          return;
        }
        void handleFatalError(error, candidate);
      }
    };
    candidate.onerror = () => {
      if (!ready && !finished) {
        finished = true;
        window.clearTimeout(timer);
        candidate.onclose = null;
        candidate.close();
        if (connectingWs === candidate) connectingWs = null;
        if (ws === candidate) {
          ws = null;
          transport = null;
        }
        reject(new Error('WebSocket unavailable'));
      }
    };
    candidate.onclose = () => {
      window.clearTimeout(timer);
      if (connectingWs === candidate) connectingWs = null;
      if (!ready && !finished) {
        finished = true;
        if (ws === candidate) {
          ws = null;
          transport = null;
        }
        reject(new Error('WebSocket closed before the ready handshake'));
        return;
      }
      if (ws !== candidate) return;
      finished = true;
      void cleanupAfterClose(false, candidate, 'Idle');
    };
  });
}

async function checkAsr() {
  asrStatusEl.textContent = 'Checking';
  asrStatusEl.dataset.state = 'idle';
  try {
    await requestJson('api/asr/health', { timeoutMs: 5000 });
    asrAvailable = true;
    asrStatusEl.textContent = 'Online';
    asrStatusEl.dataset.state = 'online';
  } catch (_) {
    asrAvailable = false;
    asrStatusEl.textContent = 'Offline';
    asrStatusEl.dataset.state = 'offline';
  }
}

function appendUtteranceAudio(rawPcm, voiceDetected, frameStartSample, frameRms) {
  if (!asrAvailable) return;
  const bytes = new Uint8Array(rawPcm);
  const frameSamples = bytes.byteLength / 2;
  const frameEndSample = frameStartSample + frameSamples;

  if (!asrUtterance) {
    if (!voiceDetected) {
      asrSpeechCandidate = null;
      return;
    }
    if (!asrSpeechCandidate) {
      asrSpeechCandidate = {
        startSample: frameStartSample,
        sampleCount: 0,
        voiceSamples: 0,
        maxRms: 0,
        asrGeneration,
        transcriptGeneration,
        parts: []
      };
    }
    asrSpeechCandidate.parts.push(bytes);
    asrSpeechCandidate.sampleCount += frameSamples;
    asrSpeechCandidate.voiceSamples += frameSamples;
    asrSpeechCandidate.maxRms = Math.max(asrSpeechCandidate.maxRms, frameRms);
    if (asrSpeechCandidate.voiceSamples >= ASR_START_VOICE_SAMPLES) {
      asrUtterance = {
        ...asrSpeechCandidate,
        lastVoiceEndSample: frameEndSample
      };
      asrSpeechCandidate = null;
    }
    return;
  }

  asrUtterance.parts.push(bytes);
  asrUtterance.sampleCount += frameSamples;
  if (voiceDetected) {
    asrUtterance.lastVoiceEndSample = frameEndSample;
    asrUtterance.voiceSamples += frameSamples;
    asrUtterance.maxRms = Math.max(asrUtterance.maxRms, frameRms);
  }

  const ended = !voiceDetected
    && frameEndSample - asrUtterance.lastVoiceEndSample >= ASR_END_SILENCE_SAMPLES;
  const rollover = asrUtterance.sampleCount >= ASR_ROLLING_SEGMENT_SAMPLES;
  if (ended || rollover) finishAsrUtterance();
}

function observeAsrNoise(inputRms) {
  if (!Number.isFinite(inputRms) || inputRms < 0) return;
  asrNoiseSamples.push(Math.min(inputRms, ASR_NOISE_SAMPLE_CEILING));
  if (asrNoiseSamples.length > ASR_NOISE_WINDOW_FRAMES) asrNoiseSamples.shift();
  const ordered = [...asrNoiseSamples].sort((left, right) => left - right);
  const index = Math.floor((ordered.length - 1) * ASR_NOISE_QUANTILE);
  asrNoiseFloorRms = ordered[Math.max(0, index)] || 0;
}

function asrVoiceThreshold(gateDb) {
  const fixed = Math.pow(10, Math.max(gateDb, ASR_MIN_GATE_DB) / 20);
  const multiplier = asrUtterance
    ? ASR_ACTIVE_NOISE_MULTIPLIER
    : ASR_START_NOISE_MULTIPLIER;
  return Math.max(fixed, asrNoiseFloorRms * multiplier);
}

function closeAsrTurnAtModelBoundary() {
  if (asrUtterance) finishAsrUtterance();
  else asrSpeechCandidate = null;
}

function finishAsrUtterance() {
  const utterance = asrUtterance;
  asrUtterance = null;
  asrSpeechCandidate = null;
  if (!utterance || utterance.voiceSamples < ASR_MIN_VOICE_SAMPLES) return;

  const pcm = new Uint8Array(utterance.sampleCount * 2);
  let offset = 0;
  for (const part of utterance.parts) {
    pcm.set(part, offset);
    offset += part.byteLength;
  }
  const start = utterance.startSample / INPUT_RATE;
  const end = start + utterance.sampleCount / INPUT_RATE;
  const utteranceAsrGeneration = utterance.asrGeneration ?? asrGeneration;
  const utteranceTranscriptGeneration =
    utterance.transcriptGeneration ?? transcriptGeneration;
  liveCaptionEl.textContent = `Transcribing ${formatTime(start, true)} - ${formatTime(end, true)}`;

  asrQueue = asrQueue.then(async () => {
    try {
      const result = await requestJson(
        `api/asr/transcribe?start_ms=${Math.round(start * 1000)}&language=${encodeURIComponent(languageSelect.value)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: pcm,
          timeoutMs: 120000
        }
      );
      if (utteranceAsrGeneration !== asrGeneration) return;
      if (result.text) {
        asrTurnCount += 1;
        asrTurnCountEl.textContent = String(asrTurnCount);
        if (utteranceTranscriptGeneration === transcriptGeneration) {
          renderUserTranscript(result, start, end);
        } else {
          liveCaptionEl.textContent = '--';
        }
        addEvent(`ASR turn ${asrTurnCount} completed`);
        try {
          await submitFinalTurn(result.text, start, end);
        } catch (error) {
          if (utteranceAsrGeneration !== asrGeneration) return;
          addEvent(`Runtime turn binding failed: ${error.message || error}`);
        }
        if (utteranceAsrGeneration !== asrGeneration) return;
      } else {
        liveCaptionEl.textContent = '--';
      }
    } catch (error) {
      if (utteranceAsrGeneration !== asrGeneration) return;
      asrAvailable = false;
      asrStatusEl.textContent = 'Offline';
      asrStatusEl.dataset.state = 'offline';
      addEvent(`ASR error: ${error.message || error}`);
      liveCaptionEl.textContent = '--';
    }
  });
}

function abandonPendingAsr() {
  asrGeneration += 1;
  transcriptGeneration += 1;
  asrUtterance = null;
  asrSpeechCandidate = null;
  asrQueue = Promise.resolve();
  liveCaptionEl.textContent = '--';
}

function rejectPendingFinalTurns(reason) {
  for (const pending of pendingFinalTurns.values()) {
    window.clearTimeout(pending.timeout);
    pending.reject(new Error(reason));
  }
  pendingFinalTurns.clear();
}

function processMicFrame(input, capturedAtSec) {
  const inputRms = rms(input);
  const inputDb = 20 * Math.log10(Math.max(inputRms, 1e-6));
  inputLevelEl.textContent = `${Math.round(inputDb)} dB`;
  meterEl.style.width = `${Math.min(100, inputRms * 500)}%`;
  updateWaveform(inputRms);
  if (capturePaused || !micContext) {
    if (micContext) observeAsrNoise(inputRms);
    return;
  }

  const rawPcm = downsampleToPcm16(input, micContext.sampleRate);
  const frameStartSample = capturedSamples;
  capturedSamples += rawPcm.byteLength / 2;
  updatePipelineTiming();
  const gateDb = Number(gateSlider.value);
  const threshold = Math.pow(10, gateDb / 20);
  const asrThreshold = asrVoiceThreshold(gateDb);
  const now = performance.now();
  const modelVoiceDetected = inputRms >= threshold;
  const asrVoiceDetected = inputRms >= asrThreshold;
  if (modelVoiceDetected) gateOpenUntil = now + NOISE_GATE_HOLD_MS;
  const modelPcm = now <= gateOpenUntil ? rawPcm : new ArrayBuffer(rawPcm.byteLength);

  appendUtteranceAudio(rawPcm, asrVoiceDetected, frameStartSample, inputRms);
  if (!asrVoiceDetected && !asrUtterance && !asrSpeechCandidate) {
    observeAsrNoise(inputRms);
  }
  if (transport === 'ws' && ws && ws.readyState === WebSocket.OPEN) {
    if (audioInputProtocol === 'metadata-json+pcm16-binary-v1') {
      audioFrameSequence += 1;
      const capturedAtMs = Date.now()
        - (micContext.currentTime - capturedAtSec) * 1000;
      ws.send(JSON.stringify({
        type: 'audio.frame',
        sequence: audioFrameSequence,
        start_sample: frameStartSample,
        sample_count: modelPcm.byteLength / 2,
        captured_at_ms: Math.round(capturedAtMs)
      }));
    }
    ws.send(modelPcm);
  }
}

async function createMicProcessor(context) {
  if (!context.audioWorklet || typeof window.AudioWorkletNode !== 'function') {
    throw new Error('This browser does not support AudioWorklet microphone capture');
  }
  await context.audioWorklet.addModule(serviceUrl('assets/mic-worklet.js'));
  const processor = new AudioWorkletNode(context, 'minicpm-mic-capture', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { frameSize: BUFFER_SIZE }
  });
  processor.port.onmessage = (event) => {
    processMicFrame(event.data.samples, event.data.capturedAtSec);
  };
  addEvent('AudioWorklet microphone ready');
  return processor;
}

async function startMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Microphone requires HTTPS or a localhost page');
  }
  asrNoiseSamples = [];
  asrNoiseFloorRms = 0;
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  micContext = new AudioContextClass();
  micSource = micContext.createMediaStreamSource(micStream);
  micProcessor = await createMicProcessor(micContext);
  silenceGain = micContext.createGain();
  silenceGain.gain.value = 0;
  micSource.connect(micProcessor);
  micProcessor.connect(silenceGain);
  silenceGain.connect(micContext.destination);
}

async function stopMic() {
  const processor = micProcessor;
  const source = micSource;
  const gain = silenceGain;
  const stream = micStream;
  const context = micContext;
  micProcessor = null;
  micSource = null;
  silenceGain = null;
  micStream = null;
  micContext = null;

  if (processor?.port) {
    processor.port.onmessage = null;
    try { processor.port.close(); } catch (_) {}
  }
  for (const node of [processor, source, gain]) {
    try { node?.disconnect(); } catch (_) {}
  }
  if (stream) {
    for (const track of stream.getTracks()) {
      try { track.stop(); } catch (_) {}
    }
  }
  if (context && context.state !== 'closed') {
    try { await context.close(); } catch (_) {}
  }
  meterEl.style.width = '0%';
  inputLevelEl.textContent = '-120 dB';
  waveformLevels.fill(0);
}

async function start() {
  if (sessionState !== 'idle') return;
  // Unlock output audio before any await can consume the click's user gesture.
  // This is especially important when screen sharing opens a permission prompt.
  ensurePlaybackContext();
  const hasConversation = snapshotTranscript().length > 0;
  archiveCurrentConversation();
  if (hasConversation) clearTranscript();
  clearControlTimers();
  resetRuntimeSignals({ clearEvents: true, resetClock: true });
  setSessionState('starting');
  setStatus('Connecting');
  setModelState('Starting');
  transportValueEl.textContent = '--';
  playbackGeneration = 0;
  stopping = false;
  capturePaused = true;
  void checkAsr();

  try {
    // Screen share must be requested while this click is still the active user
    // gesture; the handshake below would outlive it.
    await window.GanderVideo?.preacquireIfNeeded();
    if (sessionState !== 'starting' || stopping) {
      await window.GanderVideo?.stop();
      return;
    }
    await checkRuntime();
    if (sessionState !== 'starting') return;
    setStatus('Waiting for microphone');
    await startMic();
    if (sessionState !== 'starting') {
      await stopMic();
      await window.GanderVideo?.stop();
      return;
    }
    setStatus('Connecting');
    await connectWebSocket();
    if (sessionState !== 'active' && sessionState !== 'starting') return;
    startSessionTimer();
    setSessionState('active');
    await window.GanderVideo?.attach({ sendControl: sendControlEvent, addEvent });
    if (sessionState !== 'active' || stopping) {
      await window.GanderVideo?.stop();
      return;
    }
    capturePaused = false;
  } catch (error) {
    if (!['starting', 'active'].includes(sessionState)) return;
    await handleFatalError(error);
  }
}

async function stop() {
  if (!['starting', 'active'].includes(sessionState) || stopping) return;
  stopping = true;
  capturePaused = true;
  setSessionState('stopping');
  setStatus('Finishing');
  setModelState('Finishing');
  abandonPendingAsr();
  void stopMic();
  void window.GanderVideo?.stop();

  if (transport === 'ws' && ws && ws.readyState === WebSocket.OPEN) {
    const activeSocket = ws;
    activeSocket.send(JSON.stringify({ type: 'stop' }));
    stopAckHandle = window.setTimeout(() => {
      stopAckHandle = null;
      if (sessionState === 'stopping' && ws === activeSocket) {
        void cleanupAfterClose(true, activeSocket, 'Idle');
      }
    }, STOP_ACK_TIMEOUT_MS);
    return;
  }
  await cleanupAfterClose(true, ws, 'Idle');
}

async function reset() {
  if (sessionState !== 'active' || capturePaused || stopping) return;
  capturePaused = true;
  setSessionState('resetting');
  setStatus('Resetting');
  setModelState('Resetting');
  abandonPendingAsr();
  stopPlayback();
  pendingBinaryAudio = null;
  rejectPendingFinalTurns('Duplex session reset before transcript acknowledgement');
  try {
    if (transport === 'ws' && ws && ws.readyState === WebSocket.OPEN) {
      const activeSocket = ws;
      activeSocket.send(JSON.stringify({ type: 'reset' }));
      resetAckHandle = window.setTimeout(() => {
        resetAckHandle = null;
        if (sessionState === 'resetting' && ws === activeSocket) {
          void handleFatalError(new Error('Runtime reset timed out'), activeSocket);
        }
      }, RESET_ACK_TIMEOUT_MS);
      return;
    }
    throw new Error('No active duplex session to reset');
  } catch (error) {
    await handleFatalError(error);
  }
}

async function waitForRuntimeRelease(timeoutMs = 5000) {
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    try {
      const health = await requestJson('health', { timeoutMs: 1000 });
      if (!health.busy) return;
    } catch (_) {
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 150));
  }
}

async function cleanupAfterClose(forceClose, expectedWs = null, finalStatus = 'Idle') {
  if (expectedWs && ws && ws !== expectedWs) return;
  if (cleanupPromise) return cleanupPromise;

  cleanupPromise = (async () => {
    const oldWs = expectedWs || ws;
    const oldConnectingWs = connectingWs;
    clearControlTimers();
    transport = null;
    taskProtocol = null;
    audioInputProtocol = null;
    if (!expectedWs || ws === expectedWs) ws = null;
    connectingWs = null;
    capturePaused = true;
    setSessionState('stopping');
    abandonPendingAsr();
    rejectPendingFinalTurns('Duplex session closed before transcript acknowledgement');
    stopPlayback();

    for (const socket of [oldWs, oldConnectingWs]) {
      if (socket && socket.readyState < WebSocket.CLOSING) {
        try { socket.close(); } catch (_) {}
      }
    }
    await stopMic();
    await window.GanderVideo?.stop();
    if (oldWs) await waitForRuntimeRelease();

    archiveCurrentConversation();
    resetRuntimeSignals({ clearEvents: true, resetClock: true });
    transportValueEl.textContent = '--';
    setModelState('Idle');
    setStatus(finalStatus, finalStatus === 'Error' ? 'error' : 'idle');
    capturePaused = false;
    stopping = false;
    setSessionState('idle');
  })();

  try {
    await cleanupPromise;
  } finally {
    cleanupPromise = null;
  }
}

async function handleFatalError(error, expectedWs = null) {
  const message = error && error.message ? error.message : String(error);
  await cleanupAfterClose(true, expectedWs, 'Error');
  setStatus('Error', 'error');
  setModelState('Error');
  addEvent(message);
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
resetBtn.addEventListener('click', reset);
clearBtn.addEventListener('click', clearTranscript);
historyBtn.addEventListener('click', () => {
  renderHistoryArchives();
  if (typeof historyDialog.showModal === 'function') historyDialog.showModal();
  else historyDialog.setAttribute('open', '');
});
closeHistoryBtn.addEventListener('click', closeHistoryDialog);
historyDialog.addEventListener('close', releaseHistoryView);
historyDialog.addEventListener('click', (event) => {
  if (event.target === historyDialog) closeHistoryDialog();
});
clearHistoryBtn.addEventListener('click', () => {
  if (historyBtn.disabled || !window.confirm('Clear all archived conversations?')) return;
  clearStoredHistory();
  renderHistoryArchives();
});
gateSlider.addEventListener('input', () => {
  gateValue.textContent = `${gateSlider.value} dB`;
  localStorage.setItem('minicpm-noise-gate-db', gateSlider.value);
});
languageSelect.addEventListener('change', () => {
  localStorage.setItem('minicpm-asr-language', languageSelect.value);
});

const startupUrl = new URL(window.location.href);
if (startupUrl.searchParams.get('clear_history') === '1') {
  clearStoredHistory();
  startupUrl.searchParams.delete('clear_history');
  window.history.replaceState(
    null,
    '',
    `${startupUrl.pathname}${startupUrl.search}${startupUrl.hash}`
  );
}
historyIndex = loadHistoryIndex();
document.querySelectorAll('button[data-help]').forEach(enhanceButtonHelp);
const storedGate = localStorage.getItem('minicpm-noise-gate-db');
const storedGateVersion = localStorage.getItem('minicpm-noise-gate-version');
if (storedGate !== null && storedGateVersion === GATE_SETTINGS_VERSION) {
  gateSlider.value = storedGate;
}
localStorage.setItem('minicpm-noise-gate-version', GATE_SETTINGS_VERSION);
gateValue.textContent = `${gateSlider.value} dB`;
languageSelect.value = localStorage.getItem('minicpm-asr-language') || '';
updateHistoryControls();
drawWaveform();
void checkRuntime();
void checkAsr();
