'use strict';

// Camera and screen capture for duplex inference. Frames follow the training
// contract: capture-time aligned, 1 Hz, JPEG, and fitted within 448 px.

(function () {
  const FIT = 448;
  const JPEG_QUALITY = 0.7;
  const PREVIEW_HZ = 5;
  const MODE_ACK_TIMEOUT_MS = 5000;
  const FRAME_ACK_TIMEOUT_MS = 5000;
  const STORAGE_KEY = 'minicpm-video-source';

  const control = document.getElementById('videoModeControl');
  const select = document.getElementById('videoModeSelect');
  const previewWrap = document.getElementById('videoPreviewWrap');
  const canvas = document.getElementById('videoPreview');
  const note = document.getElementById('videoNote');
  const source = document.getElementById('videoSource');
  const stateEl = document.getElementById('videoState');

  if (!control || !select || !canvas || !source) return;

  const context = canvas.getContext('2d', { alpha: false });

  let host = null;
  let runtimeCapabilities = null;
  let capabilities = null;
  let screenConfig = null;
  let stream = null;
  let track = null;
  let socket = null;
  let connectingSocket = null;
  let pacer = null;
  let tick = 0;
  let frameSeq = 0;
  let framesSent = 0;
  let sending = false;
  let lastFrameBytes = 0;
  let activeSource = null;
  let pendingMode = null;
  let runtimeVideoEnabled = false;
  const pendingFrameAcks = new Map();
  let preacquired = null;
  let sessionState = 'idle';
  let lifecycleGeneration = 0;

  class VideoSetupCancelledError extends Error {
    constructor() {
      super('Video setup cancelled');
      this.name = 'VideoSetupCancelledError';
    }
  }

  function requireActiveLifecycle(generation) {
    if (generation !== lifecycleGeneration || sessionState !== 'active') {
      throw new VideoSetupCancelledError();
    }
  }

  function setState(text) {
    if (stateEl) stateEl.textContent = text;
  }

  function report(message) {
    if (host && host.addEvent) host.addEvent(message);
  }

  function frameRate() {
    const rate = Number(
      (screenConfig && screenConfig.recommended_frame_rate) ||
        (capabilities && capabilities.recommended_frame_rate) ||
        (runtimeCapabilities && runtimeCapabilities.recommended_frame_rate)
    );
    return rate > 0 ? rate : 1;
  }

  function describe() {
    if (!activeSource) return 'Off';
    return `${activeSource} ${frameRate()} Hz`;
  }

  function updateNote() {
    if (!note) return;
    if (!activeSource || !canvas.width) {
      note.textContent = '--';
      return;
    }
    const lines = [
      `${canvas.width}x${canvas.height} · ${frameRate()} Hz → frontbrain · ${framesSent} sent`
    ];
    if (activeSource === 'screen') {
      lines.push('前脑只看到 448px 缩略图，读不了屏幕上的小字。');
    }
    note.textContent = lines.join('\n');
  }

  function showPreview(visible) {
    if (previewWrap) previewWrap.hidden = !visible;
    if (!visible) {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  // ---- capture ----------------------------------------------------------

  function screenShareSupported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  }

  async function acquireStream(kind) {
    if (!navigator.mediaDevices) {
      throw new Error('Video capture requires HTTPS or a localhost page');
    }
    if (kind === 'screen') {
      if (!screenShareSupported()) throw new Error('This browser cannot share a screen');
      return navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 5 } },
        audio: false
      });
    }
    return navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
  }

  function attachStream(next) {
    stream = next;
    track = next.getVideoTracks()[0] || null;
    source.srcObject = next;
    if (track) track.addEventListener('ended', onTrackEnded, { once: true });
    return source.play().catch(() => {});
  }

  function releaseStream() {
    if (track) track.removeEventListener('ended', onTrackEnded);
    if (stream) {
      for (const item of stream.getTracks()) {
        try { item.stop(); } catch (_) {}
      }
    }
    stream = null;
    track = null;
    try { source.srcObject = null; } catch (_) {}
  }

  // ---- encode and send --------------------------------------------------

  function drawFrame() {
    const width = source.videoWidth;
    const height = source.videoHeight;
    if (!width || !height) return false;
    const scale = Math.min(FIT / width, FIT / height, 1);
    const targetWidth = Math.max(1, Math.round(width * scale));
    const targetHeight = Math.max(1, Math.round(height * scale));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    context.drawImage(source, 0, 0, targetWidth, targetHeight);
    return true;
  }

  function encode() {
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
  }

  function waitForFrameAck(frameId) {
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        pendingFrameAcks.delete(frameId);
        reject(new Error('Runtime did not acknowledge the first video frame'));
      }, FRAME_ACK_TIMEOUT_MS);
      pendingFrameAcks.set(frameId, { resolve, reject, timeout });
    });
  }

  function settleFrameAck(frameId, error = null) {
    const pending = pendingFrameAcks.get(frameId);
    if (!pending) return;
    pendingFrameAcks.delete(frameId);
    window.clearTimeout(pending.timeout);
    if (error) pending.reject(error);
    else pending.resolve();
  }

  function rejectPendingFrameAcks(error) {
    for (const frameId of Array.from(pendingFrameAcks.keys())) {
      settleFrameAck(frameId, error);
    }
  }

  async function sendFrame({ awaitAcceptance = false } = {}) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    // Keep at most one frame in flight.
    if (lastFrameBytes && socket.bufferedAmount > lastFrameBytes * 2) return false;
    const capturedAtMs = Date.now();
    const blob = await encode();
    if (!blob || !socket || socket.readyState !== WebSocket.OPEN) return false;
    const bytes = await blob.arrayBuffer();
    frameSeq += 1;
    const frameId = `browser_${frameSeq}`;
    const accepted = awaitAcceptance ? waitForFrameAck(frameId) : null;
    try {
      socket.send(JSON.stringify({
      type: 'screen.frame',
      frame_id: frameId,
      // Runtime recency checks use wall-clock timestamps.
      captured_at_ms: capturedAtMs,
      encoding: 'jpeg',
      video_source: activeSource,
      metadata: { width: canvas.width, height: canvas.height }
      }));
      socket.send(bytes);
    } catch (error) {
      settleFrameAck(frameId, error);
      throw error;
    }
    lastFrameBytes = bytes.byteLength;
    framesSent += 1;
    if (accepted) await accepted;
    return true;
  }

  async function sendInitialFrame(generation) {
    const deadline = Date.now() + FRAME_ACK_TIMEOUT_MS;
    while (!drawFrame()) {
      requireActiveLifecycle(generation);
      if (Date.now() >= deadline) {
        throw new Error('Video source did not produce an initial frame');
      }
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }
    requireActiveLifecycle(generation);
    const sent = await sendFrame({ awaitAcceptance: true });
    requireActiveLifecycle(generation);
    if (!sent) throw new Error('Initial video frame could not be sent');
  }

  function startPacer() {
    stopPacer();
    const sendEvery = Math.max(1, Math.round(PREVIEW_HZ / frameRate()));
    const step = async () => {
      pacer = null;
      if (!activeSource) return;
      const drawn = drawFrame();
      tick += 1;
      if (drawn && tick % sendEvery === 0 && !sending) {
        sending = true;
        try {
          await sendFrame();
        } catch (error) {
          report(`Video frame failed: ${error.message || error}`);
        } finally {
          sending = false;
        }
      }
      // Refresh the displayed encoded size on every draw.
      if (drawn) updateNote();
      if (activeSource) pacer = window.setTimeout(step, 1000 / PREVIEW_HZ);
    };
    // enable() sends the initial frame; continue at 1 Hz.
    pacer = window.setTimeout(step, 1000 / PREVIEW_HZ);
  }

  function stopPacer() {
    if (pacer !== null) window.clearTimeout(pacer);
    pacer = null;
    tick = 0;
  }

  // ---- screen channel ---------------------------------------------------

  function screenSocketUrl(config) {
    const url = new URL(config.path.replace(/^\//, ''), baseUrl());
    url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('session_id', config.session_id);
    url.searchParams.set('token', config.token);
    return url;
  }

  function baseUrl() {
    const base = new URL(window.location.href);
    base.hash = '';
    base.search = '';
    if (!base.pathname.endsWith('/')) base.pathname += '/';
    return base;
  }

  function openScreenSocket(config, generation) {
    return new Promise((resolve, reject) => {
      try {
        requireActiveLifecycle(generation);
      } catch (error) {
        reject(error);
        return;
      }
      const candidate = new WebSocket(screenSocketUrl(config));
      connectingSocket = candidate;
      candidate.binaryType = 'arraybuffer';
      let settled = false;
      const timer = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        if (connectingSocket === candidate) connectingSocket = null;
        try { candidate.close(); } catch (_) {}
        reject(new Error('Screen channel did not become ready'));
      }, MODE_ACK_TIMEOUT_MS);

      // Screen-channel errors remain local to video controls.
      candidate.onmessage = (event) => {
        if (typeof event.data !== 'string') return;
        let message;
        try { message = JSON.parse(event.data); } catch (_) { return; }
        if (message.type === 'screen.ready') {
          if (settled) return;
          try {
            requireActiveLifecycle(generation);
          } catch (error) {
            settled = true;
            window.clearTimeout(timer);
            if (connectingSocket === candidate) connectingSocket = null;
            try { candidate.close(); } catch (_) {}
            reject(error);
            return;
          }
          settled = true;
          window.clearTimeout(timer);
          if (connectingSocket === candidate) connectingSocket = null;
          socket = candidate;
          resolve(candidate);
          return;
        }
        if (message.type === 'screen.frame.accepted') {
          settleFrameAck(message.frame_id);
          return;
        }
        if (message.type === 'screen.frame.dropped') {
          settleFrameAck(
            message.frame_id,
            new Error(`First video frame dropped: ${message.reason || 'unknown reason'}`)
          );
          report(`Video frame dropped: ${message.reason || 'unknown reason'}`);
          return;
        }
        if (message.type === 'error') {
          rejectPendingFrameAcks(
            new Error(`Video channel error: ${message.message || 'unknown'}`)
          );
          report(`Video channel error: ${message.message || 'unknown'}`);
        }
      };
      candidate.onclose = () => {
        if (connectingSocket === candidate) connectingSocket = null;
        const wasActive = socket === candidate;
        const awaitingFrame = pendingFrameAcks.size > 0;
        if (wasActive) socket = null;
        rejectPendingFrameAcks(
          new Error('Screen channel closed before frame acknowledgement')
        );
        if (settled) {
          if (wasActive && !awaitingFrame && activeSource) {
            report('Video channel closed unexpectedly');
            revertSelect();
            void disable();
          }
          return;
        }
        settled = true;
        window.clearTimeout(timer);
        reject(new Error('Screen channel closed before it was ready'));
      };
      candidate.onerror = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        if (connectingSocket === candidate) connectingSocket = null;
        reject(new Error('Screen channel is unavailable'));
      };
    });
  }

  function closeScreenSocket() {
    const candidates = new Set([socket, connectingSocket]);
    socket = null;
    connectingSocket = null;
    lastFrameBytes = 0;
    rejectPendingFrameAcks(
      new Error('Screen channel closed before frame acknowledgement')
    );
    for (const candidate of candidates) {
      if (candidate && candidate.readyState < WebSocket.CLOSING) {
        try { candidate.close(); } catch (_) {}
      }
    }
  }

  // ---- mode negotiation -------------------------------------------------

  function cancelPendingMode(error) {
    const pending = pendingMode;
    if (!pending) return;
    pendingMode = null;
    window.clearTimeout(pending.timeout);
    pending.reject(error);
  }

  function requestMode(video, kind, generation = lifecycleGeneration) {
    if (!host || !host.sendControl) {
      return Promise.reject(new Error('Session is not connected'));
    }
    if (pendingMode) {
      return Promise.reject(new Error('A media-mode change is already in flight'));
    }
    const settled = new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        pendingMode = null;
        reject(new Error('Runtime did not acknowledge the media mode'));
      }, MODE_ACK_TIMEOUT_MS);
      pendingMode = { resolve, reject, timeout, generation };
    });
    const payload = { type: 'media.mode', video };
    if (video) payload.source = kind;
    try {
      host.sendControl(payload);
    } catch (error) {
      cancelPendingMode(error);
    }
    return settled;
  }

  async function enable(kind) {
    const generation = lifecycleGeneration;
    const acquired = preacquired && preacquired.kind === kind
      ? preacquired.stream
      : await acquireStream(kind);
    preacquired = null;
    try {
      requireActiveLifecycle(generation);
      const done = await requestMode(true, kind, generation);
      requireActiveLifecycle(generation);
      const config = done.screen;
      if (!config) throw new Error('Runtime did not offer a screen channel');
      await attachStream(acquired);
      requireActiveLifecycle(generation);
      await openScreenSocket(
        { ...config, session_id: done.session_id || sessionId() },
        generation
      );
      requireActiveLifecycle(generation);
      activeSource = kind;
      framesSent = 0;
      showPreview(true);
      await sendInitialFrame(generation);
      requireActiveLifecycle(generation);
      startPacer();
      setState(describe());
      updateNote();
      for (const warning of done.warnings || []) report(`Video caveat: ${warning}`);
      report(`Video enabled: ${kind} at ${frameRate()} Hz, ${done.estimated_tokens_per_unit} tokens/unit`);
    } catch (error) {
      for (const item of acquired.getTracks()) {
        try { item.stop(); } catch (_) {}
      }
      throw error;
    }
  }

  async function disable({ notifyServer = true } = {}) {
    activeSource = null;
    stopPacer();
    closeScreenSocket();
    releaseStream();
    showPreview(false);
    setState('Off');
    updateNote();
    if (runtimeVideoEnabled && notifyServer) {
      try {
        await requestMode(false, null);
      } catch (error) {
        report(`Video disable not acknowledged: ${error.message || error}`);
      }
    } else if (!notifyServer) {
      runtimeVideoEnabled = false;
    }
  }

  function sessionId() {
    return (capabilities && capabilities.session_id) || '';
  }

  function revertSelect() {
    select.value = '';
    window.localStorage.setItem(STORAGE_KEY, '');
  }

  function onTrackEnded() {
    // Observe the browser's native stop-sharing action.
    report('Video source ended');
    revertSelect();
    void disable();
  }

  async function applySelection() {
    const kind = select.value;
    window.localStorage.setItem(STORAGE_KEY, kind);
    // Apply pre-session choices when the session opens.
    if (sessionState !== 'active') return;
    select.disabled = true;
    try {
      if (activeSource) await disable();
      if (kind) await enable(kind);
    } catch (error) {
      if (error instanceof VideoSetupCancelledError || sessionState !== 'active') return;
      report(`Video unavailable: ${error.message || error}`);
      revertSelect();
      await disable();
    } finally {
      select.disabled = !['idle', 'active'].includes(sessionState);
    }
  }

  select.addEventListener('change', () => { void applySelection(); });

  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored) select.value = stored;

  window.GanderVideo = {
    // Health capabilities allow screen permission within the Start gesture.
    applyRuntimeCapabilities(clientVideo) {
      runtimeCapabilities = clientVideo ? { ...clientVideo } : null;
      // Keep authenticated session capabilities once ready.
      const available =
        sessionState === 'active' && capabilities
          ? capabilities
          : runtimeCapabilities;
      const enabled = Boolean(available && available.enabled);
      control.hidden = !enabled;
      if (!enabled) return;
      const sources = available.sources || [];
      for (const option of Array.from(select.options)) {
        if (!option.value) continue;
        const usable = sources.includes(option.value) &&
          (option.value !== 'screen' || screenShareSupported());
        option.hidden = !usable;
        option.disabled = !usable;
      }
      if (select.selectedOptions.length && select.selectedOptions[0].disabled) revertSelect();
      setState(describe());
    },

    // Show controls only when client video is available.
    applyCapabilities(message) {
      const screen = (message && message.screen) || null;
      capabilities = screen && screen.client_video ? { ...screen.client_video } : null;
      if (capabilities) {
        capabilities.session_id = message.session_id;
        capabilities.token = screen.token;
        capabilities.path = screen.path;
        screenConfig = screen;
      }
      const enabled = Boolean(capabilities && capabilities.enabled);
      control.hidden = !enabled;
      if (!enabled) {
        showPreview(false);
        setState('Off');
        return;
      }
      const sources = capabilities.sources || [];
      for (const option of Array.from(select.options)) {
        if (!option.value) continue;
        const usable = sources.includes(option.value) &&
          (option.value !== 'screen' || screenShareSupported());
        option.hidden = !usable;
        option.disabled = !usable;
      }
      if (select.selectedOptions.length && select.selectedOptions[0].disabled) revertSelect();
      setState(describe());
    },

    async attach(hooks) {
      host = hooks;
      // Apply the pre-Start selection.
      if (select.value && !activeSource) await applySelection();
    },

    // Handle media-mode replies in this module.
    handleServerEvent(message) {
      if (!message) return false;
      if (message.type === 'media.mode.done') {
        if (pendingMode) {
          window.clearTimeout(pendingMode.timeout);
          const { resolve, reject, generation } = pendingMode;
          pendingMode = null;
          if (generation !== lifecycleGeneration || sessionState !== 'active') {
            reject(new VideoSetupCancelledError());
          } else {
            runtimeVideoEnabled = Boolean(message.video);
            resolve(message);
          }
        }
        return true;
      }
      if (message.type === 'media.mode.rejected') {
        if (pendingMode) {
          window.clearTimeout(pendingMode.timeout);
          const { reject } = pendingMode;
          pendingMode = null;
          reject(new Error(message.reason || 'media mode rejected'));
        }
        return true;
      }
      return false;
    },

    setSessionState(state) {
      sessionState = state;
      select.disabled = !['idle', 'active'].includes(state);
      if (state === 'idle' && activeSource) void disable({ notifyServer: false });
    },

    // Request screen permission before the Start gesture expires.
    async preacquireIfNeeded() {
      const generation = lifecycleGeneration;
      const available = runtimeCapabilities || capabilities;
      if (select.value !== 'screen' || (available && !available.enabled)) return;
      try {
        const acquired = await acquireStream('screen');
        if (generation !== lifecycleGeneration || sessionState !== 'starting') {
          for (const item of acquired.getTracks()) {
            try { item.stop(); } catch (_) {}
          }
          return;
        }
        preacquired = { kind: 'screen', stream: acquired };
      } catch (error) {
        if (generation !== lifecycleGeneration || sessionState !== 'starting') return;
        report(`Screen share unavailable: ${error.message || error}`);
        revertSelect();
      }
    },

    async stop() {
      lifecycleGeneration += 1;
      cancelPendingMode(new VideoSetupCancelledError());
      if (preacquired) {
        for (const item of preacquired.stream.getTracks()) {
          try { item.stop(); } catch (_) {}
        }
        preacquired = null;
      }
      await disable({ notifyServer: false });
    }
  };
})();
