'use strict';

class MiniCPMMicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requested = Number(options.processorOptions?.frameSize);
    this.frameSize = Number.isInteger(requested) && requested > 0 ? requested : 4096;
    this.frame = new Float32Array(this.frameSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input?.length) return true;

    let inputOffset = 0;
    while (inputOffset < input.length) {
      const count = Math.min(this.frameSize - this.offset, input.length - inputOffset);
      this.frame.set(input.subarray(inputOffset, inputOffset + count), this.offset);
      this.offset += count;
      inputOffset += count;
      if (this.offset === this.frameSize) {
        const completed = this.frame;
        this.frame = new Float32Array(this.frameSize);
        this.offset = 0;
        const capturedAtSec = currentTime + inputOffset / sampleRate
          - this.frameSize / sampleRate;
        this.port.postMessage(
          { samples: completed, capturedAtSec },
          [completed.buffer]
        );
      }
    }
    return true;
  }
}

registerProcessor('minicpm-mic-capture', MiniCPMMicCaptureProcessor);
