// segmenter.worker.js
// Runs MediaPipe InteractiveSegmenter off the main thread so the UI (drag,
// button taps, HUD animation) never freezes while a segmentation call is
// in flight. The main thread sends an ImageBitmap (cheap to transfer) plus
// the normalized click point; this worker returns the parsed mask bounds.

import {
  FilesetResolver,
  InteractiveSegmenter
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite";

let segmenter = null;
let offCanvas = null;
let offCtx = null;
let simdSupported = null;

async function init(){
  simdSupported = await FilesetResolver.isSimdSupported();

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  // GPU delegate is intentionally NOT used here: it has been observed to
  // return corrupted/scrambled category mask values on some devices (the
  // whole frame reads back as a single category), which silently breaks
  // capture. CPU delegate is slower but returns correct results, and SIMD
  // (auto-detected above) is the safe speed lever we actually rely on.
  segmenter = await InteractiveSegmenter.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    outputCategoryMask: true,
    outputConfidenceMasks: false
  });

  postMessage({ type: 'ready', simdSupported });
}

function findMaskBounds(maskData, w, h, clickNormX, clickNormY){
  // Sample the mask value AT the tapped point and treat that exact value as
  // "object", everything else as "background" — robust regardless of which
  // numeric encoding the model happens to use for its two categories.
  const clickX = Math.min(w - 1, Math.max(0, Math.round(clickNormX * w)));
  const clickY = Math.min(h - 1, Math.max(0, Math.round(clickNormY * h)));
  const clickIdx = clickY * w + clickX;
  const targetVal = maskData[clickIdx];

  let minX = w, minY = h, maxX = -1, maxY = -1;
  let count = 0;
  for (let i = 0; i < w * h; i++){
    if (maskData[i] === targetVal){
      count++;
      const x = i % w, y = (i / w) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  const areaFrac = count / (w * h);
  if (areaFrac > 0.92 || count === 0){
    return { failed: true, targetVal, areaFrac, w, h };
  }
  return { minX, minY, maxX, maxY, count, w, h, targetVal, areaFrac };
}

async function runSegmentation(bitmap, normX, normY, reqId){
  if (!segmenter){ postMessage({ type: 'result', reqId, info: null }); return; }

  if (!offCanvas || offCanvas.width !== bitmap.width || offCanvas.height !== bitmap.height){
    offCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
  }
  offCtx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const t0 = performance.now();
  segmenter.segment(offCanvas, { keypoint: { x: normX, y: normY } }, (result) => {
    const elapsed = performance.now() - t0;
    const maskObj = result && result.categoryMask;
    if (!maskObj){
      postMessage({ type: 'result', reqId, info: null, inferenceMs: elapsed });
      return;
    }
    const w = maskObj.width, h = maskObj.height;
    const data = maskObj.getAsUint8Array ? maskObj.getAsUint8Array() : maskObj.getAsFloat32Array();
    const info = findMaskBounds(data, w, h, normX, normY);
    maskObj.close && maskObj.close();
    postMessage({ type: 'result', reqId, info, inferenceMs: elapsed });
  });
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init'){
    try {
      await init();
    } catch (err){
      postMessage({ type: 'error', error: String(err) });
    }
  } else if (msg.type === 'segment'){
    try {
      await runSegmentation(msg.bitmap, msg.normX, msg.normY, msg.reqId);
    } catch (err){
      postMessage({ type: 'result', reqId: msg.reqId, info: null, error: String(err) });
    }
  }
};
