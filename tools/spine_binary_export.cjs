#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");

const TRANSFORM_MODE_NAMES = [
  "Normal",
  "OnlyTranslation",
  "NoRotationOrReflection",
  "NoScale",
  "NoScaleOrReflection",
];

function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourceArg = args.source;
  const runtimeDirArg = args.runtimeDir || args["runtime-dir"];
  const versionArg = args.version;
  if (!sourceArg) {
    throw new Error("Missing required --source argument.");
  }
  if (!runtimeDirArg) {
    throw new Error("Missing required --runtime-dir argument.");
  }

  const sourcePath = path.resolve(sourceArg);
  const runtimeDir = path.resolve(runtimeDirArg);
  const blob = fs.readFileSync(sourcePath);
  const detectedVersion = versionArg || detectSpineVersion(blob);
  const spine = loadRuntime(detectedVersion, runtimeDir);
  const attachmentLoader = new NullAttachmentLoader(spine);
  const reader = new spine.SkeletonBinary(attachmentLoader);
  const skeletonData = reader.readSkeletonData(blob);
  const payload = exportSkeletonPayload(skeletonData, detectedVersion);
  process.stdout.write(JSON.stringify(payload));
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value == null || value.startsWith("--")) {
      args[key] = true;
      continue;
    }
    args[key] = value;
    index += 1;
  }
  return args;
}

function detectSpineVersion(blob) {
  const header = Buffer.from(blob.subarray(0, 512)).toString("latin1");
  const match = header.match(/\b([34]\.\d+\.\d+)\b/);
  if (!match) {
    throw new Error(
      "Could not detect Spine binary version from header. " +
        "Use a file with an embedded version string or pass --version."
    );
  }
  return match[1];
}

function loadRuntime(version, runtimeDir) {
  const requireFromRuntime = createRequire(path.join(runtimeDir, "package.json"));
  const packageName = runtimePackageFor(version);
  return requireFromRuntime(packageName);
}

function runtimePackageFor(version) {
  const family = String(version || "").trim().split(".").slice(0, 2).join(".");
  switch (family) {
    case "3.8":
      return "@pixi-spine/runtime-3.8";
    case "4.0":
      return "@pixi-spine/runtime-4.0";
    case "4.1":
      return "@pixi-spine/runtime-4.1";
    case "4.2":
      return "@esotericsoftware/spine-core";
    default:
      throw new Error(`Unsupported Spine binary version family: ${version || "<unknown>"}`);
  }
}

class NullAttachmentLoader {
  constructor(spine) {
    this.spine = spine;
  }

  newRegionAttachment(_skin, name, pathValue) {
    return new this.spine.RegionAttachment(name, pathValue || name);
  }

  newMeshAttachment(_skin, name, pathValue) {
    return new this.spine.MeshAttachment(name, pathValue || name);
  }

  newBoundingBoxAttachment(_skin, name) {
    return new this.spine.BoundingBoxAttachment(name);
  }

  newPathAttachment(_skin, name) {
    return new this.spine.PathAttachment(name);
  }

  newPointAttachment(_skin, name) {
    return new this.spine.PointAttachment(name);
  }

  newClippingAttachment(_skin, name) {
    return new this.spine.ClippingAttachment(name);
  }
}

function exportSkeletonPayload(skeletonData, detectedVersion) {
  const boneNames = skeletonData.bones.map((bone) => bone.name);
  const slotNames = skeletonData.slots.map((slot) => slot.name);
  return {
    skeleton: exportSkeletonInfo(skeletonData, detectedVersion),
    bones: skeletonData.bones.map(exportBone),
    slots: skeletonData.slots.map(exportSlot),
    skins: skeletonData.skins.map((skin) => exportSkin(skin, slotNames)),
    animations: exportAnimations(skeletonData.animations, boneNames),
  };
}

function exportSkeletonInfo(skeletonData, detectedVersion) {
  const payload = {};
  const spineVersion = skeletonData.version || detectedVersion || null;
  if (spineVersion) {
    payload.spine = spineVersion;
  }
  if (skeletonData.hash) {
    payload.hash = skeletonData.hash;
  }
  if (typeof skeletonData.x === "number") {
    payload.x = roundNumber(skeletonData.x);
  }
  if (typeof skeletonData.y === "number") {
    payload.y = roundNumber(skeletonData.y);
  }
  if (typeof skeletonData.width === "number") {
    payload.width = roundNumber(skeletonData.width);
  }
  if (typeof skeletonData.height === "number") {
    payload.height = roundNumber(skeletonData.height);
  }
  if (typeof skeletonData.fps === "number" && skeletonData.fps > 0) {
    payload.fps = roundNumber(skeletonData.fps);
  }
  if (skeletonData.imagesPath) {
    payload.images = skeletonData.imagesPath;
  }
  if (skeletonData.audioPath) {
    payload.audio = skeletonData.audioPath;
  }
  return payload;
}

function exportBone(bone) {
  const payload = {
    name: bone.name,
  };
  if (bone.parent) {
    payload.parent = bone.parent.name;
  }
  addIfMeaningful(payload, "length", bone.length, 0);
  addIfMeaningful(payload, "x", bone.x, 0);
  addIfMeaningful(payload, "y", bone.y, 0);
  addIfMeaningful(payload, "rotation", bone.rotation, 0);
  addIfMeaningful(payload, "scaleX", bone.scaleX, 1);
  addIfMeaningful(payload, "scaleY", bone.scaleY, 1);
  addIfMeaningful(payload, "shearX", bone.shearX, 0);
  addIfMeaningful(payload, "shearY", bone.shearY, 0);
  if (
    Number.isInteger(bone.transformMode) &&
    bone.transformMode >= 0 &&
    bone.transformMode < TRANSFORM_MODE_NAMES.length &&
    bone.transformMode !== 0
  ) {
    payload.transform = TRANSFORM_MODE_NAMES[bone.transformMode];
  }
  return payload;
}

function exportSlot(slot) {
  const payload = {
    name: slot.name,
    bone: slot.boneData?.name || "",
  };
  if (slot.attachmentName) {
    payload.attachment = slot.attachmentName;
  }
  return payload;
}

function exportSkin(skin, slotNames) {
  const attachments = {};
  for (const [slotIndexRaw, slotAttachmentMap] of Object.entries(skin.attachments || {})) {
    const slotIndex = Number(slotIndexRaw);
    const slotName = slotNames[slotIndex] || String(slotIndexRaw);
    const exportedAttachments = {};
    for (const [attachmentName, attachment] of Object.entries(slotAttachmentMap || {})) {
      exportedAttachments[attachmentName] = exportAttachment(attachment, attachmentName);
    }
    if (Object.keys(exportedAttachments).length) {
      attachments[slotName] = exportedAttachments;
    }
  }
  return {
    name: skin.name || "default",
    attachments,
  };
}

function exportAttachment(attachment, attachmentName) {
  const payload = {
    type: attachmentTypeName(attachment),
  };
  if (attachment?.path && attachment.path !== attachmentName) {
    payload.path = attachment.path;
  }
  return payload;
}

function attachmentTypeName(attachment) {
  const ctorName = attachment?.constructor?.name || "";
  switch (ctorName) {
    case "RegionAttachment":
      return "region";
    case "MeshAttachment":
      return "mesh";
    case "BoundingBoxAttachment":
      return "boundingbox";
    case "PathAttachment":
      return "path";
    case "PointAttachment":
      return "point";
    case "ClippingAttachment":
      return "clipping";
    default:
      return "unknown";
  }
}

function exportAnimations(animations, boneNames) {
  const payload = {};
  for (const animation of animations || []) {
    payload[animation.name] = exportAnimation(animation, boneNames);
  }
  return payload;
}

function exportAnimation(animation, boneNames) {
  const bones = {};
  const axisBuckets = new Map();

  for (const timeline of animation.timelines || []) {
    const ctorName = timeline?.constructor?.name || "";
    if (ctorName === "RotateTimeline") {
      setBoneTimeline(bones, boneNames[timeline.boneIndex], "rotate", exportCurveTimeline1(timeline, "value"));
      continue;
    }
    if (ctorName === "TranslateTimeline") {
      setBoneTimeline(bones, boneNames[timeline.boneIndex], "translate", exportCurveTimeline2(timeline, "x", "y"));
      continue;
    }
    if (ctorName === "ScaleTimeline") {
      setBoneTimeline(bones, boneNames[timeline.boneIndex], "scale", exportCurveTimeline2(timeline, "x", "y"));
      continue;
    }
    if (
      ctorName === "TranslateXTimeline" ||
      ctorName === "TranslateYTimeline" ||
      ctorName === "ScaleXTimeline" ||
      ctorName === "ScaleYTimeline"
    ) {
      collectAxisTimeline(axisBuckets, boneNames[timeline.boneIndex], ctorName, timeline);
    }
  }

  for (const [boneName, bucket] of axisBuckets.entries()) {
    if (bucket.translateX || bucket.translateY) {
      setBoneTimeline(
        bones,
        boneName,
        "translate",
        mergeAxisTimelines(bucket.translateX, bucket.translateY, 0, 0, "x", "y")
      );
    }
    if (bucket.scaleX || bucket.scaleY) {
      setBoneTimeline(
        bones,
        boneName,
        "scale",
        mergeAxisTimelines(bucket.scaleX, bucket.scaleY, 1, 1, "x", "y")
      );
    }
  }

  return Object.keys(bones).length ? { bones } : {};
}

function collectAxisTimeline(axisBuckets, boneName, timelineType, timeline) {
  if (!boneName) {
    return;
  }
  const bucket = axisBuckets.get(boneName) || {};
  if (timelineType === "TranslateXTimeline") {
    bucket.translateX = timeline;
  } else if (timelineType === "TranslateYTimeline") {
    bucket.translateY = timeline;
  } else if (timelineType === "ScaleXTimeline") {
    bucket.scaleX = timeline;
  } else if (timelineType === "ScaleYTimeline") {
    bucket.scaleY = timeline;
  }
  axisBuckets.set(boneName, bucket);
}

function setBoneTimeline(bones, boneName, key, value) {
  if (!boneName || !Array.isArray(value) || !value.length) {
    return;
  }
  bones[boneName] = bones[boneName] || {};
  bones[boneName][key] = value;
}

function exportCurveTimeline1(timeline, valueKey) {
  const keys = [];
  for (let index = 0; index < timeline.frames.length; index += 2) {
    const frameIndex = index / 2;
    const keyframe = {
      time: roundNumber(timeline.frames[index]),
      [valueKey]: roundNumber(timeline.frames[index + 1]),
    };
    applyCurveMetadata(timeline, frameIndex, keyframe);
    keys.push(keyframe);
  }
  return keys;
}

function exportCurveTimeline2(timeline, valueKey1, valueKey2) {
  const keys = [];
  for (let index = 0; index < timeline.frames.length; index += 3) {
    const frameIndex = index / 3;
    const keyframe = {
      time: roundNumber(timeline.frames[index]),
      [valueKey1]: roundNumber(timeline.frames[index + 1]),
      [valueKey2]: roundNumber(timeline.frames[index + 2]),
    };
    applyCurveMetadata(timeline, frameIndex, keyframe);
    keys.push(keyframe);
  }
  return keys;
}

function mergeAxisTimelines(
  firstTimeline,
  secondTimeline,
  firstDefault,
  secondDefault,
  firstKey,
  secondKey
) {
  const keyTimes = new Set();
  collectTimelineTimes(keyTimes, firstTimeline, 2);
  collectTimelineTimes(keyTimes, secondTimeline, 2);
  const sortedTimes = Array.from(keyTimes).sort((lhs, rhs) => lhs - rhs);
  return sortedTimes.map((time) => ({
    time: roundNumber(time),
    [firstKey]: roundNumber(sampleCurveTimeline1(firstTimeline, time, firstDefault)),
    [secondKey]: roundNumber(sampleCurveTimeline1(secondTimeline, time, secondDefault)),
  }));
}

function collectTimelineTimes(target, timeline, step) {
  if (!timeline) {
    return;
  }
  for (let index = 0; index < timeline.frames.length; index += step) {
    target.add(timeline.frames[index]);
  }
}

function sampleCurveTimeline1(timeline, time, fallback) {
  if (!timeline) {
    return fallback;
  }
  if (typeof timeline.getCurveValue === "function") {
    return timeline.getCurveValue(time);
  }
  return fallback;
}

function applyCurveMetadata(timeline, frameIndex, keyframe) {
  if (!timeline || !timeline.curves) {
    return;
  }
  if (frameIndex >= timeline.getFrameCount() - 1) {
    return;
  }
  const curveType = timeline.curves[frameIndex];
  if (curveType === 1) {
    keyframe.curve = "stepped";
  }
}

function addIfMeaningful(target, key, value, defaultValue) {
  if (typeof value !== "number") {
    return;
  }
  const normalized = roundNumber(value);
  if (normalized === roundNumber(defaultValue)) {
    return;
  }
  target[key] = normalized;
}

function roundNumber(value) {
  return Number(Number(value || 0).toFixed(4));
}

try {
  main();
} catch (error) {
  const message = error && error.stack ? error.stack : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
}
