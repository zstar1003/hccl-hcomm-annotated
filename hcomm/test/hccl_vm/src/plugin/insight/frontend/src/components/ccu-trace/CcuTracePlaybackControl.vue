<!--
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
-->

<script setup>
import { VideoPause, VideoPlay, ArrowLeftBold, ArrowRightBold, DArrowLeft, DArrowRight, Right, Refresh } from '@element-plus/icons-vue'

const props = defineProps({
  isPlaying: Boolean,
  playSpeed: Number,
  currentStep: Number,
  totalSteps: Number,
})

const emit = defineEmits([
  'step-forward', 'step-backward', 'step-first', 'step-last',
  'toggle-play', 'jump-breakpoint', 'speed-change', 'reset',
])

const speedOptions = [
  { label: '0.1s', value: 100 },
  { label: '0.5s', value: 500 },
  { label: '1s', value: 1000 },
  { label: '2s', value: 2000 },
]
</script>

<template>
  <div class="playback-control">
    <div class="playback-control__header">回放控制</div>

    <div class="playback-control__buttons">
      <button class="ctrl-btn" title="跳到首条" @click="emit('step-first')" :disabled="currentStep <= 0">
        <el-icon><DArrowLeft /></el-icon>
      </button>
      <button class="ctrl-btn" title="后退一步" @click="emit('step-backward')" :disabled="currentStep <= 0">
        <el-icon><ArrowLeftBold /></el-icon>
      </button>
      <button class="ctrl-btn ctrl-btn--play" :title="isPlaying ? '暂停' : '播放'" @click="emit('toggle-play')">
        <el-icon :size="20"><component :is="isPlaying ? VideoPause : VideoPlay" /></el-icon>
      </button>
      <button class="ctrl-btn" title="前进一步" @click="emit('step-forward')" :disabled="currentStep >= totalSteps - 1">
        <el-icon><ArrowRightBold /></el-icon>
      </button>
      <button class="ctrl-btn" title="跳到末条" @click="emit('step-last')" :disabled="currentStep >= totalSteps - 1">
        <el-icon><DArrowRight /></el-icon>
      </button>
    </div>

    <button class="ctrl-btn ctrl-btn--breakpoint" title="跳到下一断点" @click="emit('jump-breakpoint')">
      <el-icon><Right /></el-icon>
      <span>下一断点</span>
    </button>

    <button class="ctrl-btn ctrl-btn--reset" title="重置回放" @click="emit('reset')">
      <el-icon><Refresh /></el-icon>
      <span>重置</span>
    </button>

    <div class="playback-control__speed">
      <span class="speed-label">步速</span>
      <el-select size="small" :model-value="playSpeed" @change="(v) => emit('speed-change', v)" style="width: 80px">
        <el-option v-for="opt in speedOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
      </el-select>
    </div>

    <div class="playback-control__counter">
      #{{ currentStep + 1 }} / {{ totalSteps }}
    </div>
  </div>
</template>

<style scoped>
.playback-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.playback-control__header {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
}

.playback-control__buttons {
  display: flex;
  gap: 4px;
  justify-content: center;
}

.ctrl-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 32px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.05);
  color: rgba(255, 255, 255, 0.85);
  cursor: pointer;
  transition: all 0.15s;
}

.ctrl-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(255, 255, 255, 0.25);
}

.ctrl-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.ctrl-btn--play {
  width: 44px;
  background: rgba(64, 158, 255, 0.2);
  border-color: rgba(64, 158, 255, 0.4);
}

.ctrl-btn--play:hover {
  background: rgba(64, 158, 255, 0.35);
}

.ctrl-btn--breakpoint {
  width: 100%;
  gap: 6px;
  font-size: 12px;
  height: 28px;
}

.ctrl-btn--reset {
  width: 100%;
  gap: 6px;
  font-size: 12px;
  height: 28px;
  margin-top: 4px;
  background: rgba(231, 76, 60, 0.1);
  border-color: rgba(231, 76, 60, 0.3);
}

.ctrl-btn--reset:hover {
  background: rgba(231, 76, 60, 0.25);
}

.playback-control__speed {
  display: flex;
  align-items: center;
  gap: 8px;
}

.speed-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.playback-control__counter {
  text-align: center;
  font-size: 13px;
  font-family: monospace;
  color: rgba(255, 255, 255, 0.7);
}
</style>
