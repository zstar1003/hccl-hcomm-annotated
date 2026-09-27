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
import { computed, ref, watch } from 'vue'
import { Upload } from '@element-plus/icons-vue'
import { loadTraceFromFile } from '../utils/ccuTraceData'
import CcuTracePlaybackControl from '../components/ccu-trace/CcuTracePlaybackControl.vue'
import CcuTraceProgressBar from '../components/ccu-trace/CcuTraceProgressBar.vue'
import CcuTraceInstrList from '../components/ccu-trace/CcuTraceInstrList.vue'
import CcuTraceResourceDelta from '../components/ccu-trace/CcuTraceResourceDelta.vue'
import CcuTraceInstrDetail from '../components/ccu-trace/CcuTraceInstrDetail.vue'
import CcuTraceResourceOverview from '../components/ccu-trace/CcuTraceResourceOverview.vue'
import CcuTraceSqeInfo from '../components/ccu-trace/CcuTraceSqeInfo.vue'
import CcuTraceExecSummary from '../components/ccu-trace/CcuTraceExecSummary.vue'

// === 数据状态 ===
const traceData = ref(null)
const loadError = ref('')
const loading = ref(false)
const fileName = ref('')

// === 回放状态 ===
const currentStepIndex = ref(0) // 当前在 globalEntries 中的索引
const currentRankId = ref(0)
const currentDieId = ref(0)
const isPlaying = ref(false)
const playSpeed = ref(500) // ms per step
let playTimer = null

// === 断点状态 ===
const breakpoints = ref([]) // [{rankId, dieId, instrId}]

// === 拖拽分割状态 ===
const sidebarWidth = ref(280)
const bottomHeight = ref(280)
const bottomPanel1Width = ref(null) // 资源变化
const bottomPanel2Width = ref(null) // 指令细节
const layoutRef = ref(null)
const mainRef = ref(null)
const bottomRef = ref(null)
let dragging = null // 'sidebar' | 'bottom' | 'bottomP1' | 'bottomP2' | null

function onSidebarDragStart(e) {
  e.preventDefault()
  dragging = 'sidebar'
  document.addEventListener('mousemove', onDragMove)
  document.addEventListener('mouseup', onDragEnd)
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

function onBottomDragStart(e) {
  e.preventDefault()
  dragging = 'bottom'
  document.addEventListener('mousemove', onDragMove)
  document.addEventListener('mouseup', onDragEnd)
  document.body.style.cursor = 'row-resize'
  document.body.style.userSelect = 'none'
}

function onBottomPanel1DragStart(e) {
  e.preventDefault()
  dragging = 'bottomP1'
  document.addEventListener('mousemove', onDragMove)
  document.addEventListener('mouseup', onDragEnd)
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

function onBottomPanel2DragStart(e) {
  e.preventDefault()
  dragging = 'bottomP2'
  document.addEventListener('mousemove', onDragMove)
  document.addEventListener('mouseup', onDragEnd)
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

function onDragMove(e) {
  if (dragging === 'sidebar' && layoutRef.value) {
    const rect = layoutRef.value.getBoundingClientRect()
    const newWidth = e.clientX - rect.left
    sidebarWidth.value = Math.max(200, Math.min(newWidth, rect.width - 400))
  } else if (dragging === 'bottom' && mainRef.value) {
    const rect = mainRef.value.getBoundingClientRect()
    const newHeight = rect.bottom - e.clientY
    bottomHeight.value = Math.max(120, Math.min(newHeight, rect.height - 120))
  } else if (dragging === 'bottomP1' && bottomRef.value) {
    const rect = bottomRef.value.getBoundingClientRect()
    const newWidth = e.clientX - rect.left
    bottomPanel1Width.value = Math.max(100, Math.min(newWidth, rect.width - 250))
  } else if (dragging === 'bottomP2' && bottomRef.value) {
    const rect = bottomRef.value.getBoundingClientRect()
    const newWidth = e.clientX - rect.left
    // Panel 2 左边界 = panel1 宽度 + splitter，右边界 = 容器宽 - 最小 panel3 宽度
    const minLeft = (bottomPanel1Width.value || rect.width / 3) + 10
    const maxWidth = rect.width - 110
    bottomPanel2Width.value = Math.max(minLeft, Math.min(newWidth, maxWidth))
  }
}

function onDragEnd() {
  dragging = null
  document.removeEventListener('mousemove', onDragMove)
  document.removeEventListener('mouseup', onDragEnd)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

// === Computed ===
const currentEntry = computed(() => {
  if (!traceData.value || traceData.value.totalEntries === 0) return null
  return traceData.value.globalEntries[currentStepIndex.value] || null
})

const currentCcuKey = computed(() => `${currentRankId.value}:${currentDieId.value}`)

const ccuList = computed(() => traceData.value?.ccuList || [])

const progressPercent = computed(() => {
  if (!traceData.value || traceData.value.totalEntries === 0) return 0
  return Math.round(((currentStepIndex.value + 1) / traceData.value.totalEntries) * 100)
})

// 当 entry 切换 CCU 时自动更新选择器
watch(currentEntry, (entry) => {
  if (entry) {
    currentRankId.value = entry.rankId
    currentDieId.value = entry.dieId
  }
})

// === 方法 ===
async function handleFileUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return

  loading.value = true
  loadError.value = ''
  fileName.value = file.name

  try {
    traceData.value = await loadTraceFromFile(file)
    currentStepIndex.value = 0
    if (traceData.value.ccuList.length > 0) {
      currentRankId.value = traceData.value.ccuList[0].rankId
      currentDieId.value = traceData.value.ccuList[0].dieId
    }
  } catch (err) {
    loadError.value = err.message || 'Failed to load trace file'
    traceData.value = null
  } finally {
    loading.value = false
  }
}

function stepForward() {
  if (!traceData.value) return
  if (currentStepIndex.value < traceData.value.totalEntries - 1) {
    currentStepIndex.value++
  }
}

function stepBackward() {
  if (currentStepIndex.value > 0) {
    currentStepIndex.value--
  }
}

function stepToFirst() {
  currentStepIndex.value = 0
}

function stepToLast() {
  if (traceData.value) {
    currentStepIndex.value = traceData.value.totalEntries - 1
  }
}

function togglePlay() {
  if (isPlaying.value) {
    stopPlay()
  } else {
    startPlay()
  }
}

function startPlay() {
  if (!traceData.value || traceData.value.totalEntries === 0) return
  isPlaying.value = true
  playTimer = setInterval(() => {
    if (currentStepIndex.value >= traceData.value.totalEntries - 1) {
      stopPlay()
      return
    }
    // 检查断点
    const nextEntry = traceData.value.globalEntries[currentStepIndex.value + 1]
    if (nextEntry && isBreakpointHit(nextEntry)) {
      currentStepIndex.value++
      stopPlay()
      return
    }
    currentStepIndex.value++
  }, playSpeed.value)
}

function stopPlay() {
  isPlaying.value = false
  if (playTimer) {
    clearInterval(playTimer)
    playTimer = null
  }
}

function jumpToNextBreakpoint() {
  if (!traceData.value) return
  for (let i = currentStepIndex.value + 1; i < traceData.value.totalEntries; i++) {
    const entry = traceData.value.globalEntries[i]
    if (isBreakpointHit(entry)) {
      currentStepIndex.value = i
      return
    }
  }
}

function resetPlayback() {
  stopPlay()
  currentStepIndex.value = 0
  if (traceData.value && traceData.value.ccuList.length > 0) {
    currentRankId.value = traceData.value.ccuList[0].rankId
    currentDieId.value = traceData.value.ccuList[0].dieId
  }
}

function isBreakpointHit(entry) {
  return breakpoints.value.some(
    (bp) => bp.rankId === entry.rankId && bp.dieId === entry.dieId && bp.instrId === entry.instrId,
  )
}

function hasBreakpoint(rankId, dieId, instrId) {
  return breakpoints.value.some(
    (bp) => bp.rankId === rankId && bp.dieId === dieId && bp.instrId === instrId,
  )
}

function toggleBreakpoint(rankId, dieId, instrId) {
  const idx = breakpoints.value.findIndex(
    (bp) => bp.rankId === rankId && bp.dieId === dieId && bp.instrId === instrId,
  )
  if (idx >= 0) {
    breakpoints.value.splice(idx, 1)
  } else {
    breakpoints.value.push({ rankId, dieId, instrId })
  }
}

function clearBreakpoints() {
  breakpoints.value = []
}

function jumpToStep(index) {
  if (index >= 0 && traceData.value && index < traceData.value.totalEntries) {
    currentStepIndex.value = index
  }
}

function onCcuChange({ rankId, dieId }) {
  stopPlay()
  currentRankId.value = rankId
  currentDieId.value = dieId
}

function handleDragFile(e) {
  e.preventDefault()
  const file = e.dataTransfer?.files?.[0]
  if (file && file.name.endsWith('.json')) {
    const fakeEvent = { target: { files: [file] } }
    handleFileUpload(fakeEvent)
  }
}
</script>

<template>
  <div class="ccu-trace-page">
    <!-- 未加载状态：文件上传区 -->
    <div v-if="!traceData" class="ccu-trace-upload"
      @dragover.prevent @drop="handleDragFile">
      <div class="upload-card">
        <el-icon :size="48"><Upload /></el-icon>
        <h2>加载 CCU Trace 数据</h2>
        <p>拖拽 JSON 文件到此处，或点击下方按钮选择文件</p>
        <el-button type="primary" size="large" @click="$refs.fileInput.click()">
          选择 Trace JSON 文件
        </el-button>
        <input ref="fileInput" type="file" accept=".json" style="display: none" @change="handleFileUpload" />
        <p v-if="loading" class="upload-status">加载中...</p>
        <p v-if="loadError" class="upload-error">{{ loadError }}</p>
      </div>
    </div>

    <!-- 已加载状态：回放界面 -->
    <div v-else class="ccu-trace-layout" ref="layoutRef">
      <!-- 左侧栏 -->
      <aside class="ccu-trace-sidebar" :style="{ width: sidebarWidth + 'px' }">
        <CcuTracePlaybackControl
          :is-playing="isPlaying"
          :play-speed="playSpeed"
          :current-step="currentStepIndex"
          :total-steps="traceData.totalEntries"
          @step-forward="stepForward"
          @step-backward="stepBackward"
          @step-first="stepToFirst"
          @step-last="stepToLast"
          @toggle-play="togglePlay"
          @jump-breakpoint="jumpToNextBreakpoint"
          @speed-change="(v) => playSpeed = v"
          @reset="resetPlayback"
        />

        <div class="sidebar-divider" />

        <CcuTraceExecSummary
          :trace-data="traceData"
          :current-step-index="currentStepIndex"
          :current-rank-id="currentRankId"
          :current-die-id="currentDieId"
        />

        <div class="sidebar-divider" />

        <CcuTraceSqeInfo
          :trace-data="traceData"
          :current-entry="currentEntry"
        />

        <div class="sidebar-divider" />

        <!-- 断点管理 -->
        <div class="sidebar-section">
          <div class="sidebar-section__header">
            <span>断点管理</span>
            <el-button size="small" text @click="clearBreakpoints" :disabled="breakpoints.length === 0">
              清除全部
            </el-button>
          </div>
          <div class="breakpoint-list" v-if="breakpoints.length > 0">
            <div v-for="(bp, idx) in breakpoints" :key="idx" class="breakpoint-item">
              <span class="bp-marker">●</span>
              <span class="bp-label">R{{ bp.rankId }}:D{{ bp.dieId }} id={{ bp.instrId }}</span>
              <el-button size="small" text type="danger" @click="breakpoints.splice(idx, 1)">✕</el-button>
            </div>
          </div>
          <div v-else class="breakpoint-empty">点击指令列表 BP 列设置断点</div>
        </div>

        <div class="sidebar-divider" />

        <!-- 文件信息 -->
        <div class="sidebar-section">
          <div class="file-info">
            <span class="file-name" :title="fileName">{{ fileName }}</span>
            <el-button size="small" text @click="traceData = null; fileName = ''">重新加载</el-button>
          </div>
        </div>
      </aside>

      <!-- 垂直分割拖拽条 -->
      <div class="splitter splitter--vertical" @mousedown="onSidebarDragStart" />

      <!-- 右侧主区域 -->
      <main class="ccu-trace-main" ref="mainRef">
        <!-- 进度条 + CCU 选择器 -->
        <CcuTraceProgressBar
          :trace-data="traceData"
          :current-step-index="currentStepIndex"
          :current-rank-id="currentRankId"
          :current-die-id="currentDieId"
          :ccu-list="ccuList"
          :progress-percent="progressPercent"
          :current-entry="currentEntry"
          @ccu-change="onCcuChange"
          @jump-to-step="jumpToStep"
        />

        <!-- 指令列表 -->
        <CcuTraceInstrList
          :trace-data="traceData"
          :current-rank-id="currentRankId"
          :current-die-id="currentDieId"
          :current-entry="currentEntry"
          :current-step-index="currentStepIndex"
          :breakpoints="breakpoints"
          @toggle-breakpoint="toggleBreakpoint"
          @has-breakpoint="hasBreakpoint"
          @jump-to-step="jumpToStep"
        />

        <!-- 水平分割拖拽条 -->
        <div class="splitter splitter--horizontal" @mousedown="onBottomDragStart" />

        <!-- 底部面板 -->
        <div class="ccu-trace-bottom" :style="{ height: bottomHeight + 'px' }" ref="bottomRef">
          <CcuTraceResourceDelta
            :current-entry="currentEntry"
            :style="bottomPanel1Width ? { width: bottomPanel1Width + 'px', flex: 'none' } : {}"
          />
          <div class="splitter splitter--vertical" @mousedown="onBottomPanel1DragStart" />
          <CcuTraceInstrDetail
            :current-entry="currentEntry"
            :trace-data="traceData"
            :style="bottomPanel2Width ? { width: (bottomPanel2Width - (bottomPanel1Width || 0)) + 'px', flex: 'none' } : {}"
          />
          <div class="splitter splitter--vertical" @mousedown="onBottomPanel2DragStart" />
          <CcuTraceResourceOverview
            :trace-data="traceData"
            :current-rank-id="currentRankId"
            :current-die-id="currentDieId"
            :current-step-index="currentStepIndex"
            :current-entry="currentEntry"
          />
        </div>
      </main>
    </div>
  </div>
</template>

<style scoped>
.ccu-trace-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #141414;
  color: rgba(255, 255, 255, 0.92);
}

/* === 上传区 === */
.ccu-trace-upload {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-card {
  text-align: center;
  padding: 48px 64px;
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.03);
}

.upload-card h2 {
  margin: 16px 0 8px;
  font-size: 20px;
}

.upload-card p {
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 24px;
}

.upload-error {
  color: #f56c6c;
}

/* === 布局 === */
.ccu-trace-layout {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* === 左侧栏 === */
.ccu-trace-sidebar {
  min-width: 200px;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 12px;
  gap: 4px;
  flex-shrink: 0;
}

.sidebar-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.08);
  margin: 8px 0;
}

.sidebar-section {
  padding: 4px 0;
}

.sidebar-section__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 8px;
}

/* === 断点列表 === */
.breakpoint-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.breakpoint-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 4px;
  font-size: 12px;
  font-family: monospace;
}

.bp-marker {
  color: #e74c3c;
  font-size: 14px;
}

.bp-label {
  flex: 1;
}

.breakpoint-empty {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.3);
  padding: 4px 0;
}

/* === 文件信息 === */
.file-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
}

.file-name {
  color: rgba(255, 255, 255, 0.5);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 180px;
}

/* === 右侧主区域 === */
.ccu-trace-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.ccu-trace-bottom {
  display: flex;
  min-height: 120px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  overflow: hidden;
  flex-shrink: 0;
}

.ccu-trace-bottom > *:not(.splitter) {
  flex: 1;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  overflow-y: auto;
  min-width: 100px;
}

.ccu-trace-bottom > *:last-child:not(.splitter) {
  border-right: none;
}

/* === 拖拽分割条 === */
.splitter {
  flex-shrink: 0;
  z-index: 10;
}

.splitter--vertical {
  width: 5px;
  cursor: col-resize;
  background: rgba(255, 255, 255, 0.04);
  transition: background 0.15s;
}

.splitter--vertical:hover,
.splitter--vertical:active {
  background: rgba(64, 158, 255, 0.5);
}

.splitter--horizontal {
  height: 5px;
  cursor: row-resize;
  background: rgba(255, 255, 255, 0.04);
  transition: background 0.15s;
}

.splitter--horizontal:hover,
.splitter--horizontal:active {
  background: rgba(64, 158, 255, 0.5);
}
</style>
