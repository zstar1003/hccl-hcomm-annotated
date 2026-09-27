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
import { computed, reactive, ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import TopoIconSvg from './TopoIconSvg.vue'

const props = defineProps({
  serverTopoConfig: {
    type: Object,
    default: null,
  },
  serverId: {
    type: Number,
    default: 0,
  },
  superPodId: {
    type: Number,
    default: 0,
  },
})

const linkTooltip = reactive({ visible: false, x: 0, y: 0, content: '' })
const selectedDevice = ref(null)
const deviceTooltip = reactive({ visible: false, x: 0, y: 0, content: '' })
const svgScale = ref(1)
const canvasWrapRef = ref(null)
const serverViewRef = ref(null)
const splitterRef = ref(null)
const topoHeightPercent = ref(70)

const SCALE_MIN = 0.3
const SCALE_MAX = 3
const SCALE_STEP = 0.15

const typeLabelMap = { 0: '', 1: 'P2P', 2: 'P2N', 3: 'H2D' }
const typeColorMap = { 0: '#555', 1: '#67d9ff', 2: '#ffa940', 3: '#95de64' }
const typeBgMap = {
  0: 'transparent',
  1: 'rgba(103, 217, 255, 0.08)',
  2: 'rgba(255, 169, 64, 0.08)',
  3: 'rgba(149, 222, 100, 0.08)',
}

const NPU_W_SINGLE = 44
const NPU_H_SINGLE = 44
const NPU_W_MULTI = 76
const NPU_H_MULTI = 56
const DIE_PADDING_X = 3
const DIE_GAP = 2
const DIE_TOP_OFFSET = 14
const DIE_BOTTOM_PADDING = 4

const devicePositions = computed(() => {
  const config = props.serverTopoConfig
  if (!config) return { devices: [], width: 0, height: 0 }

  const count = config.deviceNum || 0
  const dieNum = (config.pinMapEntries || []).length || 1
  const cols = Math.min(count, 8)
  const rows = Math.ceil(count / cols)
  const positions = []

  const npuW = dieNum >= 2 ? NPU_W_MULTI : NPU_W_SINGLE
  const npuH = dieNum >= 2 ? NPU_H_MULTI : NPU_H_SINGLE
  const colStep = dieNum >= 2 ? 110 : 100
  const rowGap = rows > 1 ? (dieNum >= 2 ? 135 : 120) : 90
  const paddingTop = 150

  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / cols)
    const col = i % cols
    const x = col * colStep + 55
    const y = row * rowGap + paddingTop

    const dies = []
    if (dieNum >= 2) {
      const dieW = (npuW - 2 * DIE_PADDING_X - (dieNum - 1) * DIE_GAP) / dieNum
      const dieH = npuH - DIE_TOP_OFFSET - DIE_BOTTOM_PADDING
      for (let d = 0; d < dieNum; d++) {
        const dieLocalX = DIE_PADDING_X + d * (dieW + DIE_GAP)
        const dieLocalY = DIE_TOP_OFFSET
        dies.push({
          dieId: (config.pinMapEntries || [])[d]?.dieId ?? d,
          localX: dieLocalX,
          localY: dieLocalY,
          w: dieW,
          h: dieH,
          centerX: x - npuW / 2 + dieLocalX + dieW / 2,
          centerY: y - npuH / 2 + dieLocalY + dieH / 2,
        })
      }
    }

    positions.push({
      id: i,
      x,
      y,
      npuW,
      npuH,
      dies,
      dieNum,
    })
  }

  return { devices: positions, width: cols * colStep + 70, height: (rows - 1) * rowGap + paddingTop + 80 }
})

const p2pLinks = computed(() => {
  const config = props.serverTopoConfig
  if (!config || !config.links) return []

  const pinMapEntries = config.pinMapEntries || []

  function getP2PPortsForDie(dieId) {
    const entry = pinMapEntries.find((e) => e.dieId === dieId)
    if (!entry) return []
    return entry.pinMap.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0)
  }

  const portUsage = {}

  function getNextP2PPort(deviceId, dieId) {
    const key = `${deviceId}-${dieId}`
    if (portUsage[key] === undefined) {
      portUsage[key] = 0
    }
    const p2pPorts = getP2PPortsForDie(dieId)
    if (p2pPorts.length === 0) return null
    const idx = portUsage[key]
    const portId = p2pPorts[idx % p2pPorts.length]
    portUsage[key] = idx + 1
    return `${dieId}/${portId}`
  }

  const links = []
  const added = new Set()

  function addLink(from, to, srcDieId, dstDieId) {
    const key = `${from}-${srcDieId}-${to}-${dstDieId}`
    if (added.has(key)) return
    added.add(key)
    const portA = getNextP2PPort(from, srcDieId)
    const portB = getNextP2PPort(to, dstDieId)
    links.push({ from, to, type: 'p2p', portA, portB, srcDieId, dstDieId })
  }

  const fullmeshConns = config.links.fullmesh || []
  for (const conn of fullmeshConns) {
    const dieId = conn.dieId
    let devices = []
    if (conn.devices && Array.isArray(conn.devices) && conn.devices.length >= 2) {
      devices = conn.devices
    } else {
      const [start, end] = conn.devicesRange || []
      if (start === undefined || end === undefined) continue
      for (let i = start; i <= end; i++) devices.push(i)
    }
    for (let i = 0; i < devices.length; i++) {
      for (let j = i + 1; j < devices.length; j++) {
        addLink(devices[i], devices[j], dieId, dieId)
      }
    }
  }

  function expandLocalIdRange(range) {
    if (!range || !Array.isArray(range) || range.length === 0) return []
    if (range.length === 2) {
      const [start, end] = range
      const result = []
      for (let i = start; i <= end; i++) result.push(i)
      return result
    }
    return [...range]
  }

  const d2dConns = config.links.enum?.deviceToDevice || []
  for (const conn of d2dConns) {
    const srcDieId = conn.srcDieId
    const dstDieId = conn.dstDieId
    if (conn.srcLocalId && conn.dstLocalId) {
      for (const s of conn.srcLocalId) {
        for (const d of conn.dstLocalId) {
          addLink(s, d, srcDieId, dstDieId)
        }
      }
    } else if (conn.srcLocalIdRange || conn.dstLocalIdRange) {
      const srcIds = expandLocalIdRange(conn.srcLocalIdRange)
      const dstIds = expandLocalIdRange(conn.dstLocalIdRange)
      for (const s of srcIds) {
        for (const d of dstIds) {
          addLink(s, d, srcDieId, dstDieId)
        }
      }
    } else {
      const [start, end] = conn.devicesRange || []
      if (start === undefined || end === undefined) continue
      for (let i = start; i <= end; i++) {
        for (let j = i + 1; j <= end; j++) {
          addLink(i, j, srcDieId, dstDieId)
        }
      }
    }
  }

  return links
})

const p2netDevices = computed(() => {
  const config = props.serverTopoConfig
  if (!config || !config.links) return []

  const devices = new Set()
  const d2s = config.links.enum?.deviceToSwitch || []
  for (const conn of d2s) {
    const [start, end] = conn.devicesRange || []
    if (start === undefined || end === undefined) continue
    for (let i = start; i <= end; i++) {
      devices.add(i)
    }
  }
  return [...devices]
})

const layer0PortGroup = computed(() => {
  const config = props.serverTopoConfig
  if (!config || !config.portGroups) return null
  return config.portGroups.find((pg) => pg.layer === 0) || null
})

const switchPosition = computed(() => {
  const dp = devicePositions.value
  if (!dp.devices || dp.devices.length === 0) return null
  if (p2netDevices.value.length === 0) return null
  return { x: dp.width / 2, y: dp.height + 30 }
})

const svgContentWidth = computed(() => {
  const dp = devicePositions.value
  return dp.width || 470
})

const svgContentHeight = computed(() => {
  const dp = devicePositions.value
  const hasSwitch = p2netDevices.value.length > 0
  return dp.height + (hasSwitch ? 60 : 0) || 240
})

const hostIp = computed(() => {
  const sp = props.superPodId
  const srv = props.serverId
  return `${192 + sp}.${srv + 1}.5.5`
})

const selectedDevicePortMap = computed(() => {
  if (selectedDevice.value === null) return null
  const config = props.serverTopoConfig
  if (!config) return null

  const devId = selectedDevice.value
  const pinMapEntries = config.pinMapEntries || []
  const portGroups = config.portGroups || []

  const devicePortGroups = []

  for (const pg of portGroups) {
    const pgPorts = []
    for (const portStr of pg.ports || []) {
      const parts = portStr.split('/')
      let belongsToDevice = false
      if (parts.length >= 3) {
        belongsToDevice = Number(parts[0]) === devId
      } else if (parts.length === 2) {
        belongsToDevice = true
      }
      if (belongsToDevice) {
        pgPorts.push(portStr)
      }
    }
    if (pgPorts.length > 0) {
      let type = 'unknown'
      const firstPort = pgPorts[0]
      if (firstPort === 'd2h') {
        type = 'H2D'
      } else {
        const parts = firstPort.split('/')
        if (parts.length >= 2) {
          const dieId = Number(parts[0])
          const portId = Number(parts[1])
          const entry = pinMapEntries.find((e) => e.dieId === dieId)
          if (entry && portId < entry.pinMap.length) {
            const typeMap = { 0: 'unknown', 1: 'P2P', 2: 'P2N', 3: 'H2D' }
            type = typeMap[entry.pinMap[portId]] || 'unknown'
          }
        }
      }
      devicePortGroups.push({
        layer: pg.layer,
        ports: pgPorts,
        type,
      })
    }
  }

  if (pinMapEntries.length > 0) {
    const maxPortCount = Math.max(...pinMapEntries.map((e) => (e.pinMap || []).length))
    const portIds = []
    for (let i = 0; i < maxPortCount; i++) {
      portIds.push(i)
    }

    const rows = pinMapEntries.map((entry) => {
      const dieId = entry.dieId
      const pinMap = entry.pinMap || []
      const cells = []
      for (let portIdx = 0; portIdx < maxPortCount; portIdx++) {
        const val = pinMap[portIdx]
        const type = val !== undefined ? val : 0
        cells.push({
          portIdx,
          type,
          label: type === 0 ? 'NA' : (typeLabelMap[type] || '?'),
          color: type === 0 ? 'rgba(255, 255, 255, 0.2)' : (typeColorMap[type] || '#888'),
          bg: typeBgMap[type] || 'transparent',
        })
      }
      return { dieId, cells }
    })

    return {
      portIds,
      rows,
      deviceId: devId,
      serverId: props.serverId,
      superPodId: props.superPodId,
      devicePortGroups,
    }
  }

  return {
    portIds: [],
    rows: [],
    deviceId: devId,
    serverId: props.serverId,
    superPodId: props.superPodId,
    devicePortGroups,
  }
})

function getP2PPath(link) {
  const dp = devicePositions.value
  const fromDev = dp.devices[link.from]
  const toDev = dp.devices[link.to]
  if (!fromDev || !toDev) return ''

  let fromX = fromDev.x
  let fromY = fromDev.y
  let toX = toDev.x
  let toY = toDev.y

  if (fromDev.dieNum >= 2 && link.srcDieId !== undefined) {
    const die = fromDev.dies.find((d) => d.dieId === link.srcDieId)
    if (die) {
      fromX = die.centerX
      fromY = die.centerY
    }
  }

  if (toDev.dieNum >= 2 && link.dstDieId !== undefined) {
    const die = toDev.dies.find((d) => d.dieId === link.dstDieId)
    if (die) {
      toX = die.centerX
      toY = die.centerY
    }
  }

  const config = props.serverTopoConfig
  const cols = Math.min(config?.deviceNum || 0, 8)
  const rows = Math.ceil((config?.deviceNum || 0) / cols)
  const fromRow = Math.floor(link.from / cols)
  const fromCol = link.from % cols
  const toRow = Math.floor(link.to / cols)
  const toCol = link.to % cols

  if (fromRow === toRow) {
    const level = Math.abs(fromCol - toCol)
    const curveHeight = level * 20
    const midX = (fromX + toX) / 2
    const direction = fromRow < rows / 2 ? -1 : 1
    return `M ${fromX} ${fromY} C ${midX} ${fromY + direction * curveHeight}, ${midX} ${toY + direction * curveHeight}, ${toX} ${toY}`
  }

  return `M ${fromX} ${fromY} L ${toX} ${toY}`
}

function showLinkInfo(event, content) {
  linkTooltip.visible = true
  linkTooltip.x = event.clientX
  linkTooltip.y = event.clientY
  linkTooltip.content = content
}

function hideLinkInfo() {
  linkTooltip.visible = false
}

function onDeviceMouseEnter(event, devId) {
  deviceTooltip.visible = true
  deviceTooltip.x = event.clientX
  deviceTooltip.y = event.clientY
  deviceTooltip.content = `Device Local ID: ${devId}`
}

function onDeviceMouseLeave() {
  deviceTooltip.visible = false
}

function onDeviceClick(devId) {
  if (selectedDevice.value === devId) {
    selectedDevice.value = null
  } else {
    selectedDevice.value = devId
  }
}

function formatP2PLinkInfo(link) {
  const portAStr = link.portA || 'N/A'
  const portBStr = link.portB || 'N/A'
  const srcDieStr = link.srcDieId !== undefined ? ` Die${link.srcDieId}` : ''
  const dstDieStr = link.dstDieId !== undefined ? ` Die${link.dstDieId}` : ''
  return `P2P 直连\nDevice ${link.from}${srcDieStr} Port: ${portAStr}\nDevice ${link.to}${dstDieStr} Port: ${portBStr}`
}

function formatP2NetLinkInfo(devId) {
  const config = props.serverTopoConfig
  const pinMapEntries = config?.pinMapEntries || []
  const p2nPorts = []
  for (const entry of pinMapEntries) {
    for (let i = 0; i < entry.pinMap.length; i++) {
      if (entry.pinMap[i] === 2) {
        p2nPorts.push(`${entry.dieId}/${i}`)
      }
    }
  }
  const portStr = p2nPorts.length > 0 ? p2nPorts.join(', ') : 'N/A'
  const pg = layer0PortGroup.value
  const pgStr = pg ? pg.ports.join(', ') : ''
  return `P2N 连交换机\nDevice ${devId} Ports: ${portStr}${pgStr ? '\nPortGroup: ' + pgStr : ''}`
}

function zoomIn() {
  svgScale.value = Math.min(svgScale.value + SCALE_STEP, SCALE_MAX)
}

function zoomOut() {
  svgScale.value = Math.max(svgScale.value - SCALE_STEP, SCALE_MIN)
}

function resetZoom() {
  svgScale.value = 1
}

const serverViewBox = computed(() => {
  const cw = svgContentWidth.value
  const ch = svgContentHeight.value
  if (cw === 0 || ch === 0) return '0 0 470 240'
  const s = svgScale.value
  const vw = cw / s
  const vh = ch / s
  const vx = (cw - vw) / 2
  const vy = (ch - vh) / 2
  return `${vx} ${vy} ${vw} ${vh}`
})

function onCanvasWheel(event) {
  event.preventDefault()
  event.stopPropagation()
  const delta = event.deltaY > 0 ? -SCALE_STEP : SCALE_STEP
  svgScale.value = Math.max(SCALE_MIN, Math.min(SCALE_MAX, svgScale.value + delta))
}

let isDragging = false

function onSplitterMouseDown(event) {
  event.preventDefault()
  event.stopPropagation()
  isDragging = true
  document.addEventListener('mousemove', onSplitterMouseMove)
  document.addEventListener('mouseup', onSplitterMouseUp)
  document.body.style.cursor = 'row-resize'
  document.body.style.userSelect = 'none'
}

function onSplitterMouseMove(event) {
  if (!isDragging || !serverViewRef.value) return
  event.preventDefault()
  const bodyEl = serverViewRef.value.querySelector('.topo-server-view__body')
  if (!bodyEl) return
  const rect = bodyEl.getBoundingClientRect()
  const mouseY = event.clientY - rect.top
  const percent = Math.max(20, Math.min(85, (mouseY / rect.height) * 100))
  topoHeightPercent.value = percent
}

function onSplitterMouseUp() {
  isDragging = false
  document.removeEventListener('mousemove', onSplitterMouseMove)
  document.removeEventListener('mouseup', onSplitterMouseUp)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

onMounted(() => {
  if (canvasWrapRef.value) {
    canvasWrapRef.value.addEventListener('wheel', onCanvasWheel, { passive: false })
  }
})

onUnmounted(() => {
  if (canvasWrapRef.value) {
    canvasWrapRef.value.removeEventListener('wheel', onCanvasWheel)
  }
  document.removeEventListener('mousemove', onSplitterMouseMove)
  document.removeEventListener('mouseup', onSplitterMouseUp)
})

watch(() => props.serverTopoConfig, () => {
  svgScale.value = 1
  selectedDevice.value = null
  topoHeightPercent.value = 70
})
</script>

<template>
  <div class="topo-server-view" v-if="serverTopoConfig" ref="serverViewRef">
    <div class="topo-server-view__header">
      <TopoIconSvg type="server" :size="20" />
      <span>Server {{ serverId }}</span>
      <span class="topo-server-view__ip">Host: {{ hostIp }}</span>
      <div class="topo-server-view__zoom">
        <button class="topo-server-view__zoom-btn" @click="zoomOut" title="缩小">−</button>
        <span class="topo-server-view__zoom-label">{{ Math.round(svgScale * 100) }}%</span>
        <button class="topo-server-view__zoom-btn" @click="zoomIn" title="放大">+</button>
        <button class="topo-server-view__zoom-btn" @click="resetZoom" title="重置">↺</button>
      </div>
    </div>

    <div class="topo-server-view__body">
      <div
        class="topo-server-view__canvas-wrap"
        ref="canvasWrapRef"
        :style="selectedDevicePortMap ? { flex: `0 0 ${topoHeightPercent}%`, minHeight: '120px' } : {}"
      >
        <svg
          class="topo-server-view__canvas"
          :viewBox="serverViewBox"
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid meet"
        >
          <path
            v-for="link in p2pLinks"
            :key="`p2p-${link.from}-${link.srcDieId}-${link.to}-${link.dstDieId}`"
            :d="getP2PPath(link)"
            fill="none"
            stroke="rgba(103, 217, 255, 0.35)"
            stroke-width="2"
            class="topo-server-view__link-line"
            @click.stop="showLinkInfo($event, formatP2PLinkInfo(link))"
            @mouseenter="showLinkInfo($event, formatP2PLinkInfo(link))"
            @mouseleave="hideLinkInfo"
          />

          <template v-if="p2netDevices.length > 0 && switchPosition">
            <line
              v-for="devId in p2netDevices"
              :key="`p2net-${devId}`"
              :x1="devicePositions.devices[devId]?.x"
              :y1="devicePositions.devices[devId]?.y"
              :x2="switchPosition.x"
              :y2="switchPosition.y"
              stroke="rgba(255, 169, 64, 0.4)"
              stroke-width="2"
              stroke-dasharray="4 3"
              class="topo-server-view__link-line"
              @click.stop="showLinkInfo($event, formatP2NetLinkInfo(devId))"
              @mouseenter="showLinkInfo($event, formatP2NetLinkInfo(devId))"
              @mouseleave="hideLinkInfo"
            />

            <g :transform="`translate(${switchPosition.x - 24}, ${switchPosition.y - 14})`">
              <rect x="0" y="0" width="48" height="28" rx="5" fill="rgba(255, 169, 64, 0.12)" stroke="rgba(255, 169, 64, 0.6)" stroke-width="1.2" />
              <text x="24" y="17" text-anchor="middle" fill="rgba(255, 169, 64, 0.9)" font-size="10" font-weight="500">SW</text>
            </g>
          </template>

          <g
            v-for="dev in devicePositions.devices"
            :key="`dev-${dev.id}`"
            :transform="`translate(${dev.x - dev.npuW / 2}, ${dev.y - dev.npuH / 2})`"
            class="topo-server-view__device"
            :class="{ 'is-selected': selectedDevice === dev.id }"
            @mouseenter="onDeviceMouseEnter($event, dev.id)"
            @mouseleave="onDeviceMouseLeave"
            @click.stop="onDeviceClick(dev.id)"
          >
            <template v-if="dev.dieNum < 2">
              <rect x="0" y="0" :width="dev.npuW" :height="dev.npuH" rx="6" fill="rgba(103, 217, 255, 0.08)" stroke="rgba(103, 217, 255, 0.5)" stroke-width="1.2" class="topo-server-view__device-rect" />
              <text :x="dev.npuW / 2" y="18" text-anchor="middle" fill="rgba(103, 217, 255, 0.9)" font-size="9" font-weight="600">NPU</text>
              <text :x="dev.npuW / 2" y="33" text-anchor="middle" fill="rgba(255, 255, 255, 0.85)" font-size="12" font-weight="500">{{ dev.id }}</text>
            </template>
            <template v-else>
              <rect x="0" y="0" :width="dev.npuW" :height="dev.npuH" rx="6" fill="rgba(103, 217, 255, 0.05)" stroke="rgba(103, 217, 255, 0.5)" stroke-width="1.2" class="topo-server-view__device-rect" />
              <text :x="dev.npuW / 2" :y="DIE_TOP_OFFSET - 3" text-anchor="middle" fill="rgba(103, 217, 255, 0.9)" font-size="8" font-weight="600">NPU {{ dev.id }}</text>
              <g v-for="die in dev.dies" :key="`die-${dev.id}-${die.dieId}`">
                <rect :x="die.localX" :y="die.localY" :width="die.w" :height="die.h" rx="3" fill="rgba(103, 217, 255, 0.08)" stroke="rgba(103, 217, 255, 0.3)" stroke-width="0.8" class="topo-server-view__die-rect" />
                <text :x="die.localX + die.w / 2" :y="die.localY + die.h / 2 + 3" text-anchor="middle" fill="rgba(255, 255, 255, 0.75)" font-size="8" font-weight="500">D{{ die.dieId }}</text>
              </g>
            </template>
          </g>
        </svg>
      </div>

      <div
        v-if="selectedDevicePortMap"
        class="topo-server-view__splitter-bar"
        ref="splitterRef"
        @mousedown="onSplitterMouseDown"
      >
        <span class="topo-server-view__splitter-handle"></span>
      </div>

      <div
        v-if="selectedDevicePortMap"
        class="topo-server-view__portmap-wrap"
      >
        <div class="topo-server-view__portmap">
          <div class="topo-server-view__portmap-header">
            <span class="topo-server-view__portmap-title">Device {{ selectedDevicePortMap.deviceId }} 端口分配表</span>
            <span class="topo-server-view__portmap-meta">
              SP{{ selectedDevicePortMap.superPodId }} / Server {{ selectedDevicePortMap.serverId }} / Device {{ selectedDevicePortMap.deviceId }}
            </span>
            <button class="topo-server-view__portmap-close" @click="selectedDevice = null">✕</button>
          </div>
          <div class="topo-server-view__portmap-legend">
            <span class="topo-server-view__portmap-legend-item" v-for="t in [1, 2, 3]" :key="t">
              <span class="topo-server-view__portmap-legend-dot" :style="{ background: typeColorMap[t] }"></span>
              {{ typeLabelMap[t] }}
            </span>
          </div>
          <table v-if="selectedDevicePortMap.rows.length > 0" class="topo-server-view__portmap-grid">
            <thead>
              <tr>
                <th class="topo-server-view__portmap-corner">Die \ Port</th>
                <th v-for="pid in selectedDevicePortMap.portIds" :key="pid" class="topo-server-view__portmap-port-header">{{ pid }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in selectedDevicePortMap.rows" :key="row.dieId">
                <td class="topo-server-view__portmap-die-label">Die {{ row.dieId }}</td>
                <td
                  v-for="cell in row.cells"
                  :key="cell.portIdx"
                  class="topo-server-view__portmap-cell"
                  :style="{ color: cell.color, backgroundColor: cell.bg }"
                  :title="`Die${row.dieId}/Port${cell.portIdx}: ${cell.label}`"
                >
                  {{ cell.label }}
                </td>
              </tr>
            </tbody>
          </table>
          <div v-if="selectedDevicePortMap.devicePortGroups.length > 0" class="topo-server-view__portmap-portgroups">
            <div class="topo-server-view__portmap-pg-title">Port Groups</div>
            <div
              v-for="(pg, pgIdx) in selectedDevicePortMap.devicePortGroups"
              :key="pgIdx"
              class="topo-server-view__portmap-portgroup"
            >
              <span class="topo-server-view__portmap-pg-label">L{{ pg.layer }}</span>
              <span class="topo-server-view__portmap-pg-type" :class="`topo-server-view__portmap-pg-type--${pg.type.toLowerCase()}`">{{ pg.type }}</span>
              <span class="topo-server-view__portmap-pg-ports">{{ pg.ports.join(', ') }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <Teleport to="body">
      <div
        v-if="linkTooltip.visible"
        class="topo-link-tooltip"
        :style="{ left: linkTooltip.x + 12 + 'px', top: linkTooltip.y + 12 + 'px' }"
      >
        <pre class="topo-link-tooltip__content">{{ linkTooltip.content }}</pre>
      </div>
      <div
        v-if="deviceTooltip.visible"
        class="topo-link-tooltip"
        :style="{ left: deviceTooltip.x + 12 + 'px', top: deviceTooltip.y + 12 + 'px' }"
      >
        <pre class="topo-link-tooltip__content">{{ deviceTooltip.content }}</pre>
      </div>
    </Teleport>
  </div>
</template>
