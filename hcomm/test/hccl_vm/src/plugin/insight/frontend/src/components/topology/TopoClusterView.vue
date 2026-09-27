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
import { computed, ref, reactive, onMounted, onUnmounted, watch } from 'vue'
import TopoIconSvg from './TopoIconSvg.vue'
import TopoServerView from './TopoServerView.vue'

const props = defineProps({
  clusterData: {
    type: Object,
    default: null,
  },
  serverTopoConfig: {
    type: Object,
    default: null,
  },
  topoChangeToken: {
    type: Number,
    default: 0,
  },
})

const selectedServer = ref(null)
const linkTooltip = reactive({ visible: false, x: 0, y: 0, content: '' })
const clusterScale = ref(1)
const clusterMainRef = ref(null)

const CLUSTER_SCALE_MIN = 0.2
const CLUSTER_SCALE_MAX = 3
const CLUSTER_SCALE_STEP = 0.15

const SP_PADDING = 20
const SP_HEADER_H = 25
const L1_SW_H = 40
const L1_SW_GAP = 10
const SRV_W = 55
const SRV_H = 50
const SRV_GAP_X = 10
const SRV_GAP_Y = 12
const SRV_COLS = 4
const L2_SW_H = 40
const L2_GAP = 30
const SP_GAP = 40
const MARGIN = 30

const clusterLayout = computed(() => {
  const cd = props.clusterData
  if (!cd) return null

  const sps = cd.superPods || []
  const allServers = []
  const spGroups = []
  let globalIdx = 0

  for (const sp of sps) {
    const servers = (sp.servers || []).map((srv) => ({
      ...srv,
      superPodId: sp.id,
      globalIndex: globalIdx++,
    }))
    allServers.push(...servers)
    spGroups.push({
      id: sp.id,
      servers,
    })
  }

  const portGroups = props.serverTopoConfig?.portGroups || []
  const hasLayer0 = portGroups.some((pg) => pg.layer === 0)
  const hasLayer1 = portGroups.some((pg) => pg.layer === 1)
  const hasLayer2 = portGroups.some((pg) => pg.layer === 2)

  const layer1PortGroup = portGroups.find((pg) => pg.layer === 1) || null
  const layer2PortGroup = portGroups.find((pg) => pg.layer === 2) || null

  return {
    superPods: spGroups,
    allServers,
    hasLayer0,
    hasLayer1,
    hasLayer2,
    layer1PortGroup,
    layer2PortGroup,
    totalServers: allServers.length,
    totalSuperPods: sps.length,
  }
})

const svgLayout = computed(() => {
  const cl = clusterLayout.value
  if (!cl) return { width: 800, height: 600, spRects: [], l2Switch: null }

  const sps = cl.superPods
  const hasL1 = cl.hasLayer1
  const hasL2 = cl.hasLayer2

  let l2Switch = null
  if (hasL2) {
    l2Switch = { y: MARGIN, height: L2_SW_H }
  }

  const spRects = []

  for (let spIdx = 0; spIdx < sps.length; spIdx++) {
    const sp = sps[spIdx]
    const srvCount = sp.servers.length
    const srvCols = Math.min(srvCount, SRV_COLS)
    const srvRows = Math.ceil(srvCount / SRV_COLS)

    const srvAreaW = srvCols * SRV_W + (srvCols - 1) * SRV_GAP_X
    const spContentW = Math.max(srvAreaW, hasL1 ? 80 : 0) + SP_PADDING * 2
    const spW = Math.max(spContentW, 140)

    let contentH = SP_HEADER_H
    if (hasL1) {
      contentH += L1_SW_H + L1_SW_GAP
    }
    contentH += srvRows * SRV_H + (srvRows - 1) * SRV_GAP_Y
    const spH = contentH + SP_PADDING * 2

    spRects.push({
      spIdx,
      spId: sp.id,
      x: 0,
      y: 0,
      width: spW,
      height: spH,
      servers: sp.servers,
      srvCols,
      srvRows,
      l1SwitchY: 0,
      srvStartY: 0,
    })
  }

  for (let i = 0; i < spRects.length; i++) {
    const prevRight = i > 0 ? spRects[i - 1].x + spRects[i - 1].width : MARGIN
    spRects[i].x = prevRight + (i > 0 ? SP_GAP : 0)
  }

  const spStartY = hasL2 ? MARGIN + L2_SW_H + L2_GAP : MARGIN

  for (const rect of spRects) {
    rect.y = spStartY
    rect.l1SwitchY = rect.y + SP_PADDING + SP_HEADER_H
    rect.srvStartY = rect.l1SwitchY + (hasL1 ? L1_SW_H + L1_SW_GAP : 0)
  }

  const lastSp = spRects[spRects.length - 1]
  const svgW = (lastSp ? lastSp.x + lastSp.width : 0) + MARGIN
  const svgH = (lastSp ? lastSp.y + lastSp.height : 0) + MARGIN

  if (hasL2 && spRects.length > 0) {
    l2Switch.x = (spRects[0].x + lastSp.x + lastSp.width) / 2 - 60
  }

  return { width: svgW, height: svgH, spRects, l2Switch }
})

const selectedServerConfig = computed(() => {
  if (!selectedServer.value) return null
  return props.serverTopoConfig
})

function selectServer(server) {
  if (selectedServer.value?.globalIndex === server.globalIndex) {
    selectedServer.value = null
  } else {
    selectedServer.value = server
  }
}

function showLinkTooltip(event, content) {
  linkTooltip.visible = true
  linkTooltip.x = event.clientX
  linkTooltip.y = event.clientY
  linkTooltip.content = content
}

function hideLinkTooltip() {
  linkTooltip.visible = false
}

function formatPortGroup(pg) {
  if (!pg || !pg.ports) return ''
  return pg.ports.join(', ')
}

function getHostIp(superPodId, serverId) {
  return `${192 + superPodId}.${serverId + 1}.5.5`
}

function getSrvX(spRect, srvIdx) {
  const col = srvIdx % SRV_COLS
  const srvAreaW = spRect.srvCols * SRV_W + (spRect.srvCols - 1) * SRV_GAP_X
  const startX = spRect.x + (spRect.width - srvAreaW) / 2
  return startX + col * (SRV_W + SRV_GAP_X)
}

function getSrvY(spRect, srvIdx) {
  const row = Math.floor(srvIdx / SRV_COLS)
  return spRect.srvStartY + row * (SRV_H + SRV_GAP_Y)
}

function getSrvCenterX(spRect, srvIdx) {
  return getSrvX(spRect, srvIdx) + SRV_W / 2
}

function getL1SwitchCenterX(spRect) {
  return spRect.x + spRect.width / 2
}

function clusterZoomIn() {
  clusterScale.value = Math.min(clusterScale.value + CLUSTER_SCALE_STEP, CLUSTER_SCALE_MAX)
}

function clusterZoomOut() {
  clusterScale.value = Math.max(clusterScale.value - CLUSTER_SCALE_STEP, CLUSTER_SCALE_MIN)
}

function clusterResetZoom() {
  clusterScale.value = 1
}

const clusterViewBox = computed(() => {
  const layout = svgLayout.value
  if (!layout || layout.width === 0 || layout.height === 0) return '0 0 800 600'
  const s = clusterScale.value
  const vw = layout.width / s
  const vh = layout.height / s
  const vx = (layout.width - vw) / 2
  const vy = (layout.height - vh) / 2
  return `${vx} ${vy} ${vw} ${vh}`
})

function onClusterWheel(event) {
  event.preventDefault()
  event.stopPropagation()
  const delta = event.deltaY > 0 ? -CLUSTER_SCALE_STEP : CLUSTER_SCALE_STEP
  clusterScale.value = Math.max(CLUSTER_SCALE_MIN, Math.min(CLUSTER_SCALE_MAX, clusterScale.value + delta))
}

onMounted(() => {
  if (clusterMainRef.value) {
    clusterMainRef.value.addEventListener('wheel', onClusterWheel, { passive: false })
  }
})

onUnmounted(() => {
  if (clusterMainRef.value) {
    clusterMainRef.value.removeEventListener('wheel', onClusterWheel)
  }
})

watch(() => props.topoChangeToken, () => {
  selectedServer.value = null
  clusterScale.value = 1
})

const detailSize = ref(420)

function handleDetailResize(size) {
  if (typeof size === 'number') {
    detailSize.value = Math.max(300, Math.min(700, size))
  }
}
</script>

<template>
  <div class="topo-cluster-view" v-if="clusterLayout">
    <div class="topo-cluster-view__toolbar">
      <div class="topo-cluster-view__summary">
        <span class="topo-cluster-view__stat">
          <TopoIconSvg type="superpod" :size="16" />
          {{ clusterLayout.totalSuperPods }} SuperPods
        </span>
        <span class="topo-cluster-view__stat">
          <TopoIconSvg type="server" :size="16" />
          {{ clusterLayout.totalServers }} Servers
        </span>
        <span class="topo-cluster-view__stat">
          <TopoIconSvg type="npu" :size="16" />
          {{ clusterLayout.totalServers * (serverTopoConfig?.deviceNum || 0) }} NPUs
        </span>
      </div>
      <div class="topo-cluster-view__zoom">
        <button class="topo-cluster-view__zoom-btn" @click="clusterZoomOut" title="缩小">−</button>
        <span class="topo-cluster-view__zoom-label">{{ Math.round(clusterScale * 100) }}%</span>
        <button class="topo-cluster-view__zoom-btn" @click="clusterZoomIn" title="放大">+</button>
        <button class="topo-cluster-view__zoom-btn" @click="clusterResetZoom" title="重置">↺</button>
      </div>
    </div>

    <div class="topo-cluster-view__body">
      <el-splitter class="topo-cluster-splitter">
        <el-splitter-panel class="topo-cluster-splitter__main" min="300">
          <div class="topo-cluster-view__main" ref="clusterMainRef">
              <svg
                class="topo-cluster-view__clos-svg"
                :viewBox="clusterViewBox"
                width="100%"
                height="100%"
                preserveAspectRatio="xMidYMid meet"
              >
              <template v-if="svgLayout.l2Switch">
                <g class="topo-cluster-view__l2-group">
                  <rect
                    :x="svgLayout.l2Switch.x" :y="svgLayout.l2Switch.y" width="120" :height="svgLayout.l2Switch.height" rx="8"
                    fill="rgba(250, 173, 20, 0.08)" stroke="rgba(250, 173, 20, 0.5)" stroke-width="1.5"
                  />
                  <text :x="svgLayout.l2Switch.x + 60" :y="svgLayout.l2Switch.y + 25" text-anchor="middle" fill="rgba(250, 173, 20, 0.9)" font-size="13" font-weight="600">L2 SW</text>
                </g>
              </template>

              <template v-for="spRect in svgLayout.spRects" :key="`sp-${spRect.spId}`">
                <g class="topo-cluster-view__sp-group">
                  <rect
                    :x="spRect.x" :y="spRect.y" :width="spRect.width" :height="spRect.height" rx="10"
                    fill="rgba(64, 158, 255, 0.03)" stroke="rgba(64, 158, 255, 0.2)" stroke-width="1" stroke-dasharray="6 3"
                  />
                  <text :x="spRect.x + spRect.width / 2" :y="spRect.y + SP_PADDING + 16" text-anchor="middle" fill="rgba(103, 166, 255, 0.9)" font-size="12" font-weight="500">SuperPod {{ spRect.spId }}</text>

                  <template v-if="clusterLayout.hasLayer1">
                    <rect
                      :x="getL1SwitchCenterX(spRect) - 40" :y="spRect.l1SwitchY" width="80" :height="L1_SW_H" rx="6"
                      fill="rgba(250, 173, 20, 0.06)" stroke="rgba(250, 173, 20, 0.4)" stroke-width="1.2"
                    />
                    <text :x="getL1SwitchCenterX(spRect)" :y="spRect.l1SwitchY + 24" text-anchor="middle" fill="rgba(250, 173, 20, 0.85)" font-size="11" font-weight="500">L1 SW</text>
                  </template>

                  <g
                    v-for="(srv, srvIdx) in spRect.servers"
                    :key="`srv-${spRect.spId}-${srv.id}`"
                    class="topo-cluster-view__srv-node"
                    :class="{ 'is-selected': selectedServer?.globalIndex === srv.globalIndex }"
                    @click.stop="selectServer(srv)"
                  >
                    <rect
                      :x="getSrvX(spRect, srvIdx)" :y="getSrvY(spRect, srvIdx)" :width="SRV_W" :height="SRV_H" rx="6"
                      fill="rgba(103, 217, 255, 0.06)" stroke="rgba(103, 217, 255, 0.35)" stroke-width="1"
                      class="topo-cluster-view__srv-rect"
                    />
                    <text :x="getSrvCenterX(spRect, srvIdx)" :y="getSrvY(spRect, srvIdx) + 20" text-anchor="middle" fill="rgba(103, 217, 255, 0.85)" font-size="9" font-weight="500">Srv {{ srv.id }}</text>
                    <text :x="getSrvCenterX(spRect, srvIdx)" :y="getSrvY(spRect, srvIdx) + 36" text-anchor="middle" fill="rgba(255, 255, 255, 0.45)" font-size="7" font-family="monospace">{{ getHostIp(spRect.spId, srv.id) }}</text>

                    <template v-if="clusterLayout.hasLayer1">
                      <line
                        :x1="getSrvCenterX(spRect, srvIdx)" :y1="getSrvY(spRect, srvIdx)"
                        :x2="getL1SwitchCenterX(spRect)" :y2="spRect.l1SwitchY + L1_SW_H"
                        stroke="rgba(250, 173, 20, 0.3)" stroke-width="1.2" stroke-dasharray="4 3"
                        class="topo-cluster-view__link-line"
                        @click.stop="showLinkTooltip($event, `L1: Server ${srv.id} → L1 Switch (SP${spRect.spId})\nPortGroup: ${formatPortGroup(clusterLayout.layer1PortGroup)}`)"
                        @mouseenter="showLinkTooltip($event, `L1: Server ${srv.id} → L1 Switch\nPortGroup: ${formatPortGroup(clusterLayout.layer1PortGroup)}`)"
                        @mouseleave="hideLinkTooltip"
                      />
                    </template>
                  </g>

                  <template v-if="clusterLayout.hasLayer2 && svgLayout.l2Switch">
                    <line
                      :x1="getL1SwitchCenterX(spRect)" :y1="spRect.l1SwitchY"
                      :x2="svgLayout.l2Switch.x + 60" :y2="svgLayout.l2Switch.y + svgLayout.l2Switch.height"
                      stroke="rgba(250, 173, 20, 0.25)" stroke-width="1.5" stroke-dasharray="6 4"
                      class="topo-cluster-view__link-line"
                      @click.stop="showLinkTooltip($event, `L2: SuperPod ${spRect.spId} → L2 Switch\nPortGroup: ${formatPortGroup(clusterLayout.layer2PortGroup)}`)"
                      @mouseenter="showLinkTooltip($event, `L2: SuperPod ${spRect.spId} → L2 Switch\nPortGroup: ${formatPortGroup(clusterLayout.layer2PortGroup)}`)"
                      @mouseleave="hideLinkTooltip"
                    />
                  </template>
                </g>
              </template>
            </svg>
          </div>
        </el-splitter-panel>

        <el-splitter-panel
          v-if="selectedServer && selectedServerConfig"
          class="topo-cluster-splitter__detail"
          :size="detailSize"
          :min="300"
          @update:size="handleDetailResize"
        >
          <div class="topo-cluster-view__detail">
            <div class="topo-cluster-view__detail-header">
              <h4>Server {{ selectedServer.id }} (SP{{ selectedServer.superPodId }}) 内部拓扑</h4>
              <button class="topo-cluster-view__close-btn" @click="selectedServer = null">✕</button>
            </div>
            <TopoServerView
              :server-topo-config="selectedServerConfig"
              :server-id="selectedServer.id"
              :super-pod-id="selectedServer.superPodId"
            />
          </div>
        </el-splitter-panel>
      </el-splitter>
    </div>

    <Teleport to="body">
      <div
        v-if="linkTooltip.visible"
        class="topo-link-tooltip"
        :style="{ left: linkTooltip.x + 12 + 'px', top: linkTooltip.y + 12 + 'px' }"
      >
        <pre class="topo-link-tooltip__content">{{ linkTooltip.content }}</pre>
      </div>
    </Teleport>
  </div>
</template>
