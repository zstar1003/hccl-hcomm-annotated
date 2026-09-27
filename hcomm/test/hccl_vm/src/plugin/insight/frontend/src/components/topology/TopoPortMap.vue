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
import { computed } from 'vue'

const props = defineProps({
  serverTopoConfig: {
    type: Object,
    default: null,
  },
})

const typeLabelMap = {
  unused: '',
  p2p: 'P2P',
  p2net: 'P2N',
  d2h: 'H2D',
  unknown: '?',
}

const typeColorMap = {
  unused: '#555',
  p2p: '#67d9ff',
  p2net: '#ffa940',
  d2h: '#95de64',
  unknown: '#888',
}

const typeBgMap = {
  unused: 'transparent',
  p2p: 'rgba(103, 217, 255, 0.08)',
  p2net: 'rgba(255, 169, 64, 0.08)',
  d2h: 'rgba(149, 222, 100, 0.08)',
  unknown: 'transparent',
}

const gridData = computed(() => {
  const config = props.serverTopoConfig
  if (!config || !config.pinMapEntries) return null

  const entries = config.pinMapEntries
  if (entries.length === 0) return null

  const maxPortCount = Math.max(...entries.map((e) => (e.pinMap || []).length))
  const portIds = []
  for (let i = 0; i < maxPortCount; i++) {
    portIds.push(i)
  }

  const rows = entries.map((entry) => {
    const dieId = entry.dieId
    const pinMap = entry.pinMap || []
    const cells = []
    for (let portIdx = 0; portIdx < maxPortCount; portIdx++) {
      const val = pinMap[portIdx]
      const typeMap = { 0: 'unused', 1: 'p2p', 2: 'p2net', 3: 'd2h' }
      const type = typeMap[val] || 'unknown'
      cells.push({
        portIdx,
        type,
        label: typeLabelMap[type],
        color: typeColorMap[type],
        bg: typeBgMap[type],
      })
    }
    return { dieId, cells }
  })

  return { portIds, rows, maxPortCount }
})

const portGroupData = computed(() => {
  const config = props.serverTopoConfig
  if (!config || !config.portGroups) return []

  return config.portGroups.map((pg) => ({
    layer: pg.layer,
    ports: pg.ports,
    type: getPortGroupTypeLabel(pg),
    portCount: pg.ports?.length || 0,
  }))
})

function getPortGroupTypeLabel(pg) {
  if (!pg.ports || pg.ports.length === 0) return 'unknown'
  const config = props.serverTopoConfig
  if (!config || !config.pinMapEntries) return 'unknown'

  const firstPort = pg.ports[0]
  if (firstPort === 'd2h') return 'H2D'

  const parts = firstPort.split('/')
  if (parts.length < 2) return 'unknown'
  const dieId = Number(parts[0])
  const portId = Number(parts[1])
  const entry = config.pinMapEntries.find((e) => e.dieId === dieId)
  if (!entry || portId >= entry.pinMap.length) return 'unknown'
  const typeMap = { 0: 'unused', 1: 'P2P', 2: 'P2N', 3: 'H2D' }
  return typeMap[entry.pinMap[portId]] || 'unknown'
}

function getTypeColor(type) {
  return typeColorMap[type] || typeColorMap.unknown
}
</script>

<template>
  <div class="topo-port-map" v-if="serverTopoConfig">
    <div class="topo-port-map__header">
      <h4>NPU 端口分配表</h4>
      <span class="topo-port-map__soc">{{ serverTopoConfig.socVersion }}</span>
    </div>

    <div class="topo-port-map__legend">
      <span class="topo-port-map__legend-item" v-for="t in ['p2p', 'p2net', 'd2h']" :key="t">
        <span class="topo-port-map__legend-dot" :style="{ background: typeColorMap[t] }"></span>
        {{ typeLabelMap[t] }}
      </span>
    </div>

    <div class="topo-port-map__section" v-if="gridData">
      <table class="topo-port-map__grid">
        <thead>
          <tr>
            <th class="topo-port-map__corner">Die \ Port</th>
            <th v-for="pid in gridData.portIds" :key="pid" class="topo-port-map__port-header">
              {{ pid }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in gridData.rows" :key="row.dieId">
            <td class="topo-port-map__die-label">Die {{ row.dieId }}</td>
            <td
              v-for="cell in row.cells"
              :key="cell.portIdx"
              class="topo-port-map__cell"
              :class="`topo-port-map__cell--${cell.type}`"
              :style="{ color: cell.color, backgroundColor: cell.bg }"
              :title="`Die${row.dieId}/Port${cell.portIdx}: ${cell.label}`"
            >
              {{ cell.label }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="topo-port-map__section">
      <h5>Port Group</h5>
      <table class="topo-port-map__table topo-port-map__table--groups">
        <thead>
          <tr>
            <th>Layer</th>
            <th>Type</th>
            <th>Ports</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="pg in portGroupData" :key="pg.layer">
            <td class="topo-port-map__die-label">L{{ pg.layer }}</td>
            <td
              class="topo-port-map__cell"
              :style="{ color: getTypeColor(pg.type.toLowerCase() === 'p2p' ? 'p2p' : pg.type.toLowerCase() === 'p2n' ? 'p2net' : pg.type.toLowerCase() === 'h2d' ? 'd2h' : 'unknown') }"
            >
              {{ pg.type }}
            </td>
            <td class="topo-port-map__ports">{{ pg.ports.join(', ') }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
