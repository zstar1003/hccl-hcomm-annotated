/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

const DEFAULT_API_ROOT = ''

function joinApiPath(apiRoot, path) {
  const normalizedRoot = apiRoot.endsWith('/') ? apiRoot.slice(0, -1) : apiRoot
  return `${normalizedRoot}${path}`
}

async function requestJson(path, { apiRoot = DEFAULT_API_ROOT } = {}) {
  const response = await fetch(joinApiPath(apiRoot, path))
  const text = await response.text()
  const payload = text ? JSON.parse(text) : {}
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`)
  }
  return payload
}

export async function fetchTopologyList(options) {
  const payload = await requestJson('/api/topology/list', options)
  return payload.topologies || []
}

export async function fetchClusterTopology(topoName, options) {
  return requestJson(`/api/topology/cluster/${encodeURIComponent(topoName)}`, options)
}

export function parseClusterTopology(raw) {
  const superPods = []
  const spEntries = raw.super_pods || {}

  for (const [spIdStr, servers] of Object.entries(spEntries)) {
    const spId = Number(spIdStr)
    const parsedServers = (servers || []).map((srv) => ({
      id: srv.id,
      serverTopo: srv.server_topo || '',
      socVersion: srv.soc_version || '',
    }))

    const serverTopoKey = parsedServers.length > 0 ? parsedServers[0].serverTopo : ''
    const serverTopoConfig = (raw.server_topo_configs || {})[serverTopoKey] || null

    superPods.push({
      id: spId,
      servers: parsedServers,
      serverTopoConfig,
    })
  }

  superPods.sort((a, b) => a.id - b.id)

  return {
    name: raw.name || '',
    description: raw.description || '',
    superNodeNum: raw.super_node_num || 0,
    serverNum: raw.server_num || 0,
    isNewFormat: raw.is_new_format || false,
    superPods,
  }
}

export function parseServerTopoConfig(config) {
  if (!config) return null

  const deviceNum = config.device_num || 0
  const socVersion = config.soc_version || ''
  const hardwareType = config.hardware_type || ''

  const pinMapEntries = (config.device_ports_allocate_map || []).map((entry) => ({
    dieId: entry.die_id,
    pinMap: entry.pin_map || [],
  }))

  const portGroups = (config.port_group || []).map((pg) => ({
    layer: pg.layer,
    ports: pg.ports || [],
  }))

  const links = parseLinks(config.links || [])

  return {
    deviceNum,
    socVersion,
    hardwareType,
    pinMapEntries,
    portGroups,
    links,
  }
}

function normalizeDevicesRange(conn) {
  if (conn.devices_range && Array.isArray(conn.devices_range) && conn.devices_range.length >= 2) {
    const start = conn.devices_range[0]
    const end = conn.devices_range[1]
    return [start, end]
  }
  if (conn.devices && Array.isArray(conn.devices) && conn.devices.length >= 2) {
    return [conn.devices[0], conn.devices[conn.devices.length - 1]]
  }
  return []
}

function parseLinks(rawLinks) {
  const result = { fullmesh: [], enum: { deviceToDevice: [], deviceToSwitch: [] } }

  for (const link of rawLinks) {
    const mode = link.link_mode || ''
    if (mode === 'fullmesh') {
      const conns = (link.connections || []).map((c) => ({
        dieId: c.die_id,
        devicesRange: normalizeDevicesRange(c),
        devices: c.devices || null,
      }))
      result.fullmesh.push(...conns)
    } else if (mode === 'enum') {
      if (link.device_to_device_links) {
        const d2d = (link.device_to_device_links || []).map((c) => {
          const srcDieId = c.src_die_id !== undefined ? c.src_die_id : c.die_id
          const dstDieId = c.dst_die_id !== undefined ? c.dst_die_id : srcDieId
          const entry = {
            srcDieId,
            dstDieId,
            devicesRange: normalizeDevicesRange(c),
            srcLocalId: c.src_local_id || null,
            dstLocalId: c.dst_local_id || null,
            srcLocalIdRange: c.src_local_id_range || null,
            dstLocalIdRange: c.dst_local_id_range || null,
          }
          return entry
        })
        result.enum.deviceToDevice.push(...d2d)
      }
      if (link.device_to_switch_links) {
        const d2s = (link.device_to_switch_links || []).map((c) => ({
          dieId: c.die_id,
          devicesRange: normalizeDevicesRange(c),
        }))
        result.enum.deviceToSwitch.push(...d2s)
      }
    }
  }

  return result
}

export function getPinMapType(pinMap) {
  const typeMap = { 0: 'unused', 1: 'p2p', 2: 'p2net', 3: 'd2h' }
  return pinMap.map((v) => typeMap[v] || 'unknown')
}

export function getPortGroupType(portGroup, pinMapEntries) {
  if (!portGroup.ports || portGroup.ports.length === 0) return 'unknown'
  const firstPort = portGroup.ports[0]
  const parts = firstPort.split('/')
  if (parts.length < 2) return 'unknown'
  const dieId = Number(parts[0])
  const portId = Number(parts[1])
  const entry = pinMapEntries.find((e) => e.dieId === dieId)
  if (!entry || portId >= entry.pinMap.length) return 'unknown'
  const typeMap = { 0: 'unused', 1: 'p2p', 2: 'p2net', 3: 'd2h' }
  return typeMap[entry.pinMap[portId]] || 'unknown'
}
