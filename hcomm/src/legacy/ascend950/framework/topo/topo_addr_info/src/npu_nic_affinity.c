/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_nic_affinity.h"
#include "xml_parser.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/socket.h>

#include "hal.h"
#include "securec.h"
#include "topo_addr_info_log.h"
#include "topo_addr_info_perf.h"

/* ───────── 常量 ───────── */
#ifndef XML_PATH
#define XML_PATH "/var/run/ascend-topologyd/virtualTopology.xml"
#endif
#ifndef HCA_NET_PATH_TEMPLATE
#define HCA_NET_PATH_TEMPLATE "/sys/class/infiniband/%s/device/net"
#endif
#define MAX_PATH_LEN 512
#define MAX_GROUP_CNT 16
#define MAX_HCA_COUNT 64
#define MAX_NAME_LEN 64
#define MAX_IP_STR_LEN 48

/* ───────── 业务结构 ───────── */
typedef struct {
    int npuIds[MAX_NPU_COUNT];
    unsigned int npuCnt;
    unsigned int nicIdx[MAX_HCA_COUNT];
    unsigned int nicCnt;
} AffinityGroup;

typedef struct {
    char nicNames[MAX_HCA_COUNT][MAX_NAME_LEN];
    unsigned int nicCount;
    AffinityGroup groups[MAX_GROUP_CNT];
    unsigned int groupCount;
    bool affined[MAX_NPU_COUNT][MAX_HCA_COUNT]; /* groups 展开后的 NPU×NIC 亲和矩阵 */
} XmlInfo;

static bool TagIs(const TagEntry* e, const char* name) { return strcmp(e->tagName, name) == 0; }

/* ─── 硬件枚举 ─── */

static void BuildNpuBdfTable(char bdfs[MAX_NPU_COUNT][MAX_NAME_LEN])
{
    int npuCnt = hal_get_npu_count();
    if (npuCnt <= 0 || npuCnt > (int)MAX_NPU_COUNT) {
        TOPO_ERR("BuildNpuBdfTable: invalid npuCnt=%d", npuCnt);
        return;
    }
    for (int phyId = 0; phyId < npuCnt; phyId++) {
        struct dcmi_pcie_info_all pcieInfo;
        if (hal_get_device_pcie_info(phyId, &pcieInfo) == 0) {
            (void)sprintf_s(
                bdfs[phyId], MAX_NAME_LEN, "%04x:%02x:%02x.%x", pcieInfo.domain, pcieInfo.bdf_busid,
                pcieInfo.bdf_deviceid, pcieInfo.bdf_funcid);
        } else {
            bdfs[phyId][0] = '\0';
        }
    }
}

/* ─── 亲和分组构建 ─── */

static void TryAddNpuByBdf(
    const TagEntry* e, unsigned int curGroupIdx, const char npuBdfs[MAX_NPU_COUNT][MAX_NAME_LEN], XmlInfo* info)
{
    const char* busId = TagFindAttr(e, "busid");
    if (busId == NULL || busId[0] == '\0') {
        return;
    }
    if (curGroupIdx >= MAX_GROUP_CNT) {
        return;
    }

    /* 遍历所有 NPU 的 BDF 表，找与 busId 匹配的那个 NPU */
    int npuCount = hal_get_npu_count();
    unsigned int tableSize = (npuCount <= 0) ? 0 : (unsigned int)npuCount;
    if (tableSize > MAX_NPU_COUNT) {
        tableSize = MAX_NPU_COUNT;
    }
    for (unsigned int npuIdx = 0; npuIdx < tableSize; npuIdx++) {
        if (npuBdfs[npuIdx][0] == '\0') {
            continue;
        }
        if (strcmp(busId, npuBdfs[npuIdx]) != 0) {
            continue;
        }

        /* busId 匹配 → 将 NPU[npuIdx] 加入亲和组 */
        AffinityGroup* group = &info->groups[curGroupIdx];
        unsigned int groupCnt = group->npuCnt;
        if (groupCnt > MAX_NPU_COUNT) {
            groupCnt = MAX_NPU_COUNT;
        }

        /* 检查该 NPU 是否已在组内，避免重复占用槽位 */
        bool alreadyInGroup = false;
        for (unsigned int existIdx = 0; existIdx < groupCnt; existIdx++) {
            if (group->npuIds[existIdx] == (int)npuIdx) {
                alreadyInGroup = true;
                break;
            }
        }
        if (!alreadyInGroup && groupCnt < MAX_NPU_COUNT) {
            group->npuIds[groupCnt] = (int)npuIdx;
            group->npuCnt = groupCnt + 1;
        }
        break;
    }
}

/* ─── 分组上下文，降低标签处理器的参数传递 ─── */
typedef struct {
    const char (*npuBdfs)[MAX_NAME_LEN];
    XmlInfo* info;
    unsigned int* curGroupIdx;
    unsigned int* groupCount;
    unsigned int* nicCount;
    bool* inGroup;
    int* containerDepth;
} GroupCtx;

static void GroupEnterOrSkip(GroupCtx* ctx, int depth)
{
    if (*ctx->inGroup && depth <= *ctx->containerDepth) {
        *ctx->inGroup = false;
    }
    if (!*ctx->inGroup && *ctx->groupCount < MAX_GROUP_CNT) {
        *ctx->curGroupIdx = *ctx->groupCount;
        (*ctx->groupCount)++;
        *ctx->containerDepth = depth;
        *ctx->inGroup = true;
    }
}

static void HandlePciTag(const TagEntry* e, GroupCtx* ctx)
{
    if (!e->isSelfClose) {
        GroupEnterOrSkip(ctx, e->depth);
        if (*ctx->inGroup) {
            TryAddNpuByBdf(e, *ctx->curGroupIdx, ctx->npuBdfs, ctx->info);
        }
        return;
    }
    if (*ctx->inGroup) {
        TryAddNpuByBdf(e, *ctx->curGroupIdx, ctx->npuBdfs, ctx->info);
    }
}

static void HandleUbTag(const TagEntry* e, GroupCtx* ctx)
{
    if (!e->isSelfClose) {
        GroupEnterOrSkip(ctx, e->depth);
    }
}

static void HandleNpuTag(const TagEntry* e, GroupCtx* ctx)
{
    if (!*ctx->inGroup) {
        return;
    }
    const char* chipId = TagFindAttr(e, "chipphyid");
    if (chipId == NULL) {
        return;
    }
    int phyId = atoi(chipId);
    if (phyId < 0 || phyId >= (int)MAX_NPU_COUNT) {
        return;
    }
    /* 跳过当前进程不可见的设备 */
    int userDevId = -1;
    if (hal_get_userdevid_by_phyid(phyId, &userDevId) != 0) {
        return;
    }
    XmlInfo* info = ctx->info;
    if (info == NULL) {
        return;
    }

    unsigned int gIdx = *ctx->curGroupIdx;
    if (gIdx >= MAX_GROUP_CNT) {
        TOPO_ERR("HandleNpuTag: group index overflow, gIdx=%u >= MAX_GROUP_CNT=%d", gIdx, MAX_GROUP_CNT);
        return;
    }

    /* 去重：同组已有该 phyId 则跳过，防止重复占用 npuIds 槽位 */
    AffinityGroup* group = &info->groups[gIdx];
    unsigned int groupNpuCnt = group->npuCnt;
    if (groupNpuCnt > MAX_NPU_COUNT) {
        groupNpuCnt = MAX_NPU_COUNT;
    }
    for (unsigned int existIdx = 0; existIdx < groupNpuCnt; existIdx++) {
        if (group->npuIds[existIdx] == phyId) {
            return;
        }
    }
    if (groupNpuCnt < MAX_NPU_COUNT) {
        group->npuIds[groupNpuCnt] = phyId;
        group->npuCnt = groupNpuCnt + 1;
    }
}

/* 将 NIC 名去重加入 info，返回 TOPO_SUCCESS 并通过 nicIdx 输出索引 */
static TopoAddrResult DedupNetNic(XmlInfo* info, const char* name, unsigned int* nicCount, unsigned int* nicIdx)
{
    unsigned int curCnt = *nicCount;
    if (curCnt > MAX_HCA_COUNT) {
        curCnt = MAX_HCA_COUNT;
    }
    unsigned int idx = 0;
    for (; idx < curCnt; idx++) {
        if (strcmp(info->nicNames[idx], name) == 0) {
            *nicIdx = idx;
            return TOPO_SUCCESS;
        }
    }
    if (idx >= MAX_HCA_COUNT) {
        TOPO_ERR("DedupNetNic: NIC count overflow, name=%s", name);
        return TOPO_ERR_INTERNAL;
    }
    if (strcpy_s(info->nicNames[idx], sizeof(info->nicNames[0]), name) != 0) {
        return TOPO_ERR_INTERNAL;
    }
    (*nicCount)++;
    *nicIdx = idx;
    return TOPO_SUCCESS;
}

static void HandleNetTag(const TagEntry* e, GroupCtx* ctx)
{
    if (!*ctx->inGroup) {
        return;
    }
    const char* name = TagFindAttr(e, "name");
    if (name == NULL) {
        return;
    }
    XmlInfo* info = ctx->info;
    if (info == NULL) {
        return;
    }

    unsigned int nicIdx;
    if (DedupNetNic(info, name, ctx->nicCount, &nicIdx) != TOPO_SUCCESS) {
        return;
    }

    unsigned int gIdx = *ctx->curGroupIdx;
    if (gIdx >= MAX_GROUP_CNT) {
        TOPO_ERR("HandleNetTag: group index overflow, gIdx=%u >= MAX_GROUP_CNT=%d", gIdx, MAX_GROUP_CNT);
        return;
    }
    AffinityGroup* group = &info->groups[gIdx];
    /* 组内去重：同名 NIC 只加入一次，避免重复占用 nicIdx 槽位 */
    unsigned int groupNicCnt = group->nicCnt;
    if (groupNicCnt > MAX_HCA_COUNT) {
        groupNicCnt = MAX_HCA_COUNT;
    }
    for (unsigned int existIdx = 0; existIdx < groupNicCnt; existIdx++) {
        if (group->nicIdx[existIdx] == nicIdx) {
            return;
        }
    }
    if (groupNicCnt < MAX_HCA_COUNT) {
        group->nicIdx[groupNicCnt] = nicIdx;
        group->nicCnt = groupNicCnt + 1;
    }
}

/* 将 AffinityGroup 展开为 affined 二维矩阵，供 O(1) 亲和查询 */
static void BuildAffinityMatrix(XmlInfo* info)
{
    TOPO_PERF_BEGIN(BuildAffinityMatrix);
    unsigned int groupCnt = info->groupCount;
    if (groupCnt > MAX_GROUP_CNT) {
        groupCnt = MAX_GROUP_CNT;
    }

    for (unsigned int g = 0; g < groupCnt; g++) {
        unsigned int npuCnt = info->groups[g].npuCnt;
        if (npuCnt > MAX_NPU_COUNT) {
            npuCnt = MAX_NPU_COUNT;
        }
        unsigned int nicCnt = info->groups[g].nicCnt;
        if (nicCnt > MAX_HCA_COUNT) {
            nicCnt = MAX_HCA_COUNT;
        }
        for (unsigned int ni = 0; ni < npuCnt; ni++) {
            int npuId = info->groups[g].npuIds[ni];
            if (npuId < 0 || npuId >= (int)MAX_NPU_COUNT) {
                continue;
            }
            for (unsigned int nci = 0; nci < nicCnt; nci++) {
                unsigned int nicIdx = info->groups[g].nicIdx[nci];
                if (nicIdx < MAX_HCA_COUNT) {
                    info->affined[(unsigned int)npuId][nicIdx] = true;
                }
            }
        }
    }
    TOPO_PERF_END(BuildAffinityMatrix);
}

static TopoAddrResult BuildAffinityGroups(const TagEntry* tags, unsigned int tagCount, XmlInfo* info)
{
    char npuBdfs[MAX_NPU_COUNT][MAX_NAME_LEN] = {{0}};
    BuildNpuBdfTable(npuBdfs);
    (void)memset_s(info, sizeof(*info), 0, sizeof(*info));

    unsigned int curGroupIdx = 0;
    unsigned int groupCount = 0;
    unsigned int nicCount = 0;
    bool inGroup = false;
    int containerDepth = -1;

    GroupCtx ctx = {
        .npuBdfs = (const char(*)[MAX_NAME_LEN])npuBdfs,
        .info = info,
        .curGroupIdx = &curGroupIdx,
        .groupCount = &groupCount,
        .nicCount = &nicCount,
        .inGroup = &inGroup,
        .containerDepth = &containerDepth,
    };

    for (unsigned int i = 0; i < tagCount; i++) {
        const TagEntry* e = &tags[i];

        if (TagIs(e, "pci")) {
            HandlePciTag(e, &ctx);
            continue;
        }
        if (TagIs(e, "ub")) {
            HandleUbTag(e, &ctx);
            continue;
        }
        if (TagIs(e, "npu")) {
            HandleNpuTag(e, &ctx);
            continue;
        }
        if (TagIs(e, "net")) {
            HandleNetTag(e, &ctx);
            continue;
        }
    }

    info->groupCount = groupCount;
    info->nicCount = nicCount;

    BuildAffinityMatrix(info);

    if (nicCount == 0) {
        TOPO_ERR("no NICs found in XML, cannot build affinity groups");
        return TOPO_ERR_NOT_FOUND;
    }
    return TOPO_SUCCESS;
}

/* ─── 合成：ParseXml = ParseXmlTags + BuildAffinityGroups ─── */

static TopoAddrResult ParseXml(XmlInfo* info)
{
    TagEntry tags[MAX_TAG_ENTRIES];
    unsigned int tagCount = 0;
    TopoAddrResult ret = ParseXmlTags(XML_PATH, tags, &tagCount, MAX_TAG_ENTRIES);
    if (ret != TOPO_SUCCESS) {
        return ret;
    }
    return BuildAffinityGroups(tags, tagCount, info);
}

/* ─── 打印 NPU → 网卡名 → IP 分配结果 ─── */
static void LogAssignResult(
    const XmlInfo* info, int npuCount, const bool nicValid[MAX_HCA_COUNT], const char nicIps[][MAX_IP_STR_LEN],
    const char assignment[][MAX_IP_STR_LEN])
{
    for (int ni = 0; ni < npuCount; ni++) {
        if (assignment[ni][0] == '\0') {
            continue;
        }
        /* 反向查找该 IP 对应的 NIC 名 */
        const char* nicName = NULL;
        for (unsigned int j = 0; j < info->nicCount; j++) {
            if (nicValid[j] && strcmp(assignment[ni], nicIps[j]) == 0) {
                nicName = info->nicNames[j];
                break;
            }
        }
        TOPO_INFO("NPU %d → %s (%s)", ni, assignment[ni], nicName ? nicName : "?");
    }
}

/* ─── 轮询分发全量 NPU 的 RoCE IP ─── */

static TopoAddrResult DispatchIpsRoundRobin(
    int phyId, const XmlInfo* info, const bool nicValid[MAX_HCA_COUNT], const char nicIps[][MAX_IP_STR_LEN],
    char* outIp, size_t outLen)
{
    if (phyId < 0 || phyId >= (int)MAX_NPU_COUNT) {
        TOPO_ERR("DispatchIpsRoundRobin: invalid phyId=%d", phyId);
        return TOPO_ERR_PARA;
    }
    int npuCount = hal_get_npu_count();
    if (npuCount <= 0 || npuCount > (int)MAX_NPU_COUNT) {
        TOPO_ERR("DispatchIpsRoundRobin: invalid npuCount=%d", npuCount);
        return TOPO_ERR_INTERNAL;
    }
    unsigned int nicCount = info->nicCount;
    if (nicCount == 0 || nicCount > MAX_HCA_COUNT) {
        TOPO_ERR("DispatchIpsRoundRobin: invalid nicCount=%u", nicCount);
        return TOPO_ERR_INTERNAL;
    }

    char assignment[MAX_NPU_COUNT][MAX_IP_STR_LEN];
    (void)memset_s(assignment, sizeof(assignment), 0, sizeof(assignment));

    unsigned int cur = 0;
    for (int npuId = 0; npuId < npuCount; npuId++) {
        for (unsigned int j = cur; j < cur + nicCount; j++) {
            unsigned int nicIdx = j % nicCount;
            if (!nicValid[nicIdx] || !info->affined[npuId][nicIdx]) {
                continue;
            }
            if (strcpy_s(assignment[npuId], sizeof(assignment[0]), nicIps[nicIdx]) != 0) {
                continue;
            }
            cur = (j + 1) % nicCount;
            break;
        }
    }

    LogAssignResult(info, npuCount, nicValid, nicIps, assignment);

    if (assignment[phyId][0] != '\0') {
        return strcpy_s(outIp, outLen, assignment[phyId]);
    }
    TOPO_ERR("no IP assigned for phyId=%d", phyId);
    return TOPO_ERR_NOT_FOUND;
}

/* ─── 名称 → IP 转换 ─── */

/* 直接以 eth 名查 IP */
static TopoAddrResult EthToIp(const char* eth, char* ip, size_t ipLen)
{
    struct ifaddrs* ifaddr = NULL;
    if (getifaddrs(&ifaddr) == -1) {
        return TOPO_ERR_SYSCALL;
    }

    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_name == NULL) {
            continue;
        }
        if (strcmp(ifa->ifa_name, eth) != 0) {
            continue;
        }
        if (ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        struct sockaddr_in* sin = (struct sockaddr_in*)ifa->ifa_addr;
        if (inet_ntop(AF_INET, &sin->sin_addr, ip, (socklen_t)ipLen) != NULL) {
            freeifaddrs(ifaddr);
            return TOPO_SUCCESS;
        }
    }
    freeifaddrs(ifaddr);
    return TOPO_ERR_NOT_FOUND;
}

/* 以 HCA 名查 IP：/sys/class/infiniband/<hca>/device/net/<eth> */
static TopoAddrResult HcaToIp(const char* hca, char* ip, size_t ipLen)
{
    char netPath[MAX_PATH_LEN];
    if (sprintf_s(netPath, sizeof(netPath), HCA_NET_PATH_TEMPLATE, hca) < 0) {
        return TOPO_ERR_INTERNAL;
    }

    DIR* dir = opendir(netPath);
    if (dir == NULL) {
        return TOPO_ERR_NOT_FOUND;
    }

    char eth[MAX_NAME_LEN] = {0};
    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        (void)strncpy_s(eth, sizeof(eth), entry->d_name, sizeof(eth) - 1);
        break;
    }
    closedir(dir);

    if (eth[0] == '\0') {
        return TOPO_ERR_NOT_FOUND;
    }

    return EthToIp(eth, ip, ipLen);
}

/* name 可能是 eth 名或 HCA 名：先尝试 eth，再尝试 HCA */
static TopoAddrResult NameToIp(const char* name, char* ip, size_t ipLen)
{
    if (EthToIp(name, ip, ipLen) == TOPO_SUCCESS) {
        return TOPO_SUCCESS;
    }
    TopoAddrResult ret = HcaToIp(name, ip, ipLen);
    if (ret != TOPO_SUCCESS) {
        TOPO_ERR("NameToIp: cannot resolve IP for %s (eth+HCA)", name);
    }
    return ret;
}

/* ─── 打印 NPU-NIC 亲和关系（NIC 名 + IP） ─── */

static void LogAffinityInfo(const XmlInfo* info, const char nicIps[][MAX_IP_STR_LEN])
{
    unsigned int nicCnt = info->nicCount;
    if (nicCnt > MAX_HCA_COUNT) {
        nicCnt = MAX_HCA_COUNT;
    }
    int npuCount = hal_get_npu_count();
    if (npuCount <= 0 || npuCount > (int)MAX_NPU_COUNT) {
        return;
    }
    for (int npuId = 0; npuId < npuCount; npuId++) {
        for (unsigned int nicIdx = 0; nicIdx < nicCnt; nicIdx++) {
            if (info->affined[npuId][nicIdx]) {
                TOPO_INFO(
                    "[affinity] NPU%d <- %s(%s)", npuId, info->nicNames[nicIdx],
                    (nicIps[nicIdx][0] != '\0') ? nicIps[nicIdx] : "?");
            }
        }
    }
}

/* ─── 预解析全部 NIC 的 IP → 调用轮询分发 → 直接出 IP ─── */

static TopoAddrResult SelectNpuRoceIp(int npuId, const XmlInfo* info, char* ip, size_t ipLen)
{
    unsigned int nicCnt = info->nicCount;
    if (nicCnt > MAX_HCA_COUNT) {
        nicCnt = MAX_HCA_COUNT;
    }

    char nicIps[MAX_HCA_COUNT][MAX_IP_STR_LEN] = {{0}};
    bool nicValid[MAX_HCA_COUNT] = {false};
    for (unsigned int i = 0; i < nicCnt; i++) {
        if (NameToIp(info->nicNames[i], nicIps[i], sizeof(nicIps[i])) == TOPO_SUCCESS) {
            nicValid[i] = true;
        }
    }
    LogAffinityInfo(info, nicIps);
    return DispatchIpsRoundRobin(npuId, info, nicValid, nicIps, ip, ipLen);
}

/* ─── 对外接口 ─── */

TopoAddrResult GetRoceIpFromXml(int npuId, char* ip, size_t ipLen)
{
    if (ip == NULL || ipLen == 0 || npuId < 0) {
        TOPO_ERR("GetRoceIpFromXml: invalid params (ip=%p, ipLen=%zu, npuId=%d)", ip, ipLen, npuId);
        return TOPO_ERR_PARA;
    }

    /* 解析 XML，构建 NPU-NIC 亲和分组 */
    XmlInfo info;
    (void)memset_s(&info, sizeof(info), 0, sizeof(info));
    TopoAddrResult ret = ParseXml(&info);
    if (ret != TOPO_SUCCESS) {
        return ret;
    }

    /* XML 中无 NIC 定义 */
    if (info.nicCount == 0) {
        TOPO_ERR("GetRoceIpFromXml: no NICs in XML for npuId=%d", npuId);
        return TOPO_ERR_NOT_FOUND;
    }

    /* 从所有 NIC 中轮询选出当前 NPU 的 RoCE IP */
    return SelectNpuRoceIp(npuId, &info, ip, ipLen);
}
