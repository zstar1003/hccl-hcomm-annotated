/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "COMM_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "sim_ip_address.h"
#include "runtime/base.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
extern HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
extern HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
extern HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm);
extern HcclResult
HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);
extern HcclResult HcclCommDestroy(HcclComm comm);

void SimimSetThreadName(const std::string& threadStr)
{
    // 线程名应限制在15个字符内，防止被截断
    s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
    if (sRet != 0) {
        HCCL_VM_WARN("err[{}] link[{}] threadNameSet failed.", sRet, threadStr.c_str());
    }
}

HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo)
{
    (void)rootInfo;
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm)
{
    (void)nRanks;
    (void)rootInfo;
    // 解析ranktable文件，初始化通信域相关数据库表项
    const char* clusterInfo = std::getenv("RANK_TABLE_FILE");
    if (!clusterInfo) {
        HCCL_VM_ERROR("RANK_TABLE_FILE env not set, please check your config.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclCommInitClusterInfo(clusterInfo, rank, comm);
}

HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    HCCL_VM_INFO("xxxxxxxxxx");
    (void)nRanks;
    (void)rootInfo;
    const char* clusterInfo = getenv("RANK_TABLE_FILE");
    HcclCommConfig* cfg = const_cast<HcclCommConfig*>(config);
    return HcclCommInitClusterInfoConfig(clusterInfo, rank, cfg, comm);
}

HcclResult SimGetDeviceComm(uint32_t ndev, const uint32_t rank, const uint32_t logicDeviceId, HcclComm& comm)
{
    HCCL_VM_INFO("rank[{}] Get device comm...", rank);
    // 给当前线程添加名字
    SimimSetThreadName("Hccl_GetDevComm");

    if (aclrtSetDevice(logicDeviceId) != ACL_SUCCESS) {
        HCCL_VM_ERROR("set fail logicDeviceId[{}].", logicDeviceId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const char* clusterInfo = std::getenv("RANK_TABLE_FILE");
    if (!clusterInfo) {
        HCCL_VM_ERROR("RANK_TABLE_FILE env not set, please check your config.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto ret = HcclCommInitClusterInfo(clusterInfo, rank, &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_VM_ERROR("rank[{}] Get device comm failed!", rank);
        if (aclrtResetDevice(logicDeviceId) != ACL_SUCCESS) {
            HCCL_VM_ERROR("reset fail logicDeviceId[{}].", logicDeviceId);
            return ret;
        }
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    HCCL_VM_INFO("xxxxxxxxxx");
    HCCL_VM_INFO("Init all comm...");
    SimimSetThreadName("Hccl_GetCommAll");

    if (aclrtSetDevice(devices[0]) != ACL_SUCCESS) {
        HCCL_VM_ERROR("set fail devices[0][{}].", devices[0]);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(
            new (std::nothrow) std::thread(&SimGetDeviceComm, ndev, rankId, devices[rankId], std::ref(comms[rankId])));
        if (!threads[rankId]) {
            HCCL_VM_ERROR("threads[{}] start failed ", rankId);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_VM_ERROR("rank[{}] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    if (aclrtResetDevice(devices[0]) != ACL_SUCCESS) {
        HCCL_VM_ERROR("reset fail devices[0][{}].", devices[0]);
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
