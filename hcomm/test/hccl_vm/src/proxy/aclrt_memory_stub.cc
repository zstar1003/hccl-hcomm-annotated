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
#define HCCL_VM_MODULE "MEM_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl_proxy_common.h"
#include "store_sim_store_pub.h"
#include "sim_log.h"
#include "rt_external_mem.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "sim_models.h"
#include "db_sim_runner_ops.h"
#include "db_sim_runner_common.h"
#include "sim_sub_process_manager.h"

// rank 进程加载本库时先于 main 预热仅校验模式缓存，使后续引流判定拿到确定值。
// 只读 DB、不打日志（此刻日志组件可能尚未初始化）。
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 主机侧大块复用区句柄（懒加载缓存，与设备侧共用同一块 HcclCommPool）
static std::atomic<void*> g_commPoolBase{nullptr};

// 记下复用区在本进程的地址，首个大块申请时由 CAS 保证只有一个线程写入。
// 首次记录时多 acquire 一次复用区且进程内不释放，使复用区在进程退出前保持映射、地址不变，
// 后续用地址判断一块内存是否在复用区时地址不会失效。
static void CacheCommPoolBase(void* base)
{
    if (base == nullptr) {
        return;
    }
    void* expected = nullptr;
    // 写入用 release、IsInCommPool 读取用 acquire 配对，保证其它线程读到的地址是有效的。
    if (g_commPoolBase.compare_exchange_strong(expected, base, std::memory_order_release, std::memory_order_relaxed)) {
        sim::MemoryManager::GetInstance().AcquireMemByName(sim::CommPoolPolicy::kPoolName);
    }
}

std::string GenDevMemName(int deviceId)
{
    std::ostringstream oss;
    oss << "DEV" << deviceId << "_" << getpid() << "_" << std::this_thread::get_id() << "_"
        << std::chrono::steady_clock::now().time_since_epoch().count();
    return oss.str();
}

bool GetPhyMemBlockByVirPtr(const void* virPtr, uint32_t& offset, sim::PhyMemBlock& phyMem)
{
    // host侧根据虚拟地址查询时需要根据device_id匹配
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (
                  (virMem.dev_mapped_ptr <= devPtr) && (devPtr < (virMem.dev_mapped_ptr + virMem.size))
                  && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV) && (virMem.device_id == deviceKey));
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find VirtualMemBlock by virPtr:{:p}", virPtr);
        return false;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find PhyMemBlock by phyMemId:{:d}", phyMemId);
        return false;
    };
    offset = (uint32_t)(devPtr - virMemRes.first.dev_mapped_ptr);
    phyMem = *phyMemRes;
    return true;
}

uint32_t GetRankIdByVirAddr(const void* virAddr)
{
    sim::PhyMemBlock phyMem{};
    uint32_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(virAddr, offset, phyMem)) {
        return 0;
    }

    uint32_t deviceId = phyMem.device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([deviceId](const sim::Rank& rank) {
        return rank.device_id == deviceId;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("cannot find rank by device id[{}]", deviceId);
        return 0;
    }

    return rank.first.rank_id;
}

void* GetRealPtrByAddr(const void* virPtr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return ((virMem.dev_mapped_ptr <= devPtr) && (devPtr < (virMem.dev_mapped_ptr + virMem.size))
                      && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV))
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", virPtr);
        return nullptr;
    }

    void* virStartPtr = (void*)(uintptr_t)virMemRes.first.start_ptr;
    void* devStartPtr = (void*)(uintptr_t)virMemRes.first.dev_mapped_ptr;
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(virStartPtr);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not find host ptr by dev ptr:{:p}", virStartPtr);
        return nullptr;
    }

    hostPtr = ((char*)hostPtr + (devPtr - (uint64_t)(uintptr_t)devStartPtr));
    return hostPtr;
}

// 判断一个真实内存地址是否落在复用区内，主机侧和设备侧共用。
// 设备侧传进来的是设备指针对应的真实地址（用 GetHostPtrByDevPtr 取得）。
// 地址在 [基址, 基址+kPoolSize) 范围内，说明这块内存是引流到复用区的大块。
static bool IsInCommPool(const void* ptr)
{
    void* base = g_commPoolBase.load(std::memory_order_acquire);
    return base != nullptr && ptr >= base && ptr < static_cast<char*>(base) + sim::CommPoolPolicy::kPoolSize;
}

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    bool checkOnlyMode = sim::IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配。
    if (sim::CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR("malloc host size:{:d} exceeds pool ceiling, reject", size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 仅校验模式下大块引流到共享复用区 HcclCommPool
    if (sim::CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        void* base = g_commPoolBase.load(std::memory_order_acquire);
        if (base == nullptr) {
            base = sim::MemoryManager::GetInstance().AcquireMemByName(sim::CommPoolPolicy::kPoolName);
            if (base == nullptr) {
                HCCL_VM_ERROR("malloc host acquire HcclCommPool failed, size:{:d}", size);
                return ACL_ERROR_INTERNAL_ERROR;
            }
            CacheCommPoolBase(base);
        }
        *hostPtr = base;
        HCCL_VM_INFO("malloc host (SHM) addr:{:p}, size:{:d}", *hostPtr, size);
    } else {
        *hostPtr = malloc(size);
        HCCL_VM_INFO("malloc host addr:{:p}, size:{:d}", *hostPtr, size);
    }
    return ACL_SUCCESS;
}

aclError aclrtMallocHostWithCfg(void** ptr, uint64_t size, aclrtMallocConfig* cfg)
{
    (void)cfg;
    return aclrtMallocHost(ptr, size);
}

aclError aclrtFreeHost(void* hostPtr)
{
    HCCL_VM_INFO("free host addr:{:p}", hostPtr);
    // 池内地址：空操作（复用区生命周期由主进程管理，不在此 free）
    if (IsInCommPool(hostPtr)) {
        HCCL_VM_INFO("free host (SHM noop) addr:{:p}", hostPtr);
        return ACL_SUCCESS;
    }
    free(hostPtr);
    return ACL_SUCCESS;
}

bool GetDevMappedMemPtr(
    uint64_t rankId, size_t size, std::string& memName, bool& isDevMapped, void** virPtr, void** devPtr)
{
    isDevMapped = false;
    *virPtr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(rankId, size);
    if (*virPtr == nullptr) {
        HCCL_VM_ERROR("alloc vir mem rankId:{:d}, size:{:d} fail", rankId, size);
        return false;
    }

    // --check-only 模式下大块引流到复用区，设备侧无需真实地址映射
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        *devPtr = *virPtr;
        HCCL_VM_INFO("check-only pool-redirect: use virAddr {:p} as devPtr", *virPtr);
        return true;
    }

    if (sim::GetAicpuProcMgr().IsAlive()) {
        DevMemOpPayload payload{};
        strncpy(payload.memName, memName.c_str(), sizeof(payload.memName) - 1);

        uint8_t rspCmd;
        uint32_t rspLen = 0;
        RspGetDevPtrPayload rspPayload{};
        if (sim::GetAicpuProcMgr().Request(
                PIPE_CMD_GET_DEV_PTR, &payload, sizeof(payload), rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
            != 0) {
            HCCL_VM_ERROR("Request PIPE_CMD_GET_DEV_PTR failed.");
            return false;
        }

        isDevMapped = true;
        *devPtr = (void*)rspPayload.ptr;
        HCCL_VM_INFO("PIPE_CMD_GET_DEV_PTR vir mem by device {:p}", (void*)rspPayload.ptr);
    } else {
        *devPtr = *virPtr;
    }

    return true;
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx->device_id;

    auto dev = RunnerDB::GetById<sim::Device>(deviceId);
    if (!dev.has_value()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto rankId = (uint32_t)sim::GetCurrRankId();

    std::string memName = GenDevMemName(deviceId);
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(memName.c_str(), deviceId, size);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc phy mem deviceId:{:d}, size:{:d}", deviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 大块引流到复用区时缓存池基址，供 IsInCommPool 判定（与主机侧共用同一基址）。
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        CacheCommPoolBase(hostPtr);
    }

    bool isDevMapped = false;
    void* virPtr = nullptr;
    void* devMappedPtr = nullptr;
    if (!GetDevMappedMemPtr(rankId, size, memName, isDevMapped, &virPtr, &devMappedPtr)) {
        HCCL_VM_ERROR("can not alloc vir mem deviceId:{:d}, rankId:{:d}, size:{:d}", deviceId, rankId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 记录物理内存信息到数据库
    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    memcpy(phyMem.name, memName.c_str(), memName.size());
    phyMem.size = size;
    phyMem.ref_count = 1;
    auto phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    // 记录虚拟内存信息到数据库
    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    virMem.dev_mapped_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devMappedPtr));
    virMem.is_dev_access = static_cast<uint8_t>(isDevMapped);
    virMem.size = size;
    virMem.ctx_id = runner.current_ctx_id;
    virMem.rank_id = rankId;
    virMem.device_id = deviceId;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = (uint8_t)policy;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    *devPtr = isDevMapped ? devMappedPtr : virPtr;
    // 缓存设备地址到host地址
    sim::DeviceMemoryManager::GetInstance().MapDevPtrHostPtr(virPtr, hostPtr);
    HCCL_VM_INFO("malloc dev addr:{:p}, memName:{}, size:{:d}", virPtr, memName, size);
    return ACL_SUCCESS;
}

aclError aclrtMallocAlign32(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    HCCL_VM_INFO("..");
    size_t realSize = ((size + 31) / 32) * 32;
    return aclrtMalloc(devPtr, realSize, policy);
}

aclError aclrtMallocCached(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMemFlush(void* devPtr, size_t size)
{
    (void)devPtr;
    (void)size;
    HCCL_VM_INFO("..");
    return ACL_SUCCESS;
}

aclError aclrtMemInvalidate(void* devPtr, size_t size)
{
    (void)devPtr;
    (void)size;
    HCCL_VM_INFO("..");
    return ACL_SUCCESS;
}

aclError aclrtMallocWithCfg(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    (void)cfg;
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMallocForTaskScheduler(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    (void)cfg;
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtFree(void* devPtr)
{
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.dev_mapped_ptr == startPtr) && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:0x{:x}", startPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    auto deviceId = phyMemRes->device_id;
    auto dev = RunnerDB::GetById<sim::Device>(deviceId);
    if (!dev.has_value()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto devPhyId = dev->physical_id;

    if (sim::GetAicpuProcMgr().IsAlive()) {
        DevMemOpPayload freePayload{};
        strncpy(freePayload.memName, phyMemRes->name, sizeof(freePayload.memName) - 1);

        uint8_t rspCmd;
        uint32_t rspLen = 0;
        RspFreeDevPtrPayload rspPayload{};
        if (sim::GetAicpuProcMgr().Request(
                PIPE_CMD_FREE_DEV_PTR, &freePayload, sizeof(freePayload), rspCmd, &rspPayload, sizeof(rspPayload),
                rspLen)
            != 0) {
            HCCL_VM_ERROR("Request PIPE_CMD_FREE_DEV_PTR failed for {}", phyMemRes->name);
        }
    }

    // 先取这块内存的真实地址，判断它是否在复用区（必须在 Unmap 之前取）。
    // 第二次释放时映射已删、取不到地址，就按非复用区处理，天然幂等。
    void* backing = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devPtr);

    // 去缓存设备地址到host地址
    sim::DeviceMemoryManager::GetInstance().UnmapDevPtrHostPtr(devPtr);

    // 删除内存
    sim::DeviceMemoryManager::GetInstance().FreeVirMem(devPhyId, devPtr);
    if (IsInCommPool(backing)) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(phyMemRes->name, deviceId);
    }

    // 更新数据库记录，标记为已释放
    RunnerDB::Update<sim::PhyMemBlock>(phyMemId, [](sim::PhyMemBlock& memBlock) {
        memBlock.is_freed = 1;
    });
    RunnerDB::Update<sim::VirtualMemBlock>(virMemRes.first.id, [](sim::VirtualMemBlock& virtMemBlock) {
        virtMemBlock.is_freed = 1;
    });

    HCCL_VM_INFO("free dev addr:{:p}, memName:{}, size:{:d}", devPtr, phyMemRes->name, virMemRes.first.size);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    sim::PhyMemBlock phyMem{};
    uint32_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        memset(devPtr, value, count);
        HCCL_VM_INFO(
            "sys mem memset ptr:{:p}, maxCount: {:d}, value: {:d}, count: {:d}", devPtr, maxCount, value, count);
        return ACL_SUCCESS;
    }

    char* hostPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostPtr == nullptr) {
        HCCL_VM_INFO("dev mem memset ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际写入并按池释放。
    bool pooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (pooled) {
        HCCL_VM_INFO("memset skip (pool) ptr:{:p}, size:{:d}, count:{:d}", devPtr, phyMem.size, count);
    } else {
        memset(hostPtr + offset, value, count);
    }
    if (pooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }
    HCCL_VM_INFO("dev mem memset ptr:{:p}, memName:{}, count: {:d}", devPtr, phyMem.name, count);
    return ACL_SUCCESS;
}

aclError aclrtMemsetAsync(void* devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream)
{
    (void)stream;
    HCCL_VM_INFO("..");
    // 当前先不生成任务，后续有需要再根据实际情况生成任务
    return aclrtMemset(devPtr, maxCount, value, count);
}

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    (void)destMax;
    void* hostPtr = nullptr;
    void* devPtr = nullptr;
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        memcpy(dst, src, count);
        return ACL_SUCCESS;
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        hostPtr = const_cast<void*>(src);
        devPtr = dst;
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        hostPtr = dst;
        devPtr = const_cast<void*>(src);
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        auto dstAddr = GetRealPtrByAddr(dst);
        if (dstAddr == nullptr) {
            HCCL_VM_WARN(" only support self D2D memcpy");
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        auto srcAddr = GetRealPtrByAddr(src);
        if (srcAddr == nullptr) {
            HCCL_VM_WARN("only support self D2D memcpy");
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝。
        if (IsInCommPool(dstAddr)) {
            HCCL_VM_INFO("memcpy D2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(dstAddr, srcAddr, count);
        }
        return ACL_SUCCESS;
    } else {
        HCCL_VM_INFO("src:{:p} to dst:{:p}, size:{:d} type: {:d} not support", src, dst, count, (int)kind);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    sim::PhyMemBlock phyMem{};
    uint32_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        HCCL_VM_INFO("dev mem memcpy ptr:{:p}, count: {:d}", devPtr, count);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HCCL_VM_INFO("host:{:p} dev:{:p} memName:{} size:{:d} type:{:d}", hostPtr, devPtr, phyMem.name, count, (int)kind);
    char* hostDevPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostDevPtr == nullptr) {
        HCCL_VM_INFO(" dev mem memcpy ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝并按池释放。
    // D2H 的目的是主机缓冲，仍按地址判断主机缓冲是否在复用区。
    bool devPooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        if (devPooled) {
            HCCL_VM_INFO("memcpy H2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(hostDevPtr + offset, hostPtr, count);
        }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        if (IsInCommPool(hostPtr)) {
            HCCL_VM_INFO("memcpy D2H skip (dst host pool) count:{:d}", count);
        } else {
            memcpy(hostPtr, hostDevPtr + offset, count);
        }
    }

    if (devPooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }

    return ACL_SUCCESS;
}

aclError
aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)destMax;
    void* hostPtr = nullptr;
    void* devPtr = nullptr;
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        memcpy(dst, src, count);
        return ACL_SUCCESS;
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        hostPtr = const_cast<void*>(src);
        devPtr = dst;
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        hostPtr = dst;
        devPtr = const_cast<void*>(src);
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        auto curRank = (uint32_t)sim::GetCurrRankId();
        HcclTaskMetaData taskMetaData{};
        taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
        taskMetaData.commId = 0;
        taskMetaData.rankId = curRank;
        taskMetaData.streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);
        taskMetaData.taskData.transMem.srcRankId = GetRankIdByVirAddr(src);
        taskMetaData.taskData.transMem.srcOffset = (uint64_t)(uintptr_t)src;
        taskMetaData.taskData.transMem.dstRankId = GetRankIdByVirAddr(dst);
        taskMetaData.taskData.transMem.dstOffset = (uint64_t)(uintptr_t)dst;
        taskMetaData.taskData.transMem.len = count;

        uint32_t index{0};
        auto ret = InsertTaskToCollection(&taskMetaData, &index);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("InsertTaskToCollection fail");
            return ACL_ERROR_INTERNAL_ERROR;
        }

        // 下发cid
        HcclTaskCid taskCid{0, curRank, index};
        sim::Task task{};
        task.stream_id = taskMetaData.streamId;
        task.cid = taskCid.value;
        task.type = (uint8_t)HccLTaskMetaType::MEM_CPY;

        auto taskId = RunnerDB::Add<sim::Task>(task);
        return ACL_SUCCESS;
    } else {
        HCCL_VM_INFO("src:{:p} to dst:{:p}, size:{:d} type: {:d} not support", src, dst, count, (int)kind);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    sim::PhyMemBlock phyMem{};
    uint32_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        HCCL_VM_INFO("dev mem memcpy ptr:{:p}, count: {:d}", devPtr, count);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HCCL_VM_INFO("host:{:p} dev:{:p} memName:{} size:{:d} type:{:d}", hostPtr, devPtr, phyMem.name, count, (int)kind);

    char* hostDevPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostDevPtr == nullptr) {
        HCCL_VM_INFO("dev mem memcpy ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝并按池释放。
    // D2H 的目的是主机缓冲，仍按地址判断主机缓冲是否在复用区。
    bool devPooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        if (devPooled) {
            HCCL_VM_INFO("memcpy H2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(hostDevPtr + offset, hostPtr, count);
        }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        if (IsInCommPool(hostPtr)) {
            HCCL_VM_INFO("memcpy D2H skip (dst host pool) count:{:d}", count);
        } else {
            memcpy(hostPtr, hostDevPtr + offset, count);
        }
    }

    if (devPooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithCondition(
    void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

aclError aclrtMemcpyBatch(
    void** dsts, size_t* destMaxs, void** srcs, size_t* sizes, size_t numBatches, aclrtMemcpyBatchAttr* attrs,
    size_t* attrsIndexes, size_t numAttrs, size_t* failIndex)
{
    (void)attrs;
    (void)attrsIndexes;
    (void)numAttrs;
    (void)failIndex;
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpy(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpyBatchAsync(
    void** dsts, size_t* destMaxs, void** srcs, size_t* sizes, size_t numBatches, aclrtMemcpyBatchAttr* attrs,
    size_t* attrsIndexes, size_t numAttrs, size_t* failIndex, aclrtStream stream)
{
    (void)attrs;
    (void)attrsIndexes;
    (void)numAttrs;
    (void)failIndex;
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpyAsync(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT, stream);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpy2d(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind)
{
    (void)dst;
    (void)dpitch;
    (void)src;
    (void)spitch;
    (void)width;
    (void)height;
    if (kind != ACL_MEMCPY_HOST_TO_DEVICE && kind != ACL_MEMCPY_DEVICE_TO_HOST) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpy2dAsync(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind,
    aclrtStream stream)
{
    (void)dst;
    (void)dpitch;
    (void)src;
    (void)spitch;
    (void)width;
    (void)height;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t* descSize)
{
    (void)kind;
    (void)descSize;
    return ACL_SUCCESS;
}

aclError aclrtSetMemcpyDesc(void* desc, aclrtMemcpyKind kind, void* srcAddr, void* dstAddr, size_t count, void* config)
{
    (void)desc;
    (void)kind;
    (void)srcAddr;
    (void)dstAddr;
    (void)count;
    (void)config;
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithDesc(void* desc, aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)desc;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size, const aclrtPhysicalMemProp* prop, uint64_t flags)
{
    (void)prop;
    (void)flags;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx->device_id;

    std::string memName = GenDevMemName(deviceId);
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(memName.c_str(), deviceId, size);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc phy mem deviceId:{:d}, size:{:d}", deviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 大块物理内存同样引流到复用区，缓存池基址，使其被 map 后的 memcpy/memset 能按地址判定。
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        CacheCommPoolBase(hostPtr);
    }

    // 记录物理内存信息到数据库
    sim::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    memcpy(phyMem.name, memName.c_str(), memName.size());
    phyMem.size = size;
    phyMem.ref_count = 1;
    auto phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    HCCL_VM_INFO("malloc phy mem name:{}, size:{:d}, id:{:d}", memName, size, phyMemId);
    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    auto phyMemId = (uint64_t)(uintptr_t)handle;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    // 物理内存无 devPtr→host 映射，按 size 判定是否走了复用区（与申请同一判据）
    if (sim::CommPoolPolicy::ShouldRedirect(phyMemRes->size, sim::IsCheckOnlyMode())) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(phyMemRes->name, phyMemRes->device_id);
    }
    // 删除数据库记录
    RunnerDB::Delete<sim::PhyMemBlock>(phyMemId);
    HCCL_VM_INFO("free phy mem, id:{:d}", phyMemId);

    return ACL_SUCCESS;
}

aclError aclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx->device_id;

    auto dev = RunnerDB::GetById<sim::Device>(deviceId);
    if (!dev.has_value()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto devPhyId = dev->physical_id;

    std::string memName = GenDevMemName(deviceId);
    auto devPtr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(devPhyId, size);
    if (devPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc vir mem deviceId:{:d}, phyMemId:{:d}, size:{:d}", deviceId, devPhyId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 记录虚拟内存信息到数据库
    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = (uint64_t)(uintptr_t)devPtr;
    virMem.size = size;
    virMem.ctx_id = runner.current_ctx_id;
    virMem.phy_mem_id = 0;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::VIR_MEM_TYPE_DEV;
    virMem.policy = 0;
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);
    HCCL_VM_INFO("reserve vir mem, size:{:d}, devPtr:{:p}", size, devPtr);
    *virPtr = devPtr;
    return ACL_SUCCESS;
}

aclError aclrtReleaseMemAddress(void* virPtr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.start_ptr == devPtr) && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", devPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    auto ctxId = virMemRes.first.ctx_id;
    auto currCtx = RunnerDB::GetById<sim::Context>(ctxId);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not find current context:{:d}", ctxId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx->device_id;

    auto dev = RunnerDB::GetById<sim::Device>(deviceId);
    if (!dev.has_value()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto devPhyId = dev->physical_id;

    sim::DeviceMemoryManager::GetInstance().FreeVirMem(devPhyId, virPtr);
    RunnerDB::Delete<sim::VirtualMemBlock>(virMemRes.first.id);
    HCCL_VM_INFO("release vir mem, devPtr:{:p}", virPtr);

    return ACL_SUCCESS;
}

aclError aclrtMapMem(void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)size;
    (void)offset;
    (void)flags;
    uint64_t phyMemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.start_ptr == devPtr) && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", devPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    // 更新虚拟内存记录
    RunnerDB::Update<sim::VirtualMemBlock>(virMemRes.first.id, [phyMemId](sim::VirtualMemBlock& virMem) {
        virMem.phy_mem_id = phyMemId;
    });

    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void* virPtr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.start_ptr == devPtr) && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", devPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    // 更新虚拟内存记录
    RunnerDB::Update<sim::VirtualMemBlock>(virMemRes.first.id, [phyMemId](sim::VirtualMemBlock& virMem) {
        virMem.phy_mem_id = 0;
    });
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(
    aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t* shareableHandle)
{
    (void)handleType;
    (void)flags;
    uint64_t phyMemId = (uint64_t)(uintptr_t)handle;

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *shareableHandle = phyMemRes->id;
    return ACL_SUCCESS;
}

aclError aclrtDeviceGetBareTgid(int32_t* pid)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *pid = runner.pid;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t* pid, size_t pidNum)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < pidNum; i++) {
        sim::FdMemWhiteList tmp{};
        tmp.name_or_key = shareableHandle;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        RunnerDB::Add<sim::FdMemWhiteList>(tmp);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle* handle)
{
    (void)deviceId;
    uint64_t phyMemId = shareableHandle;

    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

aclError
aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp* prop, aclrtMemGranularityOptions option, size_t* granularity)
{
    (void)prop;
    (void)option;
    // 1字节对齐
    *granularity = 1;
    return ACL_SUCCESS;
}

aclError aclrtCmoAsync(void* src, size_t size, aclrtCmoType cmoType, aclrtStream stream)
{
    (void)src;
    (void)size;
    (void)cmoType;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtCmoAsyncWithBarrier(void* src, size_t size, aclrtCmoType cmoType, uint32_t barrierId, aclrtStream stream)
{
    (void)src;
    (void)size;
    (void)cmoType;
    (void)barrierId;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtCmoWaitBarrier(aclrtBarrierTaskInfo* taskInfo, aclrtStream stream, uint32_t flag)
{
    (void)taskInfo;
    (void)stream;
    (void)flag;
    return ACL_SUCCESS;
}

aclError aclrtPointerGetAttributes(const void* ptr, aclrtPtrAttributes* attributes)
{
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.start_ptr == startPtr) && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(phyMemRes->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("cannot find phy Mem device: {:d}", phyMemRes->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    attributes->location.id = devRes->logic_id;
    attributes->location.type = (aclrtMemLocationType)(virMemRes.first.src_type);
    attributes->pageSize = 0;
    return ACL_SUCCESS;
}

aclError aclrtHostRegister(void* ptr, uint64_t size, aclrtHostRegisterType type, void** devPtr)
{
    (void)size;
    (void)type;
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr](const sim::VirtualMemBlock& virMem) {
        return virMem.start_ptr == startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;
    });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    *devPtr = ptr;
    return ACL_SUCCESS;
}

aclError aclrtHostUnregister(void* ptr)
{
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr](const sim::VirtualMemBlock& virMem) {
        return virMem.start_ptr == startPtr && virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_HOST;
    });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtValueWrite(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    (void)devAddr;
    (void)value;
    (void)flag;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtValueWait(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    (void)devAddr;
    (void)value;
    (void)flag;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemGetExportKey(void* devPtr, size_t size, char* key, size_t len, uint64_t flags)
{
    (void)size;
    (void)len;
    (void)flags;
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devPtr));
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([startPtr](const sim::VirtualMemBlock& virMem) {
        return startPtr >= virMem.start_ptr && startPtr < virMem.start_ptr + virMem.size;
    });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::IpcMemRecord memRecord{};
    memRecord.vir_mem_id = virMemRes.first.id;
    memRecord.offset = startPtr - virMemRes.first.start_ptr;
    memRecord.create_pid = runner.pid;
    auto recordIdx = RunnerDB::Add<sim::IpcMemRecord>(memRecord);

    *(uint64_t*)key = recordIdx;

    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetImportPid(const char* key, int32_t* pid, size_t num)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    for (int32_t i = 0; i < num; i++) {
        sim::IpcMemWhiteList tmp{};
        tmp.name_or_key = *(uint64_t*)key;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        RunnerDB::Add<sim::IpcMemWhiteList>(tmp);
    }
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportByKey(void** devPtr, const char* key, uint64_t flags)
{
    (void)flags;
    uint64_t ipcRecordIdx = *(const uint64_t*)key;
    auto recordRes = RunnerDB::GetById<sim::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.has_value()) {
        HCCL_VM_ERROR("cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto virMemRes = RunnerDB::GetById<sim::VirtualMemBlock>(recordRes->vir_mem_id);
    if (!virMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find vir mem: {:d}", recordRes->vir_mem_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (recordRes->offset >= virMemRes->size) {
        HCCL_VM_ERROR(
            "ipc mem offset out of range, ipc record: {:d}, vir mem: {:d}, offset: {:d}, size: {:d}", ipcRecordIdx,
            recordRes->vir_mem_id, recordRes->offset, virMemRes->size);
        return ACL_ERROR_INVALID_PARAM;
    }
    *devPtr = reinterpret_cast<void*>(virMemRes->start_ptr + recordRes->offset);

    return ACL_SUCCESS;
}

aclError aclrtIpcMemClose(const char* key)
{
    uint64_t ipcRecordIdx = *reinterpret_cast<const uint64_t*>(key);
    auto recordRes = RunnerDB::GetById<sim::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.has_value()) {
        HCCL_VM_ERROR("cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.pid == recordRes->create_pid) {
        RunnerDB::Delete<sim::IpcMemRecord>(recordRes->id);
    }

    return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free, size_t* total)
{
    (void)attr;
    (void)free;
    (void)total;
    return ACL_SUCCESS;
}

aclError aclrtAllocatorRegister(aclrtStream stream, aclrtAllocatorDesc allocatorDesc)
{
    (void)stream;
    (void)allocatorDesc;
    return ACL_SUCCESS;
}

aclError aclrtAllocatorGetByStream(
    aclrtStream stream, aclrtAllocatorDesc* allocatorDesc, aclrtAllocator* allocator,
    aclrtAllocatorAllocFunc* allocFunc, aclrtAllocatorFreeFunc* freeFunc,
    aclrtAllocatorAllocAdviseFunc* allocAdviseFunc, aclrtAllocatorGetAddrFromBlockFunc* getAddrFromBlockFunc)
{
    (void)stream;
    (void)allocatorDesc;
    (void)allocator;
    (void)allocFunc;
    (void)freeFunc;
    (void)allocAdviseFunc;
    (void)getAddrFromBlockFunc;
    return ACL_SUCCESS;
}

aclError aclrtAllocatorUnregister(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithOffsetImpl(
    void** dst, size_t destMax, uint64_t dstDataOffset, const void** src, size_t count, size_t srcDataOffset,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)dst;
    (void)destMax;
    (void)dstDataOffset;
    (void)src;
    (void)count;
    (void)srcDataOffset;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetAttr(const char* key, aclrtIpcMemAttrType type, uint64_t attr)
{
    (void)key;
    (void)type;
    (void)attr;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportPidInterServer(const char* name, aclrtServerPid* serverPids, size_t num)
{
    const aclrtServerPid& rtServerPid = *serverPids;
    return aclrtIpcMemSetImportPid(name, rtServerPid.pid, rtServerPid.num);
}

rtError_t rtMalloc(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    (void)type;
    (void)moduleId;
    int ret = aclrtMalloc(devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    HCCL_VM_INFO("enter into stub rtMalloc.... addr:{:p}", *devPtr);
    return ret;
}

aclError rtFree(void* devPtr)
{
    (void)devPtr;
    HCCL_VM_ERROR("enter into rtFree not support");
    return ACL_ERROR_INTERNAL_ERROR;
}

std::map<uint8_t, HcclDataType> rtDataType2CheckerDataType
    = {{aclDataType::ACL_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
       {aclDataType::ACL_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
       {aclDataType::ACL_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
       {aclDataType::ACL_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
       {aclDataType::ACL_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {aclDataType::ACL_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
       {aclDataType::ACL_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
       {aclDataType::ACL_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {aclDataType::ACL_INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
       {aclDataType::ACL_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16}};

aclError aclrtReduceAsync(
    void* dst, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type, aclrtStream stream,
    void* reserve)
{
    (void)reserve;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t srcCtxId = 0;
    uint64_t dstCtxId = 0;

    auto iter = rtDataType2CheckerDataType.find(type);
    if (iter == rtDataType2CheckerDataType.end()) {
        HCCL_VM_ERROR("not support data type: {}", static_cast<uint32_t>(type));
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    auto dataType = iter->second;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId = 0;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.reduce.srcRankId = GetRankIdByVirAddr(src);
    taskMetaData.taskData.reduce.dstRankId = GetRankIdByVirAddr(dst);
    taskMetaData.taskData.reduce.srcOffset = (uint64_t)(uintptr_t)src;
    taskMetaData.taskData.reduce.dstOffset = (uint64_t)(uintptr_t)dst;
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count / dataSize;
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(kind);

    uint32_t index{0};
    HCCL_VM_DEBUG("Get reduce task, streamId={:d}", streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
