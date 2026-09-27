/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MR_MANAGER_H
#define HCCL_MR_MANAGER_H

#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_network_pub.h"

namespace hccl {
using HcclMrInfo = struct TagHcclMrInfo {
    void* addr;              /**< starting address of mr */
    unsigned long long size; /**< size of mr */
    int access;              /**< access of mr, reference to RaAccessFlags */
    unsigned int lkey;       /**< local addr access key */
};

using MrInfo = struct TagMrInfo {
    void* addr; /**< starting address of mr */
    void* devVirAddr;
    unsigned long long size; /**< size of mr */
    int access;              /**< access of mr, reference to RaAccessFlags */
    unsigned int lkey;       /**< local addr access key */
    MrHandle mrHandle;       /**< ibv_mr handle */
    int gloMemRef;
    int tmpMemRef;

    TagMrInfo()
        : addr(nullptr),
          devVirAddr(nullptr),
          size(0),
          access(0),
          lkey(0),
          mrHandle(nullptr),
          gloMemRef(0),
          tmpMemRef(0)
    {}

    TagMrInfo(void* addr, u64 size)
        : addr(addr),
          devVirAddr(nullptr),
          size(size),
          access(0),
          lkey(0),
          mrHandle(nullptr),
          gloMemRef(0),
          tmpMemRef(0)
    {}

    TagMrInfo(const TagMrInfo&) = default;
    TagMrInfo(TagMrInfo&&) noexcept = default;
    TagMrInfo& operator=(const TagMrInfo&) = default;
    TagMrInfo& operator=(TagMrInfo&&) noexcept = default;
    ~TagMrInfo() = default;

    TagMrInfo& operator=(const HcclMrInfo& mrInfo)
    {
        addr = mrInfo.addr;
        size = mrInfo.size;
        access = mrInfo.access;
        lkey = mrInfo.lkey;
        return *this;
    }
};

struct MrMapKey {
    u64 addr;
    u64 size;

    MrMapKey() : addr(0), size(0) {}

    MrMapKey(u64 addr, u64 size) : addr(addr), size(size) {}

    MrMapKey(const MrMapKey& that) : addr(that.addr), size(that.size) {}

    MrMapKey& operator=(const MrMapKey& that)
    {
        if (&that != this) {
            addr = that.addr;
            size = that.size;
        }
        return *this;
    }

    bool operator<(const MrMapKey& that) const
    {
        return ((this->addr < that.addr) || ((this->addr == that.addr) && (this->size < that.size)));
    }

    bool operator==(const MrMapKey& that) const { return ((this->addr == that.addr) && (this->size == that.size)); }
};

struct HostMappingKey {
    u64 addr = 0;
    u64 size = 0;
    u32 devId = 0;

    HostMappingKey(u64 addr, u64 size, u32 devId) : addr(addr), size(size), devId(devId) {}

    bool operator<(const HostMappingKey& that) const
    {
        if (addr != that.addr) {
            return addr < that.addr;
        }
        if (size != that.size) {
            return size < that.size;
        }
        return devId < that.devId;
    }
};

struct HostMappingInfo {
    void* devVirAddr = nullptr;
    int mappingRef = 0;
};

class MrManager {
public:
    static MrManager& GetInstance();
    MrManager();
    explicit MrManager(HcclNetDevCtx netDevCtx);
    ~MrManager();
    HcclResult Init(RdmaHandle rdmaHandle); // 初始化给rdmaHandle赋值，每初始化一次count++
    HcclResult Init(QpHandle qpHandle, u32 devId, bool isHostMem, std::map<MrMapKey, MrInfo>& unRegMrMap);
    HcclResult Init(RdmaHandle rdmaHandle, u32 devId, bool isHostMem);
    HcclResult DeInit(const void* handle); // 判断rdmaHandle是否一致，每去初始化一次count--
    HcclResult Init();
    HcclResult DeInit();
    HcclResult RegGlobalMr(void* addr, u64 size);       // 注册全局mr，网卡未初始化时先记录mr信息
    HcclResult GetKey(void* addr, u64 size, u32& lkey); // 拿到内存对应的key
    HcclResult ReleaseKey(void* addr, u64 size);        // 释放key操作权限
    HcclResult DeRegGlobalMr(void* addr);               // 解注册全局mr 地址必须是起始地址
    HcclResult DelayedReg(void* addr, u64 size);
    HcclResult GetDevVirAddr(void* addr, u64 size, u64& devVirAddr);
    HcclResult MapMem(void* addr, u64 size, void*& devVirAddr);
    std::map<MrMapKey, MrInfo> GetUnregMap();
    void SetHdcPara(u32 devId, bool isHostMem, bool isUseQPHandle);
    HcclResult InitUnRegMrMap(std::map<MrMapKey, MrInfo>& unRegMrMap);

    // delete copy and move constructors and assign operators
    MrManager(MrManager const&) = delete;            // Copy construct
    MrManager(MrManager&&) = delete;                 // Move construct
    MrManager& operator=(MrManager const&) = delete; // Copy assign
    MrManager& operator=(MrManager&&) = delete;      // Move assign
    std::unordered_map<void*, u64> addrSize_;
    std::vector<void*> deAddr_;
    static u64 g_devAddr;

private:
    static constexpr s32 COUNT_ONE = 1;
    bool isUseQPHandle_ = false;
    bool IsHostMem_ = false;
    u32 curDevId_ = -1;

    HcclResult InitMrManager(void* handle);
    HcclResult RegMr(void* addr, u64 size);                     // 注册全局mr
    HcclResult RegTmpMr(void* addr, u64 size, u32& lkey);       // 注册临时mr
    HcclResult GetMrInfo(MrInfo& mrInfo, bool& isInfoNotFound); // 通过地址查询mrMap中是否存在对应的内存
    HcclResult ReleaseMrResource();
    HcclResult RegMrImpl(void* addr, u64 size, HcclMrInfo& mrInfo, MrHandle& mrHandle, void*& devVirAddr);
    HcclResult DeRegMrImpl(MrInfo mrInfo);
    HcclResult UnmapMem(MrInfo mrInfo);
    std::map<HostMappingKey, HostMappingInfo>::iterator SearchMappingMap(u64 userAddr, u64 userSize);
    void TransMrInfo(void* addr, u64 size, HcclMrInfo& mrInfo);
    bool IsRequireMapping(void* addr, u64 size, void*& devVirAddr);
    HcclResult InitUnRegMrMap();

    RdmaHandle rdmaHandle_ = nullptr;
    QpHandle qpHandle_ = nullptr;
    std::atomic<int> count_;                // 网卡初始化计数
    std::map<MrMapKey, MrInfo> unRegMrMap_; // 网卡未初始化时记录内存map
    std::map<MrMapKey, MrInfo> regedMrMap_; // 全局内存map
    std::map<void*, u64> globalAddrSizeMap_;
    static std::map<HostMappingKey, HostMappingInfo> mappedHostToDevMap_;
    std::mutex addrSizeMutex_;
    std::mutex mrMapSpinMutex_;   // 自旋锁，锁外部注册全局内存map、内部注册全局内存map
    std::mutex unMrMapSpinMutex_; // 自旋锁，锁网卡未初始化时记录内存map
    static std::mutex mappedHostToDevMutex_;
    HcclNetDevCtx netDevCtx_;
};
} // namespace hccl
#endif // HCCL_MR_MANAGER_H
