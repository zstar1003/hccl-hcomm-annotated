/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ZERO_COPY_ADDRESS_MGR_H
#define ZERO_COPY_ADDRESS_MGR_H

#include <set>
#include <mutex>
#include <hccl/hccl_types.h>
#include "aicpu_operator_pub.h"

namespace hccl {
/*
 * 该类负责管理HcclCommSetMemoryRange/HcclCommUnsetMemoryRange等API注册的地址进行管理
 */
using ZeroCopyReserveAddrMap = std::unordered_map<u32, std::unordered_map<void*, LocalIpc2RemoteAddr>>;
class ZeroCopyAddressMgr {
public:
    ZeroCopyAddressMgr() = default;
    ~ZeroCopyAddressMgr() = default;

    // Set/Unset地址
    HcclResult SetMemoryRange(u32 devicePhyId, void* baseAddr, u64 length);
    HcclResult UnsetMemoryRange(u32 devicePhyId, void* baseAddr);
    // 判断地址是否仍被Set
    bool IsAddressSet(u32 devicePhyId, void* baseAddr);

    // 添加与对端内存的映射关系，只保留基地址映射
    HcclResult AddLocalIpc2RemoteAddr(u32 devicePhyId, void* localIpcBase, void* remoteAddrBase, u64 length);
    HcclResult DelLocalIpc2RemoteAddr(u32 devicePhyId, void* remoteAddrBase);
    HcclResult GetLocalIpc2RemoteAddr(u32 devicePhyId, void* remoteAddr, LocalIpc2RemoteAddr& addr);

    // 添加Activate的内存段
    HcclResult ActivateCommMemoryAddr(void* startPtr, u64 length);
    HcclResult DeactivateCommMemoryAddr(void* startPtr);

    // 管理从远端import的内存
    HcclResult AddRemoteImportAddr(void* devPtr, void* handle);
    HcclResult GetRemoteImportAddr(void* devPtr, void*& handle);
    HcclResult DelRemoteImportAddr(void* devPtr);

    // 只有[startPtr, startPtr + length)全在前面Activate接口注册的内存段中才为有效
    bool IsActivateCommMemoryAddr(void* startPtr, u64 length);

    // 判断是否与activate有交叠，只要与active的内存范围有交集就返回真
    bool IsOverlapWithActivateAddr(void* startPtr, u64 length);

    // 判断内存区间是否在Set区间内
    bool IsInSetAddressRange(u32 devicePhyId, void* startPtr, u64 length);

    void GetRingBufferAddr(u64& bufferPtr, u64& headPtr, u64& tailPtr)
    {
        bufferPtr = reinterpret_cast<u64>(devRingBufBase_);
        headPtr = reinterpret_cast<u64>(devRingHead_);
        tailPtr = reinterpret_cast<u64>(devRingTail_);
    }

    // 处理RingBuffer
    HcclResult ProcessRingBuffer(ZeroCopyRingBufferItem* ringBuffer, u32* head, u32* tail);
    // 处理引用计数
    u32 GetCommRefCnt() const;
    HcclResult IncreCommRefCnt();
    HcclResult DecreCommRefCnt();

private:
    ZeroCopyAddressMgr(const ZeroCopyAddressMgr&) = delete;
    ZeroCopyAddressMgr(ZeroCopyAddressMgr&&) = delete;
    ZeroCopyAddressMgr& operator=(const ZeroCopyAddressMgr&) = delete;
    ZeroCopyAddressMgr& operator=(ZeroCopyAddressMgr&&) = delete;

    __attribute__((weak)) HcclResult InitRingBuffer();
    __attribute__((weak)) HcclResult PushOne(ZeroCopyRingBufferItem& item);
    HcclResult ProcessOneAddrMap(const ZeroCopyRingBufferItem& item);

    // 表示一段内存[start, end)是个左闭右开的区间，即最后一个字节不可访问
    struct AddressRange {
        AddressRange(void* startPtr, u64 length) : start(reinterpret_cast<u64>(startPtr)), end(start + length) {}
        AddressRange(u64 startPtr, u64 length) : start(startPtr), end(start + length) {}
        ~AddressRange() = default;

        // 当前场景下不允许内存范围有重叠，一旦重叠就认为是相同
        // 比如 a=[0, 100) 与 b=[10, 200)就是相等的,
        // 使用下面的判断a<b与b<a均会返回false，那么它们就是相等的
        bool operator<(const AddressRange& other) const
        {
            // 因为最后一字节不可访问，所以这里可以等于
            return this->end <= other.start;
        }

        u64 start = 0;
        u64 end = 0;
    };

    u32 commRefCnt_{0};
    std::mutex lock_;
    std::mutex processRingBufferLock_;
    // 每个device保存自己的预留内存，每个内存的key都是对端基地址，value是映射关系
    ZeroCopyReserveAddrMap reserveAddrMappings_;
    std::unordered_map<u32, std::set<AddressRange>> reserveRanges_;
    DeviceMem ringBuffer_;
    DeviceMem ringBufferCtl_;
    ZeroCopyRingBufferItem* devRingBufBase_ = nullptr;
    u32* devRingHead_ = nullptr;
    u32* devRingTail_ = nullptr;

    bool needPushOne{true};
    std::set<AddressRange> validAddressRanges_{};
    std::unordered_map<void*, void*> importAddrs_{};
};
} // namespace hccl

#endif
