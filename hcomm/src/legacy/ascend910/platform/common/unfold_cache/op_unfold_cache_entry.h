/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_UNFOLD_CACHE_ENTRY_H__
#define __OP_UNFOLD_CACHE_ENTRY_H__

#include <cstdint>
#include <vector>

#include "dispatcher_task_types.h" // LinkType
#include "stream_pub.h"

// 确认ptr应该为空
#define CHK_PTR_NOTNULL(ptr)                                                                                     \
    do {                                                                                                         \
        if (UNLIKELY((ptr) != nullptr)) {                                                                        \
            HCCL_ERROR(                                                                                          \
                "[%s] errNo[0x%016llx] ptr[%s] is 0x%016llx (should be null), return HCCL_E_INTERNAL", __func__, \
                HCCL_ERROR_CODE(HCCL_E_INTERNAL), #ptr, (ptr));                                                  \
            return HCCL_E_INTERNAL;                                                                              \
        }                                                                                                        \
    } while (0)

// 确认ptrPtr不应该为空, 但*ptrPtr应该为空
#define CHK_PTRPTR_NULL(ptrPtr)     \
    do {                            \
        CHK_PTR_NULL(ptrPtr);       \
        CHK_PTR_NOTNULL(*(ptrPtr)); \
    } while (0)

namespace hccl {

// 记录算子展开的输入/输出的内存范围
// 注意: 内存的分配销毁由外部DeviceMem控制, 这里只是记录基地址和内存大小
struct OpUnfoldMemRange {
    explicit OpUnfoldMemRange();
    explicit OpUnfoldMemRange(const uint64_t curBaseAddr, const uint64_t curMemSize);
    explicit OpUnfoldMemRange(const OpUnfoldMemRange& other);
    ~OpUnfoldMemRange();

    const OpUnfoldMemRange& operator=(const OpUnfoldMemRange& other); // 拷贝赋值操作符

    HcclResult GetEndAddr(uint64_t& endAddr) const; // 获取当前内存范围的end addr (exclusive)
    HcclResult InRange(const uint64_t addr, bool& isInRange) const;

    bool isValid;
    uint64_t baseAddr;
    uint64_t memSize;
};

struct RefreshAddrInfo {
    explicit RefreshAddrInfo();
    explicit RefreshAddrInfo(const uint32_t curRankId, const uint8_t curMemType);
    explicit RefreshAddrInfo(const RefreshAddrInfo& other);
    ~RefreshAddrInfo();

    const RefreshAddrInfo& operator=(const RefreshAddrInfo& other); // 拷贝赋值操作符

    static constexpr uint8_t INVALID_MEMTYPE = 0;
    static constexpr uint8_t USER_INPUT_MEMTYPE = 1;
    static constexpr uint8_t USER_OUTPUT_MEMTYPE = 2;
    static constexpr uint8_t HCCL_INPUT_MEMTYPE = 3; // 只用于alltoallv下的rank判断

    // 注意: 如果是alltoallv的PrepareIntraData, 则rankId表示当前send对应的remote rank, 即使dst memory为local hccl input
    // 参考OpUnfoldCacheEntry::UpdateRefreshAddrInfoForAlltoallv
    uint32_t rankId; // 默认情况下表示sqeAddr在rankId下对应memType的内存范围内
    uint8_t memType; // 0: invalid; 1: user input; 2: user output; 3: hccl input
};

using FlipInfo = std::pair<size_t, uint16_t>;                // first: zero-taskid SQE idx; second: flipnum
using RanksIdx = std::pair<std::vector<uint32_t>, uint32_t>; // first: ranks; second: idx
using RankRflag = std::pair<uint32_t, bool>; // first: rank; second; recv flag (1: recv相关; 0: send相关)

// 每个remote rank各有两个NotifyId/SignalAddr, 分别用于send/recv count对应的Wait/Record同步
constexpr uint32_t NOTIFY_NUM_PER_REMOTE_RANK = 2;

// 每个通信域只需要设置一次 (只由HCCL_BUFFSIZE和通信域拓扑决定, 与OpUnfoldCacheKey相关字段无关, e.g., opType and
// workflowType)
struct AlltoallvMetadata {
    // alltoallv第一次Orchestrate之前初始化
    uint64_t sdmaDataBlockSize = 0; // alltoallv的SDMA data block size (给定通信域下, 由于HCCL input buffer size,
                                    // SDMA并发数量, 以及deviceNumInLocalPod固定, 所以SDMA data block size也是固定的)
    std::vector<OpUnfoldMemRange>
        hcclInputMemRanges; // 每个rank的HCCL input buffer memory range (给定通信域, 在初始化后即固定)
    std::unordered_map<uint32_t, RankRflag>
        notifyIdRankRflagMap; // 跨卡通信的notifyId到remote RankRflag的映射 (用于NotifyWait的刷新)
    std::unordered_map<uint64_t, RankRflag>
        signalAddrRankRflagMap; // 跨卡通信的signalAddr到remote RankRflag的映射 (用于WriteRecord的刷新)

    // alltoallv第一次Orchestrate之后初始化
    // 注意: local/remote hccl offset只由local/target rank以及sdmaDataBlockSize决定
    // 注意: 一个hccl offset可能对应多个remote rank, 需要用RanksIdx追踪多个remote ranks以及当前需要使用的remote
    // rank的索引
    std::unordered_map<uint64_t, RanksIdx>
        hcclOffsetDstRanksIdxMap; // 当前rank的hccl input buffer中的local hccl offset到remote dst RanksIdx的映射
                                  // (用于PrepareIntraData)

    AlltoallvMetadata();

    void Clear();
    HcclResult Check(const bool afterFirstOrch) const;
};

// 每次alltoallv算子执行时更新
struct AlltoallvSendRecvInfo {
    HcclDataType sendType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclDataType recvType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    std::vector<uint64_t> sendCounts;
    std::vector<uint64_t> recvCounts;
    std::vector<uint64_t> sendOffsets;
    std::vector<uint64_t> recvOffsets;

    AlltoallvSendRecvInfo();

    HcclResult Check() const;
};

// 算子展开的动态缓存条目 (每个OpUnfoldKey对应最多一个缓存条目)
class OpUnfoldCacheEntry {
public:
    OpUnfoldCacheEntry() = delete;
    explicit OpUnfoldCacheEntry(
        const std::vector<OpUnfoldMemRange>& userInputMemRanges,
        const std::vector<OpUnfoldMemRange>& userOutputMemRanges);
    ~OpUnfoldCacheEntry();

    HcclResult GetSqeArrayCount(size_t& sqeArrayCount) const;

    // 缓存不命中下的函数

    // 分成两次函数调用是为了即使算子第一次展开的SQE存在placeholder, 一次LaunchTask下发的SQE仍然能够缓存在连续内存中,
    // 减少后续cache hit的开销
    HcclResult AllocSqeArray(
        const size_t sqeCount, const int32_t streamId,
        size_t& arrayIdx); // 分配成功会将arrayIdx设置为分配的SQE数组在sqeArrays_当中的索引
    HcclResult MemcpySqeArray(
        const size_t arrayIdx, const size_t sqeStartIdx, const size_t sqeCount, const uint8_t* sqeArray,
        const uint8_t* sqeTypeArray, const AicpuDfxInfo* sqeDfxInfoArray, const bool isAlltoallv,
        const AlltoallvMetadata*
            alltoallvMetadataPtr); // 将sqeArray memcpy到sqeArrays_[arrayIdx][sqeStartIdx:sqeStartIdx+sqeCount-1]
                                   // (因为DispatcherAicpu第一次算子展开时持有的是AlltoallvMetadata的指针,
                                   // 并且如果不是alltoallv算子则值为nullptr, 所以不传入引用)

    // 根据streamId计算streamSeqIdx
    HcclResult CalcStreamSeqIdxes(Stream& mainStream, std::vector<Stream>& slaveStreams);

    // 针对alltoallv类算子, 更新src/dst RefreshAddrInfo用于后续算子执行时的地址更新
    // (i) 更新invalid memType (只有cache-memcpy placeholder才可能出现此问题)
    // 当rankSize最后一个或多个ranks的send/recv count为0时, local user input/output offset为对应内存范围的end addr
    // -> 对于LocalCopy, src/dst memType默认为invalid, 需要更新为local user input/output
    // -> 对于PrepareIntraData, src memType默认为invalid, 需要更新为local user input
    // -> 对于RemoteCopy, dst memType默认为invalid, 需要更新为local user output
    // (ii) 更新local dst rank (如果dst memType是local hccl input)
    // PrepareIntraData场景下, 目的地址为local hccl offset, 因此dstRefreshInfo.rankId默认为local rank, 需要更新为remote
    // rank
    HcclResult UpdateRefreshAddrInfoForAlltoallv(const uint32_t curRank, AlltoallvMetadata& alltoallvMetadata);

    // 缓存命中下的函数

    // 更新指定的一段连续SQE, 并将相关信息设置给对应指针, 用于后续下发task到RTSQ
    // flipSqeIdxes指的是该段连续SQE中taskid==0且flipnum!=0的SQE的索引, 即这些SQE前面需要增加FlipPlaceholder
    HcclResult UpdateAndGetSqeArray(
        const size_t arrayIdx, const std::vector<OpUnfoldMemRange>& curUserInputMemRanges,
        const std::vector<OpUnfoldMemRange>& curUserOutputMemRanges, Stream& mainStream,
        std::vector<Stream>& slaveStreams, const uint32_t opRingBufferIdx, size_t& sqeCount, uint8_t** sqeArrayPtr,
        uint8_t** sqeTypeArrayPtr, AicpuDfxInfo** sqeDfxInfoArrayPtr, Stream** streamPtrPtr,
        std::vector<FlipInfo>& flipInfos, const bool profL1Enable, std::vector<uint64_t>& profTimestamps,
        const bool isAlltoallv, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo);

    // Cache hit更新并下发entry中所有的SQE后, 由于缓存的SQE的addr-related fields被in-place更新,
    // 需要把userInputMemRanges_/userOutputMemRanges_为当前执行对应的memory ranges
    HcclResult SetInputOutputMemRanges(
        const std::vector<OpUnfoldMemRange>& curUserInputMemRanges,
        const std::vector<OpUnfoldMemRange>& curUserOutputMemRanges);

private:
    // 合并两个uint32_t成为一个uint64_t
    inline void CombineUint32ToUint64(uint64_t& addr, const uint32_t high, const uint32_t low) const
    {
        constexpr uint64_t uintBitWidth = 32;
        addr = (static_cast<uint64_t>(high) << uintBitWidth) | static_cast<uint64_t>(low);
        return;
    }

    // 拆分一个uint64_t成为两个uint32_t
    inline void SplitUint64ToUint32(const uint64_t addr, uint32_t& high, uint32_t& low) const
    {
        constexpr uint64_t uintBitWidth = 32;
        high = static_cast<uint32_t>(addr >> uintBitWidth);
        low = static_cast<uint32_t>(addr & 0xFFFFFFFFULL);
        return;
    }

    // 缓存不命中下的函数
    HcclResult CheckAndPrepareRefreshAddrInfo(
        const uint64_t sqeAddr, RefreshAddrInfo& refreshAddrInfo, const bool isAlltoallv,
        const AlltoallvMetadata*
            alltoallvMetadataPtr); // 根据range判断sqeAddr是否在某个rankid的input/output user memory范围内,
                                   // 并相应更新RefreshAddrInfo为后续缓存命中刷新地址做准备
                                   // (因为DispatcherAicpu第一次算子展开时持有的是AlltoallvMetadata的指针,
                                   // 并且如果不是alltoallv算子则值为nullptr, 所以不传入引用)
    HcclResult CheckMemTypeForAlltoallv(
        const uint8_t* sqePtr, const uint8_t sqeType, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo) const;

    // 缓存命中下的函数 (用于数据拷贝类SQE的刷新)
    HcclResult UpdateTransferSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo, const std::vector<OpUnfoldMemRange>& curUserInputMemRanges,
        const std::vector<OpUnfoldMemRange>& curUserOutputMemRanges, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo&
            alltoallvSendRecvInfo); // 针对alltoallv算子刷新数据拷贝类的SQE (memcpy / cache-memcpy placeholder)
    HcclResult GetTransferCountForAlltoallv(
        uint64_t& count, uint64_t& size, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo) const; // 针对alltoallv算子, 根据地址确定rank及数据拷贝大小
    HcclResult UpdateMemcpySqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo, const std::vector<OpUnfoldMemRange>& curUserInputMemRanges,
        const std::vector<OpUnfoldMemRange>& curUserOutputMemRanges, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新Memcpy SQE
    HcclResult UpdateMemcpyPlaceholderSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo, const std::vector<OpUnfoldMemRange>& curUserInputMemRanges,
        const std::vector<OpUnfoldMemRange>& curUserOutputMemRanges, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新CacheMemcpyPlaceholder SQE
    HcclResult RefreshSqeAddr(
        uint64_t& sqeAddr, const uint32_t rankId, const std::vector<OpUnfoldMemRange>& cachedMemRanges,
        const std::vector<OpUnfoldMemRange>& curMemRanges, const bool isAlltoallv,
        const uint64_t offset) const; // 根据range判断是否需要刷新, 根据计算/给定的offset进行刷新

    // 缓存命中下的函数 (用于同步类SQE的刷新)
    HcclResult UpdateSyncSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const RefreshAddrInfo& srcRefreshAddrInfo,
        const RefreshAddrInfo& dstRefreshAddrInfo, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo); // 针对alltoallv算子刷新同步类的SQE (notify / write-value /
                                                             // cache-notify / cache-write)
    HcclResult GetTransferCountForAlltoallv(
        uint64_t& count, uint64_t& size, const uint8_t* sqePtr, const uint8_t* sqeTypePtr,
        const AlltoallvMetadata& alltoallvMetadata, const AlltoallvSendRecvInfo& alltoallvSendRecvInfo)
        const; // 针对alltoallv算子, 根据notifyId/signalAddr确定rank及数据拷贝大小
    HcclResult UpdateNotifyPlaceholderSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新cache-notify placeholder
    HcclResult UpdateWritePlaceholderSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新cache-write placeholder
    HcclResult UpdateNotifySqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新notify SQE
    HcclResult UpdateWriteValueSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新WriteValue SQE
    HcclResult UpdateMemcpyRecordSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新MemcpyRecord SQE
    HcclResult UpdateMemcpyRecordPlaceholderSqeForAlltoallv(
        uint8_t* sqePtr, uint8_t* sqeTypePtr, const uint16_t curTaskId, const AlltoallvMetadata& alltoallvMetadata,
        const AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const uint64_t count,
        const uint64_t size); // 针对alltoallv算子刷新cache-memcpy-record placeholder SQE
    void SetCachePlaceholderHeaderForAlltoallv(const uint16_t streamId, const uint16_t taskId, uint8_t* sqePtr);

    std::vector<uint8_t*>
        sqeArrays_; // 多段连续的SQE数组 (每段连续的SQE不超过HCCL_SQE_SIZE * HCCL_PER_LAUNCH_SQE_CNT bytes)
    std::vector<uint8_t*> sqeTypeArrays_;         // 每段每个SQE的type
    std::vector<AicpuDfxInfo*> sqeDfxInfoArrays_; // 每段每个SQE的DfxInfo
    std::vector<int32_t> streamIds_;              // 每段SQE对应的actual stream ID
    std::vector<uint32_t> streamSeqIdxes_; // 每段SQE对应的sequential stream index (sequential是指将mainStream +
                                           // slaveStreams顺序起来看, 0代表mainStream, 1代表slaveStreams[0])
    std::vector<std::vector<RefreshAddrInfo>> srcRefreshAddrInfoArrays_; // 每段每个SQE中dstAddr (if any)对应的刷新信息
    std::vector<std::vector<RefreshAddrInfo>> dstRefreshAddrInfoArrays_; // 每段每个SQE中dstAddr (if any)对应的刷新信息

    std::vector<OpUnfoldMemRange> userInputMemRanges_;  // 当前通信域每个rank的user input memory range
    std::vector<OpUnfoldMemRange> userOutputMemRanges_; // 当前通信域每个rank的user output memory range
};

}; // namespace hccl

#endif // __OP_UNFOLD_CACHE_ENTRY_H__
