/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_TASK_CACHE_ENTRY_H
#define HCOMM_AICPU_TASK_CACHE_ENTRY_H

#include <cstdint>
#include <vector>
#include <unordered_map>

#include "ub_conn_lite.h"
#include "udma_data_struct.h"
#include "ub_transport_lite_impl.h"
#include "rtsq_a5.h"
#include "dfx_profiling_handler_lite.h"
#include "aicpu_ts_thread.h"
#include "sqe.h"
#include "res_pub.h"

namespace hcomm {

// 注意: 与ub_conn_lite.cc保持一致
constexpr uint32_t WRITE_WITH_NOTIFY_OPCODE = 0x5;

// 记录wqeTaskArrayInfos_中的每一段WQE数组, 对应的DbSqe在sqeArrayInfos_中的位置
struct DbSqeLocation {
    uint32_t sqeArrayIdx = 0; // sqeArrayInfos_中第几个SQE数组
    uint32_t dbSqeIdx = 0;    // sqeArrayInfos_[sqeArrayIdx]数组中第几个SQE是DbSqe

    bool operator==(const DbSqeLocation& other) const
    {
        return sqeArrayIdx == other.sqeArrayIdx && dbSqeIdx == other.dbSqeIdx;
    }
};

} // namespace hcomm

namespace std {
template <>
struct hash<hcomm::DbSqeLocation> {
    inline size_t operator()(const hcomm::DbSqeLocation& loc) const noexcept
    {
        constexpr uint32_t UINT32_BIT_WIDTH = 32;
        return (static_cast<size_t>(loc.sqeArrayIdx) << UINT32_BIT_WIDTH) | loc.dbSqeIdx;
    }
};
} // namespace std

namespace hcomm {

enum class TaskArrayType : uint8_t {
    kTaskArrayTypeInvalid = 0,
    kTaskArrayTypeSqe = 1,
    kTaskArrayTypeWqe = 2,
};

struct AddrRefreshInfo {
    explicit AddrRefreshInfo();
    explicit AddrRefreshInfo(const uint32_t curMemIdx);
    explicit AddrRefreshInfo(const AddrRefreshInfo& other);
    ~AddrRefreshInfo();

    const AddrRefreshInfo& operator=(const AddrRefreshInfo& other); // 拷贝赋值操作符

    bool needRefresh
        = false; // false: fixed memory (例如硬件地址, ccl buffer); true: dynamic memory (e.g., user memory)
    uint32_t memIdx = 0; // 第几个memory range (cachedBaseAddrs_ + cachedSizes_)
    size_t offset = 0;   // 刷新地址的偏移
};

struct SqeArrayInfo {
    uint8_t* sqeArray = nullptr;
    Hccl::RtsqA5* rtsqPtr = nullptr;
    hccl::AicpuTsThread* aicpuTsThreadPtr = nullptr;
    uint64_t sqeCount = 0;
    std::vector<AddrRefreshInfo> srcAddrRefreshInfoArray;
    std::vector<AddrRefreshInfo> dstAddrRefreshInfoArray;

    uint64_t GetSize() const
    {
        return sqeCount * Hccl::AC_SQE_SIZE + sizeof(Hccl::RtsqA5*) + sizeof(hccl::AicpuTsThread*) + sizeof(uint64_t)
               + sizeof(AddrRefreshInfo) * sqeCount + sizeof(AddrRefreshInfo) * sqeCount;
    }
};

struct WqeTaskArrayInfo {
    std::vector<Hccl::WqeTask> wqeTaskArray;
    Hccl::UbConnLite* ubConnLitePtr = nullptr;
    Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr = nullptr;
    DbSqeLocation dbSqeLocation; // 根据DbSqeLocation定位对应的SQE数组和其中的DbSqe
    std::vector<AddrRefreshInfo> locAddrRefreshInfoArray;
    std::vector<AddrRefreshInfo> rmtAddrRefreshInfoArray;

    uint64_t GetSize() const
    {
        return wqeTaskArray.size() * sizeof(Hccl::WqeTask) + sizeof(Hccl::UbConnLite*)
               + sizeof(Hccl::UbTransportLiteImpl*) + sizeof(DbSqeLocation)
               + sizeof(AddrRefreshInfo) * wqeTaskArray.size() + sizeof(AddrRefreshInfo) * wqeTaskArray.size();
    }
};

// DbSqe的临时信息, 用于cache miss时构造DbSqeLocation (DbSqe所在的SQE数组插入缓存时才能确定)
struct DbSqeTmpInfo {
    uint32_t wqeArrayIdx = 0;
    uint32_t dbSqeIdx = 0;
    bool isReportTask = false;
    Hccl::DbSqeProfInfo dbSqeProfInfo;
};

// DbSqe的profiling信息, 用于cache hit时填充DfxTaskInfo
struct DbSqeProfAndRefreshInfo {
    Hccl::DbSqeProfInfo dbSqeProfInfo;

    // 用于刷新DbSqeProfInfo中的地址, SubmitCacheEntry时设置, RefreshAndLaunch时使用
    AddrRefreshInfo locAddrRefreshInfo;
    AddrRefreshInfo rmtAddrRefreshInfo;

    uint32_t wqeArrayIdx; // 反向定位DbSqe对应的WQE数组
};

struct TokenInfo {
    bool needLocTokenIdFlag = false;
    uint32_t locTokenId = 0;

    bool needRmtTokenIdAndValueFlag = false;
    uint32_t rmtTokenId = 0;
    uint32_t rmtTokenValue = 0;
};

enum class AddrType : uint8_t { kAddrTypeUnknown = 0, kAddrTypeLocal = 1, kAddrTypeRemote = 2 };

// aicpu task cache单向依赖RtsqA5/UbConnLite, 下发SQE/WQE
// aicpu task cache单向依赖UbTransportLiteImpl, 获取token id/value
// aicpu task cache单向依赖AicpuTsThread/UbTransportLiteImpl, 按需填充DfxTaskInfo并上报profiling
// 注意: aicpu task cache通过在RtsqA5/UbConnLite注册回调函数, 捕捉下发的SQE/WQE并插入缓存
class AicpuTaskCacheEntry {
public:
    explicit AicpuTaskCacheEntry();
    ~AicpuTaskCacheEntry();

    // Cache admission (cache miss)
    HcclResult
    InitCacheEntry(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count); // 算子展开前保存地址信息
    HcclResult AddSqeArray(
        Hccl::RtsqA5* rtsqPtr, hccl::AicpuTsThread* aicpuTsThreadPtr, const uint64_t sqeCount, const uint8_t* sqeArray,
        const uint32_t streamId);
    HcclResult AddWqeArray(
        Hccl::UbConnLite* ubConnLitePtr, Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr,
        const std::vector<Hccl::WqeTask>& wqeTasks, const uint32_t streamId, const uint32_t dbSqeIdx,
        const bool isReportTask, const Hccl::DbSqeProfInfo& dbSqeProfInfo);
    HcclResult SubmitCacheEntry(); // 算子展开后, 更新AddrRefreshInfo和token信息
    inline uint64_t GetEntryBytes() const { return entryBytes_; }

    // Cache hit
    // 注意: 如果需要支持profiling, 参考AicpuTsThread和UbTransportLiteImpl填充DfxTaskInfo并经NextTaskSlot上报
    // 注意: inplace刷新缓存的task, 下发完成后需要更新缓存的user input/output memory range
    HcclResult RefreshAndLaunch(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count);

private:
    using UbTransportLiteImplHandle = void*; // UbTransportLiteImpl*

    inline static void CombineUint32ToUint64(uint64_t& addr, const uint32_t high, const uint32_t low)
    {
        constexpr uint64_t uintBitWidth = 32;
        addr = (static_cast<uint64_t>(high) << uintBitWidth) | static_cast<uint64_t>(low);
        return;
    }

    inline static void SplitUint64ToUint32(const uint64_t addr, uint32_t& high, uint32_t& low)
    {
        constexpr uint64_t uintBitWidth = 32;
        high = static_cast<uint32_t>(addr >> uintBitWidth);
        low = static_cast<uint32_t>(addr & 0xFFFFFFFFULL);
        return;
    }

    inline static bool InRange(const uint64_t baseAddr, const uint64_t memSize, const uint64_t addr);

    inline HcclResult
    AddSqeArray_(uint8_t* newSqeArray, const size_t sqeBytes, const uint8_t* sqeArray, const uint32_t streamId);

    // 插入WQE/SQE数组时, 更新AddrRefreshInfo
    HcclResult UpdateSqeAddrRefreshInfo_(
        const uint8_t* sqePtr, AddrRefreshInfo& srcAddrRefreshInfo, AddrRefreshInfo& dstAddrRefreshInfo) const;
    HcclResult UpdateWqeAddrRefreshInfoAndTokenInfo_(
        const Hccl::WqeTask& wqeTask, AddrRefreshInfo& locAddrRefreshInfo, AddrRefreshInfo& rmtAddrRefreshInfo,
        std::vector<TokenInfo>& tokenInfos);
    inline HcclResult UpdateTokenFlagsByAddrRefreshInfo_(
        const AddrRefreshInfo& addrRefreshInfo, std::vector<TokenInfo>& tokenInfos, bool isLoc) const;
    inline HcclResult UpdateAddrRefreshInfo_(
        const uint32_t addrLow, const uint32_t addrHigh, AddrRefreshInfo& addrRefreshInfo,
        const AddrType addrType = AddrType::kAddrTypeUnknown) const
    {
        // 拼接地址
        uint64_t addr = 0;
        AicpuTaskCacheEntry::CombineUint32ToUint64(addr, addrHigh, addrLow);
        return UpdateAddrRefreshInfo_(addr, addrRefreshInfo, addrType);
    }
    HcclResult UpdateAddrRefreshInfo_(
        const uint64_t addr, AddrRefreshInfo& addrRefreshInfo,
        const AddrType addrType = AddrType::kAddrTypeUnknown) const;

    // 刷新下发SQE
    inline HcclResult RefreshSqeTasks_(const SqeArrayInfo& sqeArrayInfo, const uint64_t* baseAddrs);
    inline HcclResult LaunchSqeTasks_(const SqeArrayInfo& sqeArrayInfo) const;

    // 刷新下发WQE, 并刷新对应的DbSqe
    inline HcclResult RefreshWqeTasks_(
        WqeTaskArrayInfo& wqeTaskArrayInfo, const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count);
    inline HcclResult LaunchWqeTasks_(WqeTaskArrayInfo& wqeTaskArrayInfo) const;
    inline HcclResult RefreshDbSqe_(WqeTaskArrayInfo& wqeTaskArrayInfo);

    // 根据AddrRefreshInfo刷新WQE/SQE/DbSqeProfInfo地址字段
    inline void RefreshTaskAddr_(
        uint32_t& addrLow, uint32_t& addrHigh, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const;
    inline void
    RefreshTaskAddr_(uint64_t& addr, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const;

    // 根据刷新后的新地址, 按需刷新WQE的token id/value
    inline HcclResult RefreshWqeLocTokenId_(
        uint32_t& tokenId, const AddrRefreshInfo& addrRefreshInfo, const std::vector<TokenInfo>& tokenInfos) const;
    inline HcclResult RefreshWqeRmtTokenIdAndValue_(
        uint32_t& tokenId, uint32_t& tokenValue, const AddrRefreshInfo& addrRefreshInfo,
        const std::vector<TokenInfo>& tokenInfos) const;

    // 使能profiling时, 对每个刷新的SQE填充DfxTaskInfo并经NextTaskSlot上报
    HcclResult ReportSqeArrayProfiling_(
        size_t arrayIdx, const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count);
    HcclResult ReportSqeProfiling_(
        uint8_t* sqePtr, size_t arrayIdx, uint32_t sqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
        const uint32_t count, Hccl::StreamLite* streamLite, const u32 sqId);
    HcclResult ReportDbSqeProfiling_(
        uint8_t* dbSqePtr, size_t arrayIdx, uint32_t dbSqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
        const uint32_t count, Hccl::StreamLite* streamLite, const u32 sqId, const u32 taskId);

    // SubmitCacheEntry子方法
    inline HcclResult SubmitSqeAddrRefreshInfo_();
    inline HcclResult SubmitWqeAddrRefreshInfoAndTokenInfo_();
    inline HcclResult SubmitDbSqeProfRefreshInfo_();
    inline HcclResult ValidateLaunchOrder_();

    // RefreshAndLaunch子方法
    inline HcclResult RefreshTokenInfos_(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count);
    inline HcclResult
    LaunchTasksByOrder_(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count, bool needTaskParam);
    inline HcclResult PrintRefreshResult_(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count);

    // RefreshSqeTasks_子方法
    inline HcclResult RefreshOneSqe_(
        uint8_t* sqeArrayPtr, const AddrRefreshInfo& srcAddrRefreshInfo, const AddrRefreshInfo& dstAddrRefreshInfo,
        const uint64_t* baseAddrs) const;

    // RefreshWqeTasks_子方法
    inline void DumpWqeTasksHeader_(uint64_t wqeCount, const Hccl::UbConnLite* ubConnLitePtr) const;
    inline HcclResult
    DumpWqeTasksPerWqe_(size_t wqeIdx, const Hccl::WqeTask& wqeTask, const Hccl::UbConnLite* ubConnLitePtr) const;
    inline HcclResult RefreshWqeRead_(
        Hccl::WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
        const uint64_t* baseAddrs, const std::vector<TokenInfo>& tokenInfos) const;
    inline HcclResult RefreshWqeWrite_(
        Hccl::WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
        const uint64_t* baseAddrs, const std::vector<TokenInfo>& tokenInfos);
    inline HcclResult RefreshWqeWriteWithNotify_(
        Hccl::WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
        const uint64_t* baseAddrs, const std::vector<TokenInfo>& tokenInfos) const;

    // ReportDbSqeProfiling_子方法
    inline HcclResult FillSlotUbDma_(
        Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
        Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr, Hccl::StreamLite* streamLite, u32 taskId) const;
    inline HcclResult FillSlotReduce_(
        Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
        Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr, Hccl::StreamLite* streamLite, u32 taskId) const;
    inline HcclResult RefreshDbSqeProfAddrs_(
        DbSqeProfAndRefreshInfo& profAndRefreshInfo, const uint64_t* baseAddrs, const uint64_t* memSizes,
        const uint32_t count) const;

    // ReportSqeProfiling_子方法
    inline HcclResult
    FillSlotNotify_(Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, Hccl::StreamLite* streamLite, u32 taskId) const;
    inline HcclResult
    FillSlotSdma_(Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, Hccl::StreamLite* streamLite, u32 taskId) const;
    inline void FillSlotCommonFields_(
        Hccl::DfxTaskInfo* slot, const Hccl::StreamLite* streamLite, u32 taskId, u8 linkType, u8 transportType,
        u64 channelHandle) const;
    inline u8 ConvertSdmaOpCodeToReduceOp_(uint8_t opcode) const;
    inline u8 GetUbLinkTypeVal_(const Hccl::UbTransportLiteImpl* ubTransportLiteImplPtr) const;

    // 统计当前cache entry的bytes开销
    uint64_t entryBytes_ = 0;

    // AddWqeArray时临时记录该段WQE数组对应的DbSqe的streamId, sqeIdx, 和profInfo;
    // 后续AddSqeArray时, 根据streamId才能确定对应的DbSqe在sqeArrayInfos_中的arrayIdx,
    //     从而确定DbSqeLocation并更新dbSqeLocInfoMap_;
    // 注意: 只有第一次cache miss时, 才会使用该map; 第一次算子展开完成后, 该map一定为空, 因此无需更新entryBytes_
    std::unordered_map<uint32_t, std::vector<DbSqeTmpInfo>> streamIdToDbSqeTmpInfoMap_;

    // 多段SQE数组: 每段SQE数组对应一次LaunchTask, 以及相应的RtsqA5指针
    std::vector<SqeArrayInfo> sqeArrayInfos_;

    // 多段WQE数组: 每段WQE数组对应多次ProcessOneWqe/ProcessOneWqeWithNotify (按256MiB切分, 但始终只对应**一个**DbSqe),
    //     以及相应的ubConnLite指针和DbSqeLocation
    std::vector<WqeTaskArrayInfo> wqeTaskArrayInfos_;

    // 维护DbSqeLocation-DbSqeProfAndRefreshInfo的映射 (只有profiling使能时, 才需要维护)
    // 注意: dbSqeLocInfoMap_不计入entryBytes_, 避免开启profiling与关闭profiling时aicpu task cache行为不一致
    std::unordered_map<DbSqeLocation, DbSqeProfAndRefreshInfo> dbSqeLocInfoMap_; // AddSqeArray时更新

    // 下发顺序
    std::vector<TaskArrayType> launchOrder_; // 大小一定为SQE+WQE数组之和

    // Cached memory ranges: InitCacheEntry时初始化, SubmitCacheEntry时用于计算AddrRefreshInfo,
    // RefreshAndLaunch时无需更新
    // 注意: 假设vector中前两个元素为curRank的local user input&output
    std::vector<uint64_t> cachedBaseAddrs_;
    std::vector<uint64_t> cachedMemSizes_;

    // 每个UbTransportLiteImplHandle 每段动态内存 对应的token信息
    std::unordered_map<UbTransportLiteImplHandle, std::vector<TokenInfo>> tokenInfosMap_;

    // 合并task-level config debug日志打印判断 (构造cache entry时设置)
    bool isTaskConfigDebug_ = false;
};

} // namespace hcomm

#endif // HCOMM_AICPU_TASK_CACHE_ENTRY_H
