/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEM_TRANSPORT_LITE_H
#define UB_MEM_TRANSPORT_LITE_H

#include "cast_utils.h"
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <mutex>
#include "base_transport_lite_impl.h"
#include "notify_lite.h"
#include "task_param.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_conn_lite.h"
#include "kernel_param_lite.h"
#include "hcomm_primitives.h"
#include "ub_conn_lite.h"
#include "rtsq_a5.h"
#include "aicpu_task_utils.h"

namespace hcomm {
class AicpuTaskCacheEntry;
}

namespace Hccl {

class UbTransportLiteImpl : public BaseTransportLiteImpl {
public:
    explicit UbTransportLiteImpl(
        std::vector<char>& uniqueId,
        std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback);

    UbTransportLiteImpl(std::vector<char>& uniqueId);
    void Init(std::vector<char>& uniqueId);

    ~UbTransportLiteImpl() override;

    std::string Describe() const override;

    Buffer GetRmtBuffer(u32 index) override;

    Eid GetLocEid() const;
    Eid GetRmtEid() const;
    uint64_t GetJettyHandle() const;
    uint32_t GetJettyId() const;
    uint32_t GetTpn() const;

    void Post(u32 index, const StreamLite& stream) override;

    void Wait(u32 index, const StreamLite& stream) override;

    void WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout) override;

    void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override;

    void Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override;

    void ReadReduce(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override;

    void WriteReduce(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override;

    void WriteWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream) override;

    void WriteReduceWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
        const StreamLite& stream) override;

    void BatchOneSidedWrite(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream) override;

    void BatchOneSidedRead(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream) override;

    void BatchTransfer(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<TransferOp>& transferOp, const StreamLite& stream) override;
    // 子类独有方法，支持所有操作类型，用于aicpu场景批量下发任务
    void BatchTransferAll(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<TransferOp>& transferOp, const std::vector<uint32_t>& notifyIdxs, const StreamLite& stream);

    inline void BatchTransferAllWqe_(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<TransferOp>& transferOp, const std::vector<uint32_t>& notifyIdxs, const StreamLite& stream,
        RmaConnLite* conn, u64& totalSize);

    void Drain(const StreamLite& stream) override;

    HcclResult BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite) override;
    HcclResult Fence() override;

    HcclResult Clean();
    HcclResult Resume(std::vector<char>& uniqueId);
    void SetTaskExceptionEnable(bool flag) { taskExceptionEnable_ = flag; }

    HcclResult ExecuteBatchTransfer(
        StreamLite* streamLitePtr, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum);

    // 用于aicpu task cache
    inline HcclResult SetNeedCacheTaskCallback(std::function<bool()> callback)
    {
        CHK_PTR_NULL(callback);
        needCacheTaskCallback_ = callback;
        return HCCL_SUCCESS;
    }
    inline HcclResult SetAddWqeArrayCallback(std::function<HcclResult(
                                                 UbConnLite*, UbTransportLiteImpl*, const std::vector<WqeTask>&,
                                                 const uint32_t, const uint32_t, const bool, const DbSqeProfInfo&)>
                                                 callback)
    {
        CHK_PTR_NULL(callback);
        addWqeArrayCallback_ = callback;
        return HCCL_SUCCESS;
    }

    std::function<void(u32, u32, const TaskParam&)> GetCallback() { return callback_; }

    friend class hcomm::AicpuTaskCacheEntry;

private:
    u32 notifyNum{0};
    u32 bufferNum{0};
    u32 rmtbufferNum{0};
    u32 connNum{0};
    DfxLinkType linkType_{DfxLinkType::UB};
    bool fence_{false};
    bool taskExceptionEnable_{true};

    struct RmtUbBufLite {
        u64 addr;
        u64 size;
        u32 tokenId;
        u32 tokenValue;
        u32 notifyId;
        std::string Describe() const
        {
            return StringFormat("RmtUbBufLite[addr=0x%llx, size=%llu, notifyId=%u]", addr, size, notifyId);
        }
    };

    struct LocUbBufLite {
        u64 addr;
        u64 size;
        u32 tokenId;
        u32 tokenValue;
        std::string Describe() const { return StringFormat("LocUbBufLite[addr=0x%llx, size=%llu]", addr, size); }
    };

    struct DrainNotify {
        u64 addr;
        u64 size;
        u32 tokenId;
        u32 tokenValue;
        u32 notifyId;
        std::string Describe() const
        {
            return StringFormat("DrainNotify[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
        }
    };

    std::vector<char> wqeData;    // connection返回的WQE内容
    ConnLiteOperationOut connOut; // connection的输出

    void ClearConnOut();

    using RmtUbBufLiteVec = std::vector<RmtUbBufLite>;
    using RmtUbBufLiteMap = std::map<uintptr_t, RmtUbBufLite>;
    using LocUbBufLiteMap = std::map<uintptr_t, LocUbBufLite>;
    MAKE_ENUM(RmaUbBufType, NOTIFY, BUFFER)
    RmtUbBufLiteVec rmtNotifyVec;
    RmtUbBufLiteVec rmtBufferVec;
    RmtUbBufLiteMap rmtBufferMap; // 性能优化使用
    LocUbBufLiteMap locBufferMap;

    RmtRmaBufSliceLite GetRmtNotifySliceLite(u32 index);
    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(const Buffer& rmtBuf);

    RmaBufSliceLite GetRmaBufSlicelite(const RmaBufferLite& lite) const;
    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(const RmaBufferLite& lite) const;

    std::vector<std::unique_ptr<NotifyLite>> locNotifyVec;

    std::mutex drainMtx_;
    DrainNotify drainNotify_{};
    RmtUbBufLite rmtDrainBuffer_{};

    // N秒快恢需要清理的两个资源
    std::vector<std::vector<char>> connUniqueIdVec;
    std::vector<RmaConnLite*> connVec;

    std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback_{nullptr};

    void ProfilingProcess(void* src, void* dst, u64 size, const StreamLite& stream, DmaOp dmaOp, u32 taskId);

    inline void BuildDbSqeProfInfoForProfilingProcess(
        void* loc, void* rmt, u64 size, DmaOp dmaOp, DbSqeProfInfo& dbSqeProfInfo) const
    {
        FillDbSqeProfInfoDmaPub(rmt, size, dmaOp, dbSqeProfInfo);

        // 构造DbSqeProfInfo (注意: 其他字段已在FillDbSqeProfInfo设置)
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_UB;
        dbSqeProfInfo.locAddr = ReinterpretAs<uint64_t>(loc);
    }

    void ReduceProfilingProcess(
        void* src, void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId);

    inline void BuildDbSqeProfInfoForReduceProfilingProcess(
        void* loc, void* rmt, u64 size, const ReduceIn& reduceIn, DbSqeProfInfo& dbSqeProfInfo) const
    {
        // 构造DbSqeProfInfo
        dbSqeProfInfo.isValid = true;
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_UB_REDUCE_INLINE;
        FillDbSqeProfInfoReducePub(loc, rmt, size, reduceIn, dbSqeProfInfo);
    }

    void ParseLocNotifyVec(std::vector<char>& data);

    void ParseRmtBufferVec(std::vector<char>& data, RmaUbBufType rmtType);

    void ParseLocBufferMap(std::vector<char>& data);

    void ParseDrainResource(std::vector<char>& data);

    void ParseConnVec(std::vector<char>& data);

    void BuildUbDbSendTask(const StreamLite& stream, const UbJettyLiteId& jettyLiteId, u32 pi);

    void BuildNotifyWaitTask(const StreamLite& stream, u32 notifyId);

    void CheckConnVec(const std::string& desc);

    void SetFenceConfig(SqeConfigLite& cfg);

    bool IsReportTask() const;

    void ExecProfiling(
        const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
        const BaseTransportLiteImpl::TransferOp& transferOp, const StreamLite& stream, u32 taskId);

    inline void BuildDbSqeProfInfoForExecProfiling(
        const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
        const BaseTransportLiteImpl::TransferOp& transferOp, DbSqeProfInfo& dbSqeProfInfo)
    {
        if (transferOp.reduceIn.reduceOp == ReduceOp::INVALID) {
            DmaOp dmaOp = DmaOp::HCCL_DMA_WRITE;
            if (transferOp.transType == TransferType::READ) {
                dmaOp = DmaOp::HCCL_DMA_READ;
            }
            BuildDbSqeProfInfoForProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, dmaOp, dbSqeProfInfo);
        } else {
            BuildDbSqeProfInfoForReduceProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn,
                dbSqeProfInfo);
        }
    }

    inline void AddTaskCallback(const StreamLite& stream, u32 taskId, const TaskParam& taskParam)
    {
        if (callback_ != nullptr) {
            callback_(stream.GetSqId(), taskId, taskParam);
        }
    }

    inline void FillTaskParamDmaPub(TaskParam& taskParam, void* dst, u64 size, DmaOp dmaOp) const
    {
        taskParam.taskPara.DMA.dst = dst;
        taskParam.taskPara.DMA.size = size;
        taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
        taskParam.taskPara.DMA.notifyValue = 0xffffffff;
        taskParam.taskPara.DMA.linkType = linkType_;
        taskParam.taskPara.DMA.dmaOp = dmaOp;
        taskParam.taskPara.DMA.locEid = GetLocEid();
        taskParam.taskPara.DMA.rmtEid = GetRmtEid();
    }

    inline void FillDbSqeProfInfoDmaPub(void* rmt, u64 size, DmaOp dmaOp, DbSqeProfInfo& dbSqeProfInfo) const
    {
        // 构造DbSqeProfInfo
        dbSqeProfInfo.isValid = true;
        dbSqeProfInfo.rmtAddr = ReinterpretAs<uint64_t>(rmt);
        dbSqeProfInfo.size = size;
        dbSqeProfInfo.dmaOp = dmaOp;
        dbSqeProfInfo.locEid = GetLocEid();
        dbSqeProfInfo.rmtEid = GetRmtEid();
        dbSqeProfInfo.jettyHandle = GetJettyHandle();
        dbSqeProfInfo.jettyId = GetJettyId();
        dbSqeProfInfo.tpn = GetTpn();
    }

    inline void
    FillTaskParamReducePub(TaskParam& taskParam, void* src, void* dst, u64 size, const ReduceIn& reduceIn) const
    {
        taskParam.taskPara.Reduce.src = src;
        taskParam.taskPara.Reduce.dst = dst;
        taskParam.taskPara.Reduce.size = size;
        taskParam.taskPara.Reduce.notifyValue = 1;
        taskParam.taskPara.Reduce.linkType = linkType_;
        taskParam.taskPara.Reduce.reduceOp = ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp);
        taskParam.taskPara.Reduce.dataType = DataTypeToHcclDataType(reduceIn.dataType);
        taskParam.taskPara.Reduce.locEid = GetLocEid();
        taskParam.taskPara.Reduce.rmtEid = GetRmtEid();
    }

    inline void FillDbSqeProfInfoReducePub(
        void* loc, void* rmt, u64 size, const ReduceIn& reduceIn, DbSqeProfInfo& dbSqeProfInfo) const
    {
        dbSqeProfInfo.locAddr = ReinterpretAs<uint64_t>(loc);
        dbSqeProfInfo.rmtAddr = ReinterpretAs<uint64_t>(rmt);
        dbSqeProfInfo.size = size;
        dbSqeProfInfo.locEid = GetLocEid();
        dbSqeProfInfo.rmtEid = GetRmtEid();
        dbSqeProfInfo.reduceOp = ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp);
        dbSqeProfInfo.dataType = DataTypeToHcclDataType(reduceIn.dataType);
        dbSqeProfInfo.jettyHandle = GetJettyHandle();
        dbSqeProfInfo.jettyId = GetJettyId();
        dbSqeProfInfo.tpn = GetTpn();
    }

    void ExecProfilingAll(
        const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
        const BaseTransportLiteImpl::TransferOp& transferOp, const StreamLite& stream, u32 taskId,
        const uint32_t notifyIdx);

    inline void BuildDbSqeProfInfoForExecProfilingAll(
        const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
        const BaseTransportLiteImpl::TransferOp& transferOp, const uint32_t notifyIdx, DbSqeProfInfo& dbSqeProfInfo)
    {
        if (transferOp.transType == TransferType::READ) {
            BuildDbSqeProfInfoForProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, DmaOp::HCCL_DMA_READ,
                dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::WRITE) {
            BuildDbSqeProfInfoForProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, DmaOp::HCCL_DMA_WRITE,
                dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::READ_REDUCE) {
            BuildDbSqeProfInfoForReduceProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn,
                dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::WRITE_REDUCE) {
            BuildDbSqeProfInfoForReduceProfilingProcess(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn,
                dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::WRITE_WITH_NOTIFY) {
            BuildDbSqeProfInfoForWriteWithNotify(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize,
                GetRmtNotifySliceLite(notifyIdx).GetNotifyId(), dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::WRITE_REDUCE_WITH_NOTIFY) {
            BuildDbSqeProfInfoForWriteReduceWithNotify(
                ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
                ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn,
                GetRmtNotifySliceLite(notifyIdx).GetNotifyId(), dbSqeProfInfo);
        } else if (transferOp.transType == TransferType::NOTIFY_RECORD) {
            BuildDbSqeProfInfoForNotifyRecord(
                ReinterpretAs<void*>(GetRmtNotifySliceLite(notifyIdx).GetAddr()),
                GetRmtNotifySliceLite(notifyIdx).GetSize(), GetRmtNotifySliceLite(notifyIdx).GetNotifyId(),
                dbSqeProfInfo);
        }
    }

    void
    WriteWithNotifyProfilingProcess(void* src, void* dst, u64 size, const StreamLite& stream, u32 taskId, u64 notifyId);

    inline void BuildDbSqeProfInfoForWriteWithNotify(
        void* loc, void* rmt, u64 size, u64 notifyId, DbSqeProfInfo& dbSqeProfInfo) const
    {
        FillDbSqeProfInfoDmaPub(rmt, size, DmaOp::HCCL_DMA_WRITE, dbSqeProfInfo);

        // 构造DbSqeProfInfo (注意: 其他字段已在FillDbSqeProfInfo设置)
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
        dbSqeProfInfo.locAddr = ReinterpretAs<uint64_t>(loc);
        dbSqeProfInfo.notifyId = notifyId;
    }

    void WriteReduceWithNotifyProfilingProcess(
        void* src, void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId, u64 notifyId);

    inline void BuildDbSqeProfInfoForWriteReduceWithNotify(
        void* loc, void* rmt, u64 size, const ReduceIn& reduceIn, u64 notifyId, DbSqeProfInfo& dbSqeProfInfo) const
    {
        // 构造DbSqeProfInfo
        dbSqeProfInfo.isValid = true;
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
        FillDbSqeProfInfoReducePub(loc, rmt, size, reduceIn, dbSqeProfInfo);
        dbSqeProfInfo.notifyId = notifyId;
    }

    void NotifyRecordProfilingProcess(void* dst, u64 size, const StreamLite& stream, u32 taskId, u64 notifyId);

    inline void BuildDbSqeProfInfoForNotifyRecord(void* rmt, u64 size, u64 notifyId, DbSqeProfInfo& dbSqeProfInfo) const
    {
        FillDbSqeProfInfoDmaPub(rmt, size, DmaOp::HCCL_DMA_WRITE, dbSqeProfInfo);

        // 构造DbSqeProfInfo (注意: 其他字段已在FillDbSqeProfInfo设置)
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_UB_INLINE_WRITE;
        dbSqeProfInfo.notifyId = notifyId;
    }

    // 用于aicpu task cache
    std::function<bool()> needCacheTaskCallback_{nullptr};
    std::function<HcclResult(
        UbConnLite*, UbTransportLiteImpl*, const std::vector<WqeTask>&, const uint32_t, const uint32_t, const bool,
        const DbSqeProfInfo& dbSqeProfInfo)>
        addWqeArrayCallback_{nullptr};

    // 展开下发WQE前，按需设置wqe tasks
    inline void PreLaunchWqe(UbConnLite*& ubConnLitePtr, bool& needCacheTask, RmaConnLite* connPtr)
    {
        // 校验needCacheTaskCallback_
        // 注意: A5新流程下needCacheTaskCallback_一定非空; 但A5老流程下不支持aicpu task cache,
        // needCacheTaskCallback_为空;
        //     为避免A5老流程报错, 这里为空时跳过执行而非报错
        needCacheTask = false;
        if (UNLIKELY(needCacheTaskCallback_ == nullptr)) {
            HCCL_WARNING(
                "[UbTransportLiteImpl][PreLaunchWqe] needCacheTaskCallback_ is null, keep needCacheTask as false");
        } else {
            needCacheTask = needCacheTaskCallback_();
        }

        // 校验是否需要打印WQE
        bool needDumpWqe = false;
        if ((UNLIKELY((GetPlfDebugConfigValue() & PLF_TASK) != 0)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            needDumpWqe = true;
        }

        // 如果需要缓存WQE 或者 打印WQE
        if (needCacheTask || UNLIKELY(needDumpWqe)) {
            // 校验connPtr
            if (UNLIKELY(connPtr == nullptr)) {
                THROW<InternalException>("[UbTransportLiteImpl][PreLaunchWqe] connPtr is null");
            }

            // 转换ubConnLitePtr并校验
            ubConnLitePtr = dynamic_cast<UbConnLite*>(connPtr);
            if (UNLIKELY(ubConnLitePtr == nullptr)) {
                THROW<InternalException>("[UbTransportLiteImpl][PreLaunchWqe] ubConnLitePtr is null");
            }

            HcclResult ret = ubConnLitePtr->EnableWqeTasks();
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                THROW<InternalException>(
                    "[UbTransportLiteImpl][PreLaunchWqe] "
                    "ubConnLitePtr->EnableWqeTasks failed, ret %d",
                    ret);
            }
        }
    }

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx 或者 打印正常展开的WQE
    inline void PostLaunchWqe(
        const StreamLite& stream, UbConnLite* ubConnLitePtr, bool needCacheTask, const uint32_t pendingSqeCnt,
        const bool isReportTask, const DbSqeProfInfo& dbSqeProfInfo)
    {
        // 校验是否需要打印WQE
        bool needDumpWqe = false;
        if ((UNLIKELY((GetPlfDebugConfigValue() & PLF_TASK) != 0)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            needDumpWqe = true;
        }

        // 如果需要缓存WQE 或者 打印WQE
        if (needCacheTask || UNLIKELY(needDumpWqe)) {
            // 校验ubConnLitePtr
            if (UNLIKELY(ubConnLitePtr == nullptr)) {
                THROW<InternalException>("[UbTransportLiteImpl][PostLaunchWqe] ubConnLitePtr is null");
            }

            HcclResult ret = HCCL_SUCCESS;

            // 按需缓存WQE
            if (needCacheTask) {
                // 校验addWqeArrayCallback_
                // 注意: 如果needCacheTask为true, 一定是A5新流程, 所以addWqeArrayCallback_一定非空
                if (UNLIKELY(addWqeArrayCallback_ == nullptr)) {
                    THROW<InternalException>("[UbTransportLiteImpl][PostLaunchWqe] addWqeArrayCallback_ is null");
                }

                // 调用addWqeArrayCallback_函数, 缓存wqe
                // 注意: pendingSqeCnt即下发DbSqe前, SqeRingBuffer的tailSqeIdx
                ret = addWqeArrayCallback_(
                    ubConnLitePtr, this, ubConnLitePtr->GetWqeTasks(), stream.GetId(), pendingSqeCnt, isReportTask,
                    dbSqeProfInfo);
                if (UNLIKELY(ret != HCCL_SUCCESS)) {
                    THROW<InternalException>(
                        "[UbTransportLiteImpl][PostLaunchWqe] "
                        "addWqeArrayCallback_ failed, ret %d",
                        ret);
                }
            }

            // 按需打印WQE
            if (UNLIKELY(needDumpWqe)) {
                const std::vector<WqeTask>& wqeTasks = ubConnLitePtr->GetWqeTasks();
                const uint64_t wqeCount = wqeTasks.size();
                PLF_CONFIG_INFO(
                    PLF_TASK,
                    "[UbTransportLiteImpl][PostLaunchWqe] dump %llu generated WQEs "
                    "in jetty[%u, %u, %u]",
                    wqeCount, ubConnLitePtr->GetUbJettyLiteId().GetDieId(),
                    ubConnLitePtr->GetUbJettyLiteId().GetFuncId(), ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
                for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
                    PLF_CONFIG_INFO(
                        PLF_TASK,
                        "[UbTransportLiteImpl][PostLaunchWqe] %uth generated WQE "
                        "in jetty[%u, %u, %u]",
                        wqeIdx, ubConnLitePtr->GetUbJettyLiteId().GetDieId(),
                        ubConnLitePtr->GetUbJettyLiteId().GetFuncId(), ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
                    ret = hcomm::AicpuTaskUtils::DumpWqeContent(ReinterpretAs<const uint8_t*>(&wqeTasks[wqeIdx]));
                    if (UNLIKELY(ret != HCCL_SUCCESS)) {
                        THROW<InternalException>(
                            "[UbTransportLiteImpl][PostLaunchWqe] "
                            "AicpuTaskUtils::DumpWqeContent failed, ret %d",
                            ret);
                    }
                }
            }

            // 缓存或者打印后清理wqe tasks
            ret = ubConnLitePtr->DisableWqeTasks();
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                THROW<InternalException>(
                    "[UbTransportLiteImpl][PostLaunchWqe] "
                    "ubConnLitePtr->DisableWqeTasks failed, ret %d",
                    ret);
            }
        }
    }

    void FillSlotUbDmaInfo(
        const StreamLite& stream, u32 taskId, TaskParamTypeVal taskType, u64 srcAddr, u64 dstAddr, u64 size,
        u32 notifyId) const;
    void FillSlotReduceInfo(
        const StreamLite& stream, u32 taskId, TaskParamTypeVal taskType, u64 srcAddr, u64 dstAddr, u64 size,
        u32 notifyId, u8 reduceOp) const;
    void FillSlotWaitInfo(const StreamLite& stream, u32 taskId) const;
};

} // namespace Hccl
#endif
