/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_BASE_VENDOR_OPS_H
#define RDMA_BASE_VENDOR_OPS_H

#include "cast_utils.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <securec.h>

#include "exception_util.h"
#include "internal_exception.h"
#include "log.h"
#include "rma_buf_slice_lite.h"     // RmaBufSliceLite
#include "rmt_rma_buf_slice_lite.h" // RmtRmaBufSliceLite
#include "data_type.h"              // DataType
#include "reduce_op.h"              // ReduceOp

namespace Hccl {

struct SqeConfigLite;

struct RdmaSqContextLite {
    uint32_t qpn;
    uint64_t sqVa;
    uint32_t wqeSize;
    uint32_t depth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbHwVa;
    uint64_t dbSwVa;
    uint8_t sl;
    uint64_t dbVendorSpecified;
};

struct RdmaCqContextLite {
    uint32_t cqn;
    uint64_t cqVa;
    uint32_t cqeSize;
    uint32_t cqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbHwVa;
    uint64_t dbSwVa;
    uint64_t dbVendorSpecified;
};

// DbVendorSpecified bitfield layout, matching Roce3DbEntry.dw0.bs:
//   cos      at bits 24-26 (3 bits)
//   mtuShift at bits 50-52 (3 bits)
static constexpr uint32_t UB_DB_VENDOR_COS_SHIFT = 24;
static constexpr uint32_t UB_DB_VENDOR_MTUSHIFT_SHIFT = 50;
static constexpr uint32_t UB_DB_VENDOR_FIELD_MASK = 0x7;

// Necessary helper funcs
constexpr uint32_t BITS_1BYTE = 8;
constexpr uint32_t BITS_3BYTE = 24;
constexpr uint32_t BITS_5BYTE = 40;
constexpr uint32_t BITS_7BYTE = 56;

inline uint16_t Htons16(uint16_t x) { return (((x & 0xffULL) << BITS_1BYTE) | ((x & 0xff00ULL) >> BITS_1BYTE)); }

inline uint32_t Htonl32(uint32_t x)
{
    return ((x & 0x000000ffU) << BITS_3BYTE) | ((x & 0x0000ff00U) << BITS_1BYTE) | ((x & 0x00ff0000U) >> BITS_1BYTE)
           | ((x & 0xff000000U) >> BITS_3BYTE);
}

inline uint64_t Htonll64(uint64_t x)
{
    return ((x & 0x00000000000000ffULL) << BITS_7BYTE) | ((x & 0x000000000000ff00ULL) << BITS_5BYTE)
           | ((x & 0x0000000000ff0000ULL) << BITS_3BYTE) | ((x & 0x00000000ff000000ULL) << BITS_1BYTE)
           | ((x & 0x000000ff00000000ULL) >> BITS_1BYTE) | ((x & 0x0000ff0000000000ULL) >> BITS_3BYTE)
           | ((x & 0x00ff000000000000ULL) >> BITS_5BYTE) | ((x & 0xff00000000000000ULL) >> BITS_7BYTE);
}

enum class CqPollStatus : int32_t {
    SUCCESS = 0,  /* indicate poll once cqe successfully; */
    EMPTY = -1,   /* indicate the cq is empty when poll cq; */
    ERROR = -2,   /* indicate the error when poll cq; */
    REROLL = -3,  /* 返回到repoll标识 */
    CONTINUE = 1, /* indicate continue the process */
};

class RdmaBaseOps {
public:
    RdmaBaseOps(RdmaSqContextLite* sqContext, RdmaCqContextLite* cqContext)
        : sqContext_(sqContext),
          cqContext_(cqContext)
    {}
    virtual ~RdmaBaseOps() = default;

    // 上层接口，不关心具体vendor类型
    HcclResult Read(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg)
    {
        // Read需要占用1个wr位置, 确定Sq存在空位
        constexpr int ReadWqeCount = 1;
        CHK_RET(WaitSqFree(ReadWqeCount));

        // 进入vendor特有wqe组装接口, 组装完wqe直接下发, opCode不支持直接返回HCCL_E_NOT_SUPPORT
        CHK_RET(BuildReadWqe(loc, rmt, cfg));

        // 更新Sq队列PI值
        CHK_RET(UpdateSqPI());

        return HCCL_SUCCESS;
    }

    HcclResult Write(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg)
    {
        // Write需要占用1个wr位置, 确定Sq存在空位
        constexpr int WriteWqeCount = 1;
        CHK_RET(WaitSqFree(WriteWqeCount));

        // 进入vendor特有wqe组装接口, 组装完wqe直接下发
        CHK_RET(BuildWriteWqe(loc, rmt, cfg));

        // 更新Sq队列PI值
        CHK_RET(UpdateSqPI());

        return HCCL_SUCCESS;
    }

    HcclResult WriteReduce(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, DataType dataType,
        ReduceOp reduceOp)
    {
        // Inline Reduce Write需要占用1个wr位置, 确定Sq存在空位
        constexpr int WriteReduceWqeCount = 1;
        CHK_RET(WaitSqFree(WriteReduceWqeCount));

        // 进入vendor特有wqe组装接口, 组装完wqe直接下发
        CHK_RET(BuildWriteReduceWqe(loc, rmt, cfg, dataType, reduceOp));

        // 更新Sq队列PI值
        CHK_RET(UpdateSqPI());

        return HCCL_SUCCESS;
    }

    HcclResult PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList)
    {
        auto timeLimit = std::chrono::milliseconds(timeOut);
        auto startTime = std::chrono::steady_clock::now();

        int32_t totalPollNum = 0;
        int32_t ret = 0;

        while (totalPollNum < numEntries) {
            // 逐一Poll Cq，并处理cqe
            ret = PollCqImpl(numEntries - totalPollNum, errList);
            if (ret == static_cast<int32_t>(CqPollStatus::ERROR)) {
                HCCL_ERROR("[RdmaBaseOps::%s][Poll cq] Poll Cq Error.", __func__);
                return HCCL_E_REMOTE;
            }

            if (ret > 0) {
                // Update cqe in this loop
                totalPollNum += ret;

                // Update Sq Tail
                sqTail_ += ret;

                // continue poll cqe
                continue;
            }

            if ((std::chrono::steady_clock::now() - startTime) > timeLimit) {
                HCCL_ERROR(
                    "[RdmaBaseOps::%s][Poll cq] Poll Cq timeout, expected[%d], actual[%d], lastRet[%d]", __func__,
                    numEntries, totalPollNum, ret);
                return HCCL_E_TIMEOUT;
            }
        }

        HCCL_INFO(
            "[RdmaBaseOps::%s][Poll cq] Poll Cq success, expected[%d], actual[%d]", __func__, numEntries, totalPollNum);
        return HCCL_SUCCESS;
    }

    // 准备Doorbell(厂商实现)
    virtual HcclResult BuildDoorbell(u64& dbAddr, u64& dbValue) = 0;

    // 准备CqDoorbell(厂商实现)
    virtual HcclResult BuildCqDoorbell(u64& dbAddr, u64& dbValue) = 0;

protected:
    // 软件侧只维护Sq PI，Sq CI由硬件维护
    u32 sqHead_{0};
    u32 sqTail_{0};

    u32 cqHead_{0};
    u32 cqTail_{0};
    bool cqDbFlush_ = false;

    RdmaSqContextLite* sqContext_;
    RdmaCqContextLite* cqContext_;

    // 默认超时时间 30 ms
    const std::chrono::milliseconds timeout_ = std::chrono::milliseconds(30U);

    // vendor扩展点: 每个原子op一个虚函数
    // 默认 NOT_SUPPORT, 各个vendor 只重写自己支持的
    virtual HcclResult BuildReadWqe(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg)
    {
        (void)loc;
        (void)rmt;
        (void)cfg;
        HCCL_ERROR("[RdmaBaseOps::%s] This backend does not support Read now.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult
    BuildWriteWqe(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg)
    {
        (void)loc;
        (void)rmt;
        (void)cfg;
        HCCL_ERROR("[RdmaBaseOps::%s] This backend does not support Write Now.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult BuildWriteReduceWqe(
        const RmaBufSliceLite& locNotify, const RmtRmaBufSliceLite& notify, const SqeConfigLite& cfg, DataType dataType,
        ReduceOp reduceOp)
    {
        (void)locNotify;
        (void)notify;
        (void)cfg;
        (void)dataType;
        (void)reduceOp;
        HCCL_ERROR("[RdmaBaseOps::%s] This backend does not support WriteReduce Now.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult WriteInvalidWqebb(uint32_t nextIdx)
    {
        (void)nextIdx;
        return HCCL_SUCCESS;
    }

    virtual int32_t PollCqImpl(int32_t numEntries, [[maybe_unused]] std::vector<int32_t>& errList)
    {
        (void)numEntries;
        HCCL_ERROR("[RdmaBaseOps::%s] This backend does not support PollCq now.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    // 搬运wqe(通用实现), 把 Wqe 写到 SQ
    HcclResult CommitWqe(const void* wqe, uint32_t wqeSize)
    {
        HCCL_INFO("[RdmaBaseOps::%s] Memcpy wqe start, Now SQ PI: [%u]", __func__, sqHead_);

        // 写wqe到va
        auto sqDepth = sqContext_->depth;
        uint32_t sqPIMask = sqDepth - 1;
        u8* va = ReinterpretAs<u8*>(sqContext_->sqVa + (sqHead_ & sqPIMask) * wqeSize);

        HCCL_INFO(
            "[RdmaBaseOps][Wqe Write] before copy, sqHead[%u], slot[%u], sqVa[0x%llx], dst[0x%llx], size[%u]", sqHead_,
            sqHead_ & sqPIMask, static_cast<unsigned long long>(sqContext_->sqVa),
            ReinterpretAs<unsigned long long>(va), wqeSize);

        auto ret = memcpy_sp(va, wqeSize, wqe, wqeSize);
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(StringFormat("[RdmaBaseOps::%s] memcpy_s failed, ret = %d", __func__, ret));
        }

        // pi维护用于传入DB Send用于Rtsq 敲door bell
        sqHead_ = sqHead_ + 1;

        // Write InValid Wqebb
        CHK_RET(WriteInvalidWqebb(sqHead_));

        HCCL_INFO("[RdmaBaseOps::%s] Memcpy wqe end, Now SQ PI: [%u]", __func__, sqHead_);
        return HCCL_SUCCESS;
    }

    HcclResult WaitSqFree(uint32_t wqeNum)
    {
        // wq_overflow
        bool timeOutFlag = false;
        auto startTime = std::chrono::steady_clock::now();

        HCCL_INFO("[RdmaBaseOps::%s] Operate: sqTail = %u", __func__, sqTail_);
        while (!timeOutFlag) {
            // sq 队列能放下，直接成功返回
            if (static_cast<uint32_t>(sqHead_ - sqTail_ + wqeNum) <= sqContext_->depth) {
                return HCCL_SUCCESS;
            }

            timeOutFlag = (std::chrono::steady_clock::now() - startTime) > timeout_;
        }

        // 超时处理
        HCCL_ERROR("[RdmaBaseOps::%s] Sq is Full !! Operate: sqTail = %u Failed. ", __func__, sqTail_);
        return HCCL_E_TIMEOUT;
    }

    // 将PI更新到硬件可见地址
    HcclResult UpdateSqPI() const
    {
        // 更新Sq PI指针
        uint32_t sqHeadNum = Htonl32(sqHead_);

        HCCL_INFO(
            "[RdmaBaseOps][Wqe Write] write soft PI, sqHead host[%u], dbSwVa[0x%llx]", sqHead_,
            static_cast<unsigned long long>(sqContext_->dbSwVa));

        auto status
            = memcpy_sp(ReinterpretAs<void*>(sqContext_->dbSwVa), sizeof(uint32_t), &sqHeadNum, sizeof(uint32_t));
        if (UNLIKELY(status != 0)) {
            THROW<InternalException>(StringFormat("[RdmaBaseOps::%s] Ring Sw DB failed, ret = %d", __func__, status));
        }
        return HCCL_SUCCESS;
    }
};

} // namespace Hccl
#endif // RDMA_BASE_VENDOR_OPS_H
