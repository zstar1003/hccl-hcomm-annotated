/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_VENDOR_1825_OPS_H
#define RDMA_VENDOR_1825_OPS_H

#include "cast_utils.h"
#include "rma_conn_lite.h"
#include "rdma_vendor_base_ops.h"

namespace Hccl {

// 先默认所有WQE都只占一个WQEBB
#define WQEBB_SHIFT 6

#define ROCE_WQE_OWNERBIT_SHIFT 7
#define ROCE_WQE_CTRL_VALUE 0x40
#define ROCE_SQ_VA_VALUE 0x20
#define ROCE_SQ_SIGNAL_SHIFT 7
#define ROCE_WQE_CQE_SIGNAL_SHIFT 7
#define ROCE_WQE_CMP_TASK_LEN1 1u /* wqe cl is set to 1 */
#define ROCE_WQE_CMP_TASK_LEN_SHIFT 28
#define ROCE_TASK_SEG_ALIGN 8
#define ROCE_WQE_FAST_DMA_SHIFT 10
#define ROCE_WQE_SSN_MASK 0x3
#define ROCE_WQE_SSN_SHIFT 12
#define ROCE_WQE_DATA_SEG_SHIFT 4
#define ROCE_WQE_TASK_SEG_LAST_EXT_LEN 4
#define ROCE_WQE_TASK_REDUCE_OP_OFFSET 4
#define WQE_SECTION_ALIGN_SHIFT 3

#define ROCE_WQE_NEXT_SGE_INVALID (1u << 31)

using Roce3CtrlSeg = struct {
    // Control Segment
    uint8_t owner_sl; /* ownerbit:ctrl_section_length:csl:dif_sl = 1:2:2:3; */
    uint8_t df_tsl;   /* cr:df:va:tal = 1:1:1:5; */
    uint16_t wf_bdsl; /* cf:wf:wqe_msn:fde:fast:drvsl:bdsl = 1:1:2:1:1:2:8; */
    uint32_t cl_pi;   /* cl:signature:mask_pi = 4:8:20; */
    uint64_t db;      /* used by direct wqe*/
};

union Roce3TaskSeg {
    struct {
        uint32_t xrcSrqn : 18;
        uint32_t ext : 1;
        uint32_t dif : 1;
        uint32_t rsvd0 : 3;
        uint32_t so : 1;
        uint32_t opType : 5;
        uint32_t signal : 1;
        uint32_t fence : 1;
        uint32_t se : 1; /* solited event flag; */
    } dw0;
    uint32_t value;
};

using Roce3TaskWqeSeg = struct {
    // Task Wqe Segment
    union Roce3TaskSeg tskSeg;
    uint32_t dataLen;
    uint32_t immData;
    union {
        struct {
            uint32_t lastExtLen : 8;
            uint32_t cmdLen : 8;
            uint32_t pi : 16;
        } bs;

        uint32_t feth; /* cflush feth header */
        uint32_t value;
    } dw3;
    uint64_t va;   /* to indicate remote buf address; */
    uint32_t rkey; /* to indicate remote mr buf; */
    uint32_t ulp;  /* ulp预留字段 */
};

static constexpr uint32_t DTYPE_INVALID = 0xFFu;

static const uint32_t Roce3ReduceDataTypeMap[] = {
    /* [DINT8]   = */ 0x0,           // INT8
    /* [DINT16]  = */ 0x1,           // INT16
    /* [DINT32]  = */ 0x2,           // INT32
    /* [DFP16]   = */ 0x6,           // FP16_NORMAL
    /* [DFP32]   = */ 0x7,           // FP32
    /* [DINT64]  = */ DTYPE_INVALID, // 硬件不支持
    /* [DUINT32] = */ DTYPE_INVALID, // 硬件不支持
};

static const uint32_t Roce3ReduceOpTypeMap[] = {
    /* [SUM]    = */ 0xA,           // ADD
    /* [PROD]   = */ DTYPE_INVALID, // 求积, 硬件不支持
    /* [MAX]    = */ 0x8,           // MAX
    /* [MIN]    = */ 0x9,           // MIN
    /* [EQUAL]  = */ 0xB,           // EQUAL
};

using Roce3WqeDataSeg = struct {
    uint64_t bufAddr; /* buffer address that wqe sge indicate; */
    uint32_t rLen;    /* buffer length that wqe sge indicate; */
    uint32_t leKey;   /* buffer lkey that wqe sge indicate; */
};

using Roce3WqeEntry = struct {
    Roce3CtrlSeg ctrl;
    Roce3TaskWqeSeg task;
    Roce3WqeDataSeg data;
};

// Cqe related
constexpr uint32_t ROCE_CQE_OPCODE_SHIFT = 27;
constexpr uint32_t ROCE_CQE_OPCODE_MASK = 0x1f;
constexpr uint32_t ROCE_CQ_UPDATE_CI_MASK = 0xffffff; /* cq arm db ci mask; */
constexpr uint32_t ROCE_CQE_MAX_GEN_NUM = 1024;
constexpr uint32_t ROCE_CQE_OWNERBIT_SHIFT = 31;
constexpr uint32_t ROCE_CQE_QPN_MASK = 0xfffff;

enum class RoceCqeType : uint32_t {
    ERROR = 0x1e,  /* indicate the cqe is an error cqe */
    RESIZE = 0x16, /* indicate the cqe is an resize type cqe */
    INVALID = 0x1f
};

struct Roce3CqeEntry {
    uint32_t owner_id_qpn; /* ownerbit: cqe_size:dif_en:wq_id:error_code:qpn = 1:2:1:4:4:20; */
    uint32_t op_sr_wqebb;  /* opcode:s_r:inline_r:merge:fake:rsvd:linkwqe_used:wqebb_index = 5:1:1:1:1:2:1:20; */
    uint32_t byte_cnt;     /* reflect the data size we receive; */
    uint32_t imm_data;     /* the immediate data we receive */

    union {
        struct {
            uint8_t smac_h[2];    /* smac value; */
            uint16_t vlan_id_pri; /* vlan_id:vlan_priority:reserved:smac = 12:1:3:16; */
        } mac_vlanid;
    } dw4;
    union {
        uint8_t smac_l[4]; /* smac value */
        uint32_t wqe_num;  /* merged wqe num */
    };
    uint32_t vlan_queue_index; /* vlan:force_loopback:reserved:RQPN = 2:1:5:24; */
    uint8_t syndrome;          /* record syndrome code; */
    uint8_t reserved2;
    uint16_t wqe_counter; /* the wqe index, valid for sq; */
};

// Doorbell related
constexpr uint32_t ROCE_SQ_DOORBELL_TYPE = 21;
constexpr uint32_t ROCE_INIT_SQ_DB_SGIT_IDX = 1;

using Roce3DbEntry = struct {
    union {
        struct {
            uint32_t qpn : 20;     /* indicate the sq qpn; */
            uint32_t cntxSize : 2; /* indicate the qpc size; */
            uint32_t r : 1;        /* reserved bit; */
            uint32_t c : 1;
            uint32_t cos : 3;
            uint32_t type : 5;

            uint32_t pi : 8;
            uint32_t resv : 8;
            uint32_t xrcvld : 1;
            uint32_t vxlan : 1;
            uint32_t mtuShift : 3;
            uint32_t sgidIndex : 7;
            uint32_t subType : 4;
        } bs;
        uint64_t db_value;
    } dw0;
};

enum class ROCE3_OPCODE : uint32_t { ROCE_OPCODE_RDMA_WRITE = 4U, ROCE_OPCODE_RDMA_READ = 8U };

// post_send helper
struct Roce3PostSendParams {
    const SqeConfigLite& cfg;
    const RmaBufSliceLite& loc;
    const RmtRmaBufSliceLite& rmt;
    const uint32_t opCode;

    Roce3PostSendParams(const Roce3PostSendParams&) = delete;
    Roce3PostSendParams& operator=(const Roce3PostSendParams&) = delete;
};

class Rdma1825Ops : public RdmaBaseOps {
public:
    Rdma1825Ops(RdmaSqContextLite* sqContext, RdmaCqContextLite* cqContext) : RdmaBaseOps(sqContext, cqContext) {}

    ~Rdma1825Ops() override {}

    HcclResult BuildDoorbell(u64& dbAddr, u64& dbValue) override
    {
        Roce3DbEntry dbEntry;
        dbEntry.dw0.db_value = 0;
        dbEntry.dw0.bs.r = 0;
        dbEntry.dw0.bs.c = 0;
        dbEntry.dw0.bs.cntxSize = 1;
        dbEntry.dw0.bs.qpn = sqContext_->qpn;
        dbEntry.dw0.bs.subType = 0;
        dbEntry.dw0.bs.resv = 0;
        dbEntry.dw0.bs.pi = ((sqHead_ >> 8) & 0xff);
        dbEntry.dw0.bs.sgidIndex = ROCE_INIT_SQ_DB_SGIT_IDX;
        dbEntry.dw0.bs.type = ROCE_SQ_DOORBELL_TYPE;

        dbEntry.dw0.bs.mtuShift = static_cast<uint32_t>(
            (sqContext_->dbVendorSpecified >> UB_DB_VENDOR_MTUSHIFT_SHIFT) & UB_DB_VENDOR_FIELD_MASK);
        dbEntry.dw0.bs.cos = static_cast<uint32_t>(
            (sqContext_->dbVendorSpecified >> UB_DB_VENDOR_COS_SHIFT) & UB_DB_VENDOR_FIELD_MASK);

        dbAddr = sqContext_->dbHwVa;
        dbValue = dbEntry.dw0.db_value;

        HCCL_INFO(
            "[Rdma1825Ops::%s] SQ DB ready, qpn[%u], sqHead[%u], mtuShift[%u], cos[%u], "
            "dbAddr[0x%llx], dbValue[0x%llx]",
            __func__, sqContext_->qpn, sqHead_, dbEntry.dw0.bs.mtuShift, dbEntry.dw0.bs.cos, dbAddr, dbValue);
        return HCCL_SUCCESS;
    }

    HcclResult BuildCqDoorbell(u64& dbAddr, u64& dbValue) override
    {
        if (cqDbFlush_) {
            dbAddr = cqContext_->dbSwVa;

            const uint32_t dbValue32 = Htonl32(cqTail_ & ROCE_CQ_UPDATE_CI_MASK);
            dbValue = dbValue32;

            // Ring Cq Soft DB
            auto ret = memcpy_sp(ReinterpretAs<void*>(dbAddr), sizeof(uint32_t), &dbValue32, sizeof(uint32_t));
            if (UNLIKELY(ret != 0)) {
                THROW<InternalException>(
                    StringFormat("[Rdma1825Ops::%s] write soft Cq DB failed, ret = %d", __func__, ret));
            }
            HCCL_INFO(
                "[Rdma1825Ops::%s] CQ DB updated, cqTail[%u], dbAddr[0x%llx], dbValue[0x%llx]", __func__, cqTail_,
                dbAddr, dbValue);
        } else {
            dbAddr = 0;
            dbValue = 0;
        }

        cqDbFlush_ = false;

        return HCCL_SUCCESS;
    }

protected:
    HcclResult
    BuildReadWqe(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg) override
    {
        Roce3WqeEntry wqe{};
        Roce3PostSendParams params{cfg, loc, rmt, static_cast<uint32_t>(ROCE3_OPCODE::ROCE_OPCODE_RDMA_READ)};

        CHK_RET(FillCtrlSeg(&wqe, params));
        CHK_RET(FillTaskSeg(&wqe, params));
        CHK_RET(FillDataSeg(&wqe, params));

        CHK_RET(CommitWqe(&wqe, sizeof(Roce3WqeEntry)));
        HCCL_INFO(
            "[Rdma1825Ops::%s] WQE committed, qpn[%u], sqHead[%u], loc[0x%llx], rmt[0x%llx], size[%u]", __func__,
            sqContext_->qpn, sqHead_, loc.GetAddr(), rmt.GetAddr(), loc.GetSize());
        return HCCL_SUCCESS;
    }

    HcclResult
    BuildWriteWqe(const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg) override
    {
        Roce3WqeEntry wqe{};
        Roce3PostSendParams params{cfg, loc, rmt, static_cast<uint32_t>(ROCE3_OPCODE::ROCE_OPCODE_RDMA_WRITE)};

        CHK_RET(FillCtrlSeg(&wqe, params));
        CHK_RET(FillTaskSeg(&wqe, params));
        CHK_RET(FillDataSeg(&wqe, params));

        CHK_RET(CommitWqe(&wqe, sizeof(Roce3WqeEntry)));
        HCCL_INFO(
            "[Rdma1825Ops::%s] WQE committed, qpn[%u], sqHead[%u], loc[0x%llx], rmt[0x%llx], size[%u]", __func__,
            sqContext_->qpn, sqHead_, loc.GetAddr(), rmt.GetAddr(), loc.GetSize());
        return HCCL_SUCCESS;
    }

    HcclResult BuildWriteReduceWqe(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, DataType dataType,
        ReduceOp reduceOp) override
    {
        Roce3WqeEntry wqe{};
        Roce3PostSendParams params{cfg, loc, rmt, static_cast<uint32_t>(ROCE3_OPCODE::ROCE_OPCODE_RDMA_WRITE)};

        CHK_RET(FillCtrlSeg(&wqe, params));
        CHK_RET(FillTaskSeg(&wqe, params));
        CHK_RET(FillReduceTaskSeg(&wqe, params, dataType, reduceOp));
        CHK_RET(FillDataSeg(&wqe, params));

        CHK_RET(CommitWqe(&wqe, sizeof(Roce3WqeEntry)));
        HCCL_INFO(
            "[Rdma1825Ops::%s] WQE committed, qpn[%u], sqHead[%u], loc[0x%llx], rmt[0x%llx], size[%u], "
            "dataType[%u], reduceOp[%u]",
            __func__, sqContext_->qpn, sqHead_, loc.GetAddr(), rmt.GetAddr(), loc.GetSize(),
            static_cast<uint32_t>(dataType), static_cast<uint32_t>(reduceOp));
        return HCCL_SUCCESS;
    }

    HcclResult WriteInvalidWqebb(uint32_t nextIdx) override
    {
        const uint32_t sqDepth = sqContext_->depth;
        const uint32_t sqMask = sqDepth - 1U;
        uint8_t ownerSl = ((nextIdx & sqDepth) == 0) ? 0xff : 0x7f;

        auto* dst
            = ReinterpretAs<void*>(sqContext_->sqVa + static_cast<uint64_t>(nextIdx & sqMask) * sqContext_->wqeSize);

        auto ret = memcpy_sp(dst, sizeof(ownerSl), &ownerSl, sizeof(ownerSl));
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(
                StringFormat("[Rdma1825Ops::%s] write invalid wqebb failed, ret = %d", __func__, ret));
        }
        return HCCL_SUCCESS;
    }

    int32_t PollCqImpl(int32_t numEntries, std::vector<int32_t>& errList) override
    {
        int32_t pollNum = 0;
        CqPollStatus ret = CqPollStatus::ERROR;

        while (pollNum < numEntries) {
            ret = PollOne(errList);
            if (ret != CqPollStatus::SUCCESS) {
                if (ret != CqPollStatus::EMPTY && ret != CqPollStatus::ERROR) {
                    HCCL_ERROR(
                        "[Rdma1825Ops::%s][Poll cq] Poll CQ error, ret: %d", __func__, static_cast<int32_t>(ret));
                }
                break;
            }

            // Update Cq CI
            cqTail_ += 1;

            ++pollNum;
        }

        if ((pollNum != 0) || (ret == CqPollStatus::ERROR)) {
            // Need to flush Cq DB
            cqDbFlush_ = true;
        }

        return (ret == CqPollStatus::ERROR) ? static_cast<int32_t>(ret) : pollNum;
    }

private:
    HcclResult FillCtrlSeg(Roce3WqeEntry* wqe, const Roce3PostSendParams& params) const
    {
        // 赋值本wqe的owner
        uint8_t owner = (sqHead_ & (sqContext_->depth)) == 0 ? 0 : 1;
        wqe->ctrl.owner_sl = (owner << ROCE_WQE_OWNERBIT_SHIFT) | ROCE_WQE_CTRL_VALUE;

        // 不产生cqe + task字段的长度
        wqe->ctrl.df_tsl = ((params.cfg.cqeEn == true) ? (1U << ROCE_SQ_SIGNAL_SHIFT) : 0) | ROCE_SQ_VA_VALUE;
        wqe->ctrl.df_tsl |= sizeof(Roce3TaskWqeSeg) / ROCE_TASK_SEG_ALIGN;

        // fast_dma + SSN + sge长度
        wqe->ctrl.wf_bdsl = Htons16(static_cast<uint16_t>(0 << ROCE_WQE_FAST_DMA_SHIFT));
        wqe->ctrl.wf_bdsl |= Htons16((sqHead_ & ROCE_WQE_SSN_MASK) << ROCE_WQE_SSN_SHIFT);
        wqe->ctrl.wf_bdsl |= Htons16(
            static_cast<uint16_t>(static_cast<uint32_t>(1) << (ROCE_WQE_DATA_SEG_SHIFT - WQE_SECTION_ALIGN_SHIFT)));

        // cl
        wqe->ctrl.cl_pi = Htonl32(ROCE_WQE_CMP_TASK_LEN1 << ROCE_WQE_CMP_TASK_LEN_SHIFT);

        return HCCL_SUCCESS;
    }

    HcclResult FillTaskSeg(Roce3WqeEntry* wqe, const Roce3PostSendParams& params) const
    {
        // ----- Task Seg -----
        wqe->task.tskSeg.value = 0;
        wqe->task.tskSeg.dw0.signal = !!((wqe->ctrl.df_tsl & (1U << ROCE_WQE_CQE_SIGNAL_SHIFT)) > 0);
        wqe->task.tskSeg.dw0.fence = params.cfg.fence;
        wqe->task.tskSeg.dw0.opType = params.opCode;
        wqe->task.tskSeg.dw0.se = 0;
        wqe->task.tskSeg.value = Htonl32(wqe->task.tskSeg.value);

        wqe->task.dataLen = Htonl32(params.loc.GetSize());
        wqe->task.immData = 0;
        wqe->task.dw3.value
            = (params.opCode == static_cast<uint32_t>(ROCE3_OPCODE::ROCE_OPCODE_RDMA_READ) ?
                   Htonl32(ROCE_WQE_TASK_SEG_LAST_EXT_LEN) :
                   0);
        wqe->task.va = Htonll64(params.rmt.GetAddr());
        wqe->task.rkey = Htonl32(params.rmt.GetRkey());
        wqe->task.ulp = Htonl32(params.loc.GetLkey() & 0xffff);

        return HCCL_SUCCESS;
    }

    HcclResult
    FillReduceTaskSeg(Roce3WqeEntry* wqe, const Roce3PostSendParams& params, DataType dataType, ReduceOp reduceOp) const
    {
        // Hcomm Type -> Roce3 Type
        uint32_t dataTypeIdx = static_cast<uint32_t>(dataType);
        if (dataTypeIdx >= sizeof(Roce3ReduceDataTypeMap) / sizeof(uint32_t)) {
            HCCL_ERROR("[Rdma1825Ops::%s] invalid DataType %u", __func__, dataTypeIdx);
            return HCCL_E_PARA;
        }
        uint32_t reduceDataType = Roce3ReduceDataTypeMap[dataTypeIdx];
        if (reduceDataType == DTYPE_INVALID) {
            HCCL_ERROR("[Rdma1825Ops::%s] DataType %u is not supported by inline reduce", __func__, dataTypeIdx);
            return HCCL_E_NOT_SUPPORT;
        }

        uint32_t reduceOpIdx = static_cast<uint32_t>(reduceOp);
        if (reduceOpIdx >= sizeof(Roce3ReduceOpTypeMap) / sizeof(uint32_t)) {
            HCCL_ERROR("[Rdma1825Ops::%s] invalid ReduceOpType %u", __func__, reduceOpIdx);
            return HCCL_E_PARA;
        }
        uint32_t reduceOpType = Roce3ReduceOpTypeMap[reduceOpIdx];
        if (reduceOpType == DTYPE_INVALID) {
            HCCL_ERROR("[Rdma1825Ops::%s] ReduceOpType %u is not supported by inline reduce", __func__, reduceOpIdx);
            return HCCL_E_NOT_SUPPORT;
        }

        // Reduce data
        uint32_t mptIndex = params.loc.GetLkey() & 0xffff;
        uint32_t reduceOpInfo = ((reduceOpType << ROCE_WQE_TASK_REDUCE_OP_OFFSET) | reduceDataType);

        // ----- Task Reduce Seg -----
        wqe->task.ulp = Htonl32(mptIndex | reduceOpInfo);

        return HCCL_SUCCESS;
    }

    HcclResult FillDataSeg(Roce3WqeEntry* wqe, const Roce3PostSendParams& params) const
    {
        // ----- Data Seg -----
        wqe->data.bufAddr = Htonll64(static_cast<uint64_t>(params.loc.GetAddr()));
        wqe->data.leKey = Htonl32(
            (params.loc.GetLkey() & (~ROCE_WQE_NEXT_SGE_INVALID)) | ROCE_WQE_NEXT_SGE_INVALID); // 当前sge就是最后一个
        wqe->data.rLen = Htonl32(params.loc.GetSize());

        return HCCL_SUCCESS;
    }

    CqPollStatus PollOne(std::vector<int32_t>& errList)
    {
        bool need_poll = true;
        while (need_poll) {
            Roce3CqeEntry* rcqe = nullptr;
            rcqe = Roce3GetOneCqe(cqHead_);
            if (rcqe == nullptr) {
                return CqPollStatus::EMPTY;
            }

            ++cqHead_;

            uint32_t syndrome = static_cast<uint32_t>(rcqe->syndrome);
            uint32_t ownerIdQpn = static_cast<uint32_t>(rcqe->owner_id_qpn);
            auto cqeType
                = static_cast<RoceCqeType>((rcqe->op_sr_wqebb >> ROCE_CQE_OPCODE_SHIFT) & ROCE_CQE_OPCODE_MASK);
            // 存在错误
            if (cqeType == RoceCqeType::ERROR) {
                const uint32_t cqeErrCode = (ownerIdQpn >> 20U) & 0xfU;

                HCCL_ERROR(
                    "[Rdma1825Ops::%s][Poll cq] CQE error, qpn[%u], syndrome[%u], cqeErrCode[%u], "
                    "wqeCounter[%u]",
                    __func__, ownerIdQpn & ROCE_CQE_QPN_MASK, syndrome, cqeErrCode,
                    static_cast<uint32_t>(rcqe->wqe_counter));

                // 后续优化为具体ErrorCode
                int32_t ERROR_CODE = 1;
                errList.push_back(ERROR_CODE);
                return CqPollStatus::ERROR;
            }

            HCCL_INFO(
                "[Rdma1825Ops::%s][Poll cq] CQE completed, cqHead[%u], qpn[%u], syndrome[%u], "
                "wqeCounter[%u]",
                __func__, cqHead_, ownerIdQpn & ROCE_CQE_QPN_MASK, syndrome, static_cast<uint32_t>(rcqe->wqe_counter));
            need_poll = false;
        }

        return CqPollStatus::SUCCESS;
    }

    Roce3CqeEntry* Roce3GetOneCqe(uint32_t consIndex)
    {
        uint32_t cqeSize = 64;
        uint32_t cqDepthReal = cqContext_->cqDepth + ROCE_CQE_MAX_GEN_NUM; // Must be power of 2
        uint32_t cqeMask = cqDepthReal - 1;
        uint32_t cqeSlot = consIndex & cqeMask;

        // Calculate Cqe Addr
        auto cqeAddr = ReinterpretAs<Roce3CqeEntry*>(
            ReinterpretAs<uint8_t*>(cqContext_->cqVa) + static_cast<size_t>(cqeSlot * cqeSize));

        // Memcpy Cqe
        auto ret = memcpy_sp(&cqeReadback_, sizeof(cqeReadback_), cqeAddr, sizeof(cqeReadback_));
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(StringFormat("[Rdma1825Ops::%s] memcpy_sp cqe failed, ret = %d", __func__, ret));
        }

        // Cqe Details Parse
        uint32_t opSrWqebb = cqeReadback_.op_sr_wqebb;
        uint32_t ownerIdQpn = cqeReadback_.owner_id_qpn;
        auto cqeType = static_cast<RoceCqeType>((opSrWqebb >> ROCE_CQE_OPCODE_SHIFT) & ROCE_CQE_OPCODE_MASK);
        if (cqeType == RoceCqeType::INVALID) {
            HCCL_INFO("[Rdma1825Ops::%s][Poll cq] CQE invalid, consIndex[%u], slot[%u]", __func__, consIndex, cqeSlot);
            return nullptr;
        }

        // Judge Owner bit
        uint32_t calHwOwner = static_cast<uint32_t>((consIndex & cqDepthReal) == 0);
        uint32_t curCqeOwner = !((ownerIdQpn & (1U << ROCE_CQE_OWNERBIT_SHIFT)) == 0);
        if ((calHwOwner ^ curCqeOwner) == 0) {
            HCCL_INFO(
                "[Rdma1825Ops::%s][Poll cq] CQE owner not ready, consIndex[%u], slot[%u], "
                "calHwOwner[%u], curCqeOwner[%u]",
                __func__, consIndex, cqeSlot, calHwOwner, curCqeOwner);
            return nullptr;
        }

        return &cqeReadback_;
    }

    Roce3CqeEntry cqeReadback_{};
};

} // namespace Hccl
#endif // RDMA_VENDOR_1825_OPS_H
