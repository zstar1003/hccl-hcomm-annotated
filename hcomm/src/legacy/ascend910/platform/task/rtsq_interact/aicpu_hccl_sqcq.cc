/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/aicpu_hccl_sqcq.h"
#include "adapter_hal_pub.h"
#include "sal_pub.h"
#include "hcomm_task_struct.h"
#include "common/type_def.h"

using char_t = char;

extern "C" {
drvError_t __attribute__((weak)) halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info);
drvError_t __attribute__((weak)) halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info);
drvError_t __attribute__((weak)) halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info);
drvError_t __attribute__((weak))
halTsdrvCtl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize);
drvError_t __attribute__((weak)) halEschedSubmitEvent(uint32_t devId, struct event_summary* event);
};

HcclResult QuerySqBaseAddr(uint32_t devId, uint32_t sqId, u64& outVal)
{
    CHK_PRT_RET(
        (halSqCqQuery == nullptr),
        HCCL_ERROR("halSqCqQuery is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    halSqCqQueryInfo queryinfo;
    queryinfo.tsId = 0;
    queryinfo.sqId = sqId;
    queryinfo.cqId = 0;
    queryinfo.type = DRV_NORMAL_TYPE;

    queryinfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    uint32_t ret = halSqCqQuery(devId, &queryinfo);
    if (ret != 0) {
        HCCL_ERROR("halSqCqQuery base addr failed. ret = %d sqid:%d\n", ret, queryinfo.sqId);
        return HCCL_E_DRV;
    }

    outVal = ((static_cast<u64>(queryinfo.value[1])) << UINT32_BIT_NUM) | queryinfo.value[0];

    HCCL_DEBUG("valu1:%x. value0:%x, outValue:%p", queryinfo.value[1], queryinfo.value[0], outVal);

    return HCCL_SUCCESS;
}

HcclResult QuerySqStatusByType(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t& outVal)
{
    CHK_PRT_RET(
        (halSqCqQuery == nullptr),
        HCCL_ERROR("halSqCqQuery is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    halSqCqQueryInfo queryinfo;
    queryinfo.tsId = 0;
    queryinfo.sqId = sqId;
    queryinfo.cqId = 0;
    queryinfo.type = DRV_NORMAL_TYPE;

    queryinfo.prop = type;
    uint32_t ret = halSqCqQuery(devId, &queryinfo);
    if (ret != 0) {
        HCCL_ERROR("halSqCqQuery %d failed. ret = %d sqid:%d\n", type, ret, queryinfo.sqId);
        return HCCL_E_DRV;
    }
    outVal = queryinfo.value[0];

    return HCCL_SUCCESS;
}

HcclResult QuerySqStatus(uint32_t devId, uint32_t sqId, uint32_t& sqHead, uint32_t& sqTail)
{
    HcclResult ret = QuerySqStatusByType(devId, sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail);
    if (ret != 0) {
        HCCL_ERROR(" halSqCqQuery TAIL failed. ret = %d sqid:%d\n", ret, sqId);
        return ret;
    }

    ret = QuerySqStatusByType(devId, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead);
    if (ret != 0) {
        HCCL_ERROR(" halSqCqQuery HEAD failed. ret = %d sqid:%d\n", ret, sqId);
        return ret;
    }

    return ret;
}

HcclResult ConfigSqStatusByType(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t value)
{
    CHK_PRT_RET(
        (halSqCqConfig == nullptr),
        HCCL_ERROR("halSqCqConfig is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    halSqCqConfigInfo configInfo;
    configInfo.tsId = 0;
    configInfo.sqId = sqId;
    configInfo.cqId = 0;
    configInfo.type = DRV_NORMAL_TYPE;

    configInfo.prop = type;
    configInfo.value[0] = value;
    uint32_t ret = halSqCqConfig(devId, &configInfo);
    if (ret != 0) {
        HCCL_ERROR(
            "halSqCqConfig %d failed. ret = %d sqid:%d, type:%d, value:%d", type, ret, configInfo.sqId, type, value);
        return HCCL_E_DRV;
    }
    return HCCL_SUCCESS;
}

namespace {
void ConstructLHWI(const rtStarsCondIsaRegister_t dstReg, const u64 immd, rtStarsCondOpLHWI_t& opLHWI)
{
    opLHWI.opCode = RT_STARS_COND_ISA_OP_CODE_LWI;
    opLHWI.func3 = RT_STARS_COND_ISA_LWI_FUNC3_LHWI;
    opLHWI.rd = dstReg;
    opLHWI.immd = static_cast<uint32_t>((immd >> 49U) & 0x7FFFU); // High15-immd[63:49]
}

void ConstructLLWI(const rtStarsCondIsaRegister_t dstReg, const u64 immd, rtStarsCondOpLLWI_t& opLLWI)
{
    opLLWI.opCode = RT_STARS_COND_ISA_OP_CODE_LWI;
    opLLWI.func3 = RT_STARS_COND_ISA_LWI_FUNC3_LLWI;
    opLLWI.rd = dstReg;
    opLLWI.immdHigh = static_cast<uint32_t>((immd >> 32U) & 0x1FFFFU); // Low49-immd[48:32]
    opLLWI.immdLow = static_cast<uint32_t>(immd & 0xFFFFFFFFU);        // Low49-immd[31:0]
}

void ConstructLoadImm(
    const rtStarsCondIsaRegister_t dstReg, const u64 addr, const rtStarsCondIsaLoadImmFunc3_t func3,
    rtStarsCondOpLoadImm_t& loadImm)
{
    loadImm.opCode = RT_STARS_COND_ISA_OP_CODE_LOAD_IMM;
    loadImm.rd = dstReg;
    loadImm.func3 = func3;
    loadImm.immdAddrHigh = static_cast<uint32_t>((addr >> 32U) & 0X1FFFFU); // bit[48:32]
    loadImm.immdAddrLow = static_cast<uint32_t>(addr & 0xFFFFFFFFU);        // bit[31:0]
}

void ConstructBranch(
    const rtStarsCondIsaRegister_t rs1Reg, const rtStarsCondIsaRegister_t rs2Reg,
    const rtStarsCondIsaBranchFunc3_t func3, const uint8_t instrOffset, rtStarsCondOpBranch_t& opBranch)
{
    opBranch.opCode = RT_STARS_COND_ISA_OP_CODE_BRANCH;
    opBranch.func3 = func3;
    opBranch.rs1 = rs1Reg;
    opBranch.rs2 = rs2Reg;
    opBranch.jumpInstrOffset = instrOffset & 0xFU; // Jump-immd[3:0]
}

void ConstructStore(
    const rtStarsCondIsaRegister_t addrReg, const rtStarsCondIsaRegister_t valReg, const uint16_t immdOffset,
    const rtStarsCondIsaStoreFunc3_t func3, rtStarsCondOpStore_t& opStore)
{
    opStore.opCode = RT_STARS_COND_ISA_OP_CODE_STORE;
    opStore.immdLow = static_cast<uint8_t>(immdOffset & 0x1FU); // S-immd[4:0]
    opStore.func3 = func3;
    opStore.rs1 = addrReg;
    opStore.rs2 = valReg;
    opStore.immdHigh = static_cast<uint8_t>((immdOffset & 0xFE0U) >> 5U); // S-immd[11:5]
}

void ConstructNop(rtStarsCondOpNop_t& nop)
{
    nop.opCode = RT_STARS_COND_ISA_OP_CODE_NOP;
    nop.rd = RT_STARS_COND_ISA_REGISTER_R0;
    nop.func3 = RT_STARS_COND_ISA_OP_IMM_FUNC3_NOP;
    nop.rs1 = RT_STARS_COND_ISA_REGISTER_R0;
    nop.immd = 0U;
}
} // namespace

void AddOneWaitStartSqe(
    uint16_t streamId, uint16_t taskId, u64 waitAddr, u64 curTurnCntAddr, bool last,
    rtStarsCcoreWaitStartSqe_t* const sqe, uint8_t* sqeType)
{
    *sqeType = SqeType::CCORE_WAIT_START_SQE;
    sqe->sqeHeader.type = RT_STARS_SQE_TYPE_COND;
    sqe->sqeHeader.rtStreamId = streamId;
    sqe->sqeHeader.taskId = taskId;

    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->csc = 1U;

    constexpr rtStarsCondIsaRegister_t r0 = RT_STARS_COND_ISA_REGISTER_R0;
    constexpr rtStarsCondIsaRegister_t r1 = RT_STARS_COND_ISA_REGISTER_R1;
    constexpr rtStarsCondIsaRegister_t r2 = RT_STARS_COND_ISA_REGISTER_R2;
    constexpr rtStarsCondIsaRegister_t r3 = RT_STARS_COND_ISA_REGISTER_R3;

    // load current Turn to r3
    ConstructLoadImm(r3, curTurnCntAddr, RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm1);

    // load sendcnt to r2
    ConstructLoadImm(r2, waitAddr, RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm2);
    uint8_t loadInstrOff
        = (offsetof(rtStarsCcoreWaitStartSqe_t, ldrImm2) - offsetof(rtStarsCcoreWaitStartSqe_t, ldrImm1));
    loadInstrOff = loadInstrOff / sizeof(uint32_t);

    // r2(sendCnt) < r3(curTurn)，goto reload r2
    ConstructBranch(r2, r3, RT_STARS_COND_ISA_BRANCH_FUNC3_BLTU, loadInstrOff, sqe->beq);

    if (last) {
        // load sendcount addr to r1
        ConstructLLWI(r1, waitAddr, sqe->clear.llwi1);
        ConstructLHWI(r1, waitAddr, sqe->clear.lhwi1);
        // the last turn clear sendCnt, r0(0) value store to r1(sendCnt),
        ConstructStore(r1, r0, 0U, RT_STARS_COND_ISA_STORE_FUNC3_SH, sqe->clear.sw);
        for (rtStarsCondOpNop_t& nop : sqe->clear.nop) {
            ConstructNop(nop);
        }
    } else {
        for (rtStarsCondOpNop_t& nop : sqe->nop) {
            ConstructNop(nop);
        }
    }

    HCCL_INFO(
        "WaitStart waitAddr %p, curTurnCntAddr %p, loadInstrOff %u, streamId %u, taskId %u, last %u"
        "ISA: %08x %08x %08x %08x %08x %08x %08x.",
        waitAddr, curTurnCntAddr, loadInstrOff, streamId, taskId, last, sqe->ldrImm1, sqe->ldrImm2, sqe->beq,
        sqe->clear.llwi1, sqe->clear.lhwi1, sqe->clear.sw, sqe->clear.nop[0]);
}

void AddOneWriteValueStartSqe(
    uint16_t streamId, uint16_t taskId, u64 writeAddr, u64 valueAddr, rtStarsCcoreWriteValueSqe_t* const sqe,
    uint8_t* sqeType)
{
    *sqeType = SqeType::CCORE_WRITE_VALUE_SQE;
    sqe->sqeHeader.type = RT_STARS_SQE_TYPE_COND;
    sqe->sqeHeader.rtStreamId = streamId;
    sqe->sqeHeader.taskId = taskId;

    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->csc = 1U;

    constexpr rtStarsCondIsaRegister_t r1 = RT_STARS_COND_ISA_REGISTER_R1;
    constexpr rtStarsCondIsaRegister_t r2 = RT_STARS_COND_ISA_REGISTER_R2;

    ConstructLoadImm(r1, valueAddr, RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm);
    ConstructLLWI(r2, writeAddr, sqe->llwi1);
    ConstructLHWI(r2, writeAddr, sqe->lhwi1);

    ConstructStore(r2, r1, 0U, RT_STARS_COND_ISA_STORE_FUNC3_SH, sqe->sw);
    for (rtStarsCondOpNop_t& nop : sqe->nop) {
        ConstructNop(nop);
    }

    HCCL_INFO(
        "CCore write value: writeAddr %p, valueAddr %p, streamId %u, taskId %u"
        "ISA: %08x %08x %08x %08x %08x.",
        writeAddr, valueAddr, streamId, taskId, sqe->ldrImm, sqe->llwi1, sqe->lhwi1, sqe->sw, sqe->nop[0]);
}

std::string StringLogicCqReportInfo(const rtLogicCqReport_t& reportOfOne)
{
    std::stringstream ss;
    ss << "streamId :" << reportOfOne.streamId;
    ss << " taskId :" << reportOfOne.taskId;
    ss << " errorCode :" << reportOfOne.errorCode;
    ss << " errorType :" << static_cast<uint32_t>(reportOfOne.errorType);
    ss << " sqeType :" << static_cast<uint32_t>(reportOfOne.sqeType);
    ss << " sqId :" << reportOfOne.sqId;
    ss << " sqHead :" << reportOfOne.sqHead;
    ss << " matchFlag :" << reportOfOne.matchFlag;
    ss << " dropFlag :" << reportOfOne.dropFlag;
    ss << " errorBit :" << reportOfOne.errorBit;
    ss << " accError :" << reportOfOne.accError;
    return ss.str();
}

bool IsExceptionCqe(const rtLogicCqReport_t& reportOfOne)
{
    HCCL_DEBUG("ReportOfOne info [%s]", StringLogicCqReportInfo(reportOfOne).c_str());
    if ((reportOfOne.errorType & RT_STARS_EXIST_ERROR) == 0U) { // 取低5位
        return false;
    }
    return true;
}

CqeStatus CqReportRecv(const CqeQueryInput& cqeQueryInput, rtLogicCqReport_t& cqeException)
{
    CHK_PRT_RET(
        (halCqReportRecv == nullptr),
        HCCL_ERROR("halCqReportRecv is nullptr, "
                   "Does not support this interface."),
        CqeStatus::kCqeInnerError);
    halReportRecvInfo recvInfo;
    recvInfo.type = static_cast<drvSqCqType_t>(cqeQueryInput.type);
    recvInfo.tsId = 0;
    recvInfo.report_cqe_num = 0;
    recvInfo.stream_id = cqeQueryInput.streamId;
    recvInfo.cqId = cqeQueryInput.cqId;
    recvInfo.timeout = 0;                      // 不设置超时时间，非阻塞
    recvInfo.task_id = 0xFFFF;                 // 接收所有类型
    recvInfo.cqe_addr = cqeQueryInput.cqeAddr; // 外部保证是有效的地址
    recvInfo.cqe_num = (recvInfo.type == DRV_LOGIC_TYPE ? AC_SQE_REV_MAX_CNT : MAX_REPORT_CNT);
    drvError_t ret = halCqReportRecv(cqeQueryInput.devId, &recvInfo);
    if (ret == DRV_ERROR_WAIT_TIMEOUT) {
        HCCL_DEBUG("halCqReportRecv has found nothing, ret:%d", ret);
        return CqeStatus::kCqeTimeOut;
    }
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR("halCqReportRecv failed, ret:%d", ret);
        return CqeStatus::kCqeInnerError;
    }
    if (recvInfo.type != DRV_LOGIC_TYPE) { // 非DRV_LOGIC_TYPE不支持解析
        return CqeStatus::kDefault;
    }
    uint32_t reportNum = recvInfo.report_cqe_num;
    CHK_PRT_RET(
        reportNum > AC_SQE_REV_MAX_CNT,
        HCCL_ERROR(
            "report cqe num %u should "
            "not be bigger than %u",
            reportNum, AC_SQE_REV_MAX_CNT),
        CqeStatus::kCqeUnknown);
    for (uint32_t idx = 0U; idx < reportNum; ++idx) {
        const auto& reportOfOne
            = *((reinterpret_cast<rtLogicCqReport_t*>(recvInfo.cqe_addr)) + idx); // 外部保证是有效的地址
        if (IsExceptionCqe(reportOfOne)) {
            HCCL_ERROR(
                "Task {%s} run failed of exception, idx:[%u], info:[%s]", cqeQueryInput.ToString().c_str(), idx,
                StringLogicCqReportInfo(reportOfOne).c_str());
            cqeException = reportOfOne;
            return CqeStatus::kCqeException;
        }
    }
    return CqeStatus::kDefault;
}

HcclResult StreamsKill(const uint32_t devId)
{
    CHK_PRT_RET(
        (halTsdrvCtl == nullptr),
        HCCL_ERROR("halTsdrvCtl is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    ts_ctrl_msg_body_t killIn = {};
    ts_ctrl_msg_body_t killAck = {};
    size_t ackCount = sizeof(ts_ctrl_msg_body_t);
    killIn.type = OP_ABORT_APP;
    struct tsdrv_ctrl_msg para = {};
    para.tsid = 0;
    para.msg_len = sizeof(ts_ctrl_msg_body_t);
    para.msg = static_cast<void*>(&killIn);
    const drvError_t ret = halTsdrvCtl(
        devId, TSDRV_CTL_CMD_CTRL_MSG, static_cast<void*>(&para), sizeof(tsdrv_ctrl_msg), static_cast<void*>(&killAck),
        &ackCount);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR("halTsdrvCtl failed. ret = %d\n", ret);
        return HCCL_E_DRV;
    }
    return HCCL_SUCCESS;
}

inline u64 GetCurCpuTimestamp()
{
    constexpr u64 NSEC_PER_SEC = 1000000000U;
    struct timespec timestamp;
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
}

HcclResult DeviceQuery(const uint32_t devId, const uint32_t step, const uint32_t timeout)
{
    CHK_PRT_RET(
        (halTsdrvCtl == nullptr),
        HCCL_ERROR("halTsdrvCtl is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    uint32_t status;
    uint64_t endTime;
    const uint64_t startTime = GetCurCpuTimestamp();
    bool flag = true;
    while (flag) {
        ts_ctrl_msg_body_t queryIn = {};
        ts_ctrl_msg_body_t queryAck = {};
        size_t ackCount = sizeof(ts_ctrl_msg_body_t);
        queryIn.type = OP_QUERY_ABORT_STATUS;
        queryIn.u.query_task_info.choice = APP_ABORT_STS_QUERY_BY_PID;
        struct tsdrv_ctrl_msg para = {};
        para.tsid = 0;
        para.msg_len = sizeof(ts_ctrl_msg_body_t);
        para.msg = static_cast<void*>(&queryIn);
        const drvError_t ret = halTsdrvCtl(
            devId, TSDRV_CTL_CMD_CTRL_MSG, static_cast<void*>(&para), sizeof(tsdrv_ctrl_msg),
            static_cast<void*>(&queryAck), &ackCount);
        if ((ret != DRV_ERROR_NONE) || (ackCount != sizeof(ts_ctrl_msg_body_t))) {
            HCCL_ERROR("halTsdrvCtl failed. ret = %d\n", ret);
            return HCCL_E_DRV;
        }

        status = queryAck.u.query_task_ack_info.status;
        if (status >= step) {
            flag = false;
            break;
        }
        endTime = GetCurCpuTimestamp();
        if ((timeout != 0U) && ((endTime - startTime) > timeout)) {
            HCCL_ERROR("kill query timeout.\n");
            return HCCL_E_TIMEOUT;
        }
        SaluSleep(5000U);
    }
    return HCCL_SUCCESS;
}

namespace hccl_plf {
// 把SDMA类错误码转换成Ts对应的错误码
uint16_t SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode)
{
    switch (cqeErrCode) {
        case RT_SDMA_COMPERR:
            return TS_ERROR_SDMA_LINK_ERROR;
        case RT_SDMA_COMPDATAERR:
            return TS_ERROR_SDMA_POISON_ERROR;
        case RT_SDMA_DATAERR:
            return TS_ERROR_SDMA_DDRC_ERROR;
        case TS_ERROR_RETRY_CONSTRAINT:
            return TS_ERROR_RETRY_CONSTRAINT;
        default:
            return TS_ERROR_HCCL_OTHER_ERROR;
    }
}

HcclResult SendTaskExceptionByMBox(
    const u32 localDeviceId, const u32 notifyId, const u32 tsId, const s32 userStreamId, const u32 cqeErrCode)
{
    CHK_PRT_RET(
        (halEschedSubmitEvent == nullptr),
        HCCL_ERROR("halEschedSubmitEvent is nullptr, "
                   "Does not support this interface."),
        HCCL_E_DRV);
    ts_aicpu_sqe_t aicpuSqe = {};
    u32 hostpid = 0;
    u32 vf_id = 0;
    // 调整drvQueryProcessHostPid获取pid和vf_id的值
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, &vf_id, &hostpid, nullptr));

    aicpuSqe.pid = hostpid;
    aicpuSqe.cmd_type = AICPU_RECORD;
    aicpuSqe.vf_id = vf_id;
    aicpuSqe.tid = 0U; // notify is no need tid
    aicpuSqe.u.aicpu_record.record_type = AICPU_MSG_NOTIFY_RECORD;
    aicpuSqe.u.aicpu_record.record_id = notifyId;

    aicpuSqe.ts_id = static_cast<uint8_t>(tsId);

    aicpuSqe.u.aicpu_record.fault_stream_id = static_cast<uint16_t>(userStreamId);

    aicpuSqe.u.aicpu_record.ret_code = SwitchSdmaCqeErrCodeToTsErrCode(cqeErrCode);

    struct event_summary event = {};
    event.dst_engine = TS_CPU;
    event.policy = ONLY;
    event.pid = 0;
    event.grp_id = 0;
    event.event_id = EVENT_TS_CTRL_MSG;
    event.subevent_id = 0U;
    event.msg_len = static_cast<uint32_t>(sizeof(ts_aicpu_sqe_t));
    event.msg = PtrToPtr<ts_aicpu_sqe_t, char_t>(&aicpuSqe);
    auto ret = halEschedSubmitEvent(localDeviceId, &event);
    if (ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        HCCL_ERROR(
            "[SendTaskExceptionByMBox]Send msg async to ts failed. ret=%d, streamId=%d, "
            "notifyId=%u.",
            ret, userStreamId, notifyId);
        return HCCL_E_DRV;
    }
    HCCL_RUN_INFO(
        "[SendTaskExceptionByMBox]Send msg async to ts finished. streamId=%d, notifyId=%u, msg_size=%u, "
        "hostpid=%u, vf_id=%u, errCode=%u.",
        userStreamId, notifyId, static_cast<uint32_t>(sizeof(ts_aicpu_sqe_t)), hostpid, vf_id, cqeErrCode);
    return HCCL_SUCCESS;
}
} // namespace hccl_plf
