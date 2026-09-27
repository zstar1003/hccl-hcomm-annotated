/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_exception_func.h"
#include "communicator_impl_lite_manager.h"
#include "log.h"
#include <map>
#include <vector>
#include <memory>
#include <string>

namespace Hccl {
TaskExceptionFunc& TaskExceptionFunc::GetInstance()
{
    static TaskExceptionFunc instance;
    return instance;
}

void TaskExceptionFunc::SetEnable(bool isEnable)
{
    isEnable_ = isEnable;
    if (isEnable_) {
        HCCL_INFO("[TaskExceptionFunc] Task Exception enabled.");
    } else {
        HCCL_INFO("[TaskExceptionFunc] Task Exception disabled.");
    }
}

void TaskExceptionFunc::SetDevId(uint32_t devId) { devId_ = devId; }

void TaskExceptionFunc::RegisterCallback(const Callback& callback) { callback_ = callback; }

void TaskExceptionFunc::Register(StreamLite* streamLite) { streamLiteMap_[streamLite->GetSqId()] = streamLite; }

void TaskExceptionFunc::UnRegister(const StreamLite* streamLite) { streamLiteMap_.erase(streamLite->GetSqId()); }

std::string TaskExceptionFunc::ErrorType2Str(uint8_t errorType) const
{
    static const std::map<uint8_t, std::string> errorType2Str{
        {0b1, "exception"},              // bit0:代表是否有exception
        {0b11, "bus error"},             // bit1:代表是否为bus_error
        {0b101, "rsv"},                  // bit2:代表为rsv域段（该域段硬件值可能为0或1）
        {0b1001, "sqe error"},           // bit3:代表是否为sqe_error
        {0b10001, "res conflict error"}, // bit4:代表是否为res_conflict_error
        {0b100001, "pre_p/post_p error"}, // bit5:表示软件在pre_p或post_p过程中发生了错误，写入了sq_sw_status
    };
    const auto res = errorType2Str.find(errorType);
    return res == errorType2Str.end() ? "" : res->second;
}

std::string TaskExceptionFunc::CqeStatus2Str(uint32_t errorCode) const
{
    uint8_t status = errorCode & 0xFF; // errorCode低8bit是UB CQE的status
    static const std::map<uint8_t, std::string> code2Str{
        {0x00, "OK"},
        {0x01, "Unsupported OpCode"},
        {0x02, "Local Operation Error"},
        {0x03, "Remote Operation Error"},
        {0x04, "Transaction Retry Counter Exceeded"},
        {0x05, "Transaction ACK Timeout"},
        {0x06, "Jetty work Request Flushed"},
    };
    const auto res = code2Str.find(status);
    return res == code2Str.end() ? "Reserved" : res->second;
}

std::string TaskExceptionFunc::StringLogicCqReportInfo(const rtLogicCqReport_t& reportOfOne) const
{
    std::stringstream ss;
    ss << "streamId :" << reportOfOne.streamId;
    ss << " taskId :" << reportOfOne.taskId;
    ss << " errorCode :" << reportOfOne.errorCode;
    if (reportOfOne.sqeType == 9) { // sqeType等于9时, errorCode才按照UB的格式解析, 不等于9时可不关注errorCode
        const std::string errorStatus = CqeStatus2Str(reportOfOne.errorCode);
        ss << "(" << errorStatus << ")";
    }
    ss << " errorType :" << static_cast<uint32_t>(reportOfOne.errorType);
    const std::string errorTypeStr = ErrorType2Str(reportOfOne.errorType);
    if (!errorTypeStr.empty()) {
        ss << "(" << errorTypeStr << ")";
    }
    ss << " sqeType :" << static_cast<uint32_t>(reportOfOne.sqeType);
    ss << " sqId :" << reportOfOne.sqId;
    ss << " sqHead :" << reportOfOne.sqHead;
    ss << " matchFlag :" << reportOfOne.matchFlag;
    ss << " dropFlag :" << reportOfOne.dropFlag;
    ss << " errorBit :" << reportOfOne.errorBit;
    ss << " accError :" << reportOfOne.accError;
    return ss.str();
}

unsigned int TaskExceptionFunc::GetTrailingZeros(uint8_t num) const
{
    uint8_t count = 0;
    while ((num & 1U) == 0) {
        count++;
        num >>= 1;
        if (num == 1U) {
            break;
        }
    }
    return count;
}

constexpr uint8_t RT_STARS_EXIST_ERROR = 0x3FU;

bool TaskExceptionFunc::IsExceptionCqe(const rtLogicCqReport_t& reportOfOne) const
{
    if ((reportOfOne.errorType & RT_STARS_EXIST_ERROR) == 0U) { // 取低6位
        HCCL_INFO("ReportOfOne info [%s]", StringLogicCqReportInfo(reportOfOne).c_str());
        return false;
    }
    HCCL_ERROR("ReportOfOne error info [%s]", StringLogicCqReportInfo(reportOfOne).c_str());
    return true;
}

constexpr uint32_t MAX_REPORT_CNT = 256U;
constexpr uint32_t AC_SQE_REV_MAX_CNT = 32U;
enum class CqeStatus : int64_t {
    kDefault = 0,
    kCqeException,
    kCqeTimeOut,
    kCqeInnerError,
    kCqeUnknown,
};

uint32_t TaskExceptionFunc::GetReporterInfo(const StreamLite* curStream, std::shared_ptr<halReportRecvInfo> recvInfo)
{
    recvInfo->type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo->tsId = 0;
    recvInfo->report_cqe_num = 0;
    recvInfo->timeout = 0;              // 不设置超时时间，非阻塞
    recvInfo->task_id = 0xFFFF;         // 接收所有类型
    recvInfo->cqe_num = MAX_REPORT_CNT; // 单次接收的最大cqe数量
    recvInfo->stream_id = curStream->GetId();
    recvInfo->cqId = curStream->GetCqId();
    auto exceptionInfo = reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr);

    // 接收错误信息
    drvError_t ret = halCqReportRecv(devId_, recvInfo.get());
    if (recvInfo->report_cqe_num != 0) {
        HCCL_INFO(
            "[TaskExceptionFunc]after exceptionInfo deviceId[%u], streamId[%u], taskId[%u], "
            "recvInfo->report_cqe_num[%u].",
            devId_, exceptionInfo->streamId, exceptionInfo->taskId, recvInfo->report_cqe_num);
    }

    if (ret == DRV_ERROR_WAIT_TIMEOUT) {
        HCCL_INFO("[TaskExceptionFunc]halCqReportRecv has found nothing, ret:%d", ret);
        return 1;
    }
    if (ret != DRV_ERROR_NONE) {
        HCCL_WARNING("[TaskExceptionFunc]halCqReportRecv failed, ret:%d", ret);
        return 1;
    }
    if (recvInfo->type != DRV_LOGIC_TYPE) { // 非DRV_LOGIC_TYPE不支持解析
        HCCL_WARNING(
            "[TaskExceptionFunc]halCqReportRecv type is not %d, recvInfo->type:%d", DRV_LOGIC_TYPE, recvInfo->type);
        return 1;
    }
    return 0;
}

void TaskExceptionFunc::Call()
{
    TRY_CATCH_PRINT_ERROR(
        if (!isEnable_) { return; } auto recvInfo = std::make_shared<halReportRecvInfo>();
        constexpr uint32_t cqeSize = MAX_REPORT_CNT * sizeof(rtLogicCqReport_t);
        uint8_t tmpAddr[cqeSize] = {}; // cqe byte size
        recvInfo->cqe_addr = tmpAddr;  // 外部保证是有效的地址

        std::vector<CommunicatorImplLite*> aicpuComms = CommunicatorImplLiteMgr::GetInstance().GetAll();
        for (auto aicpuComm
             : aicpuComms) {
            std::vector<StreamLite*> aicpuStreams = aicpuComm->GetStreamLiteMgr()->GetAllStreams();
            for (auto& aicpuStream : aicpuStreams) {
                if (aicpuStream == nullptr) {
                    HCCL_ERROR("[TaskExceptionFunc]stream of in aicpuComm[%s] is nullptr", aicpuComm->GetId().c_str());
                    continue;
                }
                if (GetReporterInfo(aicpuStream, recvInfo) != 0) {
                    continue;
                }
                uint32_t reportNum = recvInfo->report_cqe_num;
                if (reportNum > MAX_REPORT_CNT) {
                    HCCL_ERROR(
                        "[TaskExceptionFunc]report cqe num %u should not be greater than %u", reportNum,
                        MAX_REPORT_CNT);
                    continue;
                }
                for (uint32_t idx = 0U; idx < reportNum; ++idx) {
                    auto& reportOfOne
                        = *((reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr)) + idx); // 外部保证是有效的地址
                    if (IsExceptionCqe(reportOfOne)) {
                        callback_(aicpuComm, &reportOfOne);
                    }
                }
            }
        });
}
} // namespace Hccl
