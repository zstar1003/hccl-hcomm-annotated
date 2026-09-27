/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_EXCEPTION_FUNC_H
#define HCCL_TASK_EXCEPTION_FUNC_H

#include <unordered_map>
#include <functional>
#include <memory>
#include "stream_lite.h"
#include "ascend_hal_define.h"
#include "daemon_func.h"
#include "communicator_impl_lite.h"

namespace Hccl {
extern "C" {
drvError_t __attribute__((weak)) halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info);
drvError_t __attribute__((weak)) drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t* local_dev_id);
};

struct rtLogicCqReport_t {
    volatile uint16_t streamId;
    volatile uint16_t taskId;
    volatile uint32_t errorCode; // cqe acc_status/sq_sw_status
    volatile uint8_t errorType;  // bit0 ~ bit5 cqe stars_defined_err_code, bit 6 cqe warning bit
    volatile uint8_t sqeType;
    volatile uint16_t sqId;
    volatile uint16_t sqHead;
    volatile uint16_t matchFlag : 1;
    volatile uint16_t dropFlag : 1;
    volatile uint16_t errorBit : 1;
    volatile uint16_t accError : 1;
    volatile uint16_t reserved0 : 12;
    union {
        volatile uint64_t timeStamp;
        volatile uint16_t sqeIndex;
    } u1;
    /* Union description:
     * Internal: enque_timestamp temporarily used as dfx
     * External: reserved1
     */
    union {
        volatile uint64_t enqueTimeStamp;
        volatile uint64_t reserved1;
    } u2;
};

class TaskExceptionFunc : public DaemonFunc {
    using Callback = std::function<void(CommunicatorImplLite*, rtLogicCqReport_t*)>;

public:
    static TaskExceptionFunc& GetInstance();
    void SetEnable(bool isEnable);
    void SetDevId(uint32_t devId);
    uint32_t GetDevId() const { return devId_; }
    void Register(StreamLite* streamLite);
    void UnRegister(const StreamLite* streamLite);
    void Call() override;
    void RegisterCallback(const Callback& callback);
    bool IsExceptionCqe(const rtLogicCqReport_t& reportOfOne) const;
    unsigned int GetTrailingZeros(uint8_t num) const;
    std::string StringLogicCqReportInfo(const rtLogicCqReport_t& reportOfOne) const;
    uint32_t GetReporterInfo(const StreamLite* curStream, std::shared_ptr<halReportRecvInfo> recvInfo);
    std::string ErrorType2Str(uint8_t errorType) const;

private:
    // 私有构造函数，防止外部实例化
    TaskExceptionFunc() {};
    // 禁用拷贝构造函数
    TaskExceptionFunc(const TaskExceptionFunc&) = delete;
    // 禁用赋值运算符
    TaskExceptionFunc& operator=(const TaskExceptionFunc&) = delete;

    std::string CqeStatus2Str(uint32_t errorCode) const;

private:
    bool isEnable_{true};
    std::unordered_map<uint32_t, StreamLite*> streamLiteMap_; // keymap: sqid
    uint32_t devId_{0};
    Callback callback_;
};

} // namespace Hccl

#endif // HCCL_TASK_EXCEPTION_FUNC_H
