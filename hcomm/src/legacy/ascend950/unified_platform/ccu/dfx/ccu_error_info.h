/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ERROR_INFO_H
#define CCU_ERROR_INFO_H

#include <cstdint>
#include "ccu_rep_type.h"
namespace Hccl {

constexpr uint32_t MISSION_STATUS_MSG_LEN = 64;
constexpr uint32_t WAIT_SIGNAL_CHANNEL_SIZE = 16;
constexpr uint32_t BUF_REDUCE_ID_SIZE = 8;

MAKE_ENUM(CcuErrorType, DEFAULT, MISSION, WAIT_SIGNAL, TRANS_MEM, BUF_TRANS_MEM, BUF_REDUCE, LOOP, LOOP_GROUP)

struct CcuErrorInfo {
    // 根据不同的typeId解析不同的union类型
    CcuErrorType type;

    // 出异常的Rep类型
    CcuRep::CcuRepType repType;

    // CCU任务执行的DieId
    uint8_t dieId;

    // CCU任务执行的MissionId
    uint8_t missionId;

    // Error所在的指令Id
    uint16_t instrId;

    union {
        struct {
            char missionError[MISSION_STATUS_MSG_LEN];
        } mission;

        struct {
            uint16_t signalId;
            uint16_t signalMask;
            uint16_t signalValue;
            uint16_t paramId;
            uint64_t paramValue;
            uint16_t channelId[WAIT_SIGNAL_CHANNEL_SIZE];
        } waitSignal;

        struct {
            uint64_t locAddr;
            uint64_t locToken;
            uint64_t rmtAddr;
            uint64_t rmtToken;
            uint64_t len;
            uint16_t signalId;
            uint16_t signalMask;
            uint16_t channelId;
            uint16_t dataType;
            uint16_t opType;
        } transMem;

        struct {
            uint16_t bufId;
            uint64_t addr;
            uint64_t token;
            uint64_t len;
            uint16_t signalId;
            uint16_t signalMask;
            uint16_t channelId;
        } bufTransMem;

        struct {
            uint16_t bufIds[BUF_REDUCE_ID_SIZE];
            uint16_t count;
            uint16_t dataType;
            uint16_t outputDataType;
            uint16_t opType;
            uint16_t signalId;
            uint16_t signalMask;
            uint16_t xnIdLength;
        } bufReduce;

        struct {
            uint16_t startInstrId;
            uint16_t endInstrId;
            uint16_t loopEngineId;
            uint16_t loopCnt;        // 该Loop需要循环执行的次数
            uint16_t loopCurrentCnt; // 该Loop已经循环次数
            uint32_t addrStride;
        } loop;

        struct {
            uint16_t startLoopInsId;
            uint16_t loopInsCnt;
            uint16_t expandOffset;
            uint16_t expandCnt;
        } loopGroup;
    } msg;

    void SetBaseInfo(CcuRep::CcuRepType ccuRepType, uint8_t ccuDieId, uint8_t ccuMissionId, uint16_t insId)
    {
        this->repType = ccuRepType;
        this->dieId = ccuDieId;
        this->missionId = ccuMissionId;
        this->instrId = insId;
    }
};

} // namespace Hccl
#endif // CCU_ERROR_INFO_H
