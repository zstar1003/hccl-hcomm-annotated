/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ERROR_INFO_V1_H
#define CCU_ERROR_INFO_V1_H
#include <cstdint>
#include "ccu_rep_type_v1.h"
#include "ccu_error_info_v2.h"
#include "enum_factory.h"

namespace hcomm {
constexpr uint32_t MISSION_STATUS_MSG_LEN = 64;
constexpr uint32_t WAIT_SIGNAL_CHANNEL_SIZE = 16;
constexpr uint32_t BUF_REDUCE_ID_SIZE = 8;

MAKE_ENUM(
    CcuErrorType, DEFAULT, MISSION, WAIT_SIGNAL, TRANS_MEM, BUF_TRANS_MEM, BUF_REDUCE, LOOP, LOOP_GROUP,
    WRITE_VAR_ATOMIC, WRITE_WITH_CNT_INC, CASC_CNT_WAIT, CASC_CNT_CLEAR, LOAD_ADD_IMM, STORE_ADD_IMM)

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
            uint16_t signalId;
            uint16_t signalMask;
            uint16_t channelId;
            uint64_t addr;
            uint64_t token;
            uint64_t len;
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

        struct {
            uint16_t dstId;
            uint64_t dstValue;
            uint16_t signalId;
            uint16_t signalMask;
        } writeVarAtomic;

        struct {
            uint64_t locAddr;
            uint64_t rmtAddr;
            uint64_t len;
            uint64_t incCntAddr;
        } writeWithCntInc;

        struct {
            uint64_t targetValue;
            uint64_t currentCntValue;
        } cascCntWait;

        struct {
            uint16_t wishCntXnIdFirst; // block中的第一个高性能Xn
            uint16_t expectedCntXn;    // block中的最后一个高性能xn
        } cascCntClear;

        struct {
            uint16_t srcId;
            uint16_t srcOffsetId;
            uint16_t dstId;
            uint16_t immAddValue;
            uint64_t srcValue;
            uint64_t srcOffsetValue;
        } loadAddImm;

        struct {
            uint16_t dstId;
            uint16_t dstOffsetId;
            uint16_t srcId;
            uint16_t immAddValue;
            uint64_t dstValue;
            uint64_t dstOffsetValue;
        } storeAddImm;
    } msg;

    void SetBaseInfo(CcuRep::CcuRepType ccuRepType, uint8_t ccuDieId, uint8_t ccuMissionId, uint16_t insId)
    {
        this->repType = ccuRepType;
        this->dieId = ccuDieId;
        this->missionId = ccuMissionId;
        this->instrId = insId;
    }
};

struct ErrorInfoBase {
    int32_t deviceId;
    uint8_t dieId;
    uint8_t missionId;
    uint16_t currentInsId;
    uint16_t status;
};

union LoopXm {
    uint64_t value;
    struct {
        uint64_t loopCnt : 13;
        uint64_t gsaStride : 32;
        uint64_t loopCtxId : 8;
        uint64_t reserved : 11;
    };
};

struct CcuLoopContext {
    union {
        uint16_t value;
        uint16_t timestamp;
    } part0;

    union {
        uint16_t value;
        uint16_t timestamp;
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t timestamp : 4;
            uint16_t ckeOffset : 10;
            uint16_t msOffset : 2;
        };
    } part2;

    union {
        uint16_t value;
        struct {
            uint16_t msOffset : 9;
            uint16_t addrOffset : 7;
        };
    } part3;

    union {
        uint16_t value;
        uint16_t addrOffset;
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t addrOffset : 9;
            uint16_t ckBit : 7;
        };
    } part5;

    union {
        uint16_t value;
        uint16_t ckBit;
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t ckBit : 9;
            uint16_t perfMode : 1;
            uint16_t waitLoopCkbitValue : 6;
        };
    } part7;

    union {
        uint16_t value;
        uint16_t waitLoopCkbitValue;
    } part8;

    union {
        uint16_t value;
        struct {
            uint16_t waitLoopCkbitValue : 10;
            uint16_t currentIns : 6; // Current_ins [5:0]
        };
    } part9;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 10; // Current_ins [15:6]
            uint16_t addrStride : 6;  // Addr_stride [5:0]
        };
    } part10;

    union {
        uint16_t value;
        uint16_t addrStride; // Addr_stride [21:6]
    } part11;

    union {
        uint16_t value;
        struct {
            uint16_t addrStride : 10; // Addr_stride [31:22]
            uint16_t denyCnt : 6;
        };
    } part12;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 4;
            uint16_t currentCnt : 12; // Current_cnt [11:0]
        };
    } part13;

    union {
        uint16_t value;
        struct {
            uint16_t currentCnt : 1; // Current_cnt [12]
            uint16_t totalCnt : 13;
            uint16_t endIns : 2;
        };
    } part14;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 14;
            uint16_t startIns : 2;
        };
    } part15;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 14;
            uint16_t missionId : 2;
        };
    } part16;

    union {
        uint16_t value;
        struct {
            uint16_t missionId : 2;
            uint16_t reserved : 14;
        };
    } part17;

    uint16_t reserved[14]; // part 18-31

    uint16_t GetCurrentIns() const
    {
        return (part10.currentIns << 6) | (part9.currentIns); // part10.currentIns为[15:6]位
    }

    uint16_t GetCurrentCnt() const
    {
        return (part14.currentCnt << 12) | (part13.currentCnt); // part14.currentCnt为第[12]位
    }

    uint32_t GetAddrStride() const
    {
        const uint32_t low = static_cast<uint32_t>(part10.addrStride);
        const uint32_t mid = static_cast<uint32_t>(part11.addrStride) << 6;   // part11.addrStride为[21:6]位
        const uint32_t high = static_cast<uint32_t>(part12.addrStride) << 22; // part12.addrStride为[31:22]位
        return high | mid | low;
    }
};

struct CcuMissionContext {
    union {
        uint16_t value;
        uint16_t taskId;
    } part0;

    union {
        uint16_t value;
        uint16_t streamId;
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t taskKill : 1;
            uint16_t dieId : 2;
            uint16_t status : 13; // Status [12:0]
        };
    } part2;

    union {
        uint16_t value;
        struct {
            uint16_t status : 3; // Status [15:13]
            uint16_t counter : 8;
            uint16_t denyCnt : 5;
        };
    } part3;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 5;
            uint16_t currentIns : 11; // Current Ins [10:0]
        };
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 5; // Current Ins [15:11]
            uint16_t endIns : 11;
        };
    } part5;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 5;
            uint16_t startIns : 11;
        };
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 5;
            uint16_t profileEn : 1;
            uint16_t missionVld : 1;
            uint16_t reserved : 9;
        };
    } part7;

    uint16_t reserved[24]; // part 8-31

    uint16_t GetStatus() const
    {
        return (part3.status << 13) | (part2.status); // part3.status为[15:13]位
    }

    uint16_t GetCurrentIns() const
    {
        return (part5.currentIns << 11) | (part4.currentIns); // part5.currentIns为[15:11]位
    }

    uint16_t GetStartIns() const
    {
        return (part7.startIns << 11) | (part6.startIns); // part7.startIns[15:11]位
    }

    uint16_t GetEndIns() const
    {
        return (part6.endIns << 11) | (part5.endIns); // part6.endIns[15:11]位
    }
};

union LoopGroupXn {
    uint64_t value;
    struct {
        uint64_t reservedLow : 41;
        uint64_t loopInsCnt : 7;
        uint64_t expandOffset : 7;
        uint64_t expandCnt : 7;
        uint64_t reservedHigh : 2;
    };
};

}; // namespace hcomm

#endif // CCU_ERROR_INFO_V1_H
