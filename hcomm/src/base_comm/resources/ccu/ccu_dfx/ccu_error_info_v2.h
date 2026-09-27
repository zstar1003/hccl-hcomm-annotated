/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ERROR_INFO_V2_H
#define CCU_ERROR_INFO_V2_H
#include <cstdint>

namespace hcomm {

struct CcuLoopContextV2 {
    union {
        uint16_t value;
        uint16_t xnOffset;
    } part0;

    union {
        uint16_t value;
        struct {
            uint16_t ckeOffset : 12;
            uint16_t msOffset : 4;
        };
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t msOffset : 7;
            uint16_t addrOffset : 9;
        };
    } part2;

    union {
        uint16_t value;
        uint16_t addrOffset;
    } part3;

    union {
        uint16_t value;
        struct {
            uint16_t addrOffset : 7;
            uint16_t currentIns : 9;
        };
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 7;
            uint16_t addrStride : 9;
        };
    } part5;

    union {
        uint16_t value;
        uint16_t addrStride;
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t addrStride : 7;
            uint16_t loopWishCheckBit : 9;
        };
    } part7;

    union {
        uint16_t value;
        struct {
            uint16_t loopWishCheckBit : 7;
            uint16_t waitLoopCheckBit : 9;
        };
    } part8;

    union {
        uint16_t value;
        struct {
            uint16_t waitLoopCheckBit : 7;
            uint16_t waitLoopCheckBitVld : 1;
            uint16_t loopMode : 1;
            uint16_t denyCnt : 7;
        };
    } part9;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 3;
            uint16_t currentCnt : 13;
        };
    } part10;

    union {
        uint16_t value;
        struct {
            uint16_t totalCnt : 13;
            uint16_t endIns : 3;
        };
    } part11;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 13;
            uint16_t startIns : 3;
        };
    } part12;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 13;
            uint16_t missionId : 3;
        };
    } part13;

    union {
        uint16_t value;
        struct {
            uint16_t missionId : 1;
            uint16_t loopVld : 1;
            uint16_t mprofileen : 1;
            uint16_t reserved : 13;
        };
    } part14;
    uint16_t reserved[17];

    uint16_t GetCurrentIns() const
    {
        return (part5.currentIns << 9) | (part4.currentIns); // 偏移9位
    }

    uint16_t GetCurrentCnt() const { return part10.currentCnt; }

    uint32_t GetAddrStride() const
    {
        const uint32_t low = static_cast<uint32_t>(part5.addrStride);
        const uint32_t mid = static_cast<uint32_t>(part6.addrStride) << 9;   // 偏移9位
        const uint32_t high = static_cast<uint32_t>(part7.addrStride) << 25; // 偏移25位
        return high | mid | low;
    }
};

struct CcuMissionContextV2 {
    union {
        uint16_t value;
        uint16_t taskIdRaw; // same hw register as V1::taskId
    } part0;

    union {
        uint16_t value;
        uint16_t streamIdRaw; // same hw register as V1::streamId
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t taskKillRaw : 1;
            uint16_t dieIdRaw : 2;
            uint16_t statusLo : 13;
        };
    } part2;

    union {
        uint16_t value;
        struct {
            uint16_t statusHi : 3;
            uint16_t counter : 10;
            uint16_t denyCnt : 3;
        };
    } part3;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 7;
            uint16_t currentIns : 9;
        };
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 7;
            uint16_t endIns : 9;
        };
    } part5;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 7;
            uint16_t startIns : 9;
        };
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 7;
            uint16_t profileEn : 1;
            uint16_t missionVld : 1;
            uint16_t cqeInfoPart1 : 7;
        };
    } part7;

    union {
        uint16_t value;
        uint16_t cqeInfoPart2;
    } part8;

    union {
        uint16_t value;
        uint16_t cqeInfoPart3;
    } part9;

    union {
        uint16_t value;
        uint16_t cqeInfoPart4;
    } part10;

    union {
        uint16_t value;
        uint16_t cqeInfoPart5;
    } part11;

    union {
        uint16_t value;
        uint16_t cqeInfoPart6;
    } part12;

    union {
        uint16_t value;
        struct {
            uint16_t cqeInfoPart7 : 9;
            uint16_t reserved : 7;
        };
    } part13;

    uint16_t reserved[18];

    uint16_t GetStatus() const
    {
        return (part3.statusHi << 13) | (part2.statusLo); // 偏移13位
    }

    uint16_t GetCurrentIns() const
    {
        return (part5.currentIns << 9) | (part4.currentIns); // 偏移9位
    }

    uint16_t GetStartIns() const
    {
        return (part7.startIns << 9) | (part6.startIns); // 偏移9位
    }

    uint16_t GetEndIns() const
    {
        return (part6.endIns << 9) | (part5.endIns); // 偏移9位
    }
};

} // namespace hcomm

#endif // CCU_ERROR_INFO_V2_H
