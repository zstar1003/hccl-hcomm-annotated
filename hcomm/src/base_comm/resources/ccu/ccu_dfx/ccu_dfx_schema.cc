/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dfx_schema.h"

#include "securec.h"
#include "log.h"
#include "ccu_error_info_v1.h"
#include "ccu_error_info_v2.h"
#include "adapter_rts_common.h"

namespace hcomm {
namespace {
    struct CcumDfxCommonInfo {
        unsigned int ccumSqeRecvCnt;
        unsigned int ccumSqeSendCnt;
        unsigned int ccumMissionDfx;
        unsigned int ccumSqeDropCnt;
        unsigned int ccumSqeAddrLenErrDropCnt;
        unsigned int lqcCcuSecReg0;
        unsigned int ccumTifSqeCnt;
        unsigned int ccumTifCqeCnt;
        unsigned int ccumCifSqeCnt;
        unsigned int ccumCifCqeCnt;
    };

    struct ccumDfxInfo {
        unsigned int queryResult; // 0:success, 1:fail
        CcumDfxCommonInfo commonInfo;
    };

    struct ccumDfxInfoV2 {
        union {
            struct {
                unsigned int queryResult : 1; // 0:success, 1:fail
                unsigned int sqeRecvCnt : 1;
                unsigned int sqeSendCnt : 1;
                unsigned int missionDfx : 1;
                unsigned int sqeDropCnt : 1;
                unsigned int sqeErrDropCnt : 1;
                unsigned int secReg0 : 1;
                unsigned int tifSqeCnt : 1;
                unsigned int tifCqeCnt : 1;
                unsigned int cifSqeCnt : 1;
                unsigned int cifCqeCnt : 1;
                unsigned int mcmDfx : 1;
                unsigned int resv : 20;
            } bs;
            unsigned int validBits : 32;
        };
        union {
            struct {
                CcumDfxCommonInfo commonInfo;
                unsigned int ccumMcmDfx;
                unsigned int resv[20];
            } dfxInfo;
            unsigned int regVal[31];
        };
    };
    void PrintCcumDfxInfoV1(const void* rawData, std::ostringstream& oss)
    {
        if (rawData == nullptr) {
            oss << " [rawData is null]";
            HCCL_ERROR("[PrintCcumDfxInfoV1] rawData is null");
            return;
        }
        struct ccumDfxInfo info {};
        const auto copyRet = memcpy_s(&info, sizeof(info), rawData, sizeof(info));
        if (copyRet != EOK) {
            oss << " [decode failed]";
            HCCL_ERROR("[PrintCcumDfxInfoV1] memcpy_s failed, ret[%d]", copyRet);
            return;
        }
        if (info.queryResult != 0U) {
            HCCL_ERROR("get ccu dfx info fail, ccu dfx info not all correct");
        }
        oss << " SQE_RECV_CNT[" << info.commonInfo.ccumSqeRecvCnt << ']';
        oss << " SQE_SEND_CNT[" << info.commonInfo.ccumSqeSendCnt << ']';
        oss << " MISSION_DFX[" << info.commonInfo.ccumMissionDfx << ']';
        oss << " TIF_SQE_CNT[" << info.commonInfo.ccumTifSqeCnt << ']';
        oss << " TIF_CQE_CNT[" << info.commonInfo.ccumTifCqeCnt << ']';
        oss << " CIF_SQE_CNT[" << info.commonInfo.ccumCifSqeCnt << ']';
        oss << " CIF_CQE_CNT[" << info.commonInfo.ccumCifCqeCnt << ']';
        oss << " SQE_DROP_CNT[" << info.commonInfo.ccumSqeDropCnt << ']';
        oss << " SQE_ADDR_LEN_ERR_DROP_CNT[" << info.commonInfo.ccumSqeAddrLenErrDropCnt << ']';
        oss << " ccumIsEnable[" << (info.commonInfo.lqcCcuSecReg0 & 1U) << ']';
    }

    void PrintCcumDfxInfoV2(const void* rawData, std::ostringstream& oss)
    {
        if (rawData == nullptr) {
            oss << " [rawData is null]";
            HCCL_ERROR("[PrintCcumDfxInfoV2] rawData is null");
            return;
        }
        struct ccumDfxInfoV2 info {};
        const auto copyRet = memcpy_s(&info, sizeof(info), rawData, sizeof(info));
        if (copyRet != EOK) {
            oss << " [decode failed]";
            HCCL_ERROR("[PrintCcumDfxInfoV2] memcpy_s failed, ret[%d]", copyRet);
            return;
        }
        if (info.bs.queryResult != 0U) {
            HCCL_ERROR("get ccu dfx info fail, ccu dfx info not all correct");
        }
        auto dump = [&oss](const char* name, unsigned int value, bool hwValid) {
            oss << ' ' << name << '[';
            if (hwValid) {
                oss << value;
            } else {
                oss << "INVALID(hardware)";
                HCCL_WARNING("[CCU DFX][V2] %s marked invalid by hardware", name);
            }
            oss << ']';
        };
        dump("SQE_RECV_CNT", info.dfxInfo.commonInfo.ccumSqeRecvCnt, info.bs.sqeRecvCnt != 0U);
        dump("SQE_SEND_CNT", info.dfxInfo.commonInfo.ccumSqeSendCnt, info.bs.sqeSendCnt != 0U);
        dump("MISSION_DFX", info.dfxInfo.commonInfo.ccumMissionDfx, info.bs.missionDfx != 0U);
        dump("TIF_SQE_CNT", info.dfxInfo.commonInfo.ccumTifSqeCnt, info.bs.tifSqeCnt != 0U);
        dump("TIF_CQE_CNT", info.dfxInfo.commonInfo.ccumTifCqeCnt, info.bs.tifCqeCnt != 0U);
        dump("CIF_SQE_CNT", info.dfxInfo.commonInfo.ccumCifSqeCnt, info.bs.cifSqeCnt != 0U);
        dump("CIF_CQE_CNT", info.dfxInfo.commonInfo.ccumCifCqeCnt, info.bs.cifCqeCnt != 0U);
        dump("SQE_DROP_CNT", info.dfxInfo.commonInfo.ccumSqeDropCnt, info.bs.sqeDropCnt != 0U);
        dump(
            "SQE_ADDR_LEN_ERR_DROP_CNT", info.dfxInfo.commonInfo.ccumSqeAddrLenErrDropCnt, info.bs.sqeErrDropCnt != 0U);
        dump("ccumIsEnable", info.dfxInfo.commonInfo.lqcCcuSecReg0 & 1U, info.bs.secReg0 != 0U);
        dump("MCM_DFX", info.dfxInfo.ccumMcmDfx, info.bs.mcmDfx != 0U);
    }

    HcclResult GetCcuMissionInfoV1(const void* rawData, CcuMissionInfo* out)
    {
        if (rawData == nullptr || out == nullptr) {
            HCCL_ERROR("[GetCcuMissionInfoV1] invalid input: rawData=%p, out=%p", rawData, out);
            return HCCL_E_PARA;
        }
        CcuMissionContext ctx{};
        const auto copyRet = memcpy_s(&ctx, sizeof(ctx), rawData, sizeof(ctx));
        if (copyRet != EOK) {
            HCCL_ERROR("[GetCcuMissionInfoV1] memcpy_s failed, ret[%d]", copyRet);
            return HCCL_E_INTERNAL;
        }
        out->currentIns = ctx.GetCurrentIns();
        out->endIns = ctx.GetEndIns();
        out->startIns = ctx.GetStartIns();
        return HCCL_SUCCESS;
    }

    HcclResult GetCcuLoopInfoV1(const void* rawData, CcuLoopInfo* out)
    {
        if (rawData == nullptr || out == nullptr) {
            HCCL_ERROR("[GetCcuLoopInfoV1] invalid input: rawData=%p, out=%p", rawData, out);
            return HCCL_E_PARA;
        }
        CcuLoopContext ctx{};
        const auto copyRet = memcpy_s(&ctx, sizeof(ctx), rawData, sizeof(ctx));
        if (copyRet != EOK) {
            HCCL_ERROR("[GetCcuLoopInfoV1] memcpy_s failed, ret[%d]", copyRet);
            return HCCL_E_INTERNAL;
        }
        out->currentCnt = ctx.GetCurrentCnt();
        out->addrStride = ctx.GetAddrStride();
        return HCCL_SUCCESS;
    }

    HcclResult GetCcuMissionInfoV2(const void* rawData, CcuMissionInfo* out)
    {
        if (rawData == nullptr || out == nullptr) {
            HCCL_ERROR("[GetCcuMissionInfoV2] invalid input: rawData=%p, out=%p", rawData, out);
            return HCCL_E_PARA;
        }

        CcuMissionContextV2 ctx{};
        const auto copyRet = memcpy_s(&ctx, sizeof(ctx), rawData, sizeof(ctx));
        if (copyRet != EOK) {
            HCCL_ERROR("[GetCcuMissionInfoV2] memcpy_s failed, ret[%d]", copyRet);
            return HCCL_E_INTERNAL;
        }
        out->currentIns = ctx.GetCurrentIns();
        out->endIns = ctx.GetEndIns();
        out->startIns = ctx.GetStartIns();
        return HCCL_SUCCESS;
    }

    HcclResult GetCcuLoopInfoV2(const void* rawData, CcuLoopInfo* out)
    {
        if (rawData == nullptr || out == nullptr) {
            HCCL_ERROR("[GetCcuLoopInfoV2] invalid input: rawData=%p, out=%p", rawData, out);
            return HCCL_E_PARA;
        }

        CcuLoopContextV2 ctx{};
        const auto copyRet = memcpy_s(&ctx, sizeof(ctx), rawData, sizeof(ctx));
        if (copyRet != EOK) {
            HCCL_ERROR("[GetCcuLoopInfoV2] memcpy_s failed, ret[%d]", copyRet);
            return HCCL_E_INTERNAL;
        }
        out->currentCnt = ctx.GetCurrentCnt();
        out->addrStride = ctx.GetAddrStride();

        return HCCL_SUCCESS;
    }

    enum class CcuSchemaVersion : uint8_t { CCU_SCHEMA_V1 = 0, CCU_SCHEMA_V2 = 1, CCU_SCHEMA_COUNT = 2 };

    const CcuVersionOps CCU_V1_OPS = {
        "CCU_V1",
        PrintCcumDfxInfoV1,
        GetCcuMissionInfoV1,
        GetCcuLoopInfoV1,
    };

    const CcuVersionOps CCU_V2_OPS = {
        "CCU_V2",
        PrintCcumDfxInfoV2,
        GetCcuMissionInfoV2,
        GetCcuLoopInfoV2,
    };

    const CcuVersionOps* const CCU_OPS_TABLE[] = {
        &CCU_V1_OPS, // [0] V1
        &CCU_V2_OPS, // [1] V2
    };

    // 编译期校验：ops 表下标必须与 CcuSchemaVersion 枚举值一一对应，
    // 后续新增 schema 版本时若漏改表项会立即编译失败。
    static_assert(
        sizeof(CCU_OPS_TABLE) / sizeof(CCU_OPS_TABLE[0]) == static_cast<size_t>(CcuSchemaVersion::CCU_SCHEMA_COUNT),
        "CCU_OPS_TABLE size must match CcuSchemaVersion::CCU_SCHEMA_COUNT");

    HcclResult GetCcuSchemaVersion(CcuSchemaVersion& schemaVersion)
    {
        DevType deviceType = DevType::DEV_TYPE_COUNT;
        HcclResult ret = hrtGetDeviceType(deviceType);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[GetCcuSchemaVersion] hrtGetDeviceType failed, ret[%d].", ret);
            return ret;
        }
        if (deviceType == DevType::DEV_TYPE_950) {
            schemaVersion = CcuSchemaVersion::CCU_SCHEMA_V1;
        } else if (deviceType == DevType::DEV_TYPE_960) {
            schemaVersion = CcuSchemaVersion::CCU_SCHEMA_V2;
        } else {
            HCCL_ERROR(
                "[GetCcuSchemaVersion] Unsupported deviceType[%u], only 950/960 supported.",
                static_cast<uint32_t>(deviceType));
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ResolveCcuOps(CcuSchemaVersion schemaVersion, const CcuVersionOps*& ops)
    {
        const uint8_t versionIdx = static_cast<uint8_t>(schemaVersion);
        const size_t tableSize = sizeof(CCU_OPS_TABLE) / sizeof(CCU_OPS_TABLE[0]);
        if (versionIdx >= tableSize || CCU_OPS_TABLE[versionIdx] == nullptr) {
            HCCL_ERROR("[ResolveCcuOps] Invalid schema version[%u], table_size[%zu].", versionIdx, tableSize);
            ops = nullptr;
            return HCCL_E_INTERNAL;
        }

        ops = CCU_OPS_TABLE[versionIdx];
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult GetCcuOps(const CcuVersionOps*& ops)
{
    ops = nullptr;
    CcuSchemaVersion schemaVersion = CcuSchemaVersion::CCU_SCHEMA_COUNT;
    const HcclResult ret = GetCcuSchemaVersion(schemaVersion);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetCcuOps] GetCcuSchemaVersion failed, ret[%d].", ret);
        return ret;
    }

    const HcclResult resolveRet = ResolveCcuOps(schemaVersion, ops);
    if (resolveRet != HCCL_SUCCESS) {
        return resolveRet;
    }
    return HCCL_SUCCESS;
}
} // namespace hcomm
