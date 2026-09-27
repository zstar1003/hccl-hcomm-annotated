/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_assist_v1.h"

#include "ccu_microcode_v1.h"

#include "hcomm_adapter_rts.h"

#include "exception_util.h" // todo: 需要统一整改为不抛异常
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    constexpr uint64_t SetBits(uint16_t start, uint16_t end)
    {
        return ((uint64_t(1) << (end - start + 1)) - uint64_t(1)) << start;
    }

    constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

    uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
    {
        constexpr uint16_t ctxIdBitNum = 8;
        constexpr uint16_t ctxIdShiftBit = 45;
        constexpr uint16_t gsaBitNum = 32;
        constexpr uint16_t gsaShiftBit = 13;
        constexpr uint16_t loopNumBitNum = 13;
        constexpr uint16_t loopNumShiftBit = 0;
        return ((loopCtxId & SetBits(ctxIdBitNum)) << ctxIdShiftBit) | ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
               | ((loopIterNum & SetBits(loopNumBitNum)) << loopNumShiftBit);
    }

    uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t repeatBitNum = 7;
        constexpr uint16_t repeatNumShiftBit = 55;
        constexpr uint16_t repeatLoopBitNum = 7;
        constexpr uint16_t repeatLoopShiftBit = 48;
        constexpr uint16_t totalLoopBitNum = 7;
        constexpr uint16_t totalLoopShiftBit = 41;
        return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
               | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
               | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
    }

    uint64_t GetParallelParamV2(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t loopNumBitNum = 10;
        constexpr uint16_t loopNumShiftBit = 0;
        constexpr uint16_t repeatLoopBitNum = 9;
        constexpr uint16_t repeatLoopShiftBit = 10;
        constexpr uint16_t extendBitNum = 9;
        constexpr uint16_t extendShiftBit = 19;
        return ((totalLoopNum & SetBits(loopNumBitNum)) << loopNumShiftBit)
               | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
               | ((repeatNum & SetBits(extendBitNum)) << extendShiftBit);
    }

    uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
    {
        constexpr uint16_t gsaBitNum = 32;
        constexpr uint16_t gsaShiftBit = 21;
        constexpr uint16_t msBitNum = 11;
        constexpr uint16_t msShiftBit = 10;
        constexpr uint16_t ckeBitNum = 10;
        constexpr uint16_t ckeShiftBit = 0;
        return ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit) | ((msOffset & SetBits(msBitNum)) << msShiftBit)
               | ((ckeOffset & SetBits(ckeBitNum)) << ckeShiftBit);
    }

    uint64_t GetToken(uint64_t tokenId, uint64_t tokenValue, uint64_t tokenValid)
    {
        constexpr uint16_t tokenValidBitNum = 1;
        constexpr uint16_t tokenValidShiftBit = 52;
        constexpr uint16_t tokenIdBitNum = 20;
        constexpr uint16_t tokenIdShiftBit = 32;
        constexpr uint16_t tokenValueBitNum = 32;
        constexpr uint16_t tokenValueShiftBit = 0;
        return ((tokenValid & SetBits(tokenValidBitNum)) << tokenValidShiftBit)
               | ((tokenId & SetBits(tokenIdBitNum)) << tokenIdShiftBit)
               | ((tokenValue & SetBits(tokenValueBitNum)) << tokenValueShiftBit);
    }

    uint64_t CcuCombineTokenInfo(uint64_t tokenId, uint64_t tokenValue, uint64_t tokenValid)
    {
        return GetToken(tokenId, tokenValue, tokenValid);
    }

    uint16_t GetCcuReduceType(Hccl::ReduceOp reduceOp)
    {
        static std::map<Hccl::ReduceOp, uint16_t> ccuReduceTypeMap = {
            {Hccl::ReduceOp::SUM, CCU_REDUCE_SUM},
            {Hccl::ReduceOp::MAX, CCU_REDUCE_MAX},
            {Hccl::ReduceOp::MIN, CCU_REDUCE_MIN},
        };

        if (ccuReduceTypeMap.find(reduceOp) == ccuReduceTypeMap.end()) {
            Hccl::THROW<Hccl::CcuApiException>("Unsupported ReduceOp[%s] for Ccu", reduceOp.Describe().c_str());
        }

        return ccuReduceTypeMap[reduceOp];
    }

    uint16_t GetCcuDataType(Hccl::DataType dataType, Hccl::ReduceOp reduceOp)
    {
        static std::map<Hccl::DataType, uint16_t> ccuSumDataTypeMap = {
            {Hccl::DataType::FP32, 0},  {Hccl::DataType::FP16, 1},    {Hccl::DataType::BFP16, 2},
            {Hccl::DataType::HIF8, 3},  {Hccl::DataType::FP8E4M3, 4}, {Hccl::DataType::FP8E5M2, 5},
            {Hccl::DataType::INT8, 6},  {Hccl::DataType::UINT8, 7},   {Hccl::DataType::INT16, 8},
            {Hccl::DataType::INT32, 9},
        };

        static std::map<Hccl::DataType, uint16_t> ccuMaxMinDataTypeMap = {
            {Hccl::DataType::FP32, 0},  {Hccl::DataType::FP16, 1},  {Hccl::DataType::BFP16, 2},
            {Hccl::DataType::INT8, 6},  {Hccl::DataType::UINT8, 7}, {Hccl::DataType::INT16, 8},
            {Hccl::DataType::INT32, 9},

        };

        uint16_t ccuReduceType = GetCcuReduceType(reduceOp);
        if (ccuReduceType == CCU_REDUCE_SUM) {
            if (ccuSumDataTypeMap.find(dataType) == ccuSumDataTypeMap.end()) {
                Hccl::THROW<Hccl::CcuApiException>(
                    "Unsupported Hccl::DataType[%s] for Ccu SUM", dataType.Describe().c_str());
            }
            return ccuSumDataTypeMap[dataType];
        }

        if (ccuReduceType == CCU_REDUCE_MAX || ccuReduceType == CCU_REDUCE_MIN) {
            if (ccuMaxMinDataTypeMap.find(dataType) == ccuMaxMinDataTypeMap.end()) {
                Hccl::THROW<Hccl::CcuApiException>(
                    "Unsupported Hccl::DataType[%s] for Ccu MAX/MIN", dataType.Describe().c_str());
            }
            return ccuMaxMinDataTypeMap[dataType];
        }

        return ccuSumDataTypeMap[dataType];
    }

    uint16_t GetUBReduceType(Hccl::ReduceOp reduceOp)
    {
        static std::map<Hccl::ReduceOp, uint16_t> ubReduceTypeMap = {
            {Hccl::ReduceOp::SUM, 10},
            {Hccl::ReduceOp::MAX, 8},
            {Hccl::ReduceOp::MIN, 9},
        };

        if (ubReduceTypeMap.find(reduceOp) == ubReduceTypeMap.end()) {
            Hccl::THROW<Hccl::CcuApiException>("Unsupported reduceOp[%s] for UB Reduce", reduceOp.Describe().c_str());
        }

        return ubReduceTypeMap[reduceOp];
    }

    uint16_t GetUBDataType(Hccl::DataType dataType)
    {
        static std::map<Hccl::DataType, uint16_t> ubDataTypeMap
            = {{Hccl::DataType::FP32, 7},  {Hccl::DataType::FP16, 6},   {Hccl::DataType::BFP16, 8},
               {Hccl::DataType::INT8, 0},  {Hccl::DataType::UINT8, 3},  {Hccl::DataType::INT16, 1},
               {Hccl::DataType::INT32, 2}, {Hccl::DataType::UINT16, 4}, {Hccl::DataType::UINT32, 5}};

        if (ubDataTypeMap.find(dataType) == ubDataTypeMap.end()) {
            Hccl::THROW<Hccl::CcuApiException>(
                "Unsupported Hccl::DataType[%s] for UB Reduce", dataType.Describe().c_str());
        }
        return ubDataTypeMap[dataType];
    }

    uint64_t GetTokenInfo(uint64_t va, uint64_t size)
    {
        rtMemUbTokenInfo info{};
        info.va = va;
        info.size = size;
        if (RtsUbDevQueryInfo(QUERY_PROCESS_TOKEN, info) != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("failed to query tokenInfo.");
        }
        return CcuRep::GetToken(info.tokenId, info.tokenValue, 1);
    }

}; // namespace CcuRep
}; // namespace hcomm
