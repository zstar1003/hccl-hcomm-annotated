/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_assist.h"

#include "orion_adapter_rts.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_microcode.h"

namespace Hccl {
namespace CcuRep {

    constexpr uint64_t SetBits(uint16_t start, uint16_t end)
    {
        return ((uint64_t(1) << (end - start + 1)) - uint64_t(1)) << start;
    }

    constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

    // 辅助函数
    uint64_t GetMaxLoopIterNum()
    {
        constexpr uint16_t loopNumBitNum = 12;
        return SetBits(loopNumBitNum);
    }

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

    uint16_t ParseRepeatNumFromParallelParam(uint64_t parallelParam)
    {
        constexpr uint16_t repeatBitNum = 7;       // 7： repeat num 占 7 bits
        constexpr uint16_t repeatNumShiftBit = 55; // 55： repeat num占[61:55]位置
        return (parallelParam >> repeatNumShiftBit) & SetBits(repeatBitNum);
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

    uint64_t GetExpansionParam(uint64_t expansionNum)
    {
        constexpr uint64_t expansionNum1 = 1;
        constexpr uint64_t expansionNum2 = 2;
        constexpr uint64_t expansionNumShiftBit = 53;
        return (expansionNum == expansionNum2 ? expansionNum1 : expansionNum2)
               << expansionNumShiftBit; // Bit[53-54], 00: 1, 01: 2, 10: 4
    }

    uint16_t GetCcuReduceType(ReduceOp reduceOp)
    {
        static std::map<ReduceOp, uint16_t> ccuReduceTypeMap = {
            {ReduceOp::SUM, CCU_REDUCE_SUM},
            {ReduceOp::MAX, CCU_REDUCE_MAX},
            {ReduceOp::MIN, CCU_REDUCE_MIN},
        };

        if (ccuReduceTypeMap.find(reduceOp) == ccuReduceTypeMap.end()) {
            THROW<CcuApiException>("Unsupported ReduceOp[%s] for Ccu", reduceOp.Describe().c_str());
        }

        return ccuReduceTypeMap[reduceOp];
    }

    uint16_t GetCcuDataType(DataType dataType, ReduceOp reduceOp)
    {
        static std::map<DataType, uint16_t> ccuSumDataTypeMap = {
            {DataType::FP32, 0},    {DataType::FP16, 1},    {DataType::BFP16, 2}, {DataType::HIF8, 3},
            {DataType::FP8E4M3, 4}, {DataType::FP8E5M2, 5}, {DataType::INT8, 6},  {DataType::UINT8, 7},
            {DataType::INT16, 8},   {DataType::INT32, 9},
        };

        static std::map<DataType, uint16_t> ccuMaxMinDataTypeMap = {
            {DataType::FP32, 0},  {DataType::FP16, 1},  {DataType::BFP16, 2}, {DataType::INT8, 6},
            {DataType::UINT8, 7}, {DataType::INT16, 8}, {DataType::INT32, 9},

        };

        uint16_t ccuReduceType = GetCcuReduceType(reduceOp);
        if (ccuReduceType == CCU_REDUCE_SUM) {
            if (ccuSumDataTypeMap.find(dataType) == ccuSumDataTypeMap.end()) {
                THROW<CcuApiException>("Unsupported DataType[%s] for Ccu SUM", dataType.Describe().c_str());
            }
            return ccuSumDataTypeMap[dataType];
        }

        if (ccuReduceType == CCU_REDUCE_MAX || ccuReduceType == CCU_REDUCE_MIN) {
            if (ccuMaxMinDataTypeMap.find(dataType) == ccuMaxMinDataTypeMap.end()) {
                THROW<CcuApiException>("Unsupported DataType[%s] for Ccu MAX/MIN", dataType.Describe().c_str());
            }
            return ccuMaxMinDataTypeMap[dataType];
        }

        return ccuSumDataTypeMap[dataType];
    }

    uint16_t GetUBReduceType(ReduceOp reduceOp)
    {
        static std::map<ReduceOp, uint16_t> ubReduceTypeMap = {
            {ReduceOp::SUM, 10},
            {ReduceOp::MAX, 8},
            {ReduceOp::MIN, 9},
        };

        if (ubReduceTypeMap.find(reduceOp) == ubReduceTypeMap.end()) {
            THROW<CcuApiException>("Unsupported reduceOp[%s] for UB Reduce", reduceOp.Describe().c_str());
        }

        return ubReduceTypeMap[reduceOp];
    }

    uint16_t GetUBDataType(DataType dataType)
    {
        static std::map<DataType, uint16_t> ubDataTypeMap
            = {{DataType::FP32, 7},  {DataType::FP16, 6},   {DataType::BFP16, 8},
               {DataType::INT8, 0},  {DataType::UINT8, 3},  {DataType::INT16, 1},
               {DataType::INT32, 2}, {DataType::UINT16, 4}, {DataType::UINT32, 5}};

        if (ubDataTypeMap.find(dataType) == ubDataTypeMap.end()) {
            THROW<CcuApiException>("Unsupported DataType[%s] for UB Reduce", dataType.Describe().c_str());
        }
        return ubDataTypeMap[dataType];
    }

    uint32_t GetReduceExpansionNum(ReduceOp reduceOp, DataType dataType, DataType outputDataType)
    {
        uint32_t expansionNum = 1;

        if (reduceOp == ReduceOp::SUM && outputDataType == DataType::INVALID) {
            outputDataType = dataType;

            // 低精度数据格式可指定输出数据类型：fp32\bf16\fp16，如果没有指定，默认fp32
            if ((dataType == DataType::HIF8) || (dataType == DataType::FP8E4M3) || (dataType == DataType::FP8E5M2)
                || (dataType == DataType::INT8)) {
                outputDataType = DataType::FP32;
            }
        }
        expansionNum = DataTypeSizeGet(outputDataType) / DataTypeSizeGet(dataType);
        HCCL_INFO("Ccu low precision, expansionNum = %u", expansionNum);

        return expansionNum;
    }

    std::string GetReduceTypeStr(DataType dataType, ReduceOp opType)
    {
        static std::map<DataType, std::string> ccuRepDataTypeStr = {
            {DataType::FP32, "fp32"},   {DataType::FP16, "fp16"},       {DataType::BFP16, "bf16"},
            {DataType::HIF8, "hif8"},   {DataType::FP8E4M3, "fp8e4m3"}, {DataType::FP8E5M2, "fp8e5m2"},
            {DataType::INT8, "int8"},   {DataType::UINT8, "uint8"},     {DataType::INT16, "int16"},
            {DataType::INT32, "int32"},
        };

        static std::map<ReduceOp, std::string> ccuRepOpTypeStr = {
            {ReduceOp::SUM, "sum"},
            {ReduceOp::MAX, "max"},
            {ReduceOp::MIN, "min"},
        };

        return ccuRepDataTypeStr[dataType] + "_" + ccuRepOpTypeStr[opType];
    }

    uint64_t GetTokenInfo(uint64_t va, uint64_t size)
    {
        rtMemUbTokenInfo info;
        info.va = va;
        info.size = size;
        HrtUbDevQueryInfo(QUERY_PROCESS_TOKEN, &info);
        return CcuRep::GetToken(info.tokenId, info.tokenValue, 1);
    }

}; // namespace CcuRep
}; // namespace Hccl
