/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_params_checker.h"
#include <string>
#include "hccl_params_pub.h"
#include "data_type.h"
#include "op_type.h"
#include "string_util.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

HcclResult
OpParamsChecker::CheckOpDataTypeOpbase(const CollOpParams& opParams, bool ccuEnable, bool isDevUsed, bool isAiv)
{
    HcclResult ret = HcclResult::HCCL_E_PARA;
    if (ccuEnable) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapCcuOpbase);
    } else if (isDevUsed) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapAicpuOpbase);
    } else if (isAiv) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapAivOpbase);
    } else {
        HCCL_ERROR("[OpParamsChecker::%s] Host opbase mode is invalid.", __func__);
    }
    return ret;
}

HcclResult
OpParamsChecker::CheckOpDataTypeOffload(const CollOpParams& opParams, bool ccuEnable, bool isDevUsed, bool isAiv)
{
    HcclResult ret = HcclResult::HCCL_E_PARA;
    if (ccuEnable) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapCcuOffload);
    } else if (isDevUsed) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapAicpuOffload);
    } else if (isAiv) {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapAivOffload);
    } else {
        ret = CheckOpDataTypeByMap(opParams, opDataTypeSupportMapHostOffload);
    }
    return ret;
}

static void ReportOpTypeErrMsg(const std::string& callName, OpType opType)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({callName, opType.Describe(), "opType", "please check opType that is not supported"}));
    HCCL_ERROR(
        "[%s][%s] unsupported opType [%s].", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
        opType.Describe().c_str());
}

static void ReportInputDataTypeMC2HighPErrMsg(const std::string& callName, OpType opType, DataType inputDataType)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {callName, "[" + opType.Describe() + "][" + inputDataType.Describe() + "]", "[opType][dataType]",
             "FP32,FP16,BF16,UINT8,INT16,INT32"}));
    HCCL_ERROR(
        "[%s][%s] opType [%s] not support data type [%s].", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opType.Describe().c_str(), inputDataType.Describe().c_str());
}

static void ReportInputDataTypeMC2LowPErrMsg(const std::string& callName, OpType opType, DataType inputDataType)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {callName, "[" + opType.Describe() + "][" + inputDataType.Describe() + "]", "[opType][inputDataType]",
             "Mc2LowP input:HIF8,E4M3,E5M2,INT8"}));
    HCCL_ERROR(
        "[%s][%s] Mc2LowP opType [%s] not support input data type [%s].", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opType.Describe().c_str(), inputDataType.Describe().c_str());
}

static void ReportOutputDataTypeMC2LowPErrMsg(const std::string& callName, OpType opType, DataType outputDataType)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {callName, "[" + opType.Describe() + "][" + outputDataType.Describe() + "]", "[opType][outputDataType]",
             "Mc2LowP output:FP32,FP16,BF16"}));
    HCCL_ERROR(
        "[%s][%s] Mc2LowP opType [%s] not support output data type [%s].", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opType.Describe().c_str(), outputDataType.Describe().c_str());
}

static void ReportDataTypeNotTheSameErrMsg(
    const std::string& callName, OpType opType, DataType inputDataType, DataType outputDataType)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {callName,
             "[" + opType.Describe() + "][" + inputDataType.Describe() + "and" + outputDataType.Describe() + "]",
             "[opType][inputDataType and outputDataType]", "should be same"}));
    HCCL_ERROR(
        "[%s][%s] opType [%s] inputDataType [%s] != outputDataType [%s].", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opType.Describe().c_str(), inputDataType.Describe().c_str(),
        outputDataType.Describe().c_str());
}

HcclResult OpParamsChecker::CheckOpDataTypeMC2(const Mc2CommConfig& config)
{
    OpType opType = MC2OpType(static_cast<AicpuComType>(config.opType));
    DataType inputDataType = MC2DataType(static_cast<HcclDataType>(config.dataType));
    DataType outputDataType = MC2DataType(static_cast<HcclDataType>(config.outputDataType));

    // 支持算子情况检验
    auto iter = opDataTypeSupportMapMC2.find(opType);
    if (iter == opDataTypeSupportMapMC2.end()) {
        ReportOpTypeErrMsg(__func__, opType);
        std::string msg
            = StringFormat("[OpParamsChecker::%s] unsupported opType [%s].", __func__, opType.Describe().c_str());
        THROW<InvalidParamsException>(msg);
    }

    /* CCU数据类型校验规则
     * Reduce算子：
     *      高精度模式，当inputDataType==outputDataType时，可选类型为FP32、FP16、BF16、INT16、INT32，暂不支持UINT8；
     *      低精度模式，当inputDataType!=outputDataType时，inputDataType可选范围HIF8、E4M3、E5M2、INT8；outputDataType可选范围FP32、FP16、BF16；
     * 非Reduce算子：任意数据类型，inputDataType==outputDataType即可。
     */
    bool checkResult = false;
    if (opType == OpType::REDUCESCATTER || opType == OpType::ALLREDUCE) {
        if (inputDataType == outputDataType) {
            checkResult = dataTypeMC2HighP.test(static_cast<int>(inputDataType));
            if (!checkResult) {
                ReportInputDataTypeMC2HighPErrMsg(__func__, opType, inputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] opType [%s] not support data type [%s].", __func__,
                    opType.Describe().c_str(), inputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
        } else {
            checkResult = inputDataTypeMC2LowP.test(static_cast<int>(inputDataType));
            if (!checkResult) {
                ReportInputDataTypeMC2LowPErrMsg(__func__, opType, inputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] Mc2LowP InputDataType[%s] != OutputDataType[%s] for OpType[%s], not support "
                    "input data type [%s].",
                    __func__, inputDataType.Describe().c_str(), outputDataType.Describe().c_str(),
                    opType.Describe().c_str(), inputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
            checkResult = OutputDataTypeMC2LowP.test(static_cast<int>(outputDataType));
            if (!checkResult) {
                ReportOutputDataTypeMC2LowPErrMsg(__func__, opType, outputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] Mc2LowP InputDataType[%s] != OutputDataType[%s] for OpType[%s], not support "
                    "output data type [%s].",
                    __func__, inputDataType.Describe().c_str(), outputDataType.Describe().c_str(),
                    opType.Describe().c_str(), outputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
        }
    } else {
        if (inputDataType != outputDataType) {
            ReportDataTypeNotTheSameErrMsg(__func__, opType, inputDataType, outputDataType);
            std::string msg = StringFormat(
                "[OpParamsChecker::%s] DataType[%s] != OutputDataType[%s] for OpType[%s].", __func__,
                inputDataType.Describe().c_str(), outputDataType.Describe().c_str(), opType.Describe().c_str());
            THROW<InvalidParamsException>(msg);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult OpParamsChecker::CheckOpDataTypeMC2V2(const Mc2CcTilingInner& config)
{
    OpType opType = MC2OpType(static_cast<AicpuComType>(config.opType));
    DataType inputDataType = MC2DataType(static_cast<HcclDataType>(config.srcDataType));
    DataType outputDataType = MC2DataType(static_cast<HcclDataType>(config.dstDataType));

    // 支持算子情况检验
    auto iter = opDataTypeSupportMapMC2.find(opType);
    if (iter == opDataTypeSupportMapMC2.end()) {
        ReportOpTypeErrMsg(__func__, opType);
        std::string msg
            = StringFormat("[OpParamsChecker::%s] unsupported opType [%s].", __func__, opType.Describe().c_str());
        THROW<InvalidParamsException>(msg);
    }

    /* CCU数据类型校验规则
     * Reduce算子：
     *      高精度模式，当dataType==outputDataType时，可选类型为FP32、FP16、BF16、UINT8、INT16、INT32；
     *      低精度模式，当dataType!=outputDataType时，dataType可选范围HIF8、E4M3、E5M2、INT8；outputDataType可选范围FP32、FP16、BF16；
     * 非Reduce算子：任意数据类型，dataType==outputDataType即可。
     */
    bool checkResult = false;
    if (opType == OpType::REDUCESCATTER || opType == OpType::ALLREDUCE) {
        if (inputDataType == outputDataType) {
            checkResult = dataTypeMC2HighP.test(static_cast<int>(inputDataType));
            if (!checkResult) {
                ReportInputDataTypeMC2HighPErrMsg(__func__, opType, inputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] opType [%s] not support data type [%s].", __func__,
                    opType.Describe().c_str(), inputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
        } else {
            checkResult = inputDataTypeMC2LowP.test(static_cast<int>(inputDataType));
            if (!checkResult) {
                ReportInputDataTypeMC2LowPErrMsg(__func__, opType, inputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] Mc2LowP InputDataType[%s] != OutputDataType[%s] for OpType[%s], not support "
                    "input data type [%s].",
                    __func__, inputDataType.Describe().c_str(), outputDataType.Describe().c_str(),
                    opType.Describe().c_str(), inputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
            checkResult = OutputDataTypeMC2LowP.test(static_cast<int>(outputDataType));
            if (!checkResult) {
                ReportOutputDataTypeMC2LowPErrMsg(__func__, opType, outputDataType);
                std::string msg = StringFormat(
                    "[OpParamsChecker::%s] Mc2LowP InputDataType[%s] != OutputDataType[%s] for OpType[%s], not support "
                    "output data type [%s].",
                    __func__, inputDataType.Describe().c_str(), outputDataType.Describe().c_str(),
                    opType.Describe().c_str(), outputDataType.Describe().c_str());
                THROW<InvalidParamsException>(msg);
            }
        }
    } else {
        if (inputDataType != outputDataType) {
            ReportDataTypeNotTheSameErrMsg(__func__, opType, inputDataType, outputDataType);
            std::string msg = StringFormat(
                "[OpParamsChecker::%s] DataType[%s] != OutputDataType[%s] for OpType[%s].", __func__,
                inputDataType.Describe().c_str(), outputDataType.Describe().c_str(), opType.Describe().c_str());
            THROW<InvalidParamsException>(msg);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

DataType OpParamsChecker::GetDataType(const CollOpParams& opParams)
{
    DataType dtype = opParams.dataType;
    if (opParams.opType == OpType::ALLTOALL) {
        dtype = opParams.all2AllDataDes.sendType;
    } else if (opParams.opType == OpType::ALLTOALLV) {
        dtype = opParams.all2AllVDataDes.sendType;
    } else if (opParams.opType == OpType::ALLTOALLVC) {
        dtype = opParams.all2AllVCDataDes.sendType;
    }
    return dtype;
}

static void ReportErrMsg(const CollOpParams& opParams, DataType dtype)
{
    RPT_INPUT_ERR(
        true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"CheckOpDataTypeByMap", "[" + opParams.opType.Describe() + "][" + dtype.Describe() + "]",
             "[opType][dataType]", "please check DataType that is not supported"}));
    HCCL_ERROR(
        "[%s][%s] opType [%s] with not support data type [%s], please check input "
        "opParam.",
        LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opParams.opType.Describe().c_str(),
        dtype.Describe().c_str());
}

HcclResult OpParamsChecker::CheckOpDataTypeByMap(const CollOpParams& opParams, const DataTypeSupportMap& opData2TypeMap)
{
    auto iter = opData2TypeMap.find(opParams.opType);
    if (iter == opData2TypeMap.end()) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"CheckOpDataTypeByMap", opParams.opType.Describe(), "opType",
                 "please check opType that is not supported"}));
        HCCL_ERROR(
            "[%s][%s] invalid opType [%s], please check input opParam.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), opParams.opType.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    bool checkResult = false;
    DataType dtype = GetDataType(opParams);

    if (opParams.opType == OpType::BATCHSENDRECV) {
        HcclSendRecvItem* sendRecvItems
            = static_cast<HcclSendRecvItem*>(opParams.batchSendRecvDataDes.sendRecvItemsPtr);
        u32 itemNum = opParams.batchSendRecvDataDes.itemNum;

        for (u32 i = 0; i < itemNum; ++i) {
            dtype = HcclDataTypeToDataType((sendRecvItems + i)->dataType);
            checkResult = (iter->second).test(static_cast<int>(dtype));
            if (!checkResult) {
                ReportErrMsg(opParams, dtype);
                return HcclResult::HCCL_E_PARA;
            }
        }
    } else {
        checkResult = (iter->second).test(static_cast<int>(dtype));
        if (!checkResult) {
            ReportErrMsg(opParams, dtype);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

DataTypeBitmap OpParamsChecker::dataTypeWithReduceAiv
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16));

DataTypeBitmap OpParamsChecker::dataTypeWithoutReduceAiv
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_HIF8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E4M3))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E5M2))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E8M0));

DataTypeBitmap OpParamsChecker::dataTypeWithReduceCcu
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16));
DataTypeBitmap OpParamsChecker::dataTypeWithReduceAicpu
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT64));
DataTypeBitmap OpParamsChecker::dataTypeWithoutReduce
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_HIF8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E4M3))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E5M2))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E8M0));
DataTypeBitmap OpParamsChecker::dataTypeWithoutReduceCcuOpbase
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_UINT64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP64))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_HIF8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E4M3))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E5M2))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E8M0));
DataTypeBitmap OpParamsChecker::dataTypeWithoutReduceCcuOffload = OpParamsChecker::dataTypeWithoutReduceCcuOpbase;
DataTypeBitmap OpParamsChecker::dataTypeWithReduceHost
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16));

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapAivOpbase
    = {{OpType::REDUCESCATTER, dataTypeWithReduceAiv},   {OpType::ALLREDUCE, dataTypeWithReduceAiv},
       {OpType::ALLGATHER, dataTypeWithoutReduceAiv},    {OpType::SCATTER, dataTypeWithoutReduceAiv},
       {OpType::ALLTOALL, dataTypeWithoutReduceAiv},     {OpType::ALLTOALLV, dataTypeWithoutReduceAiv},
       {OpType::REDUCE, dataTypeWithReduceAiv},          {OpType::BROADCAST, dataTypeWithoutReduceAiv},
       {OpType::SEND, dataTypeWithoutReduceAiv},         {OpType::RECV, dataTypeWithoutReduceAiv},
       {OpType::BATCHSENDRECV, dataTypeWithoutReduceAiv}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapAivOffload
    = {{OpType::REDUCESCATTER, dataTypeWithReduceAiv}, {OpType::ALLREDUCE, dataTypeWithReduceAiv},
       {OpType::ALLGATHER, dataTypeWithoutReduceAiv},  {OpType::SCATTER, dataTypeWithoutReduceAiv},
       {OpType::ALLTOALL, dataTypeWithoutReduceAiv},   {OpType::ALLTOALLV, dataTypeWithoutReduceAiv},
       {OpType::REDUCE, dataTypeWithReduceAiv},        {OpType::BROADCAST, dataTypeWithoutReduceAiv}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapCcuOpbase
    = {{OpType::REDUCESCATTER, dataTypeWithReduceCcu},
       {OpType::ALLREDUCE, dataTypeWithReduceCcu},
       {OpType::ALLGATHER, dataTypeWithoutReduceCcuOpbase},
       {OpType::SCATTER, dataTypeWithoutReduce},
       {OpType::ALLTOALL, dataTypeWithoutReduceCcuOpbase},
       {OpType::ALLTOALLV, dataTypeWithoutReduceCcuOpbase},
       {OpType::REDUCE, dataTypeWithReduceCcu},
       {OpType::BROADCAST, dataTypeWithoutReduce},
       {OpType::REDUCESCATTERV, dataTypeWithReduceCcu},
       {OpType::ALLGATHERV, dataTypeWithoutReduceCcuOpbase}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapCcuOffload
    = {{OpType::REDUCESCATTER, dataTypeWithReduceCcu},       {OpType::ALLREDUCE, dataTypeWithReduceCcu},
       {OpType::ALLGATHER, dataTypeWithoutReduceCcuOffload}, {OpType::ALLTOALL, dataTypeWithoutReduce},
       {OpType::ALLTOALLV, dataTypeWithoutReduce},           {OpType::REDUCE, dataTypeWithReduceCcu},
       {OpType::BROADCAST, dataTypeWithoutReduce},           {OpType::REDUCESCATTERV, dataTypeWithReduceCcu},
       {OpType::ALLGATHERV, dataTypeWithoutReduceCcuOffload}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapAicpuOpbase
    = {{OpType::REDUCESCATTER, dataTypeWithReduceAicpu},
       {OpType::ALLREDUCE, dataTypeWithReduceAicpu},
       {OpType::ALLGATHER, dataTypeWithoutReduce},
       {OpType::SCATTER, dataTypeWithoutReduce},
       {OpType::ALLTOALL, dataTypeWithoutReduce},
       {OpType::ALLTOALLV, dataTypeWithoutReduce},
       {OpType::ALLTOALLVC, dataTypeWithoutReduce},
       {OpType::SEND, dataTypeWithoutReduce},
       {OpType::RECV, dataTypeWithoutReduce},
       {OpType::REDUCE, dataTypeWithReduceAicpu},
       {OpType::BROADCAST, dataTypeWithoutReduce},
       {OpType::BATCHSENDRECV, dataTypeWithoutReduce},
       {OpType::BATCHGET, dataTypeWithoutReduce},
       {OpType::BATCHPUT, dataTypeWithoutReduce}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapAicpuOffload
    = {{OpType::ALLGATHER, dataTypeWithoutReduce},   {OpType::REDUCESCATTER, dataTypeWithReduceAicpu},
       {OpType::ALLREDUCE, dataTypeWithReduceAicpu}, {OpType::ALLTOALL, dataTypeWithoutReduce},
       {OpType::ALLTOALLV, dataTypeWithoutReduce},   {OpType::ALLTOALLVC, dataTypeWithoutReduce},
       {OpType::REDUCE, dataTypeWithReduceAicpu},    {OpType::BROADCAST, dataTypeWithoutReduce},
       {OpType::SEND, dataTypeWithoutReduce},        {OpType::RECV, dataTypeWithoutReduce}};

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapHostOffload = {
    {OpType::ALLGATHER, dataTypeWithoutReduce},  {OpType::REDUCESCATTER, dataTypeWithReduceHost},
    {OpType::ALLREDUCE, dataTypeWithReduceHost}, {OpType::ALLTOALL, dataTypeWithoutReduce},
    {OpType::ALLTOALLV, dataTypeWithoutReduce},  {OpType::ALLTOALLVC, dataTypeWithoutReduce},
    {OpType::BROADCAST, dataTypeWithoutReduce},  {OpType::SEND, dataTypeWithoutReduce},
    {OpType::RECV, dataTypeWithoutReduce},
};

DataTypeBitmap OpParamsChecker::dataTypeMC2HighP = dataTypeWithReduceCcu;
DataTypeBitmap OpParamsChecker::inputDataTypeMC2LowP
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT8))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E5M2))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP8E4M3))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_HIF8));
DataTypeBitmap OpParamsChecker::OutputDataTypeMC2LowP
    = DataTypeBitmap{} | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP16))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_FP32))
      | DataTypeBitmap(1 << static_cast<int>(HcclDataType::HCCL_DATA_TYPE_BFP16));

DataTypeSupportMap OpParamsChecker::opDataTypeSupportMapMC2
    = {{OpType::ALLGATHER, dataTypeWithoutReduce}, {OpType::REDUCESCATTER, dataTypeMC2HighP},
       {OpType::ALLREDUCE, dataTypeMC2HighP},      {OpType::ALLTOALL, dataTypeWithoutReduce},
       {OpType::ALLTOALLV, dataTypeWithoutReduce}, {OpType::HALFALLTOALLV, dataTypeWithoutReduce}};

} // namespace Hccl
