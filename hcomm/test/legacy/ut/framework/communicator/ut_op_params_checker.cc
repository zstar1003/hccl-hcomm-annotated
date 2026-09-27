/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include <hccl/hccl_types.h>
#include <vector>
#include <memory>
#include "op_params_checker.h"
#include "op_type.h"
#include "data_type.h"
#include "invalid_params_exception.h"
#include "mc2_type.h"
#include "hccl_params_pub.h"

#undef private
#undef protected

using namespace Hccl;

class OpParamsCheckerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OpParamsCheckerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "OpParamsCheckerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in OpParamsCheckerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in OpParamsCheckerTest TearDown" << std::endl;
    }
};

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_ccu_opbased)
{
    CollOpParams opParams;
    bool ccuEnable = true;
    bool isDevUsed = true;
    bool isAiv = true;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_ccu_offload)
{
    CollOpParams opParams;
    bool ccuEnable = true;
    bool isDevUsed = false;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_aicpu_opbased)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce
        = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST, OpType::SEND, OpType::RECV};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_aicpu_opbased_batchsendrecv)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<HcclDataType> datatypeWithoutReduce = {
        HcclDataType::HCCL_DATA_TYPE_INT8,   HcclDataType::HCCL_DATA_TYPE_INT16,   HcclDataType::HCCL_DATA_TYPE_INT32,
        HcclDataType::HCCL_DATA_TYPE_INT64,  HcclDataType::HCCL_DATA_TYPE_UINT8,   HcclDataType::HCCL_DATA_TYPE_UINT16,
        HcclDataType::HCCL_DATA_TYPE_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT64,  HcclDataType::HCCL_DATA_TYPE_FP16,
        HcclDataType::HCCL_DATA_TYPE_FP32,   HcclDataType::HCCL_DATA_TYPE_FP64,    HcclDataType::HCCL_DATA_TYPE_BFP16,
        HcclDataType::HCCL_DATA_TYPE_HIF8,   HcclDataType::HCCL_DATA_TYPE_FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E5M2,
        HcclDataType::HCCL_DATA_TYPE_FP8E8M0};

    opParams.opType = OpType::BATCHSENDRECV;
    HcclSendRecvItem* sendRecvItemdata = nullptr;
    sendRecvItemdata = new HcclSendRecvItem[1];
    opParams.batchSendRecvDataDes.itemNum = 1;
    for (auto dtype : datatypeWithoutReduce) {
        sendRecvItemdata->dataType = dtype;
        opParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(sendRecvItemdata);
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
    delete[] sendRecvItemdata;
}

TEST_F(OpParamsCheckerTest, should_return_error_when_check_unsupported_datatype_aicpu_opbased_batchsendrecv)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<HcclDataType> datatypeWithoutReduce = {HcclDataType::HCCL_DATA_TYPE_INT128};

    opParams.opType = OpType::BATCHSENDRECV;
    HcclSendRecvItem* sendRecvItemdata = nullptr;
    sendRecvItemdata = new HcclSendRecvItem[1];
    opParams.batchSendRecvDataDes.itemNum = 1;
    for (auto dtype : datatypeWithoutReduce) {
        sendRecvItemdata->dataType = dtype;
        opParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(sendRecvItemdata);
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
    }
    delete[] sendRecvItemdata;
}

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_aicpu_offload)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(OpParamsCheckerTest, should_return_success_when_check_datatype_host_offload)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = false;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::ALLREDUCE, OpType::REDUCESCATTER};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::BROADCAST, OpType::SEND, OpType::RECV};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT8, DataType::INT16, DataType::INT32, DataType::FP16, DataType::FP32, DataType::BFP16};
    std::vector<DataType> datatypeWithoutReduce
        = {DataType::INT8,   DataType::INT16,   DataType::INT32,   DataType::INT64,  DataType::UINT8, DataType::UINT16,
           DataType::UINT32, DataType::UINT64,  DataType::FP16,    DataType::FP32,   DataType::FP64,  DataType::BFP16,
           DataType::HIF8,   DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv),
                HcclResult::HCCL_SUCCESS);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(OpParamsCheckerTest, Ut_CheckOpDataTypeOpbase_When_datatype_not_support_Expect_fail)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = false;
    bool isAiv = false;
    opParams.opType = OpType::DEBUGCASE;
    EXPECT_EQ(OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
}

TEST_F(OpParamsCheckerTest, Ut_CheckOpDataTypeOffload_When_host_offload_Expect_fail)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = false;
    bool isAiv = false;

    EXPECT_EQ(OpParamsChecker::CheckOpDataTypeOffload(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
}

TEST_F(OpParamsCheckerTest, should_return_error_when_check_unsupported_datatype_ccu_opbased)
{
    CollOpParams opParams;
    bool ccuEnable = true;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce = {OpType::ALLGATHER, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::INT64, DataType::UINT64,   DataType::UINT16,  DataType::UINT32,  DataType::FP64, DataType::INT128,
           DataType::HIF8,  DataType::BF16_SAT, DataType::FP8E4M3, DataType::FP8E5M2, DataType::UINT8};
    std::vector<DataType> datatypeWithoutReduce = {DataType::BF16_SAT};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
    }
}

TEST_F(OpParamsCheckerTest, should_return_error_when_check_unsupported_datatype_aicpu_opbased)
{
    CollOpParams opParams;
    bool ccuEnable = false;
    bool isDevUsed = true;
    bool isAiv = false;

    std::vector<OpType> optypeWithReduce = {OpType::REDUCESCATTER, OpType::ALLREDUCE, OpType::REDUCE};
    std::vector<OpType> optypeWithoutReduce
        = {OpType::ALLGATHER, OpType::SEND, OpType::RECV, OpType::SCATTER, OpType::BROADCAST};
    std::vector<DataType> datatypeWithReduce
        = {DataType::UINT8, DataType::UINT16,   DataType::UINT32,  DataType::INT128,
           DataType::HIF8,  DataType::BF16_SAT, DataType::FP8E4M3, DataType::FP8E5M2};
    std::vector<DataType> datatypeWithoutReduce = {DataType::INT128, DataType::BF16_SAT};

    for (auto optype : optypeWithReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        opParams.opType = optype;
        for (auto dtype : datatypeWithoutReduce) {
            opParams.dataType = dtype;
            EXPECT_EQ(
                OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
        }
    }

    opParams.opType = OpType::ALLTOALL;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
    }
    opParams.opType = OpType::ALLTOALLV;
    for (auto dtype : datatypeWithoutReduce) {
        opParams.all2AllVDataDes.sendType = dtype;
        EXPECT_EQ(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, ccuEnable, isDevUsed, isAiv), HcclResult::HCCL_E_PARA);
    }
}

TEST_F(OpParamsCheckerTest, should_suc_when_check_datatype_mc2_highP)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};
    std::vector<uint32_t> dataTypeHighP
        = {static_cast<uint32_t>(DataType::INT16), static_cast<uint32_t>(DataType::INT32),
           static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.dataType = dtype;
            config.outputDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.dataType = dtype;
            config.outputDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(OpParamsCheckerTest, should_suc_when_check_datatype_mc2_highP_V2)
{
    Mc2CcTilingInner config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};
    std::vector<uint32_t> dataTypeHighP
        = {static_cast<uint32_t>(DataType::INT16), static_cast<uint32_t>(DataType::INT32),
           static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.srcDataType = dtype;
            config.dstDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.srcDataType = dtype;
            config.dstDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(OpParamsCheckerTest, should_suc_when_check_datatype_mc2_lowP)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> inputDataType
        = {static_cast<uint32_t>(DataType::INT8), static_cast<uint32_t>(DataType::FP8E5M2),
           static_cast<uint32_t>(DataType::FP8E4M3), static_cast<uint32_t>(DataType::HIF8)};
    std::vector<uint32_t> outputDataType
        = {static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : inputDataType) {
            config.dataType = dtype;
            for (auto outDtype : outputDataType) {
                config.outputDataType = outDtype;
                EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2(config), HcclResult::HCCL_SUCCESS);
            }
        }
    }
}

TEST_F(OpParamsCheckerTest, should_suc_when_check_datatype_mc2_lowP_V2)
{
    Mc2CcTilingInner config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> inputDataType
        = {static_cast<uint32_t>(DataType::INT8), static_cast<uint32_t>(DataType::FP8E5M2),
           static_cast<uint32_t>(DataType::FP8E4M3), static_cast<uint32_t>(DataType::HIF8)};
    std::vector<uint32_t> outputDataType
        = {static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : inputDataType) {
            config.srcDataType = dtype;
            for (auto outDtype : outputDataType) {
                config.dstDataType = outDtype;
                EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
            }
        }
    }
}

TEST_F(OpParamsCheckerTest, should_fail_when_check_unsupported_datatype_mc2_lowP)
{
    Mc2CommConfig config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.dataType = static_cast<uint32_t>(DataType::INT16);
    config.outputDataType = static_cast<uint32_t>(DataType::FP16);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);

    config.dataType = static_cast<uint32_t>(DataType::INT8);
    config.outputDataType = static_cast<uint32_t>(DataType::INT32);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
}

TEST_F(OpParamsCheckerTest, should_fail_when_check_unsupported_datatype_mc2_lowP_V2)
{
    Mc2CcTilingInner config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.srcDataType = static_cast<uint32_t>(DataType::INT16);
    config.dstDataType = static_cast<uint32_t>(DataType::FP16);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);

    config.srcDataType = static_cast<uint32_t>(DataType::INT8);
    config.dstDataType = static_cast<uint32_t>(DataType::INT32);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);
}

TEST_F(OpParamsCheckerTest, should_fail_when_check_unsupported_datatype_mc2_highP)
{
    Mc2CommConfig config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.dataType = static_cast<uint32_t>(DataType::INT64);
    config.outputDataType = static_cast<uint32_t>(DataType::INT64);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
}

TEST_F(OpParamsCheckerTest, should_fail_when_check_unsupported_datatype_mc2_highP_V2)
{
    Mc2CcTilingInner config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.srcDataType = static_cast<uint32_t>(DataType::INT64);
    config.dstDataType = static_cast<uint32_t>(DataType::INT64);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);
}

TEST_F(OpParamsCheckerTest, should_fail_when_check_unsupported_datatype_mc2_optype_without_reduce)
{
    Mc2CommConfig config;

    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};

    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        config.dataType = static_cast<uint32_t>(DataType::INT8);
        config.outputDataType = static_cast<uint32_t>(DataType::INT16);
        EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2(config), InvalidParamsException);
    }
}

TEST_F(OpParamsCheckerTest, DescCollOpParamsTest)
{
    CollOpParams opParams;
    EXPECT_NO_THROW(opParams.DescReduceScatter(opParams));
    EXPECT_NO_THROW(opParams.DescReduce(opParams));
    EXPECT_NO_THROW(opParams.DescAllgather(opParams));
    EXPECT_NO_THROW(opParams.DescScatter(opParams));
    EXPECT_NO_THROW(opParams.DescSend(opParams));
    EXPECT_NO_THROW(opParams.DescRecv(opParams));
    EXPECT_NO_THROW(opParams.DescBroadcast(opParams));
}
