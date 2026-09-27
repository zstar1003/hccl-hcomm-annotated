/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_DATA_TYPE_H
#define MC2_DATA_TYPE_H
#include "hccl_types.h"

enum class HcclTaskStatus { HCCL_NORMAL_STATUS, HCCL_CQE_ERROR, HCCL_END_STATUS };

struct HcclOpData {
    HcclOpData() { dataDes = {0, HcclDataType::HCCL_DATA_TYPE_RESERVED, 0}; }
    HcclCMDType opType = HcclCMDType::HCCL_CMD_MAX;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_RESERVED;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclDataType outputDataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t dataCount{0};
    uint32_t root{0};
    uint32_t sendRecvRemoteRank{0};
    uint64_t input{0};
    uint64_t output{0};
    uint64_t rev[10] = {0};

    union {
        struct {
            uint64_t dataCount;
            HcclDataType dataType;
            uint64_t strideCount;
        } dataDes;
        struct {
            void* counts;
            void* displs;
            HcclDataType dataType;
        } vDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            uint64_t sendCount;
            uint64_t recvCount;
        } all2AllDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
        } all2AllVDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            void* sendCountMatrix;
        } all2AllVCDataDes;
        struct {
            void* sendRecvItemsPtr;
            uint32_t itemNum;
        } batchSendRecvDataDes;
    };
};

#endif
