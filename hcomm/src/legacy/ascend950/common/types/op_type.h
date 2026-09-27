/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_OP_TYPE_H
#define HCCLV2_OP_TYPE_H

#include <algorithm>
#include <map>
#include <unordered_map>
#include "../utils/enum_factory.h"
#include "log.h"

namespace Hccl {

MAKE_ENUM(
    OpType, ALLREDUCE, BROADCAST, ALLGATHER, REDUCESCATTER, SEND, RECV, BARRIER, ALLTOALL, REDUCE, GATHER, SCATTER,
    ALLTOALLV, ALLTOALLVC, HALFALLTOALLV, BATCHSENDRECV, BATCHGET, BATCHPUT, ALLGATHERV, REDUCESCATTERV, DEBUGCASE,
    HCCLGROUPOP, OPTYPEINVALID)

inline std::string OpTypeToString(OpType type) { return type.Describe(); }

const std::map<std::string, OpType> HCOM_OP_TYPE_STR_MAP_V2{
    {"ALLREDUCE", OpType::ALLREDUCE},
    {"BROADCAST", OpType::BROADCAST},
    {"ALLGATHER", OpType::ALLGATHER},
    {"REDUCESCATTER", OpType::REDUCESCATTER},
    {"SEND", OpType::SEND},
    {"RECV", OpType::RECV},
    {"BARRIER", OpType::BARRIER},
    {"ALLTOALL", OpType::ALLTOALL},
    {"REDUCE", OpType::REDUCE},
    {"GATHER", OpType::GATHER},
    {"SCATTER", OpType::SCATTER},
    {"ALLTOALLV", OpType::ALLTOALLV},
    {"ALLTOALLVC", OpType::ALLTOALLVC},
    {"HALFALLTOALLV", OpType::HALFALLTOALLV},
    {"BATCHSENDRECV", OpType::BATCHSENDRECV},
    {"BATCHGET", OpType::BATCHGET},
    {"BATCHPUT", OpType::BATCHPUT},
    {"ALLGATHERV", OpType::ALLGATHERV},
    {"REDUCESCATTERV", OpType::REDUCESCATTERV},
    {"DEBUGCASE", OpType::DEBUGCASE}};

const std::map<HcclCMDType, Hccl::OpType> OP_TYPE_MAP = {
    {HcclCMDType::HCCL_CMD_ALLREDUCE, Hccl::OpType::ALLREDUCE},
    {HcclCMDType::HCCL_CMD_ALLGATHER, Hccl::OpType::ALLGATHER},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, Hccl::OpType::REDUCESCATTER},
    {HcclCMDType::HCCL_CMD_SEND, Hccl::OpType::SEND},
    {HcclCMDType::HCCL_CMD_RECEIVE, Hccl::OpType::RECV},
    {HcclCMDType::HCCL_CMD_ALLTOALL, Hccl::OpType::ALLTOALL},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, Hccl::OpType::ALLTOALLV},
    {HcclCMDType::HCCL_CMD_BROADCAST, Hccl::OpType::BROADCAST},
    {HcclCMDType::HCCL_CMD_ALLGATHER_V, Hccl::OpType::ALLGATHERV},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, Hccl::OpType::REDUCESCATTERV},
    {HcclCMDType::HCCL_CMD_REDUCE, Hccl::OpType::REDUCE},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, Hccl::OpType::ALLTOALLVC},
    {HcclCMDType::HCCL_CMD_SCATTER, Hccl::OpType::SCATTER},
    {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, Hccl::OpType::BATCHSENDRECV},
    {HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, Hccl::OpType::HALFALLTOALLV},
    {HcclCMDType::HCCL_CMD_BARRIER, Hccl::OpType::BARRIER},
    {HcclCMDType::HCCL_CMD_GATHER, Hccl::OpType::GATHER},
    {HcclCMDType::HCCL_CMD_BATCH_GET, Hccl::OpType::BATCHGET},
    {HcclCMDType::HCCL_CMD_BATCH_PUT, Hccl::OpType::BATCHPUT},
};

const std::unordered_map<std::string, Hccl::OpType> OP_TYPE_STR = {
    {"HcomAllReduce", Hccl::OpType::ALLREDUCE},
    {"HcomAllGather", Hccl::OpType::ALLGATHER},
    {"HcomReduceScatter", Hccl::OpType::REDUCESCATTER},
    {"HcomSend", Hccl::OpType::SEND},
    {"HcomReceive", Hccl::OpType::RECV},
    {"HcomAllToAll", Hccl::OpType::ALLTOALL},
    {"HcomAllToAllV", Hccl::OpType::ALLTOALLV},
    {"HcomBroadcast", Hccl::OpType::BROADCAST},
    {"HcomAllGatherV", Hccl::OpType::ALLGATHERV},
    {"HcomReduceScatterV", Hccl::OpType::REDUCESCATTERV},
    {"HcomReduce", Hccl::OpType::REDUCE},
    {"HcomAllToAllVC", Hccl::OpType::ALLTOALLVC},
};
} // namespace Hccl
#endif
