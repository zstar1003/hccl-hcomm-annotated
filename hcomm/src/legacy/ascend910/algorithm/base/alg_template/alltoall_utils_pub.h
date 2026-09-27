/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_INFO_PUB_H
#define ALLTOALL_V_INFO_PUB_H

#include <list>
#include "hccl_types.h"
#include "mem_device_pub.h"

namespace hccl {

struct RemoteMem {
    DeviceMem remoteScratchPingMem;
    DeviceMem remoteScratchPongMem;
};

struct SendDataBlock {
    u64 sendLen;
    u64 userInOffset;
    u64 scratchOffset;
};

struct ReadDataBlock {
    u64 recvLen;
    u64 remoteOffset;
    u64 recvOffset;
};

struct RecvDataBlock {
    u64 recvOffset;
    u64 recvLen;
    u64 scratchOffset;
};

struct AlltoallSendRecvInfo {
    std::vector<SendDataBlock> sendInfo;
    std::vector<ReadDataBlock> readInfo;
};

// alltoallv_mesh_read_only_pub.h
struct DataTrace {
    u32 dataIndex;
    u64 dataOffset;
};

// alltoallv_for_310p_pub.h
const uint32_t COMPUTE_CONST = 2; // 计算Rank类型用到的常量
const uint32_t STEP_NUM = 5;
const uint32_t THIRD_STEP = 3;
const uint32_t MAX_RANK_GAP = 3;
const uint32_t DUO_RANK_NUM = 4;
const uint32_t ALIGN_CONST = 128;

// alltoallv_direct_fullmesh_pub.h
const uint32_t ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE = 8;   // SDMA链路上的并发数量
const uint32_t ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE = 1;   // RDMA链路上的并发数量
const uint32_t RANK_SET_COMPUTE_CONST = 2;                           // 计算对端Rank用到的常量
const uint32_t ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE = 1 * 1024 * 1024; // 大数据量走并发拷贝的标准

struct AlltoAllVBufferInfo {
    DeviceMem mem;
    u64* counts = nullptr;
    u64* displs = nullptr;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;

    AlltoAllVBufferInfo& operator=(const AlltoAllVBufferInfo& that) noexcept
    {
        if (&that != this) {
            mem = that.mem;
            counts = that.counts;
            displs = that.displs;
            dataType = that.dataType;
        }
        return *this;
    }

    AlltoAllVBufferInfo& operator=(const AlltoAllVBufferInfo&& that) noexcept
    {
        if (&that != this) {
            mem = that.mem;
            counts = that.counts;
            displs = that.displs;
            dataType = that.dataType;
        }
        return *this;
    }
};

struct OneSendRecvAddrInfo {
    u64 localOffset;
    u64 localLength;
    u64 remoteOffset;
    u64 remoteLength;
};

using StageAlltoAllVAddrInfo = std::map<u32, std::list<OneSendRecvAddrInfo>>; // key: remote rank in local communicator

class A2aPipelineMemory {
public:
    DeviceMem userInput;
    DeviceMem userOutput;
    DeviceMem scratchMem;   // 图模式使用
    DeviceMem cclInBuffer;  // 单算子模式使用
    DeviceMem cclOutBuffer; // 单算子模式使用
};

} // namespace hccl
#endif
