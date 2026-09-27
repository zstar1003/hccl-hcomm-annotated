/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCER_PUB_H
#define REDUCER_PUB_H

#include <hccl/hccl_types.h>

#include <functional>
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "transport_pub.h"

namespace hccl {

enum class DstMemType { RESULT_INPUT_MEM, RESULT_OUTPUT_MEM, RESULT_MEM_RESERVED_NEW };

struct ReducerMemoryInfo {
    u64 remoteMemOffset;
    DeviceMem localsrc;
    DeviceMem localdst;
    DeviceMem remoteRcvTemp;
};

class Reducer {
public:
    explicit Reducer(
        const HcclDataType dataType = HCCL_DATA_TYPE_RESERVED, const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
        const u64 reduceAttribute = 0x0);
    ~Reducer();

    /**********************************************************************
     功能描述  : reduce操作执行
     输入参数  : const HcclDispatcher dispatcher：
               const std::unique_ptr<Transport>& link：
               s32 remote_mem_offset  对端内存偏移
               DeviceMem& local_src 本端源数据存储空间
               DeviceMem& local_dst 本端目的数据存储空间(可与本端源数据空间相同)
               DeviceMem& remote_rcv_temp 本端用于接收对端数据的存储空间
               stream: 任务流
     输出参数  : 无
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult
    run(const HcclDispatcher dispatcher, const std::shared_ptr<Transport>& link, const u64 remoteMemOffset,
        DeviceMem& localSrc, DeviceMem& localDst, DeviceMem& remoteRcvTemp, Stream& stream,
        DstMemType resultMem = DstMemType::RESULT_INPUT_MEM,
        const UserMemType srcMemType = UserMemType::INPUT_MEM) const;

    HcclResult
    run(const HcclDispatcher dispatcher, const std::shared_ptr<Transport>& link,
        const std::vector<ReducerMemoryInfo>& reducerMems, Stream& stream,
        DstMemType resultMem = DstMemType::RESULT_INPUT_MEM) const;

    HcclResult
    run(const HcclDispatcher dispatcher, const std::shared_ptr<Transport>& link,
        const std::vector<ReducerMemoryInfo>& reducerMems, u32 notifyIdx, Stream& stream,
        DstMemType resultMem = DstMemType::RESULT_INPUT_MEM) const;

    void SetPreSyncFunc(std::function<HcclResult()> lambda);
    void SetPostSyncFunc(std::function<HcclResult()> lambda);

protected:
    std::function<HcclResult()> preSync_;
    std::function<HcclResult()> postSync_;

private:
    HcclResult
    PrepareRxMems(const std::vector<ReducerMemoryInfo>& reducerMems, std::vector<RxMemoryInfo>& rxMems) const;
    HcclResult PrepareRxWithReduceMems(
        const std::vector<ReducerMemoryInfo>& reducerMems, std::vector<RxWithReduceMemoryInfo>& rxWithReduceMems) const;

    HcclDataType dataType_;
    HcclReduceOp reductionOp_;
    const u64 reduceAttribute_;
};
} // namespace hccl

#endif /* REDUCER_PUB_H */
