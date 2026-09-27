/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_RECURSIVE_HD_PUB_H
#define ALL_GATHER_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"

namespace hccl {
class AllGatherRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    explicit AllGatherRecursiveHalvingDoubling(const HcclDispatcher dispatcher);
    ~AllGatherRecursiveHalvingDoubling() override;

    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult CalculateSlices(u64 dataBytes, const u32 rankSize) const;

    HcclResult AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult GatherInPartOneToEven(u32 rank, const std::vector<LINK>& links);

    HcclResult GatherInPartOneToOdd(u32 rank, const std::vector<LINK>& links);
};
} // namespace hccl

#endif /* __BCAST_RECURSIVE_HALVINGDOUBLING_PUB_H__ */
