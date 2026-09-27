/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARALLEL_TASK_LOADER_H
#define PARALLEL_TASK_LOADER_H

#include <vector>
#include <hccl/base.h>
#include "stream_pub.h"
#include "dispatcher.h"
#include "task_loader.h"

namespace hccl {
class ParallelTaskLoader {
public:
    explicit ParallelTaskLoader(const s32 deviceLogicId, const HcclDispatcher dispatcher);
    ~ParallelTaskLoader();

    HcclResult Prepare(std::vector<Stream*> streamsPtr, SubCommInfo level0CommInfo);

    HcclResult StartTaskLoad();
    HcclResult WaitTaskLoadFinish();
    HcclResult ClearTagCommInfo();

protected:
private:
    s32 deviceLogicId_;               // 当前设备的device id
    const HcclDispatcher dispatcher_; // dispatcher引用
    SubCommInfo commInfo_;

    std::vector<Stream*> streamsPtr_;
    std::vector<std::shared_ptr<TaskLoader>> streamTaskLoader_;
    std::vector<uint32_t> tidInfo_;
    u32 taskLoaderNum_; // taskLoader的个数和stream的个数一一对应
};
} // namespace hccl

#endif /* PARALLEL_TASK_LOADER_H */
