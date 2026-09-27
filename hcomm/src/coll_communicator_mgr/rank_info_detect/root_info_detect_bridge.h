/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ROOT_INFO_DETECT_BRIDGE_H
#define HCOMM_ROOT_INFO_DETECT_BRIDGE_H

#include <memory>

#include "hccl/hccl_types.h"
#include "rank_table_info.h"
#include "root_handle_v2.h"

namespace Hccl {

// 链接方向为 hcomm -> hccl_v2：Register/Get 与槽位符号归属 hccl_v2；运行时回调方向为
// hccl_v2 -> hcomm：兼容入口通过该内部函数表调用 RootInfoDetect provider。不得将槽位移入 hcomm 而引入
// hccl_v2 -> hcomm 的直接链接；待后续legacy功能全部迁移，可以考虑去掉bridge和provider
struct RootInfoDetectBridge {
    // 调用方持有该类型擦除的所有权句柄，保证探测对象在通信域初始化结束前不被析构。
    using DetectContext = std::shared_ptr<void>;
    using GetRootInfoFunc = HcclResult (*)(HcclRootInfo* rootInfo);
    using DetectRankTableFunc = HcclResult (*)(
        u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
        DetectContext& detectContext);

    GetRootInfoFunc getRootInfo{nullptr};
    DetectRankTableFunc detectRankTable{nullptr};
};

// hcomm 加载时注册一次完整回调表；注册完成前查询返回 nullptr。
HcclResult RegisterRootInfoDetectBridge(const RootInfoDetectBridge& bridge);
const RootInfoDetectBridge* GetRootInfoDetectBridge();

} // namespace Hccl

#endif // HCOMM_ROOT_INFO_DETECT_BRIDGE_H
