/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_info.h"
#include "platform/platform_info_def.h"
#include "platform/platform_infos_def.h"
#include "llt_hccl_stub.h"

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
namespace fe {
class PlatformInfoManager {
public:
    static PlatformInfoManager& Instance();
    uint32_t GetPlatformInfoWithOutSocVersion(PlatformInfo& platform_info, OptionalInfo& opti_compilation_info);
    uint32_t InitializePlatformInfo();
    uint32_t
    GetPlatformInfos(const std::string SoCVersion, PlatFormInfos& platform_info, OptionalInfos& opti_compilation_info);
}
} // namespace fe

namespace TbeReduce {
ge::graphStatus AutoTiling(gert::TilingContext* context);
ge::graphStatus AutoTilingParser(gert::TilingParseContext* context);
} // namespace TbeReduce
