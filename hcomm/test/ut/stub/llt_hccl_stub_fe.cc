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

PlatformInfoManager::PlatformInfoManager() : init_flag_(false), opti_compilation_info_() {}

PlatformInfoManager::~PlatformInfoManager() {}

PlatformInfoManager& PlatformInfoManager::Instance()
{
    static PlatformInfoManager PlatformInfo_Manager;
    return PlatformInfo_Manager;
}
uint32_t
PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatformInfo& platform_info, OptionalInfo& opti_compilation_info)
{
    return 0;
}

uint32_t PlatformInfoManager::InitializePlatformInfo() { return 0; }
uint32_t PlatformInfoManager::GetPlatformInfos(
    const std::string SoCVersion, PlatFormInfos& platform_info, OptionalInfos& opti_compilation_info)
{
    return 0;
}
uint32_t PlatFormInfos::GetCoreNum() const { return 2; }

void PlatFormInfos::SetCoreNum(const uint32_t& core_num) { core_num_ = core_num; }
} // namespace fe
