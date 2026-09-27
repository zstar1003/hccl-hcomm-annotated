/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENV_UB_CONFIG_H
#define ENV_UB_CONFIG_H

#include "config_mgr/base_config.h"

namespace hccl {

/**
 * @brief UB 相关环境变量配置（HCCL 语义的环境变量落在通信域管理层）。
 *
 * 通信域初始化路径（CollComm::Init）调用 Parse() 完成解析：非法配置当场报错（EI0001），
 * 合法值缓存在字段内；读取期通过 GetUbMultiChannelNum() 直接取缓存值，不再解析。
 * ResetParsed() 仅供 UT 重置解析状态。
 */
class EnvUbConfig {
public:
    HcclResult Parse();
    void ResetParsed();
    u32 GetUbMultiChannelNum() const { return ubMultiChannelNum_.Get(); }

private:
    // 默认值与范围
    static constexpr uint32_t UB_MULTI_CHANNEL_NUM_DEFAULT = 1; // UB多channel默认数量(1表示不使能)
    static constexpr uint32_t UB_MULTI_CHANNEL_NUM_MIN = 1;     // UB多channel数量最小值
    static constexpr uint32_t UB_MULTI_CHANNEL_NUM_MAX = 16;    // UB多channel数量最大值

    // 环境变量字段（解析器与校验器复用通用模板）
    hcomm::EnvField<uint32_t> ubMultiChannelNum_{
        "HCCL_UB_MULTI_CHANNEL_NUM", UB_MULTI_CHANNEL_NUM_DEFAULT, hcomm::StrToNum<uint32_t>,
        hcomm::MakeRangeValidator(UB_MULTI_CHANNEL_NUM_MIN, UB_MULTI_CHANNEL_NUM_MAX)};

    // 只缓存成功：首次（通信域初始化）解析成功后不再重复；失败不缓存，每次初始化尝试均报错
    std::mutex parseMutex_;
    bool parsed_{false};
    HcclResult parseRet_{HCCL_SUCCESS};
};

EnvUbConfig& GetEnvUbConfig();

} // namespace hccl

#endif // ENV_UB_CONFIG_H
