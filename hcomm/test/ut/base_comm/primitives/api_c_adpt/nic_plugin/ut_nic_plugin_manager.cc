/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <stdlib.h>

#include "hcomm_nic_plugin.h"
#include "nic_plugin_manager.h"

namespace {

int32_t DummyCreateEndpoint(const EndpointDesc*, void**, HcommNicEndpointOps**) { return HCCL_SUCCESS; }

int32_t DummyCreateChannel(void*, const HcommChannelDesc*, void**, HcommNicChannelOps**) { return HCCL_SUCCESS; }

TEST(NicPluginManagerValidatePluginInfo, RejectsProtocolBetweenUbgAndCustomBase)
{
    // 覆盖 nic_plugin_manager.cc 中协议区间校验:
    // (protocol < COMM_PROTOCOL_HCCS || protocol > COMM_PROTOCOL_UBG) && protocol < COMM_PROTOCOL_CUSTOM_BASE
    HcommNicPluginInfo info{};
    info.header.version = HCOMM_NIC_PLUGIN_INFO_VERSION;
    info.header.magicWord = HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD;
    info.header.size = sizeof(HcommNicPluginInfo);
    info.protocolCount = 1U;
    info.protocols[0] = static_cast<CommProtocol>(COMM_PROTOCOL_UBG + 1); // 10, 在内置区间之外且低于 CUSTOM_BASE
    EXPECT_FALSE(hcomm::ValidatePluginInfo("ut_nic_plugin.so", &info, DummyCreateEndpoint, DummyCreateChannel));
}

// 覆盖 nic_plugin_manager.cc::LoadOnePlugin 中新增的路径规范化分支:
//   - 不存在路径 /nonexistent/... -> realpath 返回 nullptr -> 提前返回
//   - /tmp(目录必然存在, 非合法共享库) -> realpath 成功 -> dlopen(canonicalPath) 失败
// LoadOnePlugin 位于匿名命名空间, 只能经 LoadAllNicPlugins -> LoadExplicitPlugins 触达,
// 故通过环境变量 HCOMM_NIC_PLUGIN_SO 以冒号分隔注入多个路径, 一次调用同时覆盖两条分支。
// 不依赖 mkstemp/临时文件, 避免写权限或路径不可用导致的误失败。
TEST(NicPluginManagerLoadAllNicPlugins, CanonicalizesPluginPathBeforeDlopen)
{
    // 保存并清理环境, 确保走 LoadExplicitPlugins 分支
    const char* savedHome = getenv("ASCEND_HOME_PATH");
    const char* savedSo = getenv("HCOMM_NIC_PLUGIN_SO");
    unsetenv("ASCEND_HOME_PATH");
    ASSERT_EQ(setenv("HCOMM_NIC_PLUGIN_SO", "/nonexistent/ut_nic_plugin_missing.so:/tmp", 1), 0);

    // 经 std::call_once 执行, 同一进程内仅首次调用生效
    hcomm::LoadAllNicPlugins();
    hcomm::LoadAllNicPlugins(); // 再次调用确保 once 路径稳定不崩溃

    // 恢复环境
    (void)unsetenv("HCOMM_NIC_PLUGIN_SO");
    if (savedSo != nullptr) {
        (void)setenv("HCOMM_NIC_PLUGIN_SO", savedSo, 1);
    }
    if (savedHome != nullptr) {
        (void)setenv("ASCEND_HOME_PATH", savedHome, 1);
    }
}

} // namespace
