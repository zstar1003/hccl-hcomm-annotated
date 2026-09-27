/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "hcomm_nic_plugin.h"

namespace {

std::string ReadRepoFile(const std::string& relativePath)
{
    const std::string fullPath = std::string(HCOMM_CODE_ROOT) + "/" + relativePath;
    std::ifstream file(fullPath);
    EXPECT_TRUE(file.is_open()) << fullPath;
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void ExpectContains(const std::string& content, const std::string& expected)
{
    EXPECT_NE(content.find(expected), std::string::npos) << expected;
}

void ExpectNotContains(const std::string& content, const std::string& unexpected)
{
    EXPECT_EQ(content.find(unexpected), std::string::npos) << unexpected;
}

std::string RetiredHostName(const std::string& suffix) { return std::string("hcomm_") + "host_" + suffix; }

std::string RetiredHostSharedObject(const std::string& suffix)
{
    return std::string("lib") + RetiredHostName(suffix) + "_plugin.so";
}

TEST(ExperimentalNicPluginContract, ExportedAbiSignaturesRemainStable)
{
    static_assert(
        std::is_same<decltype(&HcommNicPluginGetInfo), HcommNicPluginGetInfoFunc>::value,
        "HcommNicPluginGetInfo signature must match the public ABI typedef");
    static_assert(
        std::is_same<decltype(&HcommNicPluginCreateEndpoint), HcommNicPluginCreateEndpointFunc>::value,
        "HcommNicPluginCreateEndpoint signature must match the public ABI typedef");
    static_assert(
        std::is_same<decltype(&HcommNicPluginCreateChannel), HcommNicPluginCreateChannelFunc>::value,
        "HcommNicPluginCreateChannel signature must match the public ABI typedef");
}

TEST(ExperimentalNicPluginContract, CmakeTargetsAndOutputNamesUseCpuEndpointNaming)
{
    const std::string cmake = ReadRepoFile("experimental/base_comm/nic_plugin/CMakeLists.txt");
    ExpectContains(cmake, "add_library(hcomm_cpu_roce_plugin SHARED");
    ExpectContains(cmake, "hcomm_experimental_plugin_common(hcomm_cpu_roce_plugin hcomm_cpu_roce_plugin)");
    ExpectContains(cmake, "add_library(hcomm_cpu_ub_plugin SHARED");
    ExpectContains(cmake, "hcomm_experimental_plugin_common(hcomm_cpu_ub_plugin hcomm_cpu_ub_plugin)");
    ExpectContains(cmake, "install(TARGETS hcomm_cpu_roce_plugin hcomm_cpu_ub_plugin");
    ExpectNotContains(cmake, RetiredHostName("roce_plugin"));
    ExpectNotContains(cmake, RetiredHostName("ub_plugin"));
}

TEST(ExperimentalNicPluginContract, PackageManifestUsesCpuPluginSharedObjects)
{
    const std::string pluginManifest = ReadRepoFile("experimental/ExperimentalNicPlugin.xml");
    ExpectContains(pluginManifest, "<block name=\"ExperimentalNicPlugin\"");
    ExpectContains(pluginManifest, "hcomm_plugin");
    ExpectContains(pluginManifest, "libhcomm_cpu_roce_plugin.so");
    ExpectContains(pluginManifest, "libhcomm_cpu_ub_plugin.so");
    ExpectNotContains(pluginManifest, RetiredHostSharedObject("roce"));
    ExpectNotContains(pluginManifest, RetiredHostSharedObject("ub"));

    const std::string commLibManifest = ReadRepoFile("scripts/package/module/ascend/CommLib.xml");
    ExpectNotContains(commLibManifest, "hcomm_plugin");
    ExpectNotContains(commLibManifest, "libhcomm_cpu_roce_plugin.so");
    ExpectNotContains(commLibManifest, "libhcomm_cpu_ub_plugin.so");

    const std::string hcommManifest = ReadRepoFile("scripts/package/hcomm/hcomm.xml");
    ExpectContains(hcommManifest, "block_conf_path=\"ascend\"");
    ExpectContains(hcommManifest, "<block name=\"../../../../experimental/ExperimentalNicPlugin\"/>");
}

TEST(ExperimentalNicPluginContract, PluginInfoUsesCpuNamesAndExpectedProtocols)
{
    const std::string rocePlugin = ReadRepoFile("experimental/base_comm/nic_plugin/host_roce_plugin.cc");
    ExpectContains(rocePlugin, "HCOMM_NIC_PLUGIN_INFO_VERSION");
    ExpectContains(rocePlugin, "HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD");
    ExpectContains(rocePlugin, "\"hcomm_cpu_roce\"");
    ExpectContains(rocePlugin, "1U");
    ExpectContains(rocePlugin, "{COMM_PROTOCOL_ROCE, COMM_PROTOCOL_HCCS, COMM_PROTOCOL_HCCS, COMM_PROTOCOL_HCCS}");
    ExpectNotContains(rocePlugin, std::string("\"") + RetiredHostName("roce") + "\"");

    const std::string ubPlugin = ReadRepoFile("experimental/base_comm/nic_plugin/host_ub_plugin.cc");
    ExpectContains(ubPlugin, "HCOMM_NIC_PLUGIN_INFO_VERSION");
    ExpectContains(ubPlugin, "HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD");
    ExpectContains(ubPlugin, "\"hcomm_cpu_ub\"");
    ExpectContains(ubPlugin, "2U");
    ExpectContains(
        ubPlugin, "{COMM_PROTOCOL_UBC_TP, COMM_PROTOCOL_UB_CTP, COMM_PROTOCOL_RESERVED, COMM_PROTOCOL_RESERVED}");
    ExpectNotContains(ubPlugin, std::string("\"") + RetiredHostName("ub") + "\"");
}

TEST(ExperimentalNicPluginContract, ExportMacroKeepsExpectedSymbolNames)
{
    const std::string ops = ReadRepoFile("experimental/base_comm/nic_plugin/nic_plugin_ops.h");
    ExpectContains(ops, "extern \"C\" const HcommNicPluginInfo* HcommNicPluginGetInfo");
    ExpectContains(ops, "extern \"C\" int32_t HcommNicPluginCreateEndpoint");
    ExpectContains(ops, "extern \"C\" int32_t HcommNicPluginCreateChannel");
}

} // namespace
