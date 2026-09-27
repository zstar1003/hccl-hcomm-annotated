/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_plugin.h"

namespace {
const std::string kTestBase = "/tmp/hccl_plugin_ut";

void MakeManifest(
    const std::string& dir, const std::string& name, const std::string& entry, const std::string& ver = "1.0")
{
    std::system(("mkdir -p " + dir).c_str());
    std::ofstream f(dir + "/manifest.json");
    f << "{\"name\":\"" << name << "\","
      << "\"version\":\"" << ver << "\","
      << "\"entry\":\"" << entry << "\"}";
    f.close();
}

void CleanDirs() { std::system(("rm -rf " + kTestBase + "_*").c_str()); }

bool InvokePlugin(const std::string& dir)
{
    try {
        HcclPlugin p(dir);
        return true;
    } catch (const std::runtime_error& e) {
        return false;
    }
}
} // namespace

class HcclPluginTest : public testing::Test {
protected:
    void TearDown() override { CleanDirs(); }
};

// ========== Static Constants ==========

TEST_F(HcclPluginTest, MaxScanDepth_IsTwo) { EXPECT_EQ(HcclPlugin::MAX_SCAN_DEPTH, 2); }

TEST_F(HcclPluginTest, PluginPath_IsPlugin) { EXPECT_EQ(HcclPlugin::PLUGIN_PATH, "/plugin"); }

TEST_F(HcclPluginTest, ManifestFile_IsManifestJson) { EXPECT_EQ(HcclPlugin::MANIFEST_FILE, "/manifest.json"); }

TEST_F(HcclPluginTest, Manifest_FieldNames)
{
    EXPECT_EQ(HcclPlugin::Manifest::pluginName, "name");
    EXPECT_EQ(HcclPlugin::Manifest::pluginVersion, "version");
    EXPECT_EQ(HcclPlugin::Manifest::pluginEntry, "entry");
    EXPECT_EQ(HcclPlugin::Manifest::pluginDependency::hostVersion, "min_core_version");
}

TEST_F(HcclPluginTest, PluginMessage_FieldNames)
{
    EXPECT_EQ(HcclPlugin::PluginMessage::messageType, "type");
    EXPECT_EQ(HcclPlugin::PluginMessage::messageAction, "action");
    EXPECT_EQ(HcclPlugin::PluginMessage::messagePayload, "payload");
}

TEST_F(HcclPluginTest, PluginMessageType_EnumValues)
{
    EXPECT_EQ(static_cast<int>(PLUGIN_MESSAGE_TYPE::BROADCAST), 0);
    EXPECT_EQ(static_cast<int>(PLUGIN_MESSAGE_TYPE::COMMAND), 1);
    EXPECT_EQ(static_cast<int>(PLUGIN_MESSAGE_TYPE::MESSAGE), 2);
}

// ========== Constructor Error Paths ==========

TEST_F(HcclPluginTest, Ctor_MissingManifest_Throws) { EXPECT_FALSE(InvokePlugin("/nonexistent/plugin/dir")); }

TEST_F(HcclPluginTest, Ctor_BadJson_Throws)
{
    std::string d = kTestBase + "_badjson";
    std::system(("mkdir -p " + d).c_str());
    {
        std::ofstream f(d + "/manifest.json");
        f << "{bad}";
    }
    EXPECT_FALSE(InvokePlugin(d));
}

TEST_F(HcclPluginTest, Ctor_EmptyEntry_Throws)
{
    std::string d = kTestBase + "_emptyent";
    MakeManifest(d, "x", "");
    EXPECT_FALSE(InvokePlugin(d));
}

// ========== Live Process Tests ==========

TEST_F(HcclPluginTest, Live_Sleep_StartsAndStops)
{
    std::string d = kTestBase + "_sleep";
    MakeManifest(d, "sp", "sleep 1");
    {
        HcclPlugin pl(d);
        EXPECT_GT(pl.GetPid(), 0);
        EXPECT_GT(pl.GetStdinFd(), -1);
        EXPECT_EQ(pl.GetTag(), "sp");
        EXPECT_TRUE(pl.IsRunning());
    }
}

TEST_F(HcclPluginTest, Live_Cat_ReceivesMessages)
{
    std::string d = kTestBase + "_cat";
    MakeManifest(d, "cp", "cat");
    HcclPlugin pl(d);
    EXPECT_TRUE(pl.IsRunning());

    auto r = pl.SendMessage(PLUGIN_MESSAGE_TYPE::COMMAND, "hello");
    EXPECT_EQ(r, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    r = pl.SendMessage(PLUGIN_MESSAGE_TYPE::MESSAGE, "data", nlohmann::json{{"key", 42}});
    EXPECT_EQ(r, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginTest, Live_Cat_BroadcastMessage)
{
    std::string d = kTestBase + "_catbc";
    MakeManifest(d, "cp", "cat");
    HcclPlugin pl(d);
    EXPECT_TRUE(pl.IsRunning());
    EXPECT_EQ(pl.SendMessage(PLUGIN_MESSAGE_TYPE::BROADCAST, "all"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginTest, Live_QuickExit_ExitsAndStopsCleanly)
{
    std::string d = kTestBase + "_quick";
    MakeManifest(d, "tp", "sleep 0.1");
    HcclPlugin pl(d);
    EXPECT_TRUE(pl.IsRunning());
    // Wait for the child to exit naturally
    usleep(200000);
    EXPECT_FALSE(pl.IsRunning());
    // Stop should be safe even after process already exited
    EXPECT_EQ(pl.Stop(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(pl.Stop(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_FALSE(pl.IsRunning());
}
