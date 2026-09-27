/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <any>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <gtest/gtest.h>
#include <json.hpp>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_binary_data_type_pub.h"
#define private public
#define protected public
#include "sim_plugin.h"
#include "sim_plugin_manager.h"

#undef private
#undef protected

namespace {
const std::string kTestBase = "/tmp/hccl_pm_ut";
const std::string kPluginRoot = kTestBase + "/plugin";

std::string g_binLocation = kTestBase;

void CleanAll() { std::system(("rm -rf " + kTestBase + "*").c_str()); }

void MakeManifest(const std::string& dir, const std::string& name, const std::string& entry)
{
    std::system(("mkdir -p " + dir).c_str());
    std::ofstream f(dir + "/manifest.json");
    f << "{\"name\":\"" << name << "\", \"entry\":\"" << entry << "\"}";
    f.close();
}

bool InvokePlugin(const std::string& dir)
{
    try {
        HcclPlugin p(dir);
        return true;
    } catch (const std::runtime_error& e) {
        return false;
    }
}

void ResetManager()
{
    auto& m = HcclPluginManager::GetInstance();
    std::lock_guard<std::mutex> lock(m.m_mutex);
    m.m_plugins.clear();
}

void InjectPlugin(const std::string& tag, std::shared_ptr<HcclPlugin> plugin)
{
    auto& m = HcclPluginManager::GetInstance();
    std::lock_guard<std::mutex> lock(m.m_mutex);
    m.m_plugins[tag] = plugin;
}

void ClearPlugins()
{
    auto& m = HcclPluginManager::GetInstance();
    std::lock_guard<std::mutex> lock(m.m_mutex);
    m.m_plugins.clear();
}
} // namespace

std::string GetBinLocation() { return g_binLocation; }

namespace HcclSim {
HcclVmResult DumpHcclVmFlagData(HcclVmFlagData&) { return HcclVmResult::HCCL_SIM_SUCCESS; }
} // namespace HcclSim

class HcclPluginManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        CleanAll();
        ClearPlugins();
        g_binLocation = kTestBase;
    }
    void TearDown() override
    {
        ClearPlugins();
        CleanAll();
    }
};

// ========== IsMatchingPlugin ==========

TEST_F(HcclPluginManagerTest, IsMatching_MissingFile_ReturnsFalse)
{
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_FALSE(m.IsMatchingPlugin("/nonexistent/manifest.json", "any"));
}

TEST_F(HcclPluginManagerTest, IsMatching_InvalidJson_ReturnsFalse)
{
    std::string d = kTestBase + "_badjson";
    std::system(("mkdir -p " + d).c_str());
    {
        std::ofstream f(d + "/manifest.json");
        f << "{bad}";
    }
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_FALSE(m.IsMatchingPlugin(d + "/manifest.json", "any"));
}

TEST_F(HcclPluginManagerTest, IsMatching_WrongName_ReturnsFalse)
{
    std::string d = kTestBase + "_wrong";
    MakeManifest(d, "other", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_FALSE(m.IsMatchingPlugin(d + "/manifest.json", "target"));
}

TEST_F(HcclPluginManagerTest, IsMatching_CorrectNameWithEntry_ReturnsTrue)
{
    std::string d = kTestBase + "_match";
    MakeManifest(d, "myplugin", "echo hello");
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_TRUE(m.IsMatchingPlugin(d + "/manifest.json", "myplugin"));
}

TEST_F(HcclPluginManagerTest, IsMatching_CorrectNameNoEntry_ReturnsFalse)
{
    std::string d = kTestBase + "_noent";
    std::system(("mkdir -p " + d).c_str());
    {
        std::ofstream f(d + "/manifest.json");
        f << "{\"name\":\"myplugin\"}";
    }
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_FALSE(m.IsMatchingPlugin(d + "/manifest.json", "myplugin"));
}

// ========== GetPluginFolderPath ==========

TEST_F(HcclPluginManagerTest, GetFolderPath_NoPluginDir_ReturnsFalse)
{
    auto& m = HcclPluginManager::GetInstance();
    std::string path;
    EXPECT_FALSE(m.GetPluginFolderPath("anything", path));
    EXPECT_TRUE(path.empty());
}

TEST_F(HcclPluginManagerTest, GetFolderPath_FoundAtRoot_ReturnsTrue)
{
    std::string d = kPluginRoot + "/p1";
    MakeManifest(d, "p1", "sleep 1");

    auto& m = HcclPluginManager::GetInstance();
    std::string path;
    EXPECT_TRUE(m.GetPluginFolderPath("p1", path));
    EXPECT_FALSE(path.empty());
}

TEST_F(HcclPluginManagerTest, GetFolderPath_FoundInSubdir_ReturnsTrue)
{
    MakeManifest(kPluginRoot + "/deep/p1", "p2", "sleep 1");

    auto& m = HcclPluginManager::GetInstance();
    std::string path;
    EXPECT_TRUE(m.GetPluginFolderPath("p2", path));
    EXPECT_FALSE(path.empty());
}

// ========== RegisterPlugin ==========

TEST_F(HcclPluginManagerTest, RegisterPlugin_NotFound_ReturnsNotFound)
{
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_EQ(m.RegisterPlugin("ghost"), HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
}

TEST_F(HcclPluginManagerTest, RegisterPlugin_Found_StartsSuccessfully)
{
    MakeManifest(kPluginRoot + "/p3", "p3", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_EQ(m.RegisterPlugin("p3"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginManagerTest, RegisterPlugin_Duplicate_ReturnsSuccess)
{
    MakeManifest(kPluginRoot + "/p4", "p4", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    EXPECT_EQ(m.RegisterPlugin("p4"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(m.RegisterPlugin("p4"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// ========== StopPlugins ==========

TEST_F(HcclPluginManagerTest, StopPlugins_EmptyTags_ReturnsEmpty)
{
    auto results = HcclPluginManager::GetInstance().StopPlugins({});
    EXPECT_TRUE(results.empty());
}

TEST_F(HcclPluginManagerTest, StopPlugins_NotFound_ReturnsNotFound)
{
    auto results = HcclPluginManager::GetInstance().StopPlugins({"no_such"});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
}

TEST_F(HcclPluginManagerTest, StopPlugins_RunnerTag_ReturnsSuccess)
{
    auto results = HcclPluginManager::GetInstance().StopPlugins({"runner"});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginManagerTest, StopPlugins_StopsAndErasesPlugin)
{
    MakeManifest(kPluginRoot + "/p5", "p5", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("p5"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    auto results = m.StopPlugins({"p5"});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(m.m_plugins.find("p5"), m.m_plugins.end());
}

TEST_F(HcclPluginManagerTest, StopPlugins_MultipleTags_StopsAll)
{
    MakeManifest(kPluginRoot + "/a", "a", "sleep 1");
    MakeManifest(kPluginRoot + "/b", "b", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("a"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_EQ(m.RegisterPlugin("b"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    auto results = m.StopPlugins({"a", "b"});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0], HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(results[1], HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_TRUE(m.m_plugins.empty());
}

// ========== StopAllPlugins ==========

TEST_F(HcclPluginManagerTest, StopAllPlugins_Empty_ReturnsSuccess)
{
    EXPECT_EQ(HcclPluginManager::GetInstance().StopAllPlugins(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginManagerTest, StopAllPlugins_WithPlugins_StopsAll)
{
    MakeManifest(kPluginRoot + "/c", "c", "sleep 1");
    MakeManifest(kPluginRoot + "/d", "d", "sleep 1");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("c"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_EQ(m.RegisterPlugin("d"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(m.StopAllPlugins(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_TRUE(m.m_plugins.empty());
}

// ========== GetPluginStatus ==========

TEST_F(HcclPluginManagerTest, GetPluginStatus_Empty_ReturnsHeaderOnly)
{
    auto status = HcclPluginManager::GetInstance().GetPluginStatus();
    EXPECT_LE(status.size(), 1u);
    if (!status.empty()) {
        EXPECT_NE(status[0].find("PLUGIN_TAG"), std::string::npos);
    }
}

TEST_F(HcclPluginManagerTest, GetPluginStatus_WithPlugin_ShowsRunning)
{
    MakeManifest(kPluginRoot + "/e", "e", "sleep 2");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("e"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    auto status = m.GetPluginStatus();
    bool foundRunning = false;
    for (const auto& s : status) {
        if (s.find("e") != std::string::npos && s.find("RUNNING") != std::string::npos) {
            foundRunning = true;
        }
    }
    EXPECT_TRUE(foundRunning);
    ClearPlugins();
}

// ========== StartPlugins ==========

TEST_F(HcclPluginManagerTest, StartPlugins_NotFound_ReturnsNotFound)
{
    auto results = HcclPluginManager::GetInstance().StartPlugins({"no_tag"});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0], HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
}

// ========== SendMessageToPlugin ==========

TEST_F(HcclPluginManagerTest, SendMessageToPlugin_NotFound_ReturnsNotFound)
{
    EXPECT_EQ(
        HcclPluginManager::GetInstance().SendMessageToPlugin("ghost", "act"),
        HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
}

TEST_F(HcclPluginManagerTest, SendMessageToPlugin_Found_SendsMessage)
{
    MakeManifest(kPluginRoot + "/f", "f", "cat");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("f"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    auto r = m.SendMessageToPlugin("f", "hello");
    EXPECT_EQ(r, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ClearPlugins();
}

// ========== BroadcastToAllPlugin ==========

TEST_F(HcclPluginManagerTest, BroadcastToAll_Empty_ReturnsSuccess)
{
    EXPECT_EQ(HcclPluginManager::GetInstance().BroadcastToAllPlugin("bc"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(HcclPluginManagerTest, BroadcastToAll_ToPlugins_ReturnsSuccess)
{
    MakeManifest(kPluginRoot + "/g", "g", "cat");
    MakeManifest(kPluginRoot + "/h", "h", "cat");
    auto& m = HcclPluginManager::GetInstance();
    ASSERT_EQ(m.RegisterPlugin("g"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_EQ(m.RegisterPlugin("h"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(m.BroadcastToAllPlugin("all"), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ClearPlugins();
}
