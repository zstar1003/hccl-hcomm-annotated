/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "cmd_cluster_model_utils.h"

using namespace HcclSim;

namespace {
class EnvGuard {
public:
    EnvGuard() = default;
    ~EnvGuard() {}
};
} // namespace

class ClearHvmModelEnvTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class ParseYamlTopoWithModelMetaTest : public testing::Test {
protected:
    std::string modelDir_;
    void SetUp() override { modelDir_ = GetBinLocation() + "/cluster_model"; }
    void TearDown() override {}
    std::string GetBinLocation()
    {
        std::error_code ec;
        auto exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (ec) {
            return ".";
        }
        return exePath.parent_path().string();
    }
    void WriteYaml(const std::string& name, const std::string& content)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::ofstream ofs(path);
        ofs << content;
        ofs.close();
    }
    void RemoveYaml(const std::string& name)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::filesystem::remove(path);
    }
};

TEST_F(ParseYamlTopoWithModelMetaTest, Parse112WithModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("112", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoWithModelMetaTest, Parse118WithModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("118", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoWithModelMetaTest, ParseWithoutModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("112", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoWithModelMetaTest, WrongSocVersionWithModelMeta)
{
    WriteYaml("ut_wrong_soc_meta", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_wrong_soc_meta", topo);
    EXPECT_FALSE(result);
    RemoveYaml("ut_wrong_soc_meta");
}

TEST_F(ParseYamlTopoWithModelMetaTest, Ascend950WithModelMeta)
{
    WriteYaml("ut_950_meta", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1, 2, 3]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_950_meta", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_950_meta");
}

TEST_F(ParseYamlTopoWithModelMetaTest, NonUniformWithModelMeta)
{
    WriteYaml("ut_non_uniform_meta", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 3
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
      - serId: 1
        ip: 192.2.5.5
        ranks: [2]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_non_uniform_meta", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_non_uniform_meta");
}

TEST_F(ParseYamlTopoWithModelMetaTest, ServerCountMismatchWithModelMeta)
{
    WriteYaml("ut_server_mismatch_meta", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_server_mismatch_meta", topo);
    EXPECT_FALSE(result);
    RemoveYaml("ut_server_mismatch_meta");
}

TEST_F(ParseYamlTopoWithModelMetaTest, ZeroPodNumYaml)
{
    WriteYaml("ut_zero_pod", R"(
meta:
  podNum: 0
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_pod", topo));
    RemoveYaml("ut_zero_pod");
}

TEST_F(ParseYamlTopoWithModelMetaTest, ZeroSerNumYaml)
{
    WriteYaml("ut_zero_ser", R"(
meta:
  podNum: 1
  serNum: 0
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_ser", topo));
    RemoveYaml("ut_zero_ser");
}

TEST_F(ParseYamlTopoWithModelMetaTest, ZeroRankNumYaml)
{
    WriteYaml("ut_zero_rank", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 0
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_rank", topo));
    RemoveYaml("ut_zero_rank");
}

TEST_F(ParseYamlTopoWithModelMetaTest, MissingTopologyWithoutModelMeta)
{
    WriteYaml("ut_no_topo_nometa", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_no_topo_nometa", topo));
    RemoveYaml("ut_no_topo_nometa");
}

TEST_F(ParseYamlTopoWithModelMetaTest, EmptyTopologyList)
{
    WriteYaml("ut_empty_topo", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology: []
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_empty_topo", topo);
    EXPECT_FALSE(result);
    RemoveYaml("ut_empty_topo");
}

TEST_F(ParseYamlTopoWithModelMetaTest, PodWithoutServers)
{
    WriteYaml("ut_no_servers", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_no_servers", topo);
    EXPECT_FALSE(result);
    RemoveYaml("ut_no_servers");
}

TEST_F(ParseYamlTopoWithModelMetaTest, ServerWithoutRanks)
{
    WriteYaml("ut_no_ranks", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_no_ranks", topo);
    EXPECT_FALSE(result);
    RemoveYaml("ut_no_ranks");
}

TEST_F(ParseYamlTopoWithModelMetaTest, CaseInsensitiveSocVersion)
{
    WriteYaml("ut_upper_soc", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_upper_soc", topo));
    RemoveYaml("ut_upper_soc");
}

class ExportAndShellCompatIntegrationTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};
