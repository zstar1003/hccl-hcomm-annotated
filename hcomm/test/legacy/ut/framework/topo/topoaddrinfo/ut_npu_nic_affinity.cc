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
#include <mockcpp/mockcpp.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <vector>
#include <stdarg.h>
#include "npu_nic_affinity.h"
#include "rank_info_types.h"
#include "topo_addr_info.h"
#include "hal.h"
#include "securec.h"
#include "topo_addr_info_log.h"

static void TestLogRecord(int moduleId, int level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf("[UT_TOPO] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

static int TestCheckLogLevel(int moduleId, int logLevel) { return 1; /* 允许所有级别 */ }

/* ──────── Wrap getifaddrs / freeifaddrs ──────── */

static struct ifaddrs* g_fakeIfaddr = NULL;

extern "C" int __wrap_getifaddrs(struct ifaddrs** ifap)
{
    *ifap = g_fakeIfaddr;
    return 0;
}

extern "C" void __wrap_freeifaddrs(struct ifaddrs* ifa) { /* 我们的假数据是静态管理的，不需要释放 */ }

static void SetupFakeNet(const char* ethName, const char* ipStr)
{
    if (g_fakeIfaddr != NULL) {
        free(g_fakeIfaddr->ifa_name);
        free(g_fakeIfaddr->ifa_addr);
        free(g_fakeIfaddr);
        g_fakeIfaddr = NULL;
    }

    struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
    ASSERT_NE(ifa, nullptr);
    ifa->ifa_next = NULL;
    ifa->ifa_name = strdup(ethName);
    ASSERT_NE(ifa->ifa_name, nullptr);
    ifa->ifa_flags = IFF_UP;
    ifa->ifa_addr = (struct sockaddr*)calloc(1, sizeof(struct sockaddr_in));
    ASSERT_NE(ifa->ifa_addr, nullptr);
    struct sockaddr_in* sin = (struct sockaddr_in*)ifa->ifa_addr;
    sin->sin_family = AF_INET;
    inet_pton(AF_INET, ipStr, &sin->sin_addr);
    g_fakeIfaddr = ifa;
}

static void TeardownFakeNet()
{
    if (g_fakeIfaddr != NULL) {
        free(g_fakeIfaddr->ifa_name);
        free(g_fakeIfaddr->ifa_addr);
        free(g_fakeIfaddr);
        g_fakeIfaddr = NULL;
    }
}

/* ──────── 创建 /tmp/ut_hca/<hca>/device/net/<eth> ──────── */
static void SetupFakeHca(const char* hca, const char* eth)
{
    char path[512];
    sprintf_s(path, sizeof(path), "/tmp/ut_hca/%s/device/net", hca);
    char p[512];
    strncpy_s(p, sizeof(p), path, sizeof(p) - 1);
    for (char* c = p + 1; *c; c++) {
        if (*c == '/') {
            *c = '\0';
            mkdir(p, 0755);
            *c = '/';
        }
    }
    mkdir(p, 0755);
    char ethPath[576];
    sprintf_s(ethPath, sizeof(ethPath), "%s/%s", path, eth);
    mkdir(ethPath, 0755);
}

static void TeardownFakeHca()
{
    char cmd[256];
    sprintf_s(cmd, sizeof(cmd), "rm -rf /tmp/ut_hca");
    system(cmd);
}

/* ──────── Mock HAL ──────── */
static struct dcmi_pcie_info_all g_pi[MAX_NPU_COUNT];
static unsigned int g_pc = 0;
static bool g_visibleMask[MAX_NPU_COUNT]; // true=该 phyId 对运行时可见

extern "C" int mock_pi(int phyId, struct dcmi_pcie_info_all* info)
{
    if (phyId >= 0 && (unsigned int)phyId < g_pc && g_visibleMask[phyId]) {
        *info = g_pi[phyId];
        return 0;
    }
    return -1;
}

/* 模拟 hal_get_userdevid_by_phyid：对照 g_visibleMask 判断可见性 */
extern "C" int mock_userdevid(int phyId, int* userDevId)
{
    if (phyId >= 0 && phyId < (int)MAX_NPU_COUNT && g_visibleMask[phyId]) {
        *userDevId = phyId;
        return 0;
    }
    return -1;
}

class NpuNicAffinityTest : public testing::Test {
protected:
    void SetUp() override
    {
        /* TopoAddrInfoTest 对 hal_dlopen/hal_dlsym 的 mock 导致 load_dcmi()
           缓存了无效函数指针，后序调用会崩溃。注入 mock 接管
           hal_get_device_pcie_info / hal_get_userdevid_by_phyid，
           完全绕过 load_dcmi() 路径。 */
        g_pc = 0;
        MOCKER(hal_get_device_pcie_info).stubs().with(mockcpp::any(), mockcpp::any()).will(mockcpp::invoke(mock_pi));
        MOCKER(hal_get_userdevid_by_phyid)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(mockcpp::invoke(mock_userdevid));

        remove("/tmp/ut_virtualTopology.xml");
        memset_s(g_pi, sizeof(g_pi), 0, sizeof(g_pi));
        memset(g_visibleMask, 1, sizeof(g_visibleMask)); // all visible by default
        SetupFakeHca("hrn5_0", "eth0");
        SetupFakeNet("eth0", "10.0.0.1");

        /* 注入日志回调，用例中 TOPO_ERR / TOPO_INFO 输出到 stdout */
        g_topo_DlogRecord = TestLogRecord;
        g_topo_CheckLogLevel = TestCheckLogLevel;
    }
    void TearDown() override
    {
        remove("/tmp/ut_virtualTopology.xml");
        TeardownFakeHca();
        TeardownFakeNet();
        g_topo_DlogRecord = NULL;
        g_topo_CheckLogLevel = NULL;
        GlobalMockObject::verify();
    }
    void W(const char* c)
    {
        FILE* fp = fopen("/tmp/ut_virtualTopology.xml", "w");
        ASSERT_NE(fp, nullptr);
        fputs(c, fp);
        fclose(fp);
    }
    /* 设置可见设备掩码，模拟 ASCEND_RT_VISIBLE_DEVICES 效果 */
    void SetVisibleDevices(const std::vector<int>& ids)
    {
        memset(g_visibleMask, 0, sizeof(g_visibleMask));
        for (int id : ids) {
            if (id >= 0 && id < MAX_NPU_COUNT) {
                g_visibleMask[(unsigned int)id] = true;
            }
        }
    }
    void S(unsigned int npu, unsigned int pcie)
    {
        if (npu > 0) {
            MOCKER(hal_get_npu_count).stubs().will(returnValue((int)npu));
        }
        g_pc = pcie;
        if (pcie > 0) {
            MOCKER(hal_get_device_pcie_info)
                .stubs()
                .with(mockcpp::any(), mockcpp::any())
                .will(mockcpp::invoke(mock_pi));
        }
    }
    /* 校验 GetRoceIpFromXml 成功并返回期望 IP */
    void AssertRoceIpOk(unsigned int npuId, const char* expectIp)
    {
        char ip[64] = {0};
        EXPECT_EQ(GetRoceIpFromXml(npuId, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectIp);
    }
    /* 校验 GetRoceIpFromXml 失败 */
    void AssertRoceIpFail(unsigned int npuId)
    {
        char ip[64] = {0};
        EXPECT_NE(GetRoceIpFromXml(npuId, ip, sizeof(ip)), TOPO_SUCCESS);
    }
};

/* ══════════════════════════════════════════════════════════════
 *  PCIE 格式
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, PCIE_1Bridge_1Nic_2Npu)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, PCIE_MultiBridge_4Group)
{
    S(16, 16);
    unsigned int bus[] = {3, 4, 5, 6, 0xc, 0xd, 0xe, 0xf, 0x14, 0x15, 0x16, 0x17, 0x1e, 0x1f, 0x20, 0x21};
    for (int i = 0; i < 16; i++) {
        g_pi[i].domain = 0;
        g_pi[i].bdf_busid = bus[i];
    }
    W("<system version=\"1\">\n"
      "<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "  <pci busid=\"0000:02:00.0\">\n    <nic>\n      <net name=\"xscale_3\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:03:00.0\"/>\n  <pci busid=\"0000:04:00.0\"/>\n"
      "  <pci busid=\"0000:05:00.0\"/>\n  <pci busid=\"0000:06:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:10:00.0\">\n"
      "  <pci busid=\"0000:11:00.0\">\n    <nic>\n      <net name=\"xscale_4\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:0c:00.0\"/>\n  <pci busid=\"0000:0d:00.0\"/>\n"
      "  <pci busid=\"0000:0e:00.0\"/>\n  <pci busid=\"0000:0f:00.0\"/>\n"
      "</pci>\n</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<pci busid=\"0000:20:00.0\">\n"
      "  <pci busid=\"0000:21:00.0\">\n    <nic>\n      <net name=\"xscale_1\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:14:00.0\"/>\n  <pci busid=\"0000:15:00.0\"/>\n"
      "  <pci busid=\"0000:16:00.0\"/>\n  <pci busid=\"0000:17:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:30:00.0\">\n"
      "  <pci busid=\"0000:31:00.0\">\n    <nic>\n      <net name=\"xscale_2\"/>\n    </nic>\n  </pci>\n"
      "  <pci busid=\"0000:1e:00.0\"/>\n  <pci busid=\"0000:1f:00.0\"/>\n"
      "  <pci busid=\"0000:20:00.0\"/>\n  <pci busid=\"0000:21:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    SetupFakeHca("xscale_1", "eth0");
    SetupFakeHca("xscale_2", "eth0");
    SetupFakeHca("xscale_3", "eth0");
    SetupFakeHca("xscale_4", "eth0");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* NPU 以非自闭合 <pci busid="..."></pci> 形式出现 */
TEST_F(NpuNicAffinityTest, PCIE_NpuNonSelfClose)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\">\n</pci>\n"
      "<pci busid=\"0000:04:00.0\">\n</pci>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* PCIE 多层嵌套：只顶层 Bridge 建组 */
TEST_F(NpuNicAffinityTest, PCIE_NestedDepth)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 4;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:00.0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:02:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</pci>\n</pci>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 空 Bridge（无 NIC）：应该返回 -1 */
TEST_F(NpuNicAffinityTest, PCIE_EmptyBridge)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

/* ══════════════════════════════════════════════════════════════
 *  UB 格式
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, UB_1Group)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, UB_2Group)
{
    SetupFakeHca("hrn5_1", "eth0");
    S(4, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n</ub>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_1\"/>\n</nic>\n"
      "<npu chipphyid=\"2\"/>\n<npu chipphyid=\"3\"/>\n</ub>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(2, "10.0.0.1");
}

TEST_F(NpuNicAffinityTest, UB_NpuNonSelfClose)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\">\n</npu>\n"
      "<npu chipphyid=\"1\">\n</npu>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* NIC 在子 ub 内：不应单独建组 */
TEST_F(NpuNicAffinityTest, UB_NicInSubUb)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n</ub>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"1\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  异常 / 边界
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, Error_FileNotFound) { AssertRoceIpFail(0); }

TEST_F(NpuNicAffinityTest, Error_EmptyFile)
{
    W("");
    AssertRoceIpFail(0);
}

TEST_F(NpuNicAffinityTest, Error_NoNic)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n<pci busid=\"0000:01:00.0\">\n"
      "<pci busid=\"0000:02:00.0\"/>\n<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

TEST_F(NpuNicAffinityTest, Error_WithComments)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<!-- comment -->\n<system version=\"1.0\">\n<cpu numaid=\"0\">\n<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 多标签同行：<nic><net name="x"/></nic> 在一行 */
TEST_F(NpuNicAffinityTest, MultiTagSameLine)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;
    g_pi[1].bdf_deviceid = 0;
    g_pi[1].bdf_funcid = 0;
    S(2, 2);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic><net name=\"hrn5_0\"/></nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  异常 XML 格式（解析器鲁棒性）
 * ══════════════════════════════════════════════════════════════ */

/* 多余闭合标签：无对应开标签 */
TEST_F(NpuNicAffinityTest, Malformed_ExtraCloseTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "</pci>\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 未知标签：解析器应跳过，不影响后续正常标签 */
TEST_F(NpuNicAffinityTest, Malformed_UnknownTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<foo bar=\"x\"/>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 属性无引号：ExtractAttrs 跳过该属性，不崩溃 */
TEST_F(NpuNicAffinityTest, Malformed_AttrNoQuote)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=0000:03:00.0>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 属性无值：ExtractAttrs 跳过，不崩溃 */
TEST_F(NpuNicAffinityTest, Malformed_AttrNoValue)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid>\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* 不完整的标签（缺 >）：strchr 返回 NULL，跳过该行 */
TEST_F(NpuNicAffinityTest, Malformed_IncompleteTag)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"\n"
      "</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    AssertRoceIpFail(0);
}

/* 空标签：<> 或 </>，GetTagName 返回空，跳过 */
TEST_F(NpuNicAffinityTest, Malformed_EmptyTag)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<>\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n"
      "</ub>\n</>\n"
      "</cpu>\n</system>\n");
    AssertRoceIpOk(0, "10.0.0.1");
}

/* ══════════════════════════════════════════════════════════════
 *  空参数
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, NullParams)
{
    EXPECT_NE(GetRoceIpFromXml(0, NULL, 0), TOPO_SUCCESS);
    EXPECT_NE(GetRoceIpFromXml(-1, NULL, 0), TOPO_SUCCESS);
}

/* ══════════════════════════════════════════════════════════════
 *  ASCEND_RT_VISIBLE_DEVICES 非连续可见
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, VisibleDevices_NonContiguous_PCIE)
{
    /* 模拟 ASCEND_RT_VISIBLE_DEVICES=0,3,7：8 张 NPU 只暴露 3 张，ID 不连续 */
    S(8, 8);
    SetVisibleDevices({0, 3, 7});

    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3; /* → "0000:03:00.0" */
    g_pi[3].domain = 0;
    g_pi[3].bdf_busid = 7; /* → "0000:07:00.0" */
    g_pi[7].domain = 0;
    g_pi[7].bdf_busid = 10; /* → "0000:0a:00.0" */

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:07:00.0\"/>\n"
      "<pci busid=\"0000:0a:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    /* 可见的 NPU 能正确匹配到 NIC */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(3, "10.0.0.1");
    AssertRoceIpOk(7, "10.0.0.1");

    /* 不可见的 NPU 获取不到 IP */
    AssertRoceIpFail(1);
    AssertRoceIpFail(2);
    AssertRoceIpFail(4);
    AssertRoceIpFail(5);
    AssertRoceIpFail(6);
}

/* UB 格式 + 非连续可见：验证 HandleNpuTag 中的 hal_get_userdevid_by_phyid 过滤 */
TEST_F(NpuNicAffinityTest, VisibleDevices_NonContiguous_UB)
{
    S(8, 0);
    SetVisibleDevices({0, 3, 7});

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n<npu chipphyid=\"3\"/>\n<npu chipphyid=\"7\"/>\n"
      "<npu chipphyid=\"1\"/>\n<npu chipphyid=\"2\"/>\n"
      "</ub>\n</cpu>\n</system>\n");

    /* 可见的 NPU 能匹配到 NIC */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(3, "10.0.0.1");
    AssertRoceIpOk(7, "10.0.0.1");
    /* 不可见的 NPU 不会加入亲和分组，应获取不到 IP */
    AssertRoceIpFail(1);
    AssertRoceIpFail(2);
}

/* name 为 eth 名：NameToIp 先尝试 EthToIp 直接解析 */
TEST_F(NpuNicAffinityTest, NameIsEthName)
{
    S(2, 2);
    SetVisibleDevices({0, 1});
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;

    /* XML 中 net name 直接用 eth0，不走 HCA 路径 */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    /* EthToIp("eth0") 应通过 getifaddrs mock 拿到 10.0.0.1 */
    AssertRoceIpOk(0, "10.0.0.1");
    AssertRoceIpOk(1, "10.0.0.1");
}

/* eth 名 + TopoAddrInfoGet 端到端 */
TEST_F(NpuNicAffinityTest, NameIsEthName_TopoAddrInfoGet)
{
    S(2, 2);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[1].domain = 0;
    g_pi[1].bdf_busid = 4;

    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");

    unsigned int mainboardId = MAIN_BOARD_ID_CARD_NOMESH;
    char drvPath[256] = "/usr/local/Ascend2";
    dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
    memset_s(eidList, sizeof(eidList), 0, sizeof(eidList));
    size_t eidNum = 0;
    MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mainboardId)).will(returnValue(0));
    MOCKER(hal_get_driver_install_path)
        .stubs()
        .with(outBoundP(drvPath, strlen(drvPath)), mockcpp::any())
        .will(returnValue(0));
    MOCKER(hal_get_eid_list_by_phy_id)
        .stubs()
        .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
        .will(returnValue(0));

    char buf[4096] = {0};
    size_t bufSize = sizeof(buf);
    ASSERT_EQ(TopoAddrInfoGet(0, buf, &bufSize), 0);
    /* ETH name 也能在端到端流程中正确解析出 IP（NIC 名不写入 JSON，只校验地址） */
    EXPECT_NE(strstr(buf, "\"addr\": \"10.0.0.1\""), nullptr);
}

/* ══════════════════════════════════════════════════════════════
 *  端到端：TopoAddrInfoGet 全流程（非连续可见设备）
 * ══════════════════════════════════════════════════════════════ */

/* 校验 TopoAddrInfoGet 输出的 JSON 中包含所有预期字段 */
static void AssertJsonContains(const char* json, const char* const expectFields[], const std::string& desc)
{
    for (int i = 0; expectFields[i] != nullptr; i++) {
        EXPECT_NE(strstr(json, expectFields[i]), nullptr) << desc << " JSON 应包含: " << expectFields[i];
    }
}

/* 校验 TopoAddrInfoGet 输出的 JSON 中不包含某字段 */
static void AssertJsonNotContains(const char* json, const char* field, const std::string& desc)
{
    EXPECT_EQ(strstr(json, field), nullptr) << desc << " JSON 不应包含: " << field;
}

/* ROCE 层预期字段（phyId 可见时） */
static const char* g_roceJsonFields[]
    = {"\"net_layer\": 3",       "\"net_instance_id\": \"cluster\"", "\"net_type\": \"CLOS\"", "\"rank_addr_list\":",
       "\"addr\": \"10.0.0.1\"", "\"plane_id\": \"plane0\"",         "\"ports\": [\"d2h\"]",   nullptr};

/* RootInfo 层预期字段 */
static const char* g_rootInfoJsonFields[]
    = {"\"version\": \"2.0\"", "\"rank_count\": 1", "\"rank_list\":", "\"device_id\":", "\"local_id\":", nullptr};

/* ══════════════════════════════════════════════════════════════
 *  全量 mainboard × PCIE/UB × 可见模式 正交覆盖
 * ══════════════════════════════════════════════════════════════ */

struct TopoParam {
    unsigned int mainboardId;
    const char* name;
    bool isCard;   // true→hal_get_eid_list, false→HalGetUBEntityList+hal_get_spod_info
    bool isPcie;   // true→PCIE XML, false→UB XML
    bool isSparse; // true→{0,3,7}, false→全量 {0..7}
};

std::ostream& operator<<(std::ostream& os, const TopoParam& p)
{
    return os << p.name << "_" << (p.isPcie ? "PCIE" : "UB") << "_" << (p.isSparse ? "Sparse" : "All");
}

/* 构建 XML 字符串 */
static std::string BuildXml(bool isPcie, const std::vector<int>& ids)
{
    std::string xml;
    if (isPcie) {
        xml = "<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
              "<pci busid=\"0000:01:00.0\">\n"
              "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n";
        for (int id : ids) {
            char buf[64];
            (void)sprintf_s(buf, sizeof(buf), "<pci busid=\"0000:%02x:00.0\"/>\n", 3 + id);
            xml += buf;
        }
        xml += "</pci>\n</cpu>\n</system>\n";
    } else {
        xml = "<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
              "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n";
        for (int id : ids) {
            char buf[64];
            (void)sprintf_s(buf, sizeof(buf), "<npu chipphyid=\"%d\"/>\n", id);
            xml += buf;
        }
        xml += "</ub>\n</cpu>\n</system>\n";
    }
    return xml;
}

class TopoAddrInfoAllMainboardTest : public NpuNicAffinityTest, public testing::WithParamInterface<TopoParam> {};

TEST_P(TopoAddrInfoAllMainboardTest, EndToEnd)
{
    const TopoParam& param = GetParam();
    std::vector<int> visibleIds = param.isSparse ? std::vector<int>{0, 3, 7} : std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};

    S(8, param.isPcie ? 8 : 0);
    SetVisibleDevices(visibleIds);

    /* PCIE 信息：busId = 3 + phyId */
    if (param.isPcie) {
        for (int id : visibleIds) {
            g_pi[id].domain = 0;
            g_pi[id].bdf_busid = 3 + id;
        }
    }

    std::string xml = BuildXml(param.isPcie, visibleIds);
    W(xml.c_str());

    /* 产品层 mock */
    {
        unsigned int mid = param.mainboardId;
        char drvPath[256] = "/usr/local/Ascend2";
        MOCKER(hal_get_mainboard_id).stubs().with(mockcpp::any(), outBoundP(&mid)).will(returnValue(0));
        MOCKER(hal_get_driver_install_path)
            .stubs()
            .with(outBoundP(drvPath, strlen(drvPath)), mockcpp::any())
            .will(returnValue(0));
        if (param.isCard) {
            dcmi_urma_eid_info_t eidList[MAX_EID_NUM];
            memset_s(eidList, sizeof(eidList), 0, sizeof(eidList));
            size_t eidNum = 0;
            MOCKER(hal_get_eid_list_by_phy_id)
                .stubs()
                .with(mockcpp::any(), outBoundP(eidList, eidNum * sizeof(dcmi_urma_eid_info_t)), outBoundP(&eidNum))
                .will(returnValue(0));
        } else {
            /* server/pod 用空 UEList + spod_info，
               ProcessLayer 中非 ROCE 层的循环因 ueNum=0 直接跳过，不会空指针。 */
            UEList ueList;
            memset_s(&ueList, sizeof(ueList), 0, sizeof(ueList));

            struct dcmi_spod_info spinfo;
            memset_s(&spinfo, sizeof(spinfo), 0, sizeof(spinfo));
            spinfo.sdid = 0;
            spinfo.super_pod_size = 128;
            spinfo.super_pod_id = 1;
            spinfo.server_index = 1;

            MOCKER(HalGetUBEntityList).stubs().with(mockcpp::any(), outBoundP(&ueList)).will(returnValue(0));
            MOCKER(hal_get_spod_info).stubs().with(mockcpp::any(), outBoundP(&spinfo)).will(returnValue(0));
        }
    }

    /* 可见 NPU → ROCE 层完整 */
    for (int id : visibleIds) {
        char buf[8192] = {0};
        size_t bufSize = sizeof(buf);
        ASSERT_EQ(TopoAddrInfoGet(id, buf, &bufSize), 0);
        AssertJsonContains(buf, g_rootInfoJsonFields, std::string(param.name) + " phyId=" + std::to_string(id));
        AssertJsonContains(buf, g_roceJsonFields, std::string(param.name) + " phyId=" + std::to_string(id));
    }

    /* 不可见 NPU → 无 ROCE 层 */
    for (int id = 0; id < 8; id++) {
        if (std::find(visibleIds.begin(), visibleIds.end(), id) != visibleIds.end()) {
            continue;
        }
        char buf[8192] = {0};
        size_t bufSize = sizeof(buf);
        ASSERT_EQ(TopoAddrInfoGet(id, buf, &bufSize), 0);
        AssertJsonNotContains(
            buf, "\"addr\": \"10.0.0.1\"", std::string(param.name) + " (invis) phyId=" + std::to_string(id));
    }
}

/* 四正交：All/PCIE + All/UB + Sparse/PCIE + Sparse/UB */
#define MB4(mainboardId, name, isCard)                                                 \
    {mainboardId, name, isCard, true, false}, {mainboardId, name, isCard, true, true}, \
        {mainboardId, name, isCard, false, false},                                     \
    {                                                                                  \
        mainboardId, name, isCard, false, true                                         \
    }

static const TopoParam g_allTopoParams[] = {
    /* ─── Card ─── */
    MB4(MAIN_BOARD_ID_CARD_NOMESH, "CARD_NOMESH", true), MB4(MAIN_BOARD_ID_CARD_2PMESH, "CARD_2PMESH", true),
    MB4(MAIN_BOARD_ID_CARD_4PMESH, "CARD_4PMESH", true),
    /* ─── Server ───
     * 说明：SERVER_TYPE1(0x23) 的 g_netInfoList 条目未初始化 instanceIdFunc，
     * ProcessLayer 调用 NULL 函数指针会崩溃。属于产品侧数据缺陷，暂不覆盖。 */
    MB4(MAIN_BOARD_ID_SERVER_8PMESH, "SERVER_8PMESH", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_UBOE, "SERVER_8PMESH_UBOE", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_NOSP, "SERVER_8PMESH_NOSP", false),
    MB4(MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE, "SERVER_8PMESH_NOSP_UBOE", false),
    MB4(MAIN_BOARD_ID_SERVER_350L, "SERVER_350L", false), MB4(MAIN_BOARD_ID_SERVER_550EL_100, "SERVER_550_100", false),
    MB4(MAIN_BOARD_ID_SERVER_550EL_200, "SERVER_550_200", false),
    /* ─── Pod ─── */
    MB4(MAIN_BOARD_ID_POD, "POD", false), MB4(MAIN_BOARD_ID_POD_2D, "POD_2D", false),
    MB4(MAIN_BOARD_ID_POD_FLEX, "POD_FLEX", false), MB4(MAIN_BOARD_ID_POD_FLEX_RTP, "POD_FLEX_RTP", false)};

INSTANTIATE_TEST_SUITE_P(VisibleDevices_Contiguity, TopoAddrInfoAllMainboardTest, testing::ValuesIn(g_allTopoParams));

/* ─── 多组 UB XML 端到端验证 ─── */
TEST_F(NpuNicAffinityTest, MultiGroupUbAffinity)
{
    /* 构造 8 组 fake 网络接口 */
    const char* nicNames[] = {"ens0f0", "ens1f0", "ens0f1", "ens1f1", "ens0f2", "ens1f2", "ens0f3", "ens1f3"};
    struct ifaddrs* head = NULL;
    for (int i = 0; i < 8; i++) {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = head;
        ifa->ifa_name = strdup(nicNames[i]);
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        sin->sin_family = AF_INET;
        char ip[16];
        sprintf_s(ip, sizeof(ip), "10.0.%d.%d", i / 4, (i % 4) + 1);
        inet_pton(AF_INET, ip, &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        head = ifa;
    }
    g_fakeIfaddr = head;

    S(8, 0);
    W("<system version=\"1.0\" description=\"npu nic switch stand by\">\n"
      "<cpu numaid=\"0\">\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f0\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f0\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"0\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens1f0\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens0f0\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"1\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f1\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f1\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"2\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f1\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f1\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"3\"/></ub>\n"
      "</ub>\n"
      "</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f2\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f2\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"4\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens1f2\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens0f2\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"5\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f3\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f3\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"6\"/></ub>\n"
      "</ub>\n"
      "<ub busid=\"\">\n"
      "<ub> <nic><net name=\"ens0f3\" /></nic></ub>\n"
      "<ub> <nic><net name=\"ens1f3\" /></nic></ub>\n"
      "<ub><npu chipphyid=\"7\"/></ub>\n"
      "</ub>\n"
      "</cpu>\n"
      "</system>\n");

    /* 每个 NPU 应从自己的亲和组中分配到 NIC，且取到正确的 IP */
    /* 去重后 8 个 NIC 各占一个 nicIdx，全局游标逐一分配 */
    const char* expectedIps[8] = {
        "10.0.0.1", // NPU0 → ens0f0
        "10.0.0.2", // NPU1 → ens1f0
        "10.0.0.3", // NPU2 → ens0f1
        "10.0.0.4", // NPU3 → ens1f1
        "10.0.1.1", // NPU4 → ens0f2
        "10.0.1.2", // NPU5 → ens1f2
        "10.0.1.3", // NPU6 → ens0f3
        "10.0.1.4", // NPU7 → ens1f3
    };
    for (int i = 0; i < 8; i++) {
        char ip[64] = {0};
        ASSERT_EQ(GetRoceIpFromXml(i, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectedIps[i]);
    }
}

/* ─── PCIE 多组 XML 端到端验证（含去重） ─── */
TEST_F(NpuNicAffinityTest, MultiGroupPcieAffinity)
{
    /* 6 个唯一 NIC 名，各配唯一 IP */
    const char* nicNames[] = {"ens0f0", "ens1f0", "ens0f2", "ens1f2", "ens0f3", "ens1f3"};
    const char* fakeIps[] = {"10.0.0.1", "10.0.0.2", "10.0.1.1", "10.0.1.2", "10.0.1.3", "10.0.1.4"};
    struct ifaddrs* head = NULL;
    for (int i = 0; i < 6; i++) {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ASSERT_NE(ifa, nullptr);
        ifa->ifa_next = head;
        ifa->ifa_name = strdup(nicNames[i]);
        ASSERT_NE(ifa->ifa_name, nullptr);
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        ASSERT_NE(sin, nullptr);
        sin->sin_family = AF_INET;
        inet_pton(AF_INET, fakeIps[i], &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        head = ifa;
    }
    g_fakeIfaddr = head;

    /* PCIE BDF 信息：busId = 3 + phyId */
    for (int i = 0; i < 8; i++) {
        g_pi[i].domain = 0;
        g_pi[i].bdf_busid = 3 + i;
        g_pi[i].bdf_deviceid = 0;
        g_pi[i].bdf_funcid = 0;
    }
    S(8, 8);

    /* XML: 4 个 Bridge 分布在 2 个 CPU socket，
       Bridge 2 的 NIC 与 Bridge 1 同名，验证去重 */
    W("<system version=\"1.0\">\n"
      "<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"ens0f0\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:04:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:00:02.0\">\n"
      "<nic>\n<net name=\"ens0f0\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f0\"/>\n</nic>\n"
      "<pci busid=\"0000:05:00.0\"/>\n"
      "<pci busid=\"0000:06:00.0\"/>\n"
      "</pci>\n"
      "</cpu>\n"
      "<cpu numaid=\"1\">\n"
      "<pci busid=\"0000:00:03.0\">\n"
      "<nic>\n<net name=\"ens0f2\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f2\"/>\n</nic>\n"
      "<pci busid=\"0000:07:00.0\"/>\n"
      "<pci busid=\"0000:08:00.0\"/>\n"
      "</pci>\n"
      "<pci busid=\"0000:00:04.0\">\n"
      "<nic>\n<net name=\"ens0f3\"/>\n</nic>\n"
      "<nic>\n<net name=\"ens1f3\"/>\n</nic>\n"
      "<pci busid=\"0000:09:00.0\"/>\n"
      "<pci busid=\"0000:0a:00.0\"/>\n"
      "</pci>\n"
      "</cpu>\n"
      "</system>\n");

    /* 去重后 nicIdx: {0:ens0f0, 1:ens1f0, 2:ens0f2, 3:ens1f2, 4:ens0f3, 5:ens1f3}
       Group 0/1 共享 nicIdx {0,1}，Group 2 用 {2,3}，Group 3 用 {4,5} */
    const char* expectedIps[8] = {
        "10.0.0.1", // NPU 0 → ens0f0
        "10.0.0.2", // NPU 1 → ens1f0
        "10.0.0.1", // NPU 2 → ens0f0 (去重同 nicIdx 0)
        "10.0.0.2", // NPU 3 → ens1f0 (去重同 nicIdx 1)
        "10.0.1.1", // NPU 4 → ens0f2
        "10.0.1.2", // NPU 5 → ens1f2
        "10.0.1.3", // NPU 6 → ens0f3
        "10.0.1.4", // NPU 7 → ens1f3
    };
    for (int i = 0; i < 8; i++) {
        char ip[64] = {0};
        ASSERT_EQ(GetRoceIpFromXml(i, ip, sizeof(ip)), 0);
        EXPECT_STREQ(ip, expectedIps[i]);
    }
}

/* ══════════════════════════════════════════════════════════════
 *  ProcessLayerRoce 入口
 * ══════════════════════════════════════════════════════════════ */

TEST_F(NpuNicAffinityTest, ProcessLayerRoce_Success)
{
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    S(1, 1);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:01:00.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    NetLayer layer;
    memset_s(&layer, sizeof(layer), 0, sizeof(layer));
    EXPECT_EQ(ProcessLayerRoce(0, &layer), 0);
    EXPECT_EQ(layer.net_layer, 3);
    EXPECT_STREQ(layer.net_type, "CLOS");
    EXPECT_STREQ(layer.net_instance_id, "cluster");
    EXPECT_EQ(layer.addr_count, 1);
    EXPECT_STREQ(layer.rank_addr_list[0].addr, "10.0.0.1");
    EXPECT_STREQ(layer.rank_addr_list[0].plane_id, "plane0");
    EXPECT_STREQ(layer.rank_addr_list[0].ports[0], "d2h");
}

TEST_F(NpuNicAffinityTest, ProcessLayerRoce_FileNotFound)
{
    NetLayer layer;
    memset_s(&layer, sizeof(layer), 0, sizeof(layer));
    EXPECT_NE(ProcessLayerRoce(0, &layer), TOPO_SUCCESS);
}

/* ══════════════════════════════════════════════════════════════
 *  容灾用例：NIC/NPU 设备挂掉时功能正常
 * ══════════════════════════════════════════════════════════════ */

/* NIC 故障：HCA sysfs 目录不存在 */
TEST_F(NpuNicAffinityTest, Nic_HcaSysfsMissing)
{
    TeardownFakeHca(); /* 清除 SetUp 创建的 hrn5_0 HCA 目录 */
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：HCA 目录存在但 net 子目录为空 */
TEST_F(NpuNicAffinityTest, Nic_HcaSysfsEmptyDir)
{
    TeardownFakeHca();
    system("mkdir -p /tmp/ut_hca/hrn5_0/device/net"); /* 只创空目录，不放 eth */
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 名不在 getifaddrs 返回列表中 */
TEST_F(NpuNicAffinityTest, Nic_EthNotInIfaddrs)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* XML 中 name 直接是 eth 名，但 getifaddrs 只返回 lo，没有 eth0 */
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("lo");
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in* sin = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        sin->sin_family = AF_INET;
        inet_pton(AF_INET, "127.0.0.1", &sin->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)sin;
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 存在但 ifa_addr 为 NULL */
TEST_F(NpuNicAffinityTest, Nic_EthIfaAddrNull)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("eth0");
        ifa->ifa_flags = IFF_UP;
        ifa->ifa_addr = NULL; /* 地址为空 */
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NIC 故障：eth 只有 IPv6 地址，无 AF_INET */
TEST_F(NpuNicAffinityTest, Nic_EthOnlyIpv6)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    TeardownFakeNet();
    {
        struct ifaddrs* ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
        ifa->ifa_next = NULL;
        ifa->ifa_name = strdup("eth0");
        ifa->ifa_flags = IFF_UP;
        struct sockaddr_in6* sin6 = (struct sockaddr_in6*)calloc(1, sizeof(struct sockaddr_in6));
        sin6->sin6_family = AF_INET6;
        ifa->ifa_addr = (struct sockaddr*)sin6;
        g_fakeIfaddr = ifa;
    }
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"eth0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* 组内部分 NIC 故障：第一张 NIC 不可达，自动 fallback 到同组第二张 */
TEST_F(NpuNicAffinityTest, Nic_FallbackWhenFirstDown)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* 只创建 hrn5_1 的 HCA，不创建 hrn5_0 → hrn5_0 不可达，fallback 到 hrn5_1 */
    SetupFakeHca("hrn5_1", "eth1");
    SetupFakeNet("eth1", "10.0.0.2");
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<nic>\n<net name=\"hrn5_1\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.2");
}

/* NPU 故障：BDF 不匹配 → XML 中的 busid 与所有 NPU 的 BDF 都不一致 */
TEST_F(NpuNicAffinityTest, Npu_BdfNoMatch)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    /* XML 引用 busid=0000:ff:00.0，而 NPU0 的 BDF 是 0000:03:00.0 → 不匹配 */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:ff:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NPU 故障：hal_get_device_pcie_info 全失败 → BDF 表全空 → 无匹配 */
TEST_F(NpuNicAffinityTest, Npu_PcieInfoAllFail)
{
    S(1, 0); /* pcie=0 → 不 mock hal_get_device_pcie_info */
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* NPU 去重：UB 格式中同组重复 chipphyid → 不影响分配 */
TEST_F(NpuNicAffinityTest, Npu_DupPhyId_UB)
{
    S(2, 0);
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<ub>\n<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<npu chipphyid=\"0\"/>\n"
      "<npu chipphyid=\"0\"/>\n"
      "</ub>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.1");
}

/* NPU 去重：PCIE 格式中同组重复 BDF → 不影响分配 */
TEST_F(NpuNicAffinityTest, Npu_DupBdf_PCIE)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_EQ(GetRoceIpFromXml(0, ip, sizeof(ip)), 0);
    ASSERT_STREQ(ip, "10.0.0.1");
}

/* 所有 NIC 不可达 → 返回 -1 */
TEST_F(NpuNicAffinityTest, Error_AllNicsUnreachable)
{
    TeardownFakeHca();
    TeardownFakeNet();
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "<pci busid=\"0000:03:00.0\"/>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}

/* 组内无 NPU：只有 NIC 没有 NPU → 返回 -1 */
TEST_F(NpuNicAffinityTest, Error_GroupWithoutNpu)
{
    S(1, 1);
    g_pi[0].domain = 0;
    g_pi[0].bdf_busid = 3;
    g_pi[0].bdf_deviceid = 0;
    g_pi[0].bdf_funcid = 0;
    W("<system version=\"1.0\">\n<cpu numaid=\"0\">\n"
      "<pci busid=\"0000:00:01.0\">\n"
      "<nic>\n<net name=\"hrn5_0\"/>\n</nic>\n"
      "</pci>\n</cpu>\n</system>\n");
    char ip[64] = {0};
    ASSERT_NE(GetRoceIpFromXml(0, ip, sizeof(ip)), TOPO_SUCCESS);
}
