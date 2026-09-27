/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_HCCL_TEAM_COMMON_H
#define UT_HCCL_TEAM_COMMON_H

// hccl team 系列 UT（ut_hccl_team_c_adpt.cc / ut_hccl_team_mgr.cc）共享的公共段：
// private/protected 展开下的 coll_comm.h include、公共构造辅助 BuildV2HcclComm。
// include 本头前需已 include hccl_api_base_test.h（BaseInit/UtInitHcclCommConfig）与
// mockcpp（MOCKER 宏）。

#include "hccl_comm_pub.h"
#include "my_rank.h"
#include "llt_hccl_stub_rank_graph.h"

// 两个 UT 均需访问 CollComm 私有成员（单例状态/私有方法），统一在此展开宏包含
#define private public
#define protected public
#include "coll_comm.h"
#undef protected
#undef private

namespace hccl_ut {

// 构造真实 hcclComm（rank=1, rankSize=2）。rankGraphV2 必须由 fixture 持有（InitCollComm 只存裸指针）。
inline void BuildV2HcclComm(std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));

    void* commV2 = reinterpret_cast<void*>(0x2000);
    RankGraphStub rankGraphStub;
    rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1024;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = reinterpret_cast<void*>(0x1000);
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1; // 非CCU模式
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    ASSERT_EQ(ret, HCCL_SUCCESS);
}

} // namespace hccl_ut

#endif // UT_HCCL_TEAM_COMMON_H
