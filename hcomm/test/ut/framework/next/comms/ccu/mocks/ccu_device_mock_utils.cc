/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_device_mock_utils.h"

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public

#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_pfe_cfg_mgr.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_specs.h"
#include "ccu_comp.h"

#include "hcomm_adapter_hccp.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "hccp_ctx.h"

#include "ccu_drv_handle_mock.h"
#include "ccu_dev_mgr_mock.h"

#undef protected
#undef private

HcclResult InitMockCcuResourcesForDevice(int32_t devLogicId, hcomm::CcuVersion ccuVersion)
{
    auto& ccuResSpecs = hcomm::CcuResSpecifications::GetInstance(devLogicId);
    ccuResSpecs.initFlag_ = true;
    ccuResSpecs.ccuVersion_ = ccuVersion;
    ccuResSpecs.serveMode_ = hcomm::ServeMode::NORMAL;
    for (uint8_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; dieId++) {
        ccuResSpecs.dieEnableFlags_[dieId] = true;

        ccuResSpecs.resSpecs_[dieId].loopEngineNum = 200;

        ccuResSpecs.resSpecs_[dieId].msNum = 1536;
        ccuResSpecs.resSpecs_[dieId].ckeNum = 1024;

        ccuResSpecs.resSpecs_[dieId].xnNum = 3072;

        ccuResSpecs.resSpecs_[dieId].gsaNum = 3072;

        ccuResSpecs.resSpecs_[dieId].instructionNum = 32768;
        ccuResSpecs.resSpecs_[dieId].missionNum = 16;

        ccuResSpecs.resSpecs_[dieId].channelNum = 128;

        ccuResSpecs.resSpecs_[dieId].jettyNum = 128;
        ccuResSpecs.resSpecs_[dieId].wqeBBNum = 4096;

        ccuResSpecs.resSpecs_[dieId].pfeNum = 10;

        ccuResSpecs.resSpecs_[dieId].resourceAddr = 0xE7FFBF800000;
    }

    CHK_RET(hcomm::CcuPfeCfgMgr::GetInstance(devLogicId).Init());
    CHK_RET(hcomm::CcuComponent::GetInstance(devLogicId).Init());
    CHK_RET(hcomm::CcuResBatchAllocator::GetInstance(devLogicId).Init());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MockCcuResourcesDefault(int32_t devLogicId, hcomm::CcuVersion ccuVersion)
{
    MockCcuDrvHandle();
    MockCcuDevMgr();
    return InitMockCcuResourcesForDevice(devLogicId, ccuVersion);
}

static int RaGetDevEidInfoListStub(struct RaInfo info, struct HccpDevEidInfo info_list[], unsigned int* num)
{
    strcpy(info_list[0].name, "udma2");
    info_list[0].dieId = 0;
    info_list[0].chipId = info.phyId;
    info_list[0].funcId = 2;
    info_list[0].eid.in4.addr = 167772382;

    strcpy(info_list[1].name, "udma2");
    info_list[1].dieId = 0;
    info_list[1].chipId = info.phyId;
    info_list[1].funcId = 2;
    info_list[1].eid.in4.addr = 167772383;

    strcpy(info_list[2].name, "udma3");
    info_list[2].dieId = 1;
    info_list[2].chipId = info.phyId;
    info_list[2].funcId = 3;
    info_list[2].eid.in4.addr = 469762271;

    return 0;
}

// 为 ccu 两个die添加网络通信设备
void MockCcuNetworkDeviceDefault(int32_t devPhyId)
{
    // 从驱动查询的网络设备用于：
    // 1. ccu 创建环回jetty，对于期望启用ccu功能的die，至少需要1个rtp 启用的eid
    // 2. ccu 建链申请channel资源，通过eid查询所属的die，匹配不到时将会失败
    constexpr unsigned int fakeEidNum = 3;
    unsigned int eidNum = fakeEidNum;
    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&eidNum)).will(returnValue(0));

    MOCKER(RaGetDevEidInfoList).stubs().will(invoke(RaGetDevEidInfoListStub));

    MOCKER(hcomm::HccpGetUboeFlagEnable).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hcomm::HccpCheckUboeSupported).stubs().will(returnValue(false));

    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByIp)
        .stubs()
        .will(returnValue((void*)0x12345678)); // 大部分场景ctxHandle非空即可

    MOCKER_CPP(&Hccl::RdmaHandleManager::IsHandleValid).stubs().will(returnValue(true));

    std::pair<Hccl::TokenIdHandle, uint32_t> fakeTokenInfo = std::make_pair(0x12345678, 1);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetTokenIdInfo)
        .stubs()
        .will(returnValue(fakeTokenInfo)); // 打桩保证内存注册成功

    MOCKER_CPP(&Hccl::RdmaHandleManager::GetRtpEnable)
        .stubs()
        .will(returnValue(false)) // 打桩覆盖环回选择不同端口
        .then(returnValue(true));
}
