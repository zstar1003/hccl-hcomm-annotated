/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_SPECS_TEST_COMMON_H
#define CCU_RES_SPECS_TEST_COMMON_H

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>

#define private public
#define protected public
#include "ccu_res_specs_legacy.h"
#include "hccl_common_v2.h"
#undef private
#undef protected

using namespace Hccl;

class CcuResSpecsTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResSpecsTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuResSpecsTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResSpecsTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuResSpecsTest TearDown" << std::endl;
    }
};

inline HcclResult
AllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    ckeInfos.clear();
    ResInfo ckeInfo(0, num);
    ckeInfos.push_back(ckeInfo);
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult
AllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    xnInfos.clear();
    ResInfo xnInfo(0, num);
    xnInfos.push_back(xnInfo);
    return HcclResult::HCCL_SUCCESS;
}

inline void MockDoOnce(bool withUbToken = false)
{
    CustomChannelInfoIn inBuff{};
    inBuff.data.dataInfo.dataLen = 512;
    inBuff.data.dataInfo.dataArraySize = 64;

    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)),
            outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)));
    MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
    MOCKER(HrtGetSocVer).stubs().with().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(0));
    DevType deviceType = DevType::DevType::DEV_TYPE_910A3;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(deviceType));

    std::unique_ptr<RdmaHandle> handle = std::make_unique<RdmaHandle>();
    RdmaHandle handlePtr = handle.get();
    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any()).will(returnValue(handlePtr));
    // avoid RdmaHandle null pointer

    vector<HrtDevEidInfo> eidInfoListStbu;
    HrtDevEidInfo eidInfo;
    eidInfo.name = "udma0";
    eidInfo.dieId = 0;
    eidInfo.funcId = 3;

    eidInfoListStbu.push_back(eidInfo);

    MOCKER(HrtRaGetDevEidInfoList).stubs().with(mockcpp::any()).will(returnValue(eidInfoListStbu));
    if (withUbToken) {
        MOCKER(GetUbToken).stubs().will(returnValue(1));
    }
}

inline void MockCcuDriverInterfaceReturnDieEnableStub(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    CcuDieInfo dieInfo;
    dieInfo.enableFlag = true;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(CcuDieInfo), &dieInfo, sizeof(CcuDieInfo));
}

inline void MockCcuOneDieResource(
    CcuResSpecifications& ccuResSpecs, const int32_t devLogicId, const uint8_t dieId, const CcuVersion ccuVersion)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM));

    ccuResSpecs.Init();
    ccuResSpecs.ccuVersion = ccuVersion;
    ccuResSpecs.dieEnableFlags[dieId] = true;

    ccuResSpecs.resSpecs[dieId].loopEngineNum = 200;

    ccuResSpecs.resSpecs[dieId].msNum = 1536;
    ccuResSpecs.resSpecs[dieId].ckeNum = 1024;

    ccuResSpecs.resSpecs[dieId].xnNum = 3072;

    ccuResSpecs.resSpecs[dieId].gsaNum = 3072;

    ccuResSpecs.resSpecs[dieId].instructionNum = 32768;
    ccuResSpecs.resSpecs[dieId].missionNum = 16;

    ccuResSpecs.resSpecs[dieId].channelNum = 128;

    ccuResSpecs.resSpecs[dieId].jettyNum = 128;
    ccuResSpecs.resSpecs[dieId].wqeBBNum = 4096;

    ccuResSpecs.resSpecs[dieId].pfeNum = 10;

    ccuResSpecs.resSpecs[dieId].resourceAddr = 0xE7FFBF800000;

    ccuResSpecs.resSpecs[dieId].msId = 1;       // 非0即可
    ccuResSpecs.resSpecs[dieId].missionKey = 1; // 非0即可
}

#endif // CCU_RES_SPECS_TEST_COMMON_H
