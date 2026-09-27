/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

extern "C" {
#include "ut_dispatch.h"
#include "tc_rdma_agent.h"
#include "tc_ra_ping.h"
#include "tc_ra_peer.h"
#include "tc_ra_adp_tlv.h"
#include "tc_ra_tlv.h"
}

#include "ra_rs_comm.h"
#include "ra_hdc.h"
#include <stdio.h>
#include "gtest/gtest.h"
#include "tc_adp.h"
#include "tc_hdc.h"
#include "tc_host.h"
#include "tc_ra_ctx.h"
#include "tc_ra_nda.h"
#include "tc_ra_async.h"

using namespace std;

class RdmaAgent : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--RoCE RdmaAgent SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--RoCE RdmaAgent TearDown--\033[0m" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_M(RdmaAgent, TcHdcSocketBatchConnect);
TEST_M(RdmaAgent, TcHdcSocketListenStart);
TEST_M(RdmaAgent, TcIfaddr);
TEST_M(RdmaAgent, TcHost);
TEST_M(RdmaAgent, TcPeer);
TEST_M(RdmaAgent, TcHdcInit);
TEST_M(RdmaAgent, TcHdcInitFail);
TEST_M(RdmaAgent, TcHdcDeinitFail);
TEST_M(RdmaAgent, TcHdcSocketBatchClose);
TEST_M(RdmaAgent, TcHdcSocketBatchAbort);
TEST_M(RdmaAgent, TcHdcSocketListenStop);
TEST_M(RdmaAgent, TcHdcGetSockets);
TEST_M(RdmaAgent, TcHdcSocketSend);
TEST_M(RdmaAgent, TcHdcSocketRecv);
TEST_M(RdmaAgent, TcHdcQpCreateDestroy);
TEST_M(RdmaAgent, TcHdcGetQpStatus);
TEST_M(RdmaAgent, TcHdcQpConnectAsync);

/* pingMesh ut cases */
TEST_M(RdmaAgent, TcRaRsPingInit);
TEST_M(RdmaAgent, TcRaRsPingTargetAdd);
TEST_M(RdmaAgent, TcRaRsPingTaskStart);
TEST_M(RdmaAgent, TcRaRsPingGetResults);
TEST_M(RdmaAgent, TcRaRsPingTaskStop);
TEST_M(RdmaAgent, TcRaRsPingTargetDel);
TEST_M(RdmaAgent, TcRaRsPingDeinit);

TEST_M(RdmaAgent, TcHdcSocketInit);
TEST_M(RdmaAgent, TcHdcSocketDeinit);
TEST_M(RdmaAgent, TcHdcRdevInit);
TEST_M(RdmaAgent, TcHdcRdevDeinit);
TEST_M(RdmaAgent, TcHdcSocketWhiteListAdd);
TEST_M(RdmaAgent, TcHdcSocketWhiteListDel);
TEST_M(RdmaAgent, TcHdcGetIfaddrs);
TEST_M(RdmaAgent, TcHdcGetIfaddrsV2);
TEST_M(RdmaAgent, TcHdcGetIfnum);
TEST_M(RdmaAgent, TcHdcMessageProcessFail);
TEST_M(RdmaAgent, TcHdcSocketRecvFail);
TEST_M(RdmaAgent, TcRaHdcSendWrlistExtInit);
TEST_M(RdmaAgent, TcRaHdcSendWrlistExt);
TEST_M(RdmaAgent, TcRaHdcSendNormalWrlist);

TEST_M(RdmaAgent, TcHccpInit);
TEST_M(RdmaAgent, TcHccpInitFail);
TEST_M(RdmaAgent, TcHccpDeinitFail);
TEST_M(RdmaAgent, TcSocketConnect);
TEST_M(RdmaAgent, TcSocketClose);
TEST_M(RdmaAgent, TcSocketAbort);
TEST_M(RdmaAgent, TcSocketListenStart);
TEST_M(RdmaAgent, TcSocketListenStop);
TEST_M(RdmaAgent, TcSocketInfo);
TEST_M(RdmaAgent, TcSocketSend);
TEST_M(RdmaAgent, TcSocketRecv);
TEST_M(RdmaAgent, TcSocketInit);
TEST_M(RdmaAgent, TcSocketDeinit);
TEST_M(RdmaAgent, TcSetTsqpDepth);
TEST_M(RdmaAgent, TcGetTsqpDepth);
TEST_M(RdmaAgent, TcQpCreate);
TEST_M(RdmaAgent, TcQpDestroy);
TEST_M(RdmaAgent, TcQpStatus);
TEST_M(RdmaAgent, TcQpInfo);
TEST_M(RdmaAgent, TcQpConnect);
TEST_M(RdmaAgent, TcRaRsRemapMr);
TEST_M(RdmaAgent, TcRaRsGetTlsEnable0);
TEST_M(RdmaAgent, TcMrReg);
TEST_M(RdmaAgent, TcMrDreg);
TEST_M(RdmaAgent, TcSendWr);
TEST_M(RdmaAgent, TcSendWrlist);
TEST_M(RdmaAgent, TcRdevInit);
TEST_M(RdmaAgent, TcRdevDeinit);
TEST_M(RdmaAgent, TcGetNotifyBa);
TEST_M(RdmaAgent, TcSetPid);
TEST_M(RdmaAgent, TcGetVnicIp);
TEST_M(RdmaAgent, TcSocketWhiteListAdd);
TEST_M(RdmaAgent, TcSocketWhiteListDel);
TEST_M(RdmaAgent, TcGetIfaddrs);
TEST_M(RdmaAgent, TcGetIfaddrsV2);
TEST_M(RdmaAgent, TcGetIfnum);
TEST_M(RdmaAgent, TcGetInterfaceVersion);
TEST_M(RdmaAgent, TcTlvInit);
TEST_M(RdmaAgent, TcTlvDeinit);
TEST_M(RdmaAgent, TcTlvRequest);
TEST_M(RdmaAgent, TcRaRsTestCtxOps);

TEST_M(RdmaAgent, TcHostAbnormalQpModeTest);
TEST_M(RdmaAgent, TcRaPeerSocketWhiteListAdd01);
TEST_M(RdmaAgent, TcRaPeerSocketWhiteListAdd02);
TEST_M(RdmaAgent, TcRaPeerRdevInit01);
TEST_M(RdmaAgent, TcRaPeerRdevInit02);
TEST_M(RdmaAgent, TcRaPeerRdevInit03);
TEST_M(RdmaAgent, TcRaPeerRdevInit04);
TEST_M(RdmaAgent, TcRaPeerRdevDeinit01);
TEST_M(RdmaAgent, TcRaPeerRdevDeinit02);
TEST_M(RdmaAgent, TcRaPeerRdevDeinit03);
TEST_M(RdmaAgent, TcRaPeerRdevGetPortStatusSucc);
TEST_M(RdmaAgent, TcRaPeerRdevGetPortStatusFail);
TEST_M(RdmaAgent, TcRaPeerSocketWhiteListDel);
TEST_M(RdmaAgent, TcRaPeerSocketBatchConnect);
TEST_M(RdmaAgent, TcRaPeerSocketBatchAbort);
TEST_M(RdmaAgent, TcRaPeerSocketListenStart01);
TEST_M(RdmaAgent, TcRaPeerSocketListenStart02);
TEST_M(RdmaAgent, TcRaPeerSocketListenStop);
TEST_M(RdmaAgent, TcRaPeerSetRsConnParam);
TEST_M(RdmaAgent, TcRaInetPton01);
TEST_M(RdmaAgent, TcRaInetPton02);
TEST_M(RdmaAgent, TcRaSocketInit);
TEST_M(RdmaAgent, TcRaSocketInitV1);
TEST_M(RdmaAgent, TcRaSendWrlist);
TEST_M(RdmaAgent, TcRaRdevInit);
TEST_M(RdmaAgent, TcRaRdevGetPortStatus);
TEST_M(RdmaAgent, TcRaHdcRdevDeinit);
TEST_M(RdmaAgent, TcRaHdcSocketWhiteListAdd);
TEST_M(RdmaAgent, TcRaHdcSocketWhiteListDel);
TEST_M(RdmaAgent, TcRaHdcSocketAcceptCreditAdd);
TEST_M(RdmaAgent, TcRaHdcRdevInit);
TEST_M(RdmaAgent, TcRaHdcInitApart);
TEST_M(RdmaAgent, TcRaHdcQpDestroy);
TEST_M(RdmaAgent, TcRaHdcQpDestroy01);
TEST_M(RdmaAgent, TcRaGetSocketConnectInfo);
TEST_M(RdmaAgent, TcRaGetSocketListenInfo);
TEST_M(RdmaAgent, TcRaGetSocketListenResult);
TEST_M(RdmaAgent, TcPeerFail);
TEST_M(RdmaAgent, TcRaPeerInitFail001);
TEST_M(RdmaAgent, TcRaPeerSocketDeinit001);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit001);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit002);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit003);

TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit005);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrInit006);

TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit001);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit002);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit003);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit004);
TEST_M(RdmaAgent, TcHostNotifyBaseAddrUninit005);

TEST_M(RdmaAgent, TcRaPeerSendWrlist);
TEST_M(RdmaAgent, TcRaPeerSendWrlist001);

TEST_M(RdmaAgent, TcRaRecvWrlist);
TEST_M(RdmaAgent, TcRaSetQpAttrQos);
TEST_M(RdmaAgent, TcRaSetQpAttrTimeout);
TEST_M(RdmaAgent, TcRaSetQpAttrRetryCnt);
TEST_M(RdmaAgent, TcRaGetCqeErrInfo);
TEST_M(RdmaAgent, TcRaRdevGetCqeErrInfoList);
TEST_M(RdmaAgent, TcRaRsGetIfnum);

TEST_M(RdmaAgent, TcRaPeerEpollCtlAdd);
TEST_M(RdmaAgent, TcRaPeerSetTcpRecvCallback);
TEST_M(RdmaAgent, TcRaPeerEpollCtlMod);
TEST_M(RdmaAgent, TcRaPeerEpollCtlDel);
TEST_M(RdmaAgent, TcRaGetQpContext);

TEST_M(RdmaAgent, TcHostRaSendWrlistExt);
TEST_M(RdmaAgent, TcHostRaSendNormalWrlist);
TEST_M(RdmaAgent, TcRaCreateCq);
TEST_M(RdmaAgent, TcRaCreateNotmalQp);
TEST_M(RdmaAgent, TcRaPeerCqCreate);
TEST_M(RdmaAgent, TcRaPeerNormalQpCreate);
TEST_M(RdmaAgent, TcRaCreateCompChannel);
TEST_M(RdmaAgent, TcRaCreateSrq);

TEST_M(RdmaAgent, TcRaCreateEventHandle);
TEST_M(RdmaAgent, TcRaCtlEventHandle);
TEST_M(RdmaAgent, TcRaWaitEventHandle);
TEST_M(RdmaAgent, TcRaDestroyEventHandle);
TEST_M(RdmaAgent, TcRaPeerCreateEventHandle);
TEST_M(RdmaAgent, TcRaPeerCtlEventHandle);
TEST_M(RdmaAgent, TcRaPeerWaitEventHandle);
TEST_M(RdmaAgent, TcRaPeerDestroyEventHandle);
TEST_M(RdmaAgent, TcRaLoopbackQpCreate);
TEST_M(RdmaAgent, TcRaPeerLoopbackQpCreate);
TEST_M(RdmaAgent, TcRaPeerLoopbackSingleQpCreate);
TEST_M(RdmaAgent, TcRaPeerSetQpLbValue);
TEST_M(RdmaAgent, TcRaPeerGetQpLbValue);
TEST_M(RdmaAgent, TcRaPeerGetLbMax);
TEST_M(RdmaAgent, TcRaPeerGetHccnCfg);
TEST_M(RdmaAgent, TcRaPollCq);
TEST_M(RdmaAgent, TcHdcRecvWrlist);
TEST_M(RdmaAgent, TcHdcPollCq);
TEST_M(RdmaAgent, TcHdcGetLiteSupport);
TEST_M(RdmaAgent, TcRaRdevGetSupportLite);
TEST_M(RdmaAgent, TcRaRdevGetHandle);
TEST_M(RdmaAgent, TcRaIsFirstOrLastUsed);

TEST_M(RdmaAgent, TcRaRsSocketPortIsUse);
TEST_M(RdmaAgent, TcGetVnicIpInfos);
TEST_M(RdmaAgent, TcRaRsGetVnicIpInfosV1);
TEST_M(RdmaAgent, TcRaRsGetVnicIpInfos);
TEST_M(RdmaAgent, TcRaRsTypicalMrReg);
TEST_M(RdmaAgent, TcRaRsTypicalQpCreate);
TEST_M(RdmaAgent, TcRaSocketBatchAbort);
TEST_M(RdmaAgent, TcRaHdcLiteCtxInit);
TEST_M(RdmaAgent, TcRaRemapMr);
TEST_M(RdmaAgent, TcRaRegisterMr);
TEST_M(RdmaAgent, TcRaGetLbMax);
TEST_M(RdmaAgent, TcRaSetQpLbValue);
TEST_M(RdmaAgent, TcRaGetQpLbValue);
TEST_M(RdmaAgent, TcRaGetQpHyperFeature);
TEST_M(RdmaAgent, TcRaPeerGetQpHyperFeature);
TEST_M(RdmaAgent, TcRaSetDebugConfig);
TEST_M(RdmaAgent, TcRaHdcRecvHandleSendPktUnsuccess);
TEST_M(RdmaAgent, TcRaHdcGetEidByIp);
TEST_M(RdmaAgent, TcRaRsGetEidByIp);
TEST_M(RdmaAgent, TcRaPeerGetEidByIp);
TEST_M(RdmaAgent, TcRaHdcGetIpByEid);
TEST_M(RdmaAgent, TcRaRsGetIpByEid);
TEST_M(RdmaAgent, TcRaPeerGetIpByEid);
TEST_M(RdmaAgent, TcRaCtxGetAuxInfo);
TEST_M(RdmaAgent, TcRaHdcCtxGetAuxInfo);
TEST_M(RdmaAgent, TcRaRsCtxGetAuxInfo);
TEST_M(RdmaAgent, TcRaHdcCtxGetCrErrInfoList);
TEST_M(RdmaAgent, TcRaRsCtxGetCrErrInfoList);
TEST_M(RdmaAgent, TcRaHdcCtxGetJettyContext);
TEST_M(RdmaAgent, TcRaPeerCtxGetJettyContext);

/* pingMesh ut cases */
TEST_M(RdmaAgent, TcRaPingInitGetHandleAbnormal);
TEST_M(RdmaAgent, TcRaPingInitAbnormal);
TEST_M(RdmaAgent, TcRaPingTargetAddAbnormal);
TEST_M(RdmaAgent, TcRaPingTaskStartAbnormal);
TEST_M(RdmaAgent, TcRaPingGetResultsAbnormal);
TEST_M(RdmaAgent, TcRaPingTargetDelAbnoraml);
TEST_M(RdmaAgent, TcRaPingTaskStopAbnormal);
TEST_M(RdmaAgent, TcRaPingDeinitParaCheckAbnormal);
TEST_M(RdmaAgent, TcRaPingDeinitAbnoaml);
TEST_M(RdmaAgent, TcRaPing);

TEST_M(RdmaAgent, TcRaGetDevEidInfoNum);
TEST_M(RdmaAgent, TcRaGetDevEidInfoList);
TEST_M(RdmaAgent, TcRaCtxInit);
TEST_M(RdmaAgent, TcRaGetDevBaseAttr);
TEST_M(RdmaAgent, TcRaCtxDeinit);
TEST_M(RdmaAgent, TcRaCtxLmemRegister);
TEST_M(RdmaAgent, TcRaCtxRmemImport);
TEST_M(RdmaAgent, TcRaCtxChanCreate);
TEST_M(RdmaAgent, TcRaCtxTokenIdAlloc);
TEST_M(RdmaAgent, TcRaCtxCqCreate);
TEST_M(RdmaAgent, TcRaCtxQpCreate);
TEST_M(RdmaAgent, TcRaCtxQpImport);
TEST_M(RdmaAgent, TcRaCtxQpBind);
TEST_M(RdmaAgent, TcRaBatchSendWr);
TEST_M(RdmaAgent, TcRaCtxUpdateCi);
TEST_M(RdmaAgent, TcRaGetEidByIp);
TEST_M(RdmaAgent, TcRaGetIpByEid);
TEST_M(RdmaAgent, TcRaCtxGetCrErrInfoList);
TEST_M(RdmaAgent, TcRaCtxGetJettyContext);
TEST_M(RdmaAgent, TcRaCtxNotifyEvent);
TEST_M(RdmaAgent, TcRaHdcCtxNotifyEvent);

TEST_M(RdmaAgent, TcRaRsAsyncHdcSessionConnect);
TEST_M(RdmaAgent, TcRaHdcAsyncSendPkt);
TEST_M(RdmaAgent, TcRaHdcPoolAddTask);
TEST_M(RdmaAgent, TcRaHdcAsyncRecvPkt);
TEST_M(RdmaAgent, TcHdcAsyncRecvPkt);
TEST_M(RdmaAgent, TcRaHdcPoolCreate);
TEST_M(RdmaAgent, TcRaAsyncHandlePkt);
TEST_M(RdmaAgent, TcRaHdcAsyncHandleSocketListenStart);
TEST_M(RdmaAgent, TcRaHdcAsyncHandleQpImport);
TEST_M(RdmaAgent, TcRaPeerCtxInit);
TEST_M(RdmaAgent, TcRaPeerCtxDeinit);
TEST_M(RdmaAgent, TcRaPeerGetDevEidInfoNum);
TEST_M(RdmaAgent, TcRaPeerGetDevEidInfoList);
TEST_M(RdmaAgent, TcRaPeerCtxTokenIdAlloc);
TEST_M(RdmaAgent, TcRaPeerCtxTokenIdFree);
TEST_M(RdmaAgent, TcRaPeerCtxLmemRegister);
TEST_M(RdmaAgent, TcRaPeerCtxLmemUnregister);
TEST_M(RdmaAgent, TcRaPeerCtxRmemImport);
TEST_M(RdmaAgent, TcRaPeerCtxRmemUnimport);
TEST_M(RdmaAgent, TcRaPeerCtxChanCreate);
TEST_M(RdmaAgent, TcRaPeerCtxChanDestroy);
TEST_M(RdmaAgent, TcRaPeerCtxCqCreate);
TEST_M(RdmaAgent, TcRaPeerCtxCqDestroy);
TEST_M(RdmaAgent, TcRaPeerCtxQpCreate);
TEST_M(RdmaAgent, TcRaCtxPrepareQpCreate);
TEST_M(RdmaAgent, TcRaPeerCtxQpDestroy);
TEST_M(RdmaAgent, TcRaPeerCtxQpImport);
TEST_M(RdmaAgent, TcRaPeerCtxQpUnimport);
TEST_M(RdmaAgent, TcRaPeerCtxQpBind);
TEST_M(RdmaAgent, TcRaPeerCtxQpUnbind);

TEST_M(RdmaAgent, TcRaCtxLmemRegisterAsync);
TEST_M(RdmaAgent, TcRaCtxLmemUnregisterAsync);
TEST_M(RdmaAgent, TcRaCtxQpCreateAsync);
TEST_M(RdmaAgent, TcRaCtxQpDestroyAsync);
TEST_M(RdmaAgent, TcRaCtxQpImportAsync);
TEST_M(RdmaAgent, TcRaCtxQpUnimportAsync);
TEST_M(RdmaAgent, TcRaSocketSendAsync);
TEST_M(RdmaAgent, TcRaSocketRecvAsync);
TEST_M(RdmaAgent, TcRaGetAsyncReqResult);
TEST_M(RdmaAgent, TcRaSocketBatchConnectAsync);
TEST_M(RdmaAgent, TcRaSocketListenStartAsync);
TEST_M(RdmaAgent, TcRaSocketListenStopAsync);
TEST_M(RdmaAgent, TcRaSocketBatchCloseAsync);
TEST_M(RdmaAgent, TcRaHdcAsyncInitSession);
TEST_M(RdmaAgent, TcRaGetEidByIpAsync);
TEST_M(RdmaAgent, TcRaHdcGetEidByIpAsync);
TEST_M(RdmaAgent, TcRaGetIpByEidAsync);
TEST_M(RdmaAgent, TcRaHdcGetIpByEidAsync);

TEST_M(RdmaAgent, TcRaTlvInit);
TEST_M(RdmaAgent, TcRaTlvDeinit);
TEST_M(RdmaAgent, TcRaTlvRequest);
TEST_M(RdmaAgent, TcRaHdcTlvRequest);
TEST_M(RdmaAgent, TcRaRsTlvInit);
TEST_M(RdmaAgent, TcRaRsTlvDeinit);
TEST_M(RdmaAgent, TcRaRsTlvRequest);
TEST_M(RdmaAgent, TcRaRsTlvRequestV2);
TEST_M(RdmaAgent, TcRaGetTpInfoListAsync);
TEST_M(RdmaAgent, TcRaHdcGetTpInfoListAsync);
TEST_M(RdmaAgent, TcRaRsGetTpInfoList);
TEST_M(RdmaAgent, TcRaCtxGetTpInfoList);
TEST_M(RdmaAgent, TcRaPeerCtxGetTpInfoList);
TEST_M(RdmaAgent, TcRaCtxGetTpAttr);
TEST_M(RdmaAgent, TcRaPeerCtxGetTpAttr);
TEST_M(RdmaAgent, TcRaCtxSetTpAttr);
TEST_M(RdmaAgent, TcRaPeerCtxSetTpAttr);
TEST_M(RdmaAgent, TcRaGetSecRandom);
TEST_M(RdmaAgent, TcRaRsGetSecRandom);
TEST_M(RdmaAgent, TcRaGetTlsEnable);
TEST_M(RdmaAgent, TcRaRsGetTlsEnable);
TEST_M(RdmaAgent, TcRaGetHccnCfg);
TEST_M(RdmaAgent, TcRaRsGetHccnCfg);
TEST_M(RdmaAgent, TcHdcAsyncDelReqHandle);
TEST_M(RdmaAgent, TcRaHdcDeinitAsyncAll);

TEST_M(RdmaAgent, TcRaCtxQpDestroyBatchAsync);
TEST_M(RdmaAgent, TcQpDestroyBatchParamCheck);
TEST_M(RdmaAgent, TcRaHdcCtxQpDestroyBatchAsync);
TEST_M(RdmaAgent, TcRaRsCtxQpDestroyBatch);
TEST_M(RdmaAgent, TcRaCtxQpQueryBatch);
TEST_M(RdmaAgent, TcQpQueryBatchParamCheck);
TEST_M(RdmaAgent, TcRaHdcCtxQpQueryBatch);
TEST_M(RdmaAgent, TcRaRsCtxQpQueryBatch);
TEST_M(RdmaAgent, TcRaNdaGetDirectFlag);
TEST_M(RdmaAgent, TcRaPeerNdaGetDirectFlag);
TEST_M(RdmaAgent, TcRaNdaQpCreate);
TEST_M(RdmaAgent, TcRaPeerNdaQpCreate);

TEST_F(RdmaAgent, RaRsHasCapability_CapabilityInvalid_ReturnFalse)
{
    bool ret = RaRsHasCapability(RA_CAP_INVALID, 0, 1);
    EXPECT_EQ(ret, false);
}

TEST_F(RdmaAgent, RaRsHasCapability_CapabilityBeyondInvalid_ReturnFalse)
{
    bool ret = RaRsHasCapability(RA_CAP_INVALID + 1, 0, 1);
    EXPECT_EQ(ret, false);
}

TEST_F(RdmaAgent, RaRsHasCapability_ValidCapabilityRoceAndNetMet_ReturnTrue)
{
    bool ret = RaRsHasCapability(RA_CAP_DRV_SHAREPOOL_NON_PIN, 0, 1);
    EXPECT_EQ(ret, true);
}

TEST_F(RdmaAgent, RaRsHasCapability_ValidCapabilityRoceMetNetNotMet_ReturnFalse)
{
    bool ret = RaRsHasCapability(RA_CAP_DRV_SHAREPOOL_NON_PIN, 0, 0);
    EXPECT_EQ(ret, false);
}

TEST_F(RdmaAgent, RaRsHasCapability_ValidCapabilityRoceMetAndNetMet_ReturnTrue)
{
    bool ret = RaRsHasCapability(RA_CAP_DRV_SHAREPOOL_NON_PIN, 1, 1);
    EXPECT_EQ(ret, true);
}

TEST_F(RdmaAgent, RaHdcIsBroken_WithSocketClose_ReturnTrue)
{
    int val = -DRV_ERROR_SOCKET_CLOSE;
    bool ret = RaHdcIsBroken(-DRV_ERROR_SOCKET_CLOSE);
    EXPECT_EQ(ret, true);
}

TEST_F(RdmaAgent, RaHdcIsBroken_WithZero_ReturnFalse)
{
    bool ret = RaHdcIsBroken(0);
    EXPECT_EQ(ret, false);
}

TEST_F(RdmaAgent, RaHdcIsBroken_WithNonSocketCloseError_ReturnFalse)
{
    bool ret = RaHdcIsBroken(DRV_ERROR_WAIT_TIMEOUT);
    EXPECT_EQ(ret, false);
}

TEST_F(RdmaAgent, RaHdcIsBroken_WithPositiveOne_ReturnFalse)
{
    bool ret = RaHdcIsBroken(1);
    EXPECT_EQ(ret, false);
}
