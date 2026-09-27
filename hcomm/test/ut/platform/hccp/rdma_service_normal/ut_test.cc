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
#include "tc_ut_rs.h"
#include "tc_ut_rs_ping.h"
#include "tc_ut_rs_ping_urma.h"
#include "tc_ut_rs_tlv.h"
}

#include <stdio.h>
#include <mockcpp/mockcpp.hpp>
#include "gtest/gtest.h"
#include "tc_ut_rs_ub.h"
#include "tc_ut_rs_ctx.h"
#include "tc_ut_rs_nda.h"

using namespace std;

class RS : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--RoCE RS SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--RoCE RS TearDown--\033[0m" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_M(RS, TcRsSocketListen);
TEST_M(RS, TcRsSocketListenIpv6);

TEST_M(RS, TcRsQpCreate);
TEST_M(RS, TcRsQpCreateWithAttrs);
TEST_M(RS, TcRsMemPool);
TEST_M(RS, TcRsGetQpStatus);
TEST_M(RS, TcRsGetNotifyBa);
TEST_M(RS, TcRsSetupSharemem);

TEST_M(RS, TcRsMrAbnormal);

TEST_M(RS, TcRsSendWr);
TEST_M(RS, TcRsSendWrlistNormal);
TEST_M(RS, TcRsSendWrlistExp);

TEST_M(RS, TcRsSocketOps);

TEST_M(RS, TcRsWhiteList);
TEST_M(RS, TcRsPeerGetIfaddrs);
TEST_M(RS, TcRsGetIfnum);
TEST_M(RS, TcRsGetInterfaceVersion);

TEST_M(RS, TcRsGetCurTime);
TEST_M(RS, TcRsGetHostRdevIndex);
TEST_M(RS, TcRsRdevCbInit);
TEST_M(RS, TcRsSslFree);
TEST_M(RS, TcRsSslDeinit);
TEST_M(RS, TcRsSslCaKyInit);
TEST_M(RS, Tcrs_check_pridata);
TEST_M(RS, Tctls_load_cert);
TEST_M(RS, Tcrs_ssl_err_string);
TEST_M(RS, tc_rs_socket_fill_wlist_by_phyID);
TEST_M(RS, TcRsNotifyCfgSet);
TEST_M(RS, TcRsServerSendWlistCheckResult);

TEST_M(RS, TcRsSocketDeinit2);
TEST_M(RS, tc_RsApiInit);

TEST_M(RS, TcRsDrvQueryNotifyAndAllocPd);
TEST_M(RS, TcRsSendNormalWrlist);
TEST_M(RS, TcRsDrvSendExp);
TEST_M(RS, TcRsDrvNormalQpCreateInit);
TEST_M(RS, TcRsRegisterMr);
TEST_M(RS, TcRsEpollCtlMod);
TEST_M(RS, TcRsEpollCtlMod01);
TEST_M(RS, TcRsEpollCtlMod02);
TEST_M(RS, TcRsEpollCtlMod03);
TEST_M(RS, TcRsEpollCtlDel);
TEST_M(RS, TcRsEpollCtlDel01);
TEST_M(RS, TcRsSetTcpRecvCallback);
TEST_M(RS, TcRsEpollEventInHandle);
TEST_M(RS, TcRsEpollEventInHandle01);
TEST_M(RS, TcRsEpollTcpRecv);
TEST_M(RS, TcRsEpollEventSslAcceptInHandle);
TEST_M(RS, TcRsSendExpWrlist);
TEST_M(RS, TcRsDrvPollCqHandle);
TEST_M(RS, TcRsNormalQpCreate);
TEST_M(RS, TcRsQueryEvent);
TEST_M(RS, TcRsCreateCq);
TEST_M(RS, TcRsCreateNormalQp);
TEST_M(RS, TcRsCreateCompChannel);
TEST_M(RS, TcRsGetCqeErrInfo);
TEST_M(RS, TcRsGetCqeErrInfoNum);
TEST_M(RS, TcRsGetCqeErrInfoList);
TEST_M(RS, TcRsSaveCqeErrInfo);
TEST_M(RS, TcRsCqeCallbackProcess);
TEST_M(RS, TcRsCreateSrq);
TEST_M(RS, TcRsGetIpv6ScopeId);
TEST_M(RS, TcRsCreateEventHandle);
TEST_M(RS, TcRsCtlEventHandle);
TEST_M(RS, TcRsWaitEventHandle);
TEST_M(RS, TcRsDestroyEventHandle);
TEST_M(RS, TcRsEpollCreateEpollfd);
TEST_M(RS, TcRsEpollDestroyFd);
TEST_M(RS, TcRsEpollWaitHandle);
TEST_M(RS, TcRsGetVnicIpInfo);
TEST_M(RS, TcRsTypicalRegisterMr);
TEST_M(RS, TcRsTypicalQpModify);
TEST_M(RS, TcRsGetQpHyperFeature);
TEST_M(RS, TcRsRemapMr);
TEST_M(RS, tc_RsGetApiVersion);
TEST_M(RS, TcRsGetTlsEnable);
TEST_M(RS, TcRsGetSecRandom);
TEST_M(RS, TcRsGetHccnCfg);
TEST_M(RS, TcRsJfcCallbackProcess);

/**
 ******************** Beginning of UB TEST ***************************
 */

TEST_M(RS, TcRsUbGetRdevCb);
TEST_M(RS, TcRsUrmaApiInitAbnormal);

TEST_M(RS, TcRsGetDevEidInfoNum);
TEST_M(RS, TcRsGetDevEidInfoList);
TEST_M(RS, TcRsCtxInit);
TEST_M(RS, TcRsCtxDeinit);
TEST_M(RS, TcRsCtxTokenIdAlloc);
TEST_M(RS, TcRsCtxTokenIdFree);
TEST_M(RS, TcRsCtxLmemReg);
TEST_M(RS, TcRsCtxLmemUnreg);
TEST_M(RS, TcRsCtxRmemImport);
TEST_M(RS, TcRsCtxRmemUnimport);
TEST_M(RS, TcRsCtxChanCreate);
TEST_M(RS, TcRsCtxChanDestroy);
TEST_M(RS, TcRsCtxCqCreate);
TEST_M(RS, TcRsCtxCqDestroy);
TEST_M(RS, TcRsCtxQpCreate);
TEST_M(RS, TcRsCtxQpDestroy);
TEST_M(RS, TcRsCtxQpImport);
TEST_M(RS, TcRsCtxQpUnimport);
TEST_M(RS, TcRsCtxQpBind);
TEST_M(RS, TcRsCtxQpUnbind);
TEST_M(RS, TcRsCtxBatchSendWr);
TEST_M(RS, TcRsCtxUpdateCi);
TEST_M(RS, TcRsCtxEsched);
TEST_M(RS, TcRsGetEidByIp);
TEST_M(RS, TcRsGetIpByEid);

/* pingMesh ut cases */
TEST_M(RS, TcRsPayloadHeaderResvCustomCheck);
TEST_M(RS, TcRsPingHandleInit);
TEST_M(RS, TcRsPingHandleDeinit);
TEST_M(RS, TcRsPingInit);
TEST_M(RS, TcRsGetPingCb);
TEST_M(RS, TcRsPingClientPostSend);
TEST_M(RS, TcRsPingGetResults);
TEST_M(RS, TcRsPingTaskStop);
TEST_M(RS, TcRsPingTargetDel);
TEST_M(RS, TcRsPingDeinit);
TEST_M(RS, TcRsPingCompareRdmaInfo);
TEST_M(RS, TcRsPingRoceFindTargetNode);
TEST_M(RS, TcRsPongFindTargetNode);
TEST_M(RS, TcRsPongFindAllocTargetNode);
TEST_M(RS, TcRsPingPollSendCq);
TEST_M(RS, TcRsPingServerPostSend);
TEST_M(RS, TcRsPingPostRecv);
TEST_M(RS, TcRsPingClientPollCq);
TEST_M(RS, TcRsEpollEventPingHandle);
TEST_M(RS, TcRsPingGetTripTime);
TEST_M(RS, TcRsPingCbInitMutex);
TEST_M(RS, TcRsPingResolveResponsePacket);
TEST_M(RS, TcRsPingServerPollCq);
TEST_M(RS, TcRsPingCommonDeinitLocalBuffer);
TEST_M(RS, TcRsPingCommonDeinitLocalQp);
TEST_M(RS, TcRsPingRocePollScq);
TEST_M(RS, TcRsPingPongInitLocalInfo);
TEST_M(RS, TcRsPingHandle);
TEST_M(RS, TcRsPingRocePingCbDeinit);

TEST_M(RS, TcRsEpollEventPingHandleUrma);
TEST_M(RS, TcRsPingInitDeinitUrma);
TEST_M(RS, TcRsPingTargetAddDelUrma);
TEST_M(RS, TcRsPingUrmaPostSend);
TEST_M(RS, TcRsPingUrmaPollScq);
TEST_M(RS, TcRsPingClientPollCqUrma);
TEST_M(RS, TcRsPingServerPollCqUrma);
TEST_M(RS, TcRsPingGetResultsUrma);
TEST_M(RS, TcRsPingServerPostSendUrma);
TEST_M(RS, TcRsPongJettyFindAllocTargetNode);
TEST_M(RS, TcRsPingCommonPollSendJfc);
TEST_M(RS, TcRsPongJettyFindTargetNode);
TEST_M(RS, TcRsPongJettyResolveResponsePacket);
TEST_M(RS, TcRsPingCommonImportJetty);
TEST_M(RS, TcRsPingUrmaResetRecvBuffer);
TEST_M(RS, TcRsPingCommonJfrPostRecv);

TEST_M(RS, TcRsPeerSocketRecv);
TEST_M(RS, TcRsSocketGetServerSocketErrInfo);
TEST_M(RS, TcRsSocketAcceptCreditAdd);
TEST_M(RS, TcRsEpollEventSslRecvTagInHandle);

TEST_M(RS, TcRsGetTpInfoList);
TEST_M(RS, TcRsUbCtxDrvJettyImport);
TEST_M(RS, TcRsUbCtxInit);
TEST_M(RS, TcRsUbCtxJfcDestroy);
TEST_M(RS, TcRsUbCtxExtJettyDelete);
TEST_M(RS, TcRsUbCtxChanCreate);
TEST_M(RS, TcRsUbInitSegCb);
TEST_M(RS, TcRsUbCtxLmemReg);
TEST_M(RS, TcRsUbCtxJfcCreateFail);
TEST_M(RS, TcRsUbCtxInitJettyCb);
TEST_M(RS, TcRsUbCtxJettyCreateFail);
TEST_M(RS, TcRsUbCtxJettyImportFail);
TEST_M(RS, TcRsUbCtxBatchSendWrFail);
TEST_M(RS, TcRsUbGetEidByIp);
TEST_M(RS, TcRsUbGetIpByEid);
TEST_M(RS, TcRsCtxGetAuxInfo);
TEST_M(RS, TcRsUbCtxGetAuxInfo);

TEST_M(RS, TcRsNslbInit);
TEST_M(RS, TcRsNslbDeinit);
TEST_M(RS, TcRsNslbRequest);
TEST_M(RS, TcRsTlvAssembleSendData);
TEST_M(RS, TcRsGetTlvCb);
TEST_M(RS, TcRsEpollNslbEventHandle);
TEST_M(RS, TcRsNslbApiInit);
TEST_M(RS, TcRsFreeDevList);
TEST_M(RS, TcRsFreeRdevList);
TEST_M(RS, TcRsFreeUdevList);
TEST_M(RS, TcRsCcuRequest);
TEST_M(RS, TcDlCcuApiInit);
TEST_M(RS, TcRsCtxQpDestroyBatch);
TEST_M(RS, TcRsCtxQpQueryBatch);
TEST_M(RS, TcRsUbCtxJettyDestroyBatch);
TEST_M(RS, TcRsUbCtxQueryJettyBatch);
TEST_M(RS, TcRsNetApiInitDeinit);
TEST_M(RS, TcRsNetAllocJfcId);
TEST_M(RS, TcRsNetFreeJfcId);
TEST_M(RS, TcRsNetAllocJettyId);
TEST_M(RS, TcRsNetFreeJettyId);
TEST_M(RS, TcRsNetGetCqeBaseAddr);
TEST_M(RS, TcRsCcuGetCqeBaseAddr);
TEST_M(RS, tc_RsNslbNetcoInitDeinit);
TEST_M(RS, TcRsGetTpAttr);
TEST_M(RS, TcRsSetTpAttr);
TEST_M(RS, TcRsUbGetTpAttr);
TEST_M(RS, TcRsUbSetTpAttr);
TEST_M(RS, TcRsCtxGetCrErrInfoList);
TEST_M(RS, TcRsCtxNotifyEvent);
TEST_M(RS, TcRsCtxNotifyEventAubdfxFail);
TEST_M(RS, TcRsAubdfxApiInitRollback);
TEST_M(RS, TcRsRetryTimeoutExceptionCheck);
TEST_M(RS, TcRsSetQpLbValue);
TEST_M(RS, TcRsGetQpLbValue);
TEST_M(RS, TcRsGetLbMax);
TEST_M(RS, TcRsNdaGetDirectFlag);
TEST_M(RS, TcRsNdaQpCreateFailed);
TEST_M(RS, TcRsNdaQpCreate);
TEST_M(RS, TcRsNdaQpCreateEx);
TEST_M(RS, TcRsNdaUbAllocFailed);
TEST_M(RS, TcRsNdaDbMmapHostVa);
TEST_M(RS, TcRsNdaDbMmapUbRes);
TEST_M(RS, TcRsNdaCqCreateFailed);
TEST_M(RS, TcRsNdaCqCreate);
