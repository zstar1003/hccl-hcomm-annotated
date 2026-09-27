/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TC_UT_RS_H
#define _TC_UT_RS_H

#include <stdio.h>

#define RS_TEST_MEM_SIZE 32
#define RS_TEST_MEM_PAGE_SIZE 4096

#define rs_ut_msg(fmt, args...) fprintf(stderr, "\t>>>>> " fmt, ##args)

void TcRsAbnormal();
void TcRsAbnormal2();
void TcRsInit();
void TcRsSocketListen();
void TcRsSocketListenIpv6();
void TcRsSocketConnect();
void TcRsGetSockets();

void TcRsGetTsqpDepth();
void TcRsSetTsqpDepth();
void TcRsQpCreate();
void TcRsQpCreateWithAttrs();
void TcRsGetQpStatus();
void TcRsGetNotifyBa();
void TcRsSetupSharemem();
void TcRsSendWr();
void TcRsSendWrlistNormal();
void TcRsSendWrlistExp();
void TcRsGetSqIndex();
void TcRsPostRecv();
void TcRsMemPool();

void TcRsMrCreate();
void TcRsMrAbnormal();

void TcRsCqHandle();
void TcRsSocketOps();

void TcRsDfx();

void TcRsSocketInit();
void TcRsDeinit();
void TcRsDeinit2();
void TcRsSocketDeinit1();
void TcRsSocketDeinit2();
void TcRsConnServerCreate();
void TcRsConnServerCreate1();
void TcRsConnServerCreate2();
void TcRsConnServerCreate3();
void TcRsWhiteList();
void TcRsSslTest1();
void TcRsGetIfaddrs();
void TcRsGetIfaddrsV2();
void TcRsPeerGetIfaddrs();
void TcRsGetIfnum();
void TcRsGetInterfaceVersion();
void TcRsGetCurTime();
void tc_RsRdev2rdevCb();
void TcRsCompareIpGid();
void TcRsQueryGid();
void TcRsGetHostRdevIndex();
void TcRsGetIbCtxAndRdevIndex();
void TcRsRdevCbInit();
void TcRsRdevInit();
void TcRsSslFree();
void TcRsDrvConnect();
void TcRsQpn2qpcb();
void TcRsServerValidAsyncInit();
void TcRsDrvPostRecv();
void TcRsSslDeinit();
void Tcrs_tls_inner_enable();
void TcRsSslInnerInit();
void TcRsSslCaKyInit();
void Tcrs_ssl_crl_init();
void Tcrs_check_pridata();
void Tcrs_ssl_load_ca();
void Tcrs_ssl_get_ca_data();
void Tcrs_ssl_get_ca_data1();
void Tcrs_ssl_get_crl_data1();
void Tcrs_ssl_put_certs();
void Tcrs_ssl_check_mng_and_cert_chain();
void Tcrs_ssl_check_cert_chain();
void Tcrs_ssl_skid_get_from_chain();
void Tcrs_ssl_verify_cert_chain();
void Tctls_get_cert_chain();
void Tcrs_ssl_get_leaf_cert();
void Tctls_load_cert();
void Tcrs_tls_peer_cert_verify();
void Tcrs_ssl_err_string();
void TcRsServerSendWlistCheckResult();
void TcRsServerValidAsyncInit();
void TcRsEpollEventSslAcceptInHandle();
void TcRsDrvSslBindFd();
void tc_rs_socket_fill_wlist_by_phyID();
void TcRsGetVnicIp();
void TcRsNotifyCfgSet();
void TcRsNotifyCfgGet();
void TcCryptoDecryptWithAesGcm();
void TcRsListenInvalidPort();
void TcRsDrvQpNormalFail();
void tc_RsApiInit();
void TcRsRecvWrlist();
void TcRsDrvPostRecv();
void TcRsDrvRegNotifyMr();
void TcRsDrvQueryNotifyAndAllocPd();
void TcRsSendNormalWrlist();
void TcRsDrvSendExp();
void TcRsDrvNormalQpCreateInit();
void TcRsRegisterMr();
void TcRsEpollCtlAdd();
void TcRsEpollCtlAdd01();
void TcRsEpollCtlAdd02();
void TcRsEpollCtlAdd03();
void TcRsEpollCtlMod();
void TcRsEpollCtlMod01();
void TcRsEpollCtlMod02();
void TcRsEpollCtlMod03();
void TcRsEpollCtlDel();
void TcRsEpollCtlDel01();
void TcRsSetTcpRecvCallback();
void TcRsEpollEventInHandle();
void TcRsEpollEventInHandle01();
void TcRsSocketListenBindListen();
void TcRsEpollTcpRecv();
void TcRsSendExpWrlist();
void TcRsDrvPollCqHandle();
void TcRsNormalQpCreate();
void TcRsQueryEvent();
void TcRsCreateCq();
void TcRsCreateNormalQp();
void TcRsCreateCompChannel();
void TcRsGetCqeErrInfo();
void TcRsGetCqeErrInfoNum();
void TcRsGetCqeErrInfoList();
void TcRsSaveCqeErrInfo();
void TcRsCqeCallbackProcess();
void TcRsCreateSrq();
void TcRsGetIpv6ScopeId();
void TcRsCreateEventHandle();
void TcRsCtlEventHandle();
void TcRsWaitEventHandle();
void TcRsDestroyEventHandle();
void TcRsEpollCreateEpollfd();
void TcRsEpollDestroyFd();
void TcRsEpollWaitHandle();
void TcSslverify_callback();
void Tcrs_ssl_verify_cert();
void TcRsGetVnicIpInfo();
void TcRsRemapMr();
void tc_RsGetApiVersion();
void TcRsGetTlsEnable();
void TcRsGetSecRandom();
void TcRsGetHccnCfg();
void TcDlHalSetClearUserConfig();

void TcRsTypicalRegisterMr();
void TcRsTypicalQpModify();
void TcRsGetQpHyperFeature();

void Tcrs_ssl_get_cert();
void tc_rs_ssl_X509_store_init();
void Tcrs_ssl_skids_subjects_get();
void Tcrs_ssl_put_cert_ca_pem();
void Tcrs_ssl_put_cert_end_pem();
void Tcrs_ssl_check_mng_and_cert_chain();
void Tcrs_remove_certs();
void tc_rs_ssl_X509_store_add_cert();
void TcRsPeerSocketRecv();
void TcRsSocketGetServerSocketErrInfo();
void TcRsSocketAcceptCreditAdd();
void TcRsEpollEventSslRecvTagInHandle();
void TcRsFreeDevList(void);
void TcRsFreeRdevList(void);
void TcRsFreeUdevList(void);
void TcRsRetryTimeoutExceptionCheck();
void TcRsSetQpLbValue();
void TcRsGetQpLbValue();
void TcRsGetLbMax();
#endif
