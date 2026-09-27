/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include "rs.h"
#include "ra_rs_err.h"
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <fnmatch.h>
#include "securec.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_rdma_inner.h"
#include "rs_nda.h"
#include "rs_epoll.h"
#include "rs_tls.h"
#include "ssl_adp.h"
#include "rs_socket.h"
#include "dl_ibverbs_function.h"
#include "dl_hal_function.h"
#include "rs_drv_rdma.h"
#include "file_opt.h"
#include "config_log.h"
#include "network_comm.h"
#ifdef CONFIG_TLV
#include "rs_tlv.h"
#endif
#include "ra_rs_ctx.h"
#include "rs_ctx.h"
#include "rs_esched.h"
#include "dl_net_function.h"
#include "rs_ub.h"
#include "rs_ctx_inner.h"

__thread struct rs_cb *gRsCb = NULL;           // lint !e17
struct rs_cb *gRsCbList[RS_MAX_DEV_NUM] = {0}; // lint !e17
int gInitCounter[RS_MAX_DEV_NUM] = {0};

/* set current phyId g_rs_cb */
void RsSetCtx(unsigned int phyId)
{
    gRsCb = gRsCbList[phyId];
}

RS_ATTRI_VISI_DEF void RsSetDebugConfig(uint64_t debugConfig)
{
    HccpSetDebugConfig(debugConfig);
}

/* get current g_rs_cb */
static struct rs_cb *RsGetCurRsCb(void)
{
    for (int i = 0; i < RS_MAX_DEV_NUM; i++) {
        if (gRsCbList[i] != NULL) {
            hccp_info_others("[rs_get_cur_rs_cb], phyId[%u], rsCb[%p]", i, gRsCbList[i]);
            return gRsCbList[i];
        }
    }
    return NULL;
}

RS_ATTRI_VISI_DEF void RsGetCurTime(struct timeval *time)
{
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_VOID(time);
    ret = gettimeofday(time, NULL);
    if (ret) {
        hccp_warn_others("gettimeofday unsuccessful, ret[%d] expect 0", ret);
        ret = memset_s(time, sizeof(struct timeval), 0, sizeof(struct timeval));
        if (ret) {
            hccp_warn_others("memset_s unsuccessful, ret[%d] expect 0", ret);
        }
    }

    return;
}

RS_ATTRI_VISI_DEF void HccpTimeInterval(struct timeval *endTime, struct timeval *startTime, float *msec)
{
    RS_CHECK_POINTER_NULL_RETURN_VOID(endTime);
    RS_CHECK_POINTER_NULL_RETURN_VOID(startTime);
    RS_CHECK_POINTER_NULL_RETURN_VOID(msec);

    /* if low position is sufficient, then borrow one from the high position */
    if (endTime->tv_usec < startTime->tv_usec) {
        endTime->tv_sec -= 1;
        endTime->tv_usec += MS_PER_SECOND_I * MS_PER_SECOND_I;
    }

    *msec = (float)((endTime->tv_sec - startTime->tv_sec) * MS_PER_SECOND_F +
                    (endTime->tv_usec - startTime->tv_usec) / US_PER_MS_F);

    return;
}

RS_ATTRI_VISI_DEF void RsHeartbeatAlivePrint(struct RsPthreadInfo *pthreadInfo)
{
    float timeCost = 0.0;
    struct timeval now;

    if (pthreadInfo == NULL) {
        hccp_err("pthread_info is NULL!");
        return;
    }

    RsGetCurTime(&now);
    HccpTimeInterval(&now, &pthreadInfo->lastCheckTime, &timeCost);
    if (timeCost >= RS_HEARTBEAT_TIME || timeCost <= 0) {
        hccp_info_others("pthread[%s] is alive!", pthreadInfo->pthreadName);
        RsGetCurTime(&pthreadInfo->lastCheckTime);
    }

    return;
}

int RsDev2rscb(uint32_t chipId, struct rs_cb **rsCb, bool initFlag)
{
    if (gRsCb == NULL) {
        if (initFlag == false) {
            hccp_warn_init("No device initialized !");
        }
        return -ENODEV;
    }

    if (chipId == gRsCb->chipId) {
        *rsCb = gRsCb;
        return 0;
    }

    hccp_warn_others("get rs cb unsuccessful for dev %u !", chipId);
    *rsCb = NULL;

    return -ENODEV;
}

int RsGetHccpMode(unsigned int chipId)
{
    struct rs_cb *rsCb = NULL;
    int ret;

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed(%d)", ret), ret);
    return (int)rsCb->hccpMode;
}

int RsDev2conncb(uint32_t chipId, struct RsConnCb **connCb)
{
    int ret;
    struct rs_cb *rsCb = NULL;

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed(%d)", ret), ret);

    *connCb = &(rsCb->connCb);

    return 0;
}

int RsGetRdevCb(struct rs_cb *rsCb, unsigned int rdevIndex, struct RsRdevCb **rdevCb)
{
    struct RsRdevCb *rdevCbTmp = NULL;
    struct RsRdevCb *rdevCbTmp2 = NULL;

    RS_LIST_GET_HEAD_ENTRY(rdevCbTmp, rdevCbTmp2, &rsCb->rdevList, list, struct RsRdevCb);
    for (; (&rdevCbTmp->list) != &rsCb->rdevList;
         rdevCbTmp = rdevCbTmp2, rdevCbTmp2 = list_entry(rdevCbTmp2->list.next, struct RsRdevCb, list)) {
        if (rdevCbTmp->rdevIndex == rdevIndex) {
            *rdevCb = rdevCbTmp;
            return 0;
        }
    }

    *rdevCb = NULL;
    hccp_err("rdev_cb for rdev_index[%u] is not available!", rdevIndex);

    return -ENODEV;
}

int RsRdev2rdevCb(unsigned int chipId, unsigned int rdevIndex, struct RsRdevCb **rdevCb)
{
    int ret;
    struct rs_cb *rsCb = NULL;

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed for chipId:%u, ret:%d", chipId, ret), -ENODEV);

    ret = RsGetRdevCb(rsCb, rdevIndex, rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_rdev_cb failed!, ret %d, rdevIndex %u", ret, rdevIndex), ret);

    return 0;
}

STATIC int RsPthreadMutexInit(struct rs_cb *rscb, struct RsInitConfig *cfg)
{
    int ret;
    int err;

    RS_CHECK_POINTER_NULL_RETURN_INT(cfg);
    RS_CHECK_POINTER_NULL_RETURN_INT(rscb);
    rscb->chipId = cfg->chipId;
    rscb->hccpMode = cfg->hccpMode;
    rscb->connCb.rscb = rscb;

    ret = pthread_mutex_init(&rscb->mutex, NULL);
    CHK_PRT_RETURN(ret, hccp_err("rscb mutex_init failed ret %d!, normal ret 0", ret), -ESYSFUNC);
    ret = pthread_mutex_init(&rscb->connCb.connMutex, NULL);
    if (ret) {
        hccp_err("conn_cb mutex_init failed ret %d, normal ret 0!", ret);
        err = pthread_mutex_destroy(&rscb->mutex);
        hccp_dbg("pthread destroy ret %d", err);
        return -ESYSFUNC;
    }

    hccp_info_init("mutex init ok");

    RS_INIT_LIST_HEAD(&rscb->connCb.listenList);
    RS_INIT_LIST_HEAD(&rscb->connCb.serverAcceptList);
    RS_INIT_LIST_HEAD(&rscb->connCb.clientConnList);
    RS_INIT_LIST_HEAD(&rscb->connCb.serverConnList);
    RS_INIT_LIST_HEAD(&rscb->connCb.whiteList);
    RS_INIT_LIST_HEAD(&rscb->rdevList);
    RS_INIT_LIST_HEAD(&rscb->udevList);
    RS_INIT_LIST_HEAD(&rscb->heterogTcpFdList);
    rscb->connCb.wlistEnable = cfg->whiteListStatus;
    return 0;
}

STATIC int RsGetChipLogicId(unsigned int chipId, enum NetworkMode hccpMode, unsigned int *logicId)
{
    (void)hccpMode;
    int ret = 0;

    ret = DlDrvDeviceGetIndexByPhyId(chipId, logicId);
    CHK_PRT_RETURN(ret != 0, hccp_err("hal get logicId failed, chipId[%u], ret[%d]", chipId, ret), -ENODEV);

    return 0;
}

#ifdef CUSTOM_INTERFACE
STATIC int RsInitNetAdapt(struct rs_cb *rscb)
{
    int ret = 0;

    if (rscb->protocol != PROTOCOL_UDMA) {
        return 0;
    }

    ret = RsNetAdaptInit();
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_net_adapt_init chipId[%u] logic_devid[%u] failed, ret=%d", rscb->chipId, rscb->logicId, ret), ret);

    return ret;
}
#endif

#ifdef CUSTOM_INTERFACE
STATIC void RsDeInitNetAdapt(struct rs_cb *rscb)
{
    if (rscb->protocol != PROTOCOL_UDMA) {
        return;
    }

    RsNetAdaptUninit();
}
#endif

STATIC int RsInitRscbCfg(struct rs_cb *rscb)
{
    enum ProductType productType;
    struct timeval start, end;
    float timeCost = 0.0;
    int ret;

    ret = RsGetChipLogicId(rscb->chipId, rscb->hccpMode, &rscb->logicId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_chip_logic_id failed, ret[%d]", ret), ret);

    productType = RsGetProductType(rscb->logicId);
    CHK_PRT_RETURN(productType == PRODUCT_TYPE_INVALID, hccp_err("RsGetProductType failed, logicId:%u", rscb->logicId),
        -EINVAL);
#ifdef CUSTOM_INTERFACE
    if (RsIsUdmaSupported() || RsIsRdmaSupported()) {
        ret = RsGetChipProtocol(rscb->chipId, rscb->hccpMode, &rscb->protocol, rscb->logicId);
        CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_chip_protocol failed, ret[%d]", ret), ret);
        // make sure RsRoceGetApiVersion and RsNetGetApiVersion are valid
        ret = RsCtxApiInit(rscb->hccpMode, rscb->protocol);
        CHK_PRT_RETURN(ret != 0, hccp_err("rs_ctx_api_init failed, ret[%d]", ret), ret);
        ret = RsEschedInit(rscb);
        if (ret != 0) {
            hccp_err("rs_esched_init chipId[%u] logic_devid[%u] failed, ret=%d productType=%d", rscb->chipId,
                rscb->logicId, ret, productType);
            goto esched_init_err;
        }
    }

    ret = RsInitNetAdapt(rscb);
    if (ret != 0) {
        goto net_adapt_init_err;
    }

#endif

    ret = rs_ssl_init(rscb);
    if (ret != 0) {
        hccp_err("init ssl failed, ret[%d]", ret);
        goto ssl_init_err;
    }

    RsGetCurTime(&start);
    ret = RsEpollConnectHandleInit(rscb);
    if (ret != 0) {
        hccp_err("create pthread failed, ret[%d]", ret);
        goto create_pthread_err;
    }

    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_info_init("rs_epoll_connect_handle_init ok cost [%f] ms", timeCost);
    return 0;

create_pthread_err:
    rs_ssl_deinit(rscb);
ssl_init_err:
#ifdef CUSTOM_INTERFACE
    if (RsIsUdmaSupported() || RsIsRdmaSupported()) {
        RsDeInitNetAdapt(rscb);
    net_adapt_init_err:
        RsEschedDeinit(rscb->protocol);
    esched_init_err:
        (void)RsCtxApiDeinit(rscb->hccpMode, rscb->protocol);
    }
#endif
    return ret;
}

STATIC void RsDeinitRscbCfg(struct rs_cb *rscb)
{
    int tryAgain = RS_TRY_TIME;
    eventfd_t event = 1;
    int ret;

#ifdef CUSTOM_INTERFACE
    if (RsIsUdmaSupported() || RsIsRdmaSupported()) {
        RsDeInitNetAdapt(rscb);
        RsEschedDeinit(rscb->protocol);
        (void)RsCtxApiDeinit(rscb->hccpMode, rscb->protocol);
    }
#endif
    rs_ssl_deinit(rscb);
    // deinit resources in rs_epoll_connect_handle_init
    // deinit epoll thread, send event to eventfd to waking up epoll handle thread
    ret = (int)write(rscb->connCb.eventfd, &event, sizeof(eventfd_t));
    if (ret != sizeof(eventfd_t)) {
        hccp_warn_others("eventfd_write unsuccessful(0x%x), chipId:%u, errno:%d", ret, rscb->chipId, errno);
    }
    while (((rscb->state & RS_STATE_HALT) == 0) && (tryAgain != 0)) {
        usleep(RS_USLEEP_TIME);
        tryAgain--;
    };
    if (tryAgain == 0) {
        hccp_warn_others("try_again exhausted, epoll thread quit unsuccessful, rscb state:%u", rscb->state);
    }
    rscb->state &= ~RS_STATE_HALT;

    // deinit connect thread, already been RS_CONN_EXIT_FLAG, no need to change conn_flag
    if (rscb->connFlag != RS_CONN_EXIT_FLAG) {
        rscb->connFlag = 0;
    }
    tryAgain = RS_TRY_TIME;
    while ((rscb->connFlag != RS_CONN_EXIT_FLAG) && (tryAgain != 0)) {
        usleep(RS_USLEEP_TIME);
        tryAgain--;
    }
    if (tryAgain == 0) {
        hccp_warn_others("try_again exhausted, connect thread quit unsuccessful, rscb connFlag:%d", rscb->connFlag);
    }

    RsDestroyEpoll(rscb);
}

RS_ATTRI_VISI_DEF int RsInit(struct RsInitConfig *cfg)
{
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(cfg);
    ret = DlHalInit();
    if (ret != 0) {
        hccp_err("[init][rs_init]dl_hal_init failed, ret = %d", ret);
        return ret;
    }

    int counter = __sync_fetch_and_add(&(gInitCounter[cfg->chipId]), 1);
    if (counter > 0) {
        hccp_warn_init("rs has been init for device %u!", cfg->chipId);
        return 0;
    }
    ret = RsDev2rscb(cfg->chipId, &rscb, true);
    CHK_PRT_RETURN(ret == 0, hccp_err("rs_cb exist for device %u! do NOT init it again!", cfg->chipId), -EEXIST);

    rscb = calloc(1, sizeof(struct rs_cb));
    CHK_PRT_RETURN(rscb == NULL, hccp_err("calloc rscb failed"), -ENOMEM);

    ret = RsPthreadMutexInit(rscb, cfg);
    if (ret != 0) {
        hccp_err("Init mutex failed, ret[%d]", ret);
        goto pthread_mutex_err;
    }

    ret = RsInitRscbCfg(rscb);
    if (ret != 0) {
        hccp_err("rs init rscb configure failed,ret:%d", ret);
        pthread_mutex_destroy(&rscb->mutex);
        pthread_mutex_destroy(&rscb->connCb.connMutex);
        goto pthread_mutex_err;
    }

    rscb->fdMap = calloc(1, sizeof(void *) * RS_MAX_FD_NUM);
    if (rscb->fdMap == NULL) {
        hccp_err("no memory for fd_map");
        ret = -ENOMEM;
        goto fd_map_err;
    }

    ret = getifaddrs(&rscb->ifaddrList);
    if (ret != 0) {
        hccp_err("getifaddrs failed, ret:%d", ret);
        goto getifaddrs_err;
    }

    gRsCbList[cfg->chipId] = gRsCb;

    hccp_run_info("rs init success, chipId[%u]", cfg->chipId);
    return 0;

getifaddrs_err:
    free(rscb->fdMap);
    rscb->fdMap = NULL;

fd_map_err:
    pthread_mutex_destroy(&rscb->mutex);
    pthread_mutex_destroy(&rscb->connCb.connMutex);
    RsDeinitRscbCfg(rscb);

pthread_mutex_err:
    free(rscb);
    rscb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsGetTlsEnable(unsigned int phyId, bool *tlsEnable)
{
    struct rs_cb *rsCb = NULL;
    int ret;

    CHK_PRT_RETURN(tlsEnable == NULL, hccp_err("param err, tlsEnable is NULL"), -EINVAL);
    ret = RsGetRsCb(phyId, &rsCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, phyId(%u) invalid, ret(%d)", phyId, ret), ret);

    *tlsEnable = (rsCb->sslEnable == 0) ? false : true;
    return 0;
}

STATIC const char *RsGetHccnCfgFilePath(int mode)
{
#define HCCN_OFFLINE_CFGFILE_PATH "/etc/hccl.cfg"
#define HCCN_PEER_ONLINE_CFGFILE_PATH "/etc/hcomm.cfg"
    if (mode == NETWORK_OFFLINE) {
        return HCCN_OFFLINE_CFGFILE_PATH;
    }
    if (mode == NETWORK_PEER_ONLINE) {
        return HCCN_PEER_ONLINE_CFGFILE_PATH;
    }
    return NULL;
}

RS_ATTRI_VISI_DEF int RsGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    const char *keyName[HCCN_CFG_KEY_INVALID] = {
        "udp_port_mode", "multi_qp_count", "multi_qp_udp_ports", "resv_mem", "qos_dscp"};
    unsigned int cfg_key = (unsigned int)key;
    const char *cfgFilePath = NULL;
    unsigned int valLen = 0;
    unsigned int bufLen;
    int ret = 0;

    CHK_PRT_RETURN(info == NULL, hccp_err("param err, info is NULL"), -EINVAL);
    CHK_PRT_RETURN(value == NULL || valueLen == NULL, hccp_err("param err, value or valueLen is NULL"), -EINVAL);
    CHK_PRT_RETURN(cfg_key >= HCCN_CFG_KEY_INVALID,
        hccp_err("param err, key[%u] should < [%d]", cfg_key, HCCN_CFG_KEY_INVALID), -EINVAL);

    cfgFilePath = RsGetHccnCfgFilePath(info->mode);
    CHK_PRT_RETURN(cfgFilePath == NULL,
        hccp_err("param err, do not support mode[%d] phyId(%u)", info->mode, info->phyId), -ENOTSUPP);

    bufLen = *valueLen;
    CHK_PRT_RETURN(bufLen < HCCN_CFG_MSG_DATA_LEN,
        hccp_err("param err, bufLen[%u] should >= [%u]", bufLen, HCCN_CFG_MSG_DATA_LEN), -EINVAL);

    *valueLen = 0;
    ret = FileReadCfg(cfgFilePath, (int)info->phyId, keyName[cfg_key], value, bufLen);
    CHK_PRT_RETURN(ret == FILE_OPT_INNER_PARAM_ERR || ret == FILE_OPT_SYS_READ_FILE_ERR,
        hccp_run_warn("get hccn cfg file unsuccessful, ret(%d)", ret), 0);
    CHK_PRT_RETURN(ret == FILE_OPT_NO_MEM_ERR, hccp_err("value_len > buf_len[%d], ret(%d)", bufLen, ret), -ENOMEM);
    CHK_PRT_RETURN(ret != 0, hccp_run_warn("get hccn cfg [%s] unsuccessful, ret(%d)", keyName[cfg_key], ret), 0);

    valLen = (unsigned int)strlen(value);
    *valueLen = (valLen == 0) ? valLen : (valLen + 1);
    return 0;
}

RS_ATTRI_VISI_DEF int RsBindHostpid(unsigned int chipId, pid_t pid)
{
#define QUERY_BIND_HOST_PID_TIME_US 10000
#define QUERY_BIND_HOST_PID_CNT 12000
    struct rs_cb *rsCb = NULL;
    unsigned int hostPid;
    pid_t devPid;
    int ret;
    int i;

    // get current hccp pid on device
    devPid = getpid();
    CHK_PRT_RETURN(devPid < 0, hccp_err("getpid failed, ret:%d errno:%d", devPid, errno), -EINVAL);

    // query corresponding host_pid every 10ms, total timeout cost 120s
    for (i = 0; i < QUERY_BIND_HOST_PID_CNT; i++) {
        ret = DlDrvQueryProcessHostPid(devPid, NULL, NULL, &hostPid, NULL);
        if (ret == DRV_ERROR_NONE) {
            break;
        }

        usleep(QUERY_BIND_HOST_PID_TIME_US);
    }

    if (i >= QUERY_BIND_HOST_PID_CNT) {
        hccp_err("query process host_pid failed, i:%d >= %d ret:%d", i, QUERY_BIND_HOST_PID_CNT, ret);
        return -EINVAL;
    }

    if (pid != (pid_t)hostPid) {
        hccp_err("check process failed, pid from tsd: %d, process hostPid: %u", pid, hostPid);
        return -EINVAL;
    }

    hccp_dbg("dl_drv_query_process_host_pid success, total retry cnt:%d", i);

    // save host_pid for later setup sharemem
    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed, ret:%d, chipId:%u", ret, chipId), -ENODEV);
    rsCb->hostPid = pid;

    return 0;
}

#ifdef CUSTOM_INTERFACE
STATIC int RsSetRscbGrpId(struct rs_cb *rsCb, unsigned int devId)
{
    GrpQueryGroupIdInfo grpQueryOut = {0};
    unsigned int chipId = rsCb->chipId;
    GrpQueryGroupId grpQueryIn = {0};
    struct MemInfo memInfo = {0};
    unsigned int outLen;
    unsigned int grpId;
    int ret;

    // query grp_name
    ret = DlHalMemGetInfoEx(devId, MEM_INFO_TYPE_SVM_GRP_INFO, &memInfo);
    CHK_PRT_RETURN(ret, hccp_err("dl_hal_mem_get_info_ex failed, ret:%d chipId:%u devId:%u", ret, chipId, devId), ret);

    hccp_dbg("query group name success, chipId:%u devId:%u grp_name:%s", chipId, devId, memInfo.grp_info.name);

    // query grp_id
    ret = memcpy_s(&grpQueryIn.grpName, BUFF_GRP_NAME_LEN, &memInfo.grp_info.name, SVM_GRP_NAME_LEN);
    CHK_PRT_RETURN(ret, hccp_err("memcpy_s failed, ret:%d chipId:%u devId:%u", ret, chipId, devId), ret);
    outLen = (unsigned int)sizeof(grpQueryOut);
    ret = DlHalGrpQuery(GRP_QUERY_GROUP_ID, &grpQueryIn, sizeof(grpQueryIn), &grpQueryOut, &outLen);
    CHK_PRT_RETURN(ret, hccp_err("dl_hal_grp_query failed, ret:%d chipId:%u devId:%u", ret, chipId, devId), ret);
    grpId = (unsigned int)grpQueryOut.groupId;

    // set grp_id
    rsCb->grpId = grpId;

    hccp_dbg("query group id success, chipId:%u devId:%u grpId:%u grp_name:%s", chipId, devId, grpId,
        grpQueryIn.grpName);
    return 0;
}

STATIC int RsBindSibling(struct rs_cb *rsCb, int hostPid, unsigned int vfId, unsigned int devId)
{
#define QUERY_BIND_SIBLING_TIME_US 10000
#define QUERY_BIND_SIBLING_CNT 12000
    struct halQueryDevpidInfo pidInfo = {0};
    pid_t aicpuPid;
    int ret;
    int i;

    // query aicpu pid
    pidInfo.hostpid = hostPid;
    pidInfo.devid = devId;
    pidInfo.proc_type = DEVDRV_PROCESS_CP1;
    ret = DlHalQueryDevPid(pidInfo, &aicpuPid);
    CHK_PRT_RETURN(ret != 0, hccp_err("dl_hal_query_dev_pid failed, ret:%d devId:%u", ret, devId), ret);

    // try to bind sibling every 10ms, total timeout cost 120s
    for (i = 0; i < QUERY_BIND_SIBLING_CNT; i++) {
        ret = DlHalMemBindSibling(hostPid, aicpuPid, vfId, devId, SVM_MEM_BIND_SP_GRP);
        if (ret == DRV_ERROR_NONE) {
            break;
        }

        usleep(QUERY_BIND_SIBLING_TIME_US);
    }

    if (i >= QUERY_BIND_SIBLING_CNT) {
        hccp_err("bind sibling to setup sharemem failed, i:%d >= %d ret:%d", i, QUERY_BIND_SIBLING_CNT, ret);
        return -EINVAL;
    }

    rsCb->aicpuPid = aicpuPid;
    hccp_dbg("dl_hal_mem_bind_sibling success, total retry cnt:%d", i);

    return 0;
}

int RsSetupSharemem(struct rs_cb *rsCb, bool backupFlag, unsigned int backupPhyid)
{
    unsigned int chipId = rsCb->chipId;
    pid_t pid = rsCb->hostPid;
    int64_t deviceInfo = 0;
    unsigned int logicId;
    int ret;

    // setup sharemem or skipped already, no need to setup again
    if (rsCb->grpSetupFlag) {
        hccp_dbg("grp_setup_flag:%d grp_id:%u chipId:%u", rsCb->grpSetupFlag, rsCb->grpId, chipId);
        return 0;
    }

    ret = DlDrvDeviceGetIndexByPhyId(chipId, &logicId);
    CHK_PRT_RETURN(ret, hccp_err("dl_drv_device_get_index_by_phy_id failed, ret:%d chipId:%u", ret, chipId), ret);
    ret = DlHalGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_VERSION, &deviceInfo);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("dl_hal_get_device_info failed, ret:%d logicId:%u chipId:%u", ret, logicId, chipId), ret);
    // not 910b/910_93 and not protocol udma, skip to setup share mem
    if (DlHalPlatGetChip((uint64_t)deviceInfo) != CHIP_TYPE_910B_910_93 && rsCb->protocol != PROTOCOL_UDMA) {
        hccp_info_others("logicId:%u chipId:%u protocol:%d skip to setup share mem", logicId, chipId, rsCb->protocol);
        rsCb->grpSetupFlag = true;
        return 0;
    }

    // use backup info to setup share mem
    if (backupFlag) {
        ret = DlDrvGetLocalDevIdByHostDevId(backupPhyid, &logicId);
        CHK_PRT_RETURN(ret != 0, hccp_err("DlDrvGetLocalDevIdByHostDevId failed, phyId(%u), ret(%d)", backupPhyid, ret),
            ret);
        hccp_dbg("setup sharemem with backup, phyId:%u logicId:%u", backupPhyid, logicId);
    }

    // bind sibling, default vfid is 0; query & save grp_id on rs_cb
    ret = RsBindSibling(rsCb, pid, 0, logicId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_bind_sibling failed, ret:%d logicId:%u chipId:%u", ret, logicId, chipId),
        ret);

    // query & save grp_id on rs_cb
    ret = RsSetRscbGrpId(rsCb, logicId);
    CHK_PRT_RETURN(ret, hccp_err("rs_set_rscb_grp_id failed, ret:%d logicId:%u chipId:%u", ret, logicId, chipId), ret);

    rsCb->grpSetupFlag = true;
    return 0;
}
#endif

STATIC int RsCompareIpGid(struct rdev rdevInfo, union ibv_gid *gid)
{
    return RsDrvCompareIpGid(rdevInfo.family, rdevInfo.localIp, gid);
}

int RsQueryGid(struct rdev rdevInfo, struct ibv_context *ibCtxTmp, uint8_t ibPort, int *gidIdx)
{
    static const char *portStates[] = {"Nop", "Down", "Init", "Armed", "", "Active Defer"};
    struct ibv_port_attr attr = {0};
    enum ibv_gid_type_sysfs type;
    union ibv_gid gidTmp;
    int ret;
    int i;

    CHK_PRT_RETURN(gidIdx == NULL, hccp_err("gid_idx is NULL"), -EINVAL);

    ret = RsIbvQueryPort(ibCtxTmp, ibPort, &attr);
    CHK_PRT_RETURN(ret, hccp_err("ibv_query_port failed, ret %d ibPort %u", ret, ibPort), -EOPENSRC);

    for (i = 0; i < attr.gid_tbl_len; i++) {
        ret = RsIbvQueryGidType(ibCtxTmp, ibPort, (unsigned int)i, &type);
        CHK_PRT_RETURN(ret, hccp_err("query gid type failed i %d, ret %d", i, ret), -EOPENSRC);
        if (type != IBV_GID_TYPE_SYSFS_ROCE_V2) {
            continue;
        }
        ret = RsIbvQueryGid(ibCtxTmp, ibPort, i, &gidTmp);
        CHK_PRT_RETURN(ret, hccp_err("query gid failed i %d, ret %d", i, ret), -EOPENSRC);
        ret = RsCompareIpGid(rdevInfo, &gidTmp);
        if (ret == 0) {
            CHK_PRT_RETURN(attr.state != IBV_PORT_ACTIVE,
                hccp_err("port number %u state is %s", ibPort, portStates[attr.state]), -ENOLINK);
            *gidIdx = i;
            return 0;
        }
    }

    if (i == attr.gid_tbl_len) {
        return -EEXIST;
    }
    return 0;
}

STATIC int RsGetDevRdevIndex(struct RsRdevCb *rdevCb, unsigned int *rdevIndex, int index)
{
    (void)rdevCb;
    (void)rdevIndex;
    (void)index;
#ifdef CUSTOM_INTERFACE
    struct roce_dev_data rdevData = {0}; // lint !e565
    int retVal;

    if (RsIsCustomInterfaceSupported()) {
        RS_PTHREAD_MUTEX_LOCK(&rdevCb->rsCb->mutex);
        /*lint -e132*/
        rdevCb->devName = RsIbvGetDeviceName(rdevCb->devList[index]); // lint !e101
        retVal = RsRoceGetRoceDevData(rdevCb->devName, &rdevData);    // lint !e101
        /*lint +e132*/
        if (retVal) {
            hccp_err("rs_roce_get_roce_dev_data failed, retVal:%d, devName:%s", retVal, rdevCb->devName);
            RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
            return retVal;
        }
        *rdevIndex = rdevData.rdev_index; // rdev_index is same to port_id
        rdevCb->rdevIndex = *rdevIndex;
        RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
    }
#endif
    return 0;
}

STATIC int RsGetHostRdevIndex(struct rdev rdevInfo, struct RsRdevCb *rdevCb, unsigned int *rdevIndex, int index)
{
    struct RsRdevCb *rdevCbTmp2 = NULL;
    struct RsRdevCb *rdevCbTmp = NULL;
    unsigned int tmpRdevIndex = 0;

    RS_PTHREAD_MUTEX_LOCK(&rdevCb->rsCb->mutex);
    rdevCb->devName = RsIbvGetDeviceName(rdevCb->devList[index]);
    if (rdevCb->devName == NULL) {
        hccp_err("rs_ibv_get_device_name failed, errno:%d", errno);
        RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
        return -EINVAL;
    }

    struct RsIpAddrInfo localIp;
    int ret = RsConvertIpAddr(rdevInfo.family, &rdevInfo.localIp, &localIp);
    if (ret != 0) {
        hccp_err("convert(ntop) ip failed, ret:%d", ret);
        RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
        return ret;
    }

    RS_LIST_GET_HEAD_ENTRY(rdevCbTmp, rdevCbTmp2, &rdevCb->rsCb->rdevList, list, struct RsRdevCb);
    for (; (&rdevCbTmp->list) != &rdevCb->rsCb->rdevList;
         rdevCbTmp = rdevCbTmp2, rdevCbTmp2 = list_entry(rdevCbTmp2->list.next, struct RsRdevCb, list)) {
        tmpRdevIndex = rdevCbTmp->rdevIndex;
        if (!RsCompareIpAddr(&rdevCbTmp->localIp, &localIp)) {
            *rdevIndex = tmpRdevIndex;
            rdevCb->rdevIndex = *rdevIndex;
            rdevCb->localIp = localIp;
            RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
            return 0;
        }
    }

    *rdevIndex = tmpRdevIndex + 1;
    rdevCb->rdevIndex = *rdevIndex;
    rdevCb->localIp = localIp;
    RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rsCb->mutex);
    return 0;
}

STATIC int RsGetIbCtxAndRdevIndex(struct rdev rdevInfo, struct RsRdevCb *rdevCb, unsigned int *rdevIndex)
{
    struct ibv_context *ibCtxTmp = NULL;
    int gidIndex = -1;
    int ret;
    int i;

    for (i = 0; (i < rdevCb->devNum) && (rdevCb->devList[i] != NULL); ++i) { // lint !e101
        ibCtxTmp = RsIbvOpenDevice(rdevCb->devList[i]);
        CHK_PRT_RETURN(ibCtxTmp == NULL, hccp_err("ibv_open_device failed !"), -ENODEV);
        ret = RsQueryGid(rdevInfo, ibCtxTmp, rdevCb->ibPort, &gidIndex);
        if (ret == 0) {
            if (rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
                ret = RsGetHostRdevIndex(rdevInfo, rdevCb, rdevIndex, i);
            } else {
                ret = RsGetDevRdevIndex(rdevCb, rdevIndex, i);
            }
            if (ret != 0) {
                hccp_err("get index failed, ret:%d", ret);
                RsIbvCloseDevice(ibCtxTmp);
                return ret;
            }
            ret = RsIbvQueryDevice(ibCtxTmp, &rdevCb->deviceAttr);
            if (ret != 0) {
                hccp_err("query device failed, ret:%d", ret);
                RsIbvCloseDevice(ibCtxTmp);
                return ret;
            }
            rdevCb->ibCtx = ibCtxTmp;
            return 0;
        } else if (ret == -EEXIST) {
            RsIbvCloseDevice(ibCtxTmp);
        } else {
            RsIbvCloseDevice(ibCtxTmp);
            hccp_err("rs_query_gid failed, ret:%d", ret);
            return ret;
        }
    }

    CHK_PRT_RETURN(i == rdevCb->devNum,
        hccp_err("can not find ib_ctx for phyId[%u] local_ip[0x%x] in dev_list!", rdevInfo.phyId,
            rdevInfo.localIp.addr.s_addr),
        -EEXIST);
    return 0;
}

int RsGetRsCb(unsigned int phyId, struct rs_cb **rsCb)
{
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error ! phyId:%u", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsDev2rscb(chipId, rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed, ret:%d", ret), -ENODEV);
    return 0;
}

STATIC int RsGetSqDepthAndQpMaxNum(struct RsRdevCb *rdevCb, unsigned int rdevIndex)
{
    (void)rdevCb;
    (void)rdevIndex;
#ifdef CUSTOM_INTERFACE
    unsigned int tempDepth = 0;
    unsigned int qpMaxNum = 0;
    unsigned int sqDepth = 0;
    int ret;

    if (RsIsCustomInterfaceSupported()) {
        ret = RsRoceGetTsqpDepth(rdevCb->devName, rdevIndex, &tempDepth, &qpMaxNum, &sqDepth);
        CHK_PRT_RETURN(ret,
            hccp_err("rs_roce_get_tsqp_depth failed, ret:%d, devName:%s, rdevIndex:%u", ret, rdevCb->devName,
                rdevIndex),
            ret);

        rdevCb->txDepth = sqDepth;
        rdevCb->rxDepth = sqDepth;
        rdevCb->qpMaxNum = qpMaxNum;
        hccp_run_info("qp_max_num:%u, sqDepth:%u", qpMaxNum, sqDepth);
    }
#endif
    return 0;
}

STATIC int RsSetupPdAndNotify(struct RsRdevCb *rdevCb)
{
    int ret;

    ret = RsDrvQueryNotifyAndAllocPd(rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_query_notify_and_alloc_pd failed, ret[%d]", ret), ret);

    ret = RsDrvRegNotifyMr(rdevCb);
    if (ret) {
        hccp_err("reg notify mr failed, ret[%d]", ret);
        goto dealloc_pd;
    }

    return 0;
dealloc_pd:
    RsIbvDeallocPd(rdevCb->ibPd);
    return ret;
}

STATIC int RsRdevCbInfoInit(struct rdev rdevInfo, struct rs_cb *rsCb, struct RsRdevCb *rdevCb)
{
    int ret;

    rdevCb->ibPort = RS_PORT_DEF;
    rdevCb->rsCb = rsCb;
    rdevCb->notifyVaBase = rsCb->notifyVaBase;
    rdevCb->notifySize = rsCb->notifySize;

    rdevCb->localIp.family = (uint32_t)rdevInfo.family;
    rdevCb->localIp.binAddr = rdevInfo.localIp;
    ret = RsInetNtop(rdevInfo.family, &(rdevInfo.localIp), rdevCb->localIp.readAddr, RS_MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("rs_inet_ntop failed, ret %d", ret), -EINVAL);

    return 0;
}

STATIC int RsRdevCbInit(struct rdev rdevInfo, struct RsRdevCb *rdevCb, struct rs_cb *rsCb, unsigned int *rdevIndex)
{
    int ret;

    ret = RsRdevCbInfoInit(rdevInfo, rsCb, rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_rdev_cb_info_init failed, ret %d", ret), ret);

    ret = pthread_mutex_init(&rdevCb->rdevMutex, NULL);
    CHK_PRT_RETURN(ret, hccp_err("rdev_cb mutex_init failed ret %d!, normal ret 0", ret), -ESYSFUNC);

    ret = pthread_mutex_init(&rdevCb->cqeErrCntMutex, NULL);
    if (ret) {
        hccp_err("rdev_cb cqe_err_cnt_mutex init failed ret %d!, normal ret 0", ret);
        goto destroy_rdev_mutex;
    }

    RS_PTHREAD_MUTEX_LOCK(&rdevCb->rdevMutex);
    RS_INIT_LIST_HEAD(&rdevCb->qpList);
    RS_INIT_LIST_HEAD(&rdevCb->typicalMrList);
    RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rdevMutex);

    ret = RsGetIbCtxAndRdevIndex(rdevInfo, rdevCb, rdevIndex);
    if (ret) {
        hccp_err("rs_get_ib_ctx_and_rdev_index failed, ret:%d", ret);
        goto destroy_cqe_mutex;
    }

    ret = RsGetSqDepthAndQpMaxNum(rdevCb, *rdevIndex);
    if (ret) {
        hccp_err("rs_get_sq_depth_and_qp_max_num failed, ret[%d], rdevIndex[%u]", ret, *rdevIndex);
        goto close_dev;
    }

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        ret = RsRoceMmapAiDbReg(rdevCb->ibCtx, (unsigned int)rdevCb->rsCb->aicpuPid);
        if (ret) {
            hccp_err("rs_roce_mmap_ai_db_reg failed, ret[%d], rdevIndex[%u]", ret, *rdevIndex);
            goto close_dev;
        }
    }
#endif

    ret = RsInitNdaCb(rdevCb);
    if (ret != 0) {
        hccp_err("RsInitNdaCb failed, ret[%d], rdevIndex[%u]", ret, *rdevIndex);
        goto unmmap_ai_db;
    }

    ret = RsSetupPdAndNotify(rdevCb);
    if (ret != 0) {
        hccp_err("RsSetupPdAndNotify failed, ret[%d], rdevIndex[%u]", ret, *rdevIndex);
        goto free_nda_cb;
    }

    return 0;

free_nda_cb:
    RsDeinitNdaCb(rdevCb);
unmmap_ai_db:
#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        (void)RsRoceUnmmapAiDbReg(rdevCb->ibCtx);
    }
#endif
close_dev:
    RsIbvCloseDevice(rdevCb->ibCtx);
destroy_cqe_mutex:
    pthread_mutex_destroy(&rdevCb->cqeErrCntMutex);
destroy_rdev_mutex:
    pthread_mutex_destroy(&rdevCb->rdevMutex);
    return ret;
}

int RsRetryTimeoutExceptionCheck(struct SensorNode *sensorNode)
{
    int ret = 0;

    /* sensor may not support, handle is 0 */
    if (sensorNode->sensorHandle == 0) {
        return 0;
    }

    /*
     * The notification alarm framework does not filter alarms. In this example, only one notification
     * alarm is reported by a single process, which does not need to be accurate. Therefore, no lock is used.
     */
    if (sensorNode->sensorUpdateCnt == 0) {
        ret = DlHalSensorNodeUpdateState(sensorNode->logicDevid, sensorNode->sensorHandle,
            RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE, GENERAL_EVENT_TYPE_ONE_TIME);
        if (ret == 0) {
            sensorNode->sensorUpdateCnt++;
        }
    }

    return ret;
}

STATIC int RsRdevInitWithBackupInfo(struct rdev rdevInfo, struct RsBackupInfo backupInfo, unsigned int notifyType,
    unsigned int *rdevIndex)
{
    unsigned int phyId = rdevInfo.phyId;
    struct RsRdevCb *rdevCb = NULL;
    struct rs_cb *rsCb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(rdevIndex);

    ret = RsApiInit();
    CHK_PRT_RETURN(ret, hccp_err("RsApiInit failed! ret[%d]", ret), ret);

    ret = RsGetRsCb(phyId, &rsCb);
    if (ret) {
        hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret);
        goto get_rs_cb_fail;
    }

    rdevCb = calloc(1, sizeof(struct RsRdevCb));
    if (rdevCb == NULL) {
        hccp_err("calloc for rdev_cb failed");
        ret = -ENOMEM;
        goto get_rs_cb_fail;
    }

    rdevCb->backupInfo.backupFlag = backupInfo.backupFlag;
    (void)memcpy_s(&rdevCb->backupInfo.rdevInfo, sizeof(struct rdev), &backupInfo.rdevInfo, sizeof(struct rdev));
#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        // setup sharemem for aicpu rdma unfold
        ret = RsSetupSharemem(rsCb, rdevCb->backupInfo.backupFlag, rdevCb->backupInfo.rdevInfo.phyId);
        if (ret != 0) {
            hccp_err("[init][rs_rdev]RsSetupSharemem failed, phyId(%u), ret(%d)", phyId, ret);
            goto free_rs_cb;
        }
    }
#endif

    rdevCb->notifyType = notifyType;
    rdevCb->devList = RsIbvGetDeviceList(&(rdevCb->devNum));
    if (rdevCb->devList == NULL || rdevCb->devNum == 0) {
        hccp_err("dev_list is NULL, or devNum[%d] is 0", rdevCb->devNum);
        ret = -EINVAL;
        goto free_rs_cb;
    }

    ret = RsSensorNodeRegister(phyId, rsCb);
    if (ret != 0) {
        hccp_err("[init][rs_rdev]rs_sensor_node_register failed, phyId(%u), ret(%d)", phyId, ret);
        goto free_dev_list;
    }

    hccp_info_rma("ibv_get_device_list phyId[%d] dev_num[%d]", phyId, rdevCb->devNum);

    ret = RsRdevCbInit(rdevInfo, rdevCb, rsCb, rdevIndex);
    if (ret != 0) {
        RsSensorNodeUnregister(rdevCb->rsCb);
        hccp_err("rs_rdev_cb_init failed ret %d!, normal ret 0", ret);
        goto free_dev_list;
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
    RsListAddTail(&rdevCb->list, &rsCb->rdevList);
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);
    hccp_run_info("rdev init success, phyId:%u, localIp:%s, rdevIndex:%u", phyId, rdevCb->localIp.readAddr, *rdevIndex);
    return 0;

free_dev_list:
    RsIbvFreeDeviceList(rdevCb->devList);
free_rs_cb:
    free(rdevCb);
    rdevCb = NULL;
get_rs_cb_fail:
    RsApiDeinit();
    return ret;
}

RS_ATTRI_VISI_DEF int RsRdevInitWithBackup(struct rdev rdevInfo, struct rdev backupRdevInfo, unsigned int notifyType,
    unsigned int *rdevIndex)
{
    struct RsBackupInfo backupInfo = {0};

    backupInfo.backupFlag = true;
    (void)memcpy_s(&backupInfo.rdevInfo, sizeof(struct rdev), &backupRdevInfo, sizeof(struct rdev));

    return RsRdevInitWithBackupInfo(rdevInfo, backupInfo, notifyType, rdevIndex);
}

RS_ATTRI_VISI_DEF int RsRdevInit(struct rdev rdevInfo, unsigned int notifyType, unsigned int *rdevIndex)
{
    struct RsBackupInfo backupInfo = {0};

    return RsRdevInitWithBackupInfo(rdevInfo, backupInfo, notifyType, rdevIndex);
}

STATIC void RsDestroyQpList(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb *rdevCb, struct RsQpCb *qpCb,
    struct RsQpCb *qpCb2)
{
    int ret;

    if (!RsListEmpty(&rdevCb->qpList)) {
        hccp_warn_rma("qp list do not empty!");
        RS_LIST_GET_HEAD_ENTRY(qpCb, qpCb2, &rdevCb->qpList, list, struct RsQpCb);
        for (; (&qpCb->list) != &rdevCb->qpList;
             qpCb = qpCb2, qpCb2 = list_entry(qpCb2->list.next, struct RsQpCb, list)) {
            hccp_info_rma("qpn[%u] will be destroyed", qpCb->ibQp->qp_num);
            if (qpCb->ibQpEx != NULL) {
                ret = RsNdaQpDestroy(phyId, rdevIndex, qpCb->ibQp->qp_num);
            } else {
                ret = RsQpDestroy(phyId, rdevIndex, qpCb->ibQp->qp_num);
            }
            if (ret != 0) {
                hccp_err("rs_qp_destroy failed, ret:%d", ret);
            }
        }
    }

    return;
}

STATIC void RsFreeTypicalMrCb(struct RsRdevCb *devCb)
{
    struct RsListHead *typicalMrList = &devCb->typicalMrList;
    struct RsMrCb *mrCurr = NULL;
    struct RsMrCb *mrNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&devCb->rdevMutex);
    RS_LIST_GET_HEAD_ENTRY(mrCurr, mrNext, typicalMrList, list, struct RsMrCb);
    for (; (&mrCurr->list) != typicalMrList;
         mrCurr = mrNext, mrNext = list_entry(mrNext->list.next, struct RsMrCb, list)) {
        (void)RsDrvMrDereg(mrCurr->ibMr);
        RsListDel(&mrCurr->list);
        free(mrCurr);
        mrCurr = NULL;
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);

    hccp_info_rma("rs_free_typical_mr_cb is succ");
}

RS_ATTRI_VISI_DEF int RsRdevDeinit(unsigned int phyId, unsigned int notifyType, unsigned int rdevIndex)
{
    (void)notifyType;
    struct RsRdevCb *rdevCb = NULL;
    struct RsQpCb *qpCb2 = NULL;
    struct RsQpCb *qpCb = NULL;
    unsigned int chipId;
    int ret;

    hccp_info_init("rdev deinit start, phyId:%u, rdevIndex:%u", phyId, rdevIndex);
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error ! phyId:%u", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chipId[%u] failed, ret %d", chipId, ret), ret);

    if (rdevCb->notifyType != NO_USE && rdevCb->notifyMr != NULL) {
        ret = RsDrvMrDereg(rdevCb->notifyMr);
        if (ret) {
            hccp_err("rs_drv_mr_dereg failed, ret %d", ret);
        }
    }

    hccp_info_rma("poll_cqe_num[%d]", rdevCb->pollCqeNum);

    RsDestroyQpList(phyId, rdevIndex, rdevCb, qpCb, qpCb2);

    RsFreeTypicalMrCb(rdevCb);

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        (void)RsRoceUnmmapAiDbReg(rdevCb->ibCtx);
    }
#endif

    RsIbvDeallocPd(rdevCb->ibPd);

    RsDeinitNdaCb(rdevCb);

    RsIbvCloseDevice(rdevCb->ibCtx);

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        RsCloseBackupIbCtx(rdevCb);
    }
#endif

    pthread_mutex_destroy(&rdevCb->cqeErrCntMutex);

    pthread_mutex_destroy(&rdevCb->rdevMutex);

    RsIbvFreeDeviceList(rdevCb->devList);

    RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
    RsListDel(&rdevCb->list);
    RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
    RsSensorNodeUnregister(rdevCb->rsCb);
    RsApiDeinit();
    hccp_run_info("rdev deinit success, phyId:%u, rdevIndex:%u", phyId, rdevIndex);
    free(rdevCb);
    rdevCb = NULL;
    return 0;
}

STATIC void RsHeterogTcpFreeFdNode(struct RsHeterogTcpFdInfo *fdNode)
{
    int fd;

    RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
    fd = fdNode->fd;
    RsListDel(&fdNode->list);
    free(fdNode);
    fdNode = NULL;
    gRsCb->fdMap[fd] = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
}

RS_ATTRI_VISI_DEF int RsEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event)
{
    struct RsHeterogTcpFdInfo *fdNode = NULL;
    unsigned int tmpEvent = event;
    int fd = RS_FD_INVALID;
    int ret;

    if (event == RA_EPOLLONESHOT) {
        tmpEvent = EPOLLIN | EPOLLET | EPOLLONESHOT;
    } else if (event == RA_EPOLLIN) {
        tmpEvent = EPOLLIN;
    } else {
        hccp_err("unknown event[%u]", tmpEvent);
        return -EINVAL;
    }

    if (gRsCb == NULL) {
        gRsCb = RsGetCurRsCb();
        if (gRsCb == NULL) {
            hccp_err("[rs_epoll_ctl_add]rs_get_cur_rs_cb failed rs_cb(NULL)");
            return -EINVAL;
        }
    }
    tmpEvent = tmpEvent | EPOLLRDHUP;
    fdNode = calloc(1, sizeof(struct RsHeterogTcpFdInfo));
    CHK_PRT_RETURN(fdNode == NULL, hccp_err("no memory for fd_node"), -ENOMEM);

    fd = ((const struct SocketPeerInfo *)fdHandle)->fd;
    fdNode->fd = fd;
    RS_PTHREAD_MUTEX_LOCK(&gRsCb->mutex);
    RsListAddTail(&fdNode->list, &gRsCb->heterogTcpFdList);
    gRsCb->fdMap[fd] = fdHandle;
    RS_PTHREAD_MUTEX_ULOCK(&gRsCb->mutex);
    ret = RsEpollCtl(gRsCb->connCb.epollfd, EPOLL_CTL_ADD, fd, tmpEvent);
    if (ret != 0) {
        hccp_err("[rs_epoll_ctl_add]RsEpollCtl failed ret(%d), fd:%d, event:%u", ret, fd, event);
        goto out;
    }
    return 0;
out:
    RsHeterogTcpFreeFdNode(fdNode);
    fdNode = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsEpollCtlMod(const void *fdHandle, enum RaEpollEvent event)
{
    unsigned int tmpEvent = event;
    int fd = RS_FD_INVALID;
    int ret;

    if (event == RA_EPOLLONESHOT) {
        tmpEvent = EPOLLIN | EPOLLET | EPOLLONESHOT;
    } else if (event == RA_EPOLLIN) {
        tmpEvent = EPOLLIN;
    } else {
        hccp_err("unknown event[%u]", event);
        return -EINVAL;
    }

    tmpEvent = tmpEvent | EPOLLRDHUP;
    fd = ((const struct SocketPeerInfo *)fdHandle)->fd;

    if (gRsCb == NULL) {
        gRsCb = RsGetCurRsCb();
        if (gRsCb == NULL) {
            hccp_err("[rs_epoll_ctl_mod]rs_get_cur_rs_cb failed rs_cb(NULL)");
            return -EINVAL;
        }
    }

    ret = RsEpollCtl(gRsCb->connCb.epollfd, EPOLL_CTL_MOD, fd, tmpEvent);
    CHK_PRT_RETURN(ret, hccp_err("[rs_epoll_ctl_mod]RsEpollCtl failed ret(%d), fd:%d, event:%u", ret, fd, event), ret);
    return 0;
}

RS_ATTRI_VISI_DEF int RsEpollCtlDel(int fd)
{
    int ret;
    struct RsHeterogTcpFdInfo *fdNode = NULL;
    struct RsHeterogTcpFdInfo *fdNode1 = NULL;

    if (gRsCb == NULL) {
        gRsCb = RsGetCurRsCb();
        if (gRsCb == NULL) {
            hccp_err("[rs_epoll_ctl_del]rs_get_cur_rs_cb failed rs_cb(NULL)");
            return -EINVAL;
        }
    }
    RS_LIST_GET_HEAD_ENTRY(fdNode, fdNode1, &gRsCb->heterogTcpFdList, list, struct RsHeterogTcpFdInfo);
    for (; (&fdNode->list) != &gRsCb->heterogTcpFdList;
         fdNode = fdNode1, fdNode1 = list_entry(fdNode1->list.next, struct RsHeterogTcpFdInfo, list)) {
        if (fdNode->fd == fd) {
            // 删除节点
            RsHeterogTcpFreeFdNode(fdNode);
            fdNode = NULL;
            break; // lint !e108
        }
    }

    // 为了兼容epoll不同版本，这里加EPOLLIN参数
    ret = RsEpollCtl(gRsCb->connCb.epollfd, EPOLL_CTL_DEL, fd, EPOLLIN);
    CHK_PRT_RETURN(ret, hccp_err("[rs_epoll_ctl_del]RsEpollCtl failed ret(%d), fd:%d", ret, fd), ret);
    return 0;
}

RS_ATTRI_VISI_DEF void RsSetTcpRecvCallback(const void *callback)
{
    if (gRsCb == NULL) {
        hccp_err("param error, gRsCb is NULL");
        return;
    }
    gRsCb->tcpRecvCallback = (void (*)(const void *))callback;
}

STATIC void RsFreeAcceptOneNode(struct rs_cb *rscb, struct RsAcceptInfo *accept)
{
    int ret;

    ret = RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, accept->connFd, EPOLLIN);
    if (ret) {
        hccp_err("epoll ctl del fd %d failed, ret:%d", accept->connFd, ret);
    }

    RS_PTHREAD_MUTEX_LOCK(&rscb->connCb.connMutex);
    RsListDel(&accept->list);
    RS_PTHREAD_MUTEX_ULOCK(&rscb->connCb.connMutex);

    if (rscb->sslEnable == RS_SSL_ENABLE) {
        if (accept->ssl == NULL) {
            hccp_warn_socket("[Server] accept->ssl is NULL, it maybe has not establish tls link");
        } else {
            ssl_adp_shutdown(accept->ssl);
            ssl_adp_free(accept->ssl);
            accept->ssl = NULL;
        }
    }

    RS_CLOSE_RETRY_FOR_EINTR(ret, accept->connFd);

    hccp_info_socket("free accept_server IP:%s, port:%d, connFd:%d", accept->serverIpAddr.readAddr, accept->sockPort,
        accept->connFd);
    accept->connFd = RS_FD_INVALID;

    free(accept);
    accept = NULL;
}

STATIC void RsFreeAccpetList(struct rs_cb *rscb)
{
    struct RsAcceptInfo *accept = NULL;
    struct RsAcceptInfo *accept2 = NULL;

    if (!RsListEmpty(&rscb->connCb.serverAcceptList)) {
        hccp_warn_socket("Server accept list do not empty!");
        RS_LIST_GET_HEAD_ENTRY(accept, accept2, &rscb->connCb.serverAcceptList, list, struct RsAcceptInfo);
        for (; (&accept->list) != &rscb->connCb.serverAcceptList;
             accept = accept2, accept2 = list_entry(accept2->list.next, struct RsAcceptInfo, list)) {
            RsFreeAcceptOneNode(rscb, accept);
            accept = NULL;
        }
    }

    return;
}

STATIC void RsFreeDesignatedAccpetNode(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    struct RsAcceptInfo *accept = NULL;
    struct RsAcceptInfo *accept2 = NULL;

    if (!RsListEmpty(&rscb->connCb.serverAcceptList)) {
        RS_LIST_GET_HEAD_ENTRY(accept, accept2, &rscb->connCb.serverAcceptList, list, struct RsAcceptInfo);
        for (; (&accept->list) != &rscb->connCb.serverAcceptList;
             accept = accept2, accept2 = list_entry(accept2->list.next, struct RsAcceptInfo, list)) {
            if (!RsCompareIpAddr(&accept->serverIpAddr, localIp)) {
                RsFreeAcceptOneNode(rscb, accept);
                accept = NULL;
            }
        }
    }

    return;
}

STATIC void RsFreeConnOneNode(struct rs_cb *rscb, struct RsConnInfo *conn)
{
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&rscb->connCb.connMutex);
    RsListDel(&conn->list);
    RS_PTHREAD_MUTEX_ULOCK(&rscb->connCb.connMutex);

    if (rscb->sslEnable == RS_SSL_ENABLE) {
        if (conn->ssl == NULL) {
            hccp_warn_socket("[Client] conn->ssl is NULL, it maybe has not establish tls link");
        } else {
            ssl_adp_shutdown(conn->ssl);
            ssl_adp_free(conn->ssl);
            conn->ssl = NULL;
        }
    }

    RS_CLOSE_RETRY_FOR_EINTR(ret, conn->connfd);

    hccp_info_socket("free for conn IP:%s, port:%d, connfd:%d, state:%u", conn->clientIp.readAddr, conn->port,
        conn->connfd, conn->state);

    conn->connfd = RS_FD_INVALID;
    conn->state = RS_CONN_STATE_RESET;

    free(conn);
    conn = NULL;
}

STATIC void RsFreeClientConnList(struct rs_cb *rscb)
{
    struct RsConnInfo *conn = NULL;
    struct RsConnInfo *conn2 = NULL;

    if (!RsListEmpty(&rscb->connCb.clientConnList)) {
        hccp_warn_socket("Client conn node do not empty!");
        RS_LIST_GET_HEAD_ENTRY(conn, conn2, &rscb->connCb.clientConnList, list, struct RsConnInfo);
        for (; (&conn->list) != &rscb->connCb.clientConnList;
             conn = conn2, conn2 = list_entry(conn2->list.next, struct RsConnInfo, list)) {
            RsFreeConnOneNode(rscb, conn);
            conn = NULL;
        }
    }

    return;
}

STATIC void RsFreeDesignatedClientConnNode(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    struct RsConnInfo *conn = NULL;
    struct RsConnInfo *conn2 = NULL;

    if (!RsListEmpty(&rscb->connCb.clientConnList)) {
        RS_LIST_GET_HEAD_ENTRY(conn, conn2, &rscb->connCb.clientConnList, list, struct RsConnInfo);
        for (; (&conn->list) != &rscb->connCb.clientConnList;
             conn = conn2, conn2 = list_entry(conn2->list.next, struct RsConnInfo, list)) {
            if (!RsCompareIpAddr(&conn->clientIp, localIp)) {
                hccp_warn_socket("Client conn node for IP[%s] do not empty!", localIp->readAddr);
                RsFreeConnOneNode(rscb, conn);
                conn = NULL;
            }
        }
    }

    return;
}

STATIC void RsFreeServerConnList(struct rs_cb *rscb)
{
    struct RsConnInfo *conn = NULL;
    struct RsConnInfo *conn2 = NULL;

    if (!RsListEmpty(&rscb->connCb.serverConnList)) {
        hccp_warn_socket("Server conn node do not empty!");
        RS_LIST_GET_HEAD_ENTRY(conn, conn2, &rscb->connCb.serverConnList, list, struct RsConnInfo);
        for (; (&conn->list) != &rscb->connCb.serverConnList;
             conn = conn2, conn2 = list_entry(conn2->list.next, struct RsConnInfo, list)) {
            RsFreeConnOneNode(rscb, conn);
            conn = NULL;
        }
    }

    return;
}

STATIC void RsFreeDesignatedServerConnNode(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    struct RsConnInfo *conn = NULL;
    struct RsConnInfo *conn2 = NULL;

    if (!RsListEmpty(&rscb->connCb.serverConnList)) {
        RS_LIST_GET_HEAD_ENTRY(conn, conn2, &rscb->connCb.serverConnList, list, struct RsConnInfo);
        for (; (&conn->list) != &rscb->connCb.serverConnList;
             conn = conn2, conn2 = list_entry(conn2->list.next, struct RsConnInfo, list)) {
            if (!RsCompareIpAddr(&conn->serverIp, localIp)) {
                hccp_warn_socket("Server conn node for IP[%s] do not empty!", localIp->readAddr);
                RsFreeConnOneNode(rscb, conn);
                conn = NULL;
            }
        }
    }
    return;
}

STATIC void RsFreeListenOneNode(struct rs_cb *rscb, struct RsListenInfo *listen)
{
    int ret;

    ret = RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, listen->listenFd, EPOLLIN);
    if (ret) {
        hccp_err("delete from epoll failed, ret:%d, epollfd:%d, listenFd:%d", ret, rscb->connCb.epollfd,
            listen->listenFd);
    }

    RS_PTHREAD_MUTEX_LOCK(&rscb->connCb.connMutex);
    RsListDel(&listen->list);
    RS_PTHREAD_MUTEX_ULOCK(&rscb->connCb.connMutex);

    RS_CLOSE_RETRY_FOR_EINTR(ret, listen->listenFd);

    hccp_info_socket("free Listen IP:%s, port:%d, listenFd:%d, state:%u", listen->serverIpAddr.readAddr,
        ntohs(listen->sockPort), listen->listenFd, listen->state);

    listen->listenFd = RS_FD_INVALID;
    listen->state = RS_CONN_STATE_RESET;

    free(listen);
}

STATIC void RsFreeListenList(struct rs_cb *rscb)
{
    struct RsListenInfo *listen = NULL;
    struct RsListenInfo *listen2 = NULL;

    if (!RsListEmpty(&rscb->connCb.listenList)) {
        hccp_warn_socket("Server listen node do not empty!");
        RS_LIST_GET_HEAD_ENTRY(listen, listen2, &rscb->connCb.listenList, list, struct RsListenInfo);
        for (; (&listen->list) != &rscb->connCb.listenList;
             listen = listen2, listen2 = list_entry(listen2->list.next, struct RsListenInfo, list)) {
            RsFreeListenOneNode(rscb, listen);
            listen = NULL;
        }
    }

    return;
}

STATIC void RsFreeDesignatedListenNode(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    struct RsListenInfo *listen = NULL;
    struct RsListenInfo *listen2 = NULL;

    if (!RsListEmpty(&rscb->connCb.listenList)) {
        RS_LIST_GET_HEAD_ENTRY(listen, listen2, &rscb->connCb.listenList, list, struct RsListenInfo);
        for (; (&listen->list) != &rscb->connCb.listenList;
             listen = listen2, listen2 = list_entry(listen2->list.next, struct RsListenInfo, list)) {
            if (!RsCompareIpAddr(&listen->serverIpAddr, localIp)) {
                RsFreeListenOneNode(rscb, listen);
                listen = NULL;
            }
        }
    }

    return;
}

STATIC void RsWhiteListNodeFree(struct rs_cb *rscb, struct RsWhiteList *wlist)
{
    struct RsWhiteListInfo *wlistNode = NULL;
    struct RsWhiteListInfo *wlistNode1 = NULL;

    if (!RsListEmpty(&wlist->whiteList)) {
        RS_LIST_GET_HEAD_ENTRY(wlistNode, wlistNode1, &wlist->whiteList, list, struct RsWhiteListInfo);
        for (; (&wlistNode->list) != &wlist->whiteList;
             wlistNode = wlistNode1, wlistNode1 = list_entry(wlistNode1->list.next, struct RsWhiteListInfo, list)) {
            RS_PTHREAD_MUTEX_LOCK(&rscb->connCb.connMutex);
            RsListDel(&wlistNode->list);
            RS_PTHREAD_MUTEX_ULOCK(&rscb->connCb.connMutex);

            hccp_info_socket("free White list client IP:%s, tag:%s", wlistNode->clientIp.readAddr, wlistNode->tag);
            free(wlistNode);
            wlistNode = NULL;
        }
    }
}

STATIC void RsFreeWhiteOneNode(struct rs_cb *rscb, struct RsWhiteList *wlist)
{
    RsWhiteListNodeFree(rscb, wlist);

    RS_PTHREAD_MUTEX_LOCK(&rscb->connCb.connMutex);
    RsListDel(&wlist->list);
    RS_PTHREAD_MUTEX_ULOCK(&rscb->connCb.connMutex);

    hccp_info_socket("White list server IP:%s", wlist->serverIp.readAddr);
    free(wlist);
    wlist = NULL;
}

STATIC void RsFreeWhiteList(struct rs_cb *rscb)
{
    struct RsWhiteList *wlist = NULL;
    struct RsWhiteList *wlist2 = NULL;

    if (!RsListEmpty(&rscb->connCb.whiteList)) {
        hccp_warn_socket("Server white list do not empty!");
        RS_LIST_GET_HEAD_ENTRY(wlist, wlist2, &rscb->connCb.whiteList, list, struct RsWhiteList);
        for (; (&wlist->list) != &rscb->connCb.whiteList;
             wlist = wlist2, wlist2 = list_entry(wlist2->list.next, struct RsWhiteList, list)) {
            RsFreeWhiteOneNode(rscb, wlist);
            wlist = NULL;
        }
    }

    return;
}

STATIC void RsFreeDesignatedWhiteNode(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    struct RsWhiteList *wlist = NULL;
    struct RsWhiteList *wlist2 = NULL;

    if (!RsListEmpty(&rscb->connCb.whiteList)) {
        RS_LIST_GET_HEAD_ENTRY(wlist, wlist2, &rscb->connCb.whiteList, list, struct RsWhiteList);
        for (; (&wlist->list) != &rscb->connCb.whiteList;
             wlist = wlist2, wlist2 = list_entry(wlist2->list.next, struct RsWhiteList, list)) {
            if (!RsCompareIpAddr(&wlist->serverIp, localIp)) {
                RsFreeWhiteOneNode(rscb, wlist);
                wlist = NULL;
            }
        }
    }

    return;
}

STATIC void RsFreeSocketList(struct rs_cb *rscb, struct RsIpAddrInfo *localIp)
{
    RsFreeDesignatedAccpetNode(rscb, localIp);
    RsFreeDesignatedClientConnNode(rscb, localIp);

    RsFreeDesignatedServerConnNode(rscb, localIp);

    RsFreeDesignatedListenNode(rscb, localIp);

    RsFreeDesignatedWhiteNode(rscb, localIp);

    return;
}

RS_ATTRI_VISI_DEF int RsSocketDeinit(struct rdev rdevInfo)
{
    int ret;
    unsigned int phyId = rdevInfo.phyId;
    unsigned int chipId;
    struct rs_cb *rscb = NULL;

    hccp_info_init("rs socket deinit start, phyId:%u", phyId);
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error ! phyId:%u", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    CHK_PRT_RETURN((rdevInfo.family != AF_INET) && (rdevInfo.family != AF_INET6),
        hccp_err("family[%d] invalid", rdevInfo.family), -EPROTONOSUPPORT);

    if (rdevInfo.family == AF_INET) {
        unsigned int *localIp = NULL;
        localIp = &(rdevInfo.localIp.addr.s_addr);
        ret = RsSocketNodeid2vnic(*localIp, localIp);
        hccp_info_init("socket deinit local IP is 0x%llx, ret:%d", *localIp, ret);
    }

    struct RsIpAddrInfo localIp;
    RsConvertIpAddr(rdevInfo.family, &rdevInfo.localIp, &localIp);

    ret = RsDev2rscb(chipId, &rscb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rscb failed for chipId:%u, ret:%d", chipId, ret), -ENODEV);

    RS_PTHREAD_MUTEX_LOCK(&rscb->mutex);
    RsFreeSocketList(rscb, &localIp);
    RS_PTHREAD_MUTEX_ULOCK(&rscb->mutex);
    hccp_run_info("socket deinit success, phyId:%u, localIp:%s", phyId, localIp.readAddr);
    return 0;
}

STATIC void RsFreeRdevList(struct rs_cb *rsCb)
{
    struct RsRdevCb *rdevCbCurr = NULL;
    struct RsRdevCb *rdevCbNext = NULL;
    unsigned int phyId = 0;
    int ret;

    if (RsListEmpty(&rsCb->rdevList)) {
        return;
    }

    hccp_run_warn("Rdev list is not empty!");
    ret = rsGetDevIDByLocalDevID(rsCb->chipId, &phyId);
    if (ret != 0) {
        hccp_err("chipId[%u] invalid, ret %d", rsCb->chipId, ret);
        return;
    }

    RS_LIST_GET_HEAD_ENTRY(rdevCbCurr, rdevCbNext, &rsCb->rdevList, list, struct RsRdevCb);
    for (; (&rdevCbCurr->list) != &rsCb->rdevList;
         rdevCbCurr = rdevCbNext, rdevCbNext = list_entry(rdevCbNext->list.next, struct RsRdevCb, list)) {
        ret = RsRdevDeinit(phyId, rdevCbCurr->notifyType, rdevCbCurr->rdevIndex);
        if (ret != 0) {
            hccp_err("rs_rdev_deinit failed, ret:%d, phyId:%u", ret, phyId);
        }
    }

    return;
}

STATIC void RsFreeUdevList(struct rs_cb *rsCb)
{
    struct RsUbDevCb *udevCbCurr = NULL;
    struct RsUbDevCb *udevCbNext = NULL;
    int ret;

    if (RsListEmpty(&rsCb->udevList)) {
        return;
    }

    hccp_run_warn("Udev list is not empty!");
    RS_LIST_GET_HEAD_ENTRY(udevCbCurr, udevCbNext, &rsCb->udevList, list, struct RsUbDevCb);
    for (; (&udevCbCurr->list) != &rsCb->udevList;
         udevCbCurr = udevCbNext, udevCbNext = list_entry(udevCbNext->list.next, struct RsUbDevCb, list)) {
        ret = RsUbCtxDeinit(udevCbCurr);
        if (ret != 0) {
            hccp_err("rs_ub_ctx_deinit failed, ret:%d", ret);
        }
    }

    return;
}

STATIC void RsFreeHeterogTcpFdList(struct rs_cb *rsCb)
{
    struct RsHeterogTcpFdInfo *fdNode = NULL;
    struct RsHeterogTcpFdInfo *fdNode1 = NULL;

    if (!RsListEmpty(&rsCb->heterogTcpFdList)) {
        hccp_warn_socket("heterog_tcp_fd_list do not empty!");
        RS_LIST_GET_HEAD_ENTRY(fdNode, fdNode1, &rsCb->heterogTcpFdList, list, struct RsHeterogTcpFdInfo);
        for (; (&fdNode->list) != &rsCb->heterogTcpFdList;
             fdNode = fdNode1, fdNode1 = list_entry(fdNode1->list.next, struct RsHeterogTcpFdInfo, list)) {
            hccp_info_socket(">>>>>fd_node->fd:%d", fdNode->fd);
            // 删除节点
            RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
            RsListDel(&fdNode->list);
            free(fdNode);
            fdNode = NULL;
            RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);
        }
    }

    return;
}

STATIC void RsListFree(struct rs_cb *rscb)
{
    RsFreeAccpetList(rscb);
    RsFreeClientConnList(rscb);

    RsFreeServerConnList(rscb);

    RsFreeListenList(rscb);

    RsFreeWhiteList(rscb);

    return;
}

STATIC void RsSslFree(struct rs_cb *rscb)
{
    if (rscb->sslEnable == RS_SSL_ENABLE) {
        if (rscb->skidSubjectCb != NULL) {
            if (memset_s(rscb->skidSubjectCb, sizeof(struct RsCertSkidSubjectCb), 0,
                    sizeof(struct RsCertSkidSubjectCb))) {
                hccp_warn_others("memset_s for skid_subject_cb unsuccessful");
            }
            free(rscb->skidSubjectCb);
            rscb->skidSubjectCb = NULL;
        }
        ssl_adp_ctx_free(rscb->serverSslCtx);
        rscb->serverSslCtx = NULL;
        ssl_adp_ctx_free(rscb->clientSslCtx);
        rscb->clientSslCtx = NULL;
    }
}

STATIC void RsDeinitFreeRscb(struct rs_cb *rscb)
{
    RS_PTHREAD_MUTEX_LOCK(&rscb->mutex);
    RsListFree(rscb);

    free(rscb->fdMap);
    rscb->fdMap = NULL;
    freeifaddrs(rscb->ifaddrList);
    rscb->ifaddrList = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&rscb->mutex);
    RsFreeRdevList(rscb);
    RsFreeUdevList(rscb);
    RsSslFree(rscb);
    RsFreeHeterogTcpFdList(rscb);
#ifdef CONFIG_TLV
    if (RsIsTlvSupported()) {
        if (rscb->tlvCb.initFlag) {
            RsTlvDeinit(rscb->tlvCb.phyId);
        }
    }
#endif
    pthread_mutex_destroy(&rscb->mutex);
    pthread_mutex_destroy(&rscb->connCb.connMutex);
    RsDestroyEpoll(rscb);

#ifdef CUSTOM_INTERFACE
    if (RsIsUdmaSupported() || RsIsRdmaSupported()) {
        RsDeInitNetAdapt(rscb);
        RsEschedDeinit(rscb->protocol);
        (void)RsCtxApiDeinit(rscb->hccpMode, rscb->protocol);
    }
#endif

    free(rscb);
    rscb = NULL;
    gRsCb = NULL;
}

RS_ATTRI_VISI_DEF int RsDeinit(struct RsInitConfig *cfg)
{
    struct rs_cb *rscb = gRsCb;
    unsigned int chipId;
    eventfd_t event;
    int ret;

    CHK_PRT_RETURN(cfg == NULL, hccp_err("param error, cfg is NULL"), -EINVAL);

    chipId = cfg->chipId;
    if (__sync_fetch_and_sub(&(gInitCounter[chipId]), 1) > 1) {
        return 0;
    }
    if (rscb && (chipId == rscb->chipId)) {
        event = 1;
        /* send event to eventfd to waking up epoll handle thread */
        ret = (int)write(rscb->connCb.eventfd, &event, sizeof(eventfd_t));
        CHK_PRT_RETURN(ret != sizeof(eventfd_t),
            hccp_err("eventfd_write failed(0x%x), chipId:%u, errno:%d", ret, chipId, errno), -EFILEOPER);

        hccp_info_socket("epoll wait up ok, rscb->connFlag:%d", rscb->connFlag);
        // already been RS_CONN_EXIT_FLAG, no need to change conn_flag
        if (rscb->connFlag != RS_CONN_EXIT_FLAG) {
            rscb->connFlag = 0;
        }
        int tryAgain = RS_TRY_TIME;
        while (((rscb->state & RS_STATE_HALT) == 0) && tryAgain > 0) {
            usleep(RS_USLEEP_TIME);
            tryAgain--;
        };

        if (tryAgain == 0) {
            hccp_warn_others("try_again exhausted, rscb state:%u", rscb->state);
        }

        tryAgain = RS_TRY_TIME;
        while ((rscb->connFlag != RS_CONN_EXIT_FLAG) && tryAgain > 0) {
            usleep(RS_USLEEP_TIME);
            tryAgain--;
        }

        CHK_PRT_RETURN(tryAgain == 0, hccp_warn_others("connect thread quit unsuccessful"), -EAGAIN);
        rscb->state &= ~RS_STATE_HALT;
        RsDeinitFreeRscb(rscb);
        gRsCbList[chipId] = NULL;
        DlHalDeinit();

        hccp_run_info("rs_deinit chipId[%u] ok", chipId);

        return 0;
    }

    DlHalDeinit();
    return -ENODEV;
}

RS_ATTRI_VISI_DEF int RsGetVnicIp(unsigned int phyId, unsigned int *vnicIp)
{
    int64_t deviceInfo = 0;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);
    CHK_PRT_RETURN(vnicIp == NULL, hccp_err("vnic_ip is null!"), -EINVAL);

    ret = DlHalGetDeviceInfo(phyId, MODULE_TYPE_SYSTEM, INFO_TYPE_VNIC_IP, &deviceInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("phyId:%u dl_hal_get_device_info failed! ret:%d", phyId, ret), ret);

    *vnicIp = (unsigned int)deviceInfo;
    return 0;
}

STATIC int RsGetVnicIpInfo(unsigned int phyId, unsigned int id, enum IdType type, struct IpInfo *info)
{
    int64_t deviceInfo = 0;
    unsigned int vnicIp;
    int ret;

    // get vnic ip by id with different type
    if (type == PHY_ID_VNIC_IP) {
        ret = DlHalGetDeviceInfo(id, MODULE_TYPE_SYSTEM, INFO_TYPE_VNIC_IP, &deviceInfo);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("cur_phy_id:%u dl_hal_get_device_info failed! phyId:%u ret:%d", phyId, id, ret), ret);
    } else if (type == SDID_VNIC_IP) {
        ret = DlHalGetDeviceInfo(id, MODULE_TYPE_SYSTEM, INFO_TYPE_SPOD_VNIC_IP, &deviceInfo);
        CHK_PRT_RETURN(ret != 0, hccp_err("phyId:%u dl_hal_get_device_info failed! sdid:0x%x ret:%d", phyId, id, ret),
            ret);
    } else {
        hccp_err("phyId:%u get vnic ip failed! id:0x%x, invalid type:%u", phyId, id, type);
        return -EINVAL;
    }

    // prepare ip info, only support IPv4
    vnicIp = (unsigned int)deviceInfo;
    info->family = AF_INET;
    info->ip.addr.s_addr = vnicIp;

    hccp_dbg("phyId:%u query id:%u type:%u got vnic_ip:%u", phyId, id, type, vnicIp);

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[])
{
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(ids == NULL, hccp_err("phyId:%u, ids is null!", phyId), -EINVAL);
    CHK_PRT_RETURN(infos == NULL, hccp_err("phyId:%u, infos is null!", phyId), -EINVAL);

    for (i = 0; i < num; i++) {
        ret = RsGetVnicIpInfo(phyId, ids[i], type, &infos[i]);
        if (ret != 0) {
            hccp_err("phyId:%u get vnic ip info failed! ids[%u]:0x%x type:%u", phyId, i, ids[i], type);
            return ret;
        }
    }

    return 0;
}

int rsGetLocalDevIDByHostDevID(unsigned int phyId, unsigned int *chipId)
{
    CHK_PRT_RETURN(gRsCb == NULL, hccp_warn_init("No device initialized !"), -ENODEV);

    if (gRsCb->hccpMode == NETWORK_PEER_ONLINE) {
        *chipId = phyId;
        return 0;
    } else {
        // to be compatible with mdev scenario: ignore phyId to get current devId(chipId)
        return DlDrvQueryProcessHostPid(getpid(), chipId, NULL, NULL, NULL);
    }
}

int rsGetDevIDByLocalDevID(unsigned int chipId, unsigned int *phyId)
{
    CHK_PRT_RETURN(gRsCb == NULL, hccp_warn_init("No device initialized !"), -ENODEV);

    if (gRsCb->hccpMode == NETWORK_PEER_ONLINE) {
        *phyId = chipId;
        return 0;
    } else {
        return DlDrvGetDevIdByLocalDevId(chipId, phyId);
    }
}

RS_ATTRI_VISI_DEF int RsSetQpAttrQos(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct QosAttr *attr)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    qpCb->qosAttr.tc = attr->tc;
    qpCb->qosAttr.sl = attr->sl;

    hccp_info_rma("set qp qos attr: qpn[%u] tc[%u] sl[%u]", qpn, attr->tc, attr->sl);
    return 0;
}

RS_ATTRI_VISI_DEF int RsSetQpAttrTimeout(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    unsigned int *timeout)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    qpCb->timeout = *timeout;

    hccp_info_rma("set qp qos attr: qpn[%u] timeout[%u]", qpn, *timeout);
    return 0;
}

RS_ATTRI_VISI_DEF int RsSetQpAttrRetryCnt(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    unsigned int *retryCnt)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    qpCb->retryCnt = *retryCnt;

    hccp_info_rma("set qp qos attr: qpn[%u] retry_cnt[%u]", qpn, *retryCnt);
    return 0;
}

RS_ATTRI_VISI_DEF int RsGetCqeErrInfo(struct CqeErrInfo *info)
{
    int ret;

    ret = RsDrvGetCqeErrInfo(info);
    CHK_PRT_RETURN(ret, hccp_err("get failed! ret:%d", ret), ret);
    return 0;
}

RS_ATTRI_VISI_DEF int RsGetCqeErrInfoNum(unsigned int phyId, unsigned int rdevIdx, unsigned int *num)
{
    struct RsRdevCb *rdevCb = NULL;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs get cqe err param error, phyId[%u]", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIdx, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chipId[%u] failed, ret %d", chipId, ret),
        ret);

    *num = rdevCb->cqeErrCnt;

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetCqeErrInfoList(unsigned int phyId, unsigned int rdevIdx, struct CqeErrInfo *info,
    unsigned int *num)
{
    struct RsQpCb *qpCbCurr = NULL;
    struct RsQpCb *qpCbNext = NULL;
    struct RsRdevCb *rdevCb = NULL;
    unsigned int cqeErrIdx = 0;
    unsigned int numTmp = *num;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs get cqe err param error, phyId[%u]", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIdx, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chipId[%u] failed, ret %d", chipId, ret),
        ret);

    if (RsListEmpty(&rdevCb->qpList)) {
        *num = 0;
        return 0;
    }

    RS_LIST_GET_HEAD_ENTRY(qpCbCurr, qpCbNext, &rdevCb->qpList, list, struct RsQpCb);
    for (; (&qpCbCurr->list) != &rdevCb->qpList;
         qpCbCurr = qpCbNext, qpCbNext = list_entry(qpCbNext->list.next, struct RsQpCb, list)) {
        if (qpCbCurr->cqeErrInfo.info.status != 0) {
            RS_PTHREAD_MUTEX_LOCK(&qpCbCurr->cqeErrInfo.mutex);
            info[cqeErrIdx].status = qpCbCurr->cqeErrInfo.info.status;
            info[cqeErrIdx].qpn = qpCbCurr->cqeErrInfo.info.qpn;
            info[cqeErrIdx].time = qpCbCurr->cqeErrInfo.info.time;
            qpCbCurr->cqeErrInfo.info.status = 0;
            RS_PTHREAD_MUTEX_ULOCK(&qpCbCurr->cqeErrInfo.mutex);
            RS_PTHREAD_MUTEX_LOCK(&qpCbCurr->rdevCb->cqeErrCntMutex);
            qpCbCurr->rdevCb->cqeErrCnt--;
            RS_PTHREAD_MUTEX_ULOCK(&qpCbCurr->rdevCb->cqeErrCntMutex);
            cqeErrIdx++;
            if (cqeErrIdx == numTmp) {
                break;
            }
        }
    }

    *num = cqeErrIdx;

    return 0;
}

int RsQueryMrCb(struct RsRdevCb *devCb, uint64_t addr, struct RsMrCb **mrCb, struct RsListHead *mrList)
{
    struct RsMrCb *mrCurr = NULL;
    struct RsMrCb *mrNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&devCb->rdevMutex);
    RS_LIST_GET_HEAD_ENTRY(mrCurr, mrNext, mrList, list, struct RsMrCb);
    for (; (&mrCurr->list) != mrList; mrCurr = mrNext, mrNext = list_entry(mrNext->list.next, struct RsMrCb, list)) {
        if ((mrCurr->mrInfo.addr <= addr) && (addr < mrCurr->mrInfo.addr + mrCurr->mrInfo.len)) {
            *mrCb = mrCurr;
            RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);
            return 0;
        }
    }

    *mrCb = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);

    hccp_info_rma("cannot find mrcb for addr@0x%lx !", addr);

    return -ENODEV;
}

STATIC int RsGetLinuxVersion(struct RsLinuxVersionInfo *verInfo)
{
#define LINUX_VERSION_MAX_CHAR 1024
#define LINUX_VERSION_TYPE_NUM 3
#define LINUX_VERSION_STR "Linux version "
    char buffer[LINUX_VERSION_MAX_CHAR] = {0};
    char *versionStr;
    int retClose = 0;
    int ret = 0;
    int fd;

    fd = open("/proc/version", O_RDONLY);
    CHK_PRT_RETURN(fd < 0, hccp_run_warn("open proc/version unsuccessful, errno[%d] fd[%d]", errno, fd), -EFILEOPER);

    do {
        ret = (int)read(fd, buffer, sizeof(buffer) - 1);
    } while ((ret < 0) && (errno == EINTR));

    if (ret < 0) {
        hccp_run_warn("read fd unsuccessful[%d]", ret);
        RS_CLOSE_RETRY_FOR_EINTR(retClose, fd);
        return -EFILEOPER;
    }

    versionStr = strstr(buffer, LINUX_VERSION_STR);
    if (versionStr == NULL) {
        hccp_run_warn("can't get Linux version");
        RS_CLOSE_RETRY_FOR_EINTR(retClose, fd);
        return -EFILEOPER;
    }
    versionStr += strlen(LINUX_VERSION_STR);
    if (sscanf_s(versionStr, "%d.%d.%d", &verInfo->major, &verInfo->minor, &verInfo->patch) != LINUX_VERSION_TYPE_NUM) {
        hccp_run_warn("can't extract Linux version");
        RS_CLOSE_RETRY_FOR_EINTR(retClose, fd);
        return -EFILEOPER;
    }

    RS_CLOSE_RETRY_FOR_EINTR(retClose, fd);
    return retClose;
}

RS_ATTRI_VISI_DEF int RsGetSecRandom(unsigned int *value)
{
#define SEC_LINUX_VERSION_MAJOR 5
#define SEC_LINUX_VERSION_MINOR 18
#define SEC_LINUX_VERSION_PATCH 0
    struct RsLinuxVersionInfo verInfo = {0};
    int ret;

    ret = RsGetLinuxVersion(&verInfo);
    CHK_PRT_RETURN(ret, hccp_run_warn("[rs_get_random]get_linux_version unsuccessful ret(%d)", ret), ret);

    // linux_version > 5.18, urandom is secure
    if (verInfo.major > SEC_LINUX_VERSION_MAJOR ||
        (verInfo.major == SEC_LINUX_VERSION_MAJOR && verInfo.minor > SEC_LINUX_VERSION_MINOR) ||
        (verInfo.major == SEC_LINUX_VERSION_MAJOR && verInfo.minor == SEC_LINUX_VERSION_MINOR &&
            verInfo.patch > SEC_LINUX_VERSION_PATCH)) {
        ret = RsDrvGetRandomNum((int *)value);
    } else {
        hccp_run_warn("[rs_get_random]linux_version is not secure version");
        return -ENOTSUPP;
    }

    if (ret != 0) {
        hccp_run_warn("[get][get_random]rs_get_sec_random unsuccessful, ret(%d)", ret);
    }
    return ret;
}

RS_ATTRI_VISI_DEF enum ProductType RsGetProductType(int devId)
{
    static enum ProductType type = PRODUCT_TYPE_NO_VALUE;
    static halChipInfo chipInfo = {0};
    int ret;

    if (type != PRODUCT_TYPE_NO_VALUE) { // Cache result after first query
        hccp_run_info("[Get][ChipInfo]chip name is %s, type:%d", chipInfo.name, type);
        return type;
    }

    DlHalInit();
    ret = DlHalGetChipInfo(devId, &chipInfo);
    DlHalDeinit();

    CHK_PRT_RETURN(ret != 0, hccp_err("[Get][ChipInfo]DlHalGetChipInfo failed ret:%d", ret), PRODUCT_TYPE_INVALID);

    if (fnmatch("910_93[a-zA-Z1-9_]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_910_93;
    } else if (fnmatch("910B[a-zA-Z1-9_]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_910B;
    } else if (fnmatch("960[a-zA-Z1-9_]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_960;
    } else if (fnmatch("910[a-zA-Z1-9]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_910;
    } else if (fnmatch("310p[a-zA-Z1-9]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_310p;
    } else if (fnmatch("950[a-zA-Z1-9]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_950;
    } else if (fnmatch("350[a-zA-Z1-9]*", (const char *)chipInfo.name, 0) == 0) {
        type = PRODUCT_TYPE_350;
    } else {
        type = PRODUCT_TYPE_OTHERS;
    }

    hccp_run_info("[Get][ChipInfo]chip name is %s, type:%d", chipInfo.name, type);
    return type;
}
