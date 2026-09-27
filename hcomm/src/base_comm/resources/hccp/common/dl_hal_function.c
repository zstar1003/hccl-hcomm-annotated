/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pthread.h>
#include "errno.h"
#include "ascend_hal_dl.h"
#include "network_comm.h"
#include "dl_hal_function.h"

#define DL_API_IS_NULL_CHECK(handle, ptr, str)                                                                         \
    do {                                                                                                               \
        if ((handle) == NULL) {                                                                                        \
            roce_err("g_hal_api_handle is NULL!");                                                                     \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
        if ((ptr) == NULL) {                                                                                           \
            roce_err("[%s] is NULL!", (str));                                                                          \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

static pthread_mutex_t gHalApiLock = PTHREAD_MUTEX_INITIALIZER;
static void *gHalApiHandle = NULL;
static struct DlHalOps gHalOps;
static int gHalApiRefcnt = 0;

static void DlHalApiDevInit(void)
{
    gHalOps.dlDrvGetDevNum = (int (*)(unsigned int *numDev))AscendHalDlsym(gHalApiHandle, "drvGetDevNum");

    gHalOps.dlDrvGetLocalDevIdByHostDevId = (int (*)(uint32_t devId, uint32_t *chipId))AscendHalDlsym(gHalApiHandle,
        "drvGetLocalDevIDByHostDevID");

    gHalOps.dlDrvGetDevIdByLocalDevId = (int (*)(uint32_t localDevId, uint32_t *devId))AscendHalDlsym(gHalApiHandle,
        "drvGetDevIDByLocalDevID");

    gHalOps.dlDrvDeviceGetIndexByPhyId = (int (*)(uint32_t phyId, uint32_t *devIndex))AscendHalDlsym(gHalApiHandle,
        "drvDeviceGetIndexByPhyId");

    gHalOps.dlDrvDeviceGetPhyIdByIndex = (int (*)(unsigned int devIndex,
        unsigned int *phyId))AscendHalDlsym(gHalApiHandle, "drvDeviceGetPhyIdByIndex");

    gHalOps.dlHalGetPhyDevIdByudevId = (int (*)(unsigned int udevId,
        unsigned int *phyDevId))AscendHalDlsym(gHalApiHandle, "halGetPhyDevIdByUdevId");

    gHalOps.dlDrvDeviceGetBareTgid = (pid_t(*)(void))AscendHalDlsym(gHalApiHandle, "drvDeviceGetBareTgid");

    gHalOps.dlHalGetDeviceInfo = (int (*)(uint32_t devId, int32_t moduleType, int32_t infoType,
        int64_t *value))AscendHalDlsym(gHalApiHandle, "halGetDeviceInfo");

    gHalOps.dlHalQueryDevPid = (drvError_t(*)(struct halQueryDevpidInfo info, pid_t * devPid))
        AscendHalDlsym(gHalApiHandle, "halQueryDevpid");

    gHalOps.dlDrvQueryProcessHostPid = (drvError_t(*)(int pid, unsigned int *chipId, unsigned int *vfid,
        unsigned int *hostPid, unsigned int *cpType))AscendHalDlsym(gHalApiHandle, "drvQueryProcessHostPid");

    gHalOps.dlDrvGetProcessSign = (int (*)(struct process_sign *sign))AscendHalDlsym(gHalApiHandle,
        "drvGetProcessSign");

    gHalOps.dlDrvGetPlatformInfo = (int (*)(uint32_t *info))AscendHalDlsym(gHalApiHandle, "drvGetPlatformInfo");

    gHalOps.dlHalGetChipInfo = (int (*)(unsigned int devId, halChipInfo *chipInfo))AscendHalDlsym(gHalApiHandle,
        "halGetChipInfo");
}

static void DlHalApiHdcInit(void)
{
    gHalOps.dlHalHdcGetSessionAttr = (int (*)(HDC_SESSION session, int attr, int *value))AscendHalDlsym(gHalApiHandle,
        "halHdcGetSessionAttr");

    gHalOps.dlDrvHdcGetCapacity = (hdcError_t(*)(struct drvHdcCapacity * capacity))
        AscendHalDlsym(gHalApiHandle, "drvHdcGetCapacity");

    gHalOps.dlDrvHdcClientCreate = (hdcError_t(*)(HDC_CLIENT * client, int maxSessionNum, int serviceType, int flag))
        AscendHalDlsym(gHalApiHandle, "drvHdcClientCreate");

    gHalOps.dlDrvHdcClientDestroy = (hdcError_t(*)(HDC_CLIENT client))AscendHalDlsym(gHalApiHandle,
        "drvHdcClientDestroy");

    gHalOps.dlDrvHdcSessionConnect = (hdcError_t(*)(int peerNode, int peerDevid, HDC_CLIENT client,
        HDC_SESSION *session))AscendHalDlsym(gHalApiHandle, "drvHdcSessionConnect");

    gHalOps.dlDrvHdcServerCreate = (hdcError_t(*)(int devid, int serviceType,
        HDC_SERVER *pServer))AscendHalDlsym(gHalApiHandle, "drvHdcServerCreate");

    gHalOps.dlDrvHdcServerDestroy = (hdcError_t(*)(HDC_SERVER server))AscendHalDlsym(gHalApiHandle,
        "drvHdcServerDestroy");

    gHalOps.dlDrvHdcSessionAccept = (hdcError_t(*)(HDC_SERVER server, HDC_SESSION * session))
        AscendHalDlsym(gHalApiHandle, "drvHdcSessionAccept");

    gHalOps.dlDrvHdcSessionClose = (hdcError_t(*)(HDC_SESSION session))AscendHalDlsym(gHalApiHandle,
        "drvHdcSessionClose");

    gHalOps.dlDrvHdcFreeMsg = (hdcError_t(*)(struct drvHdcMsg * msg)) AscendHalDlsym(gHalApiHandle, "drvHdcFreeMsg");

    gHalOps.dlDrvHdcReuseMsg = (hdcError_t(*)(struct drvHdcMsg * msg)) AscendHalDlsym(gHalApiHandle, "drvHdcReuseMsg");

    gHalOps.dlDrvHdcAddMsgBuffer = (hdcError_t(*)(struct drvHdcMsg * msg, char *pBuf, int len))
        AscendHalDlsym(gHalApiHandle, "drvHdcAddMsgBuffer");

    gHalOps.dlDrvHdcGetMsgBuffer = (hdcError_t(*)(struct drvHdcMsg * msg, int index, char **pBuf, int *pLen))
        AscendHalDlsym(gHalApiHandle, "drvHdcGetMsgBuffer");

    gHalOps.dlHalHdcRecv = (hdcError_t(*)(HDC_SESSION session, struct drvHdcMsg * pMsg, int bufLen, UINT64 flag,
        int *recvBufCount, UINT32 timeout)) AscendHalDlsym(gHalApiHandle, "halHdcRecv");

    gHalOps.dlHalHdcSend = (hdcError_t(*)(HDC_SESSION session, struct drvHdcMsg * pMsg, UINT64 flag, UINT32 timeout))
        AscendHalDlsym(gHalApiHandle, "halHdcSend");

    gHalOps.dlDrvHdcAllocMsg = (hdcError_t(*)(HDC_SESSION session, struct drvHdcMsg * *ppMsg, int count))
        AscendHalDlsym(gHalApiHandle, "drvHdcAllocMsg");

    gHalOps.dlDrvHdcSetSessionReference = (hdcError_t(*)(HDC_SESSION session))AscendHalDlsym(gHalApiHandle,
        "drvHdcSetSessionReference");

    gHalOps.dlHalHdcSessionConnectEx = (hdcError_t(*)(int peer_node, int peer_devid, int peer_pid, HDC_CLIENT client,
        HDC_SESSION *pSession))AscendHalDlsym(gHalApiHandle, "halHdcSessionConnectEx");
}

static void DlHalApiMemInit(void)
{
    gHalOps.dlHalMemAlloc = (int (*)(void **pp, unsigned long long size,
        unsigned long long flag))AscendHalDlsym(gHalApiHandle, "halMemAlloc");

    gHalOps.dlHalMemFree = (int (*)(void *pp))AscendHalDlsym(gHalApiHandle, "halMemFree");

#ifndef HNS_ROCE_LLT
    gHalOps.dlHalMemCtl = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
        size_t *outSizeRet))AscendHalDlsym(gHalApiHandle, "halMemCtl");
#endif
    gHalOps.dlHalMemGetInfoEx = (drvError_t(*)(unsigned int devId, unsigned int type,
        struct MemInfo *info))AscendHalDlsym(gHalApiHandle, "halMemGetInfoEx");

    gHalOps.dlHalMemBindSibling = (drvError_t(*)(int hostPid, int aicpuPid, unsigned int vfid, unsigned int devId,
        unsigned int flag))AscendHalDlsym(gHalApiHandle, "halMemBindSibling");

    gHalOps.dlHalBuffAllocAlignEx = (int (*)(uint64_t size, unsigned int align, unsigned long flag, int grpId,
        void **buff))AscendHalDlsym(gHalApiHandle, "halBuffAllocAlignEx");

    gHalOps.dlHalBuffFree = (int (*)(void *buff))AscendHalDlsym(gHalApiHandle, "halBuffFree");

    gHalOps.dlHalBindCgroup = (int (*)(BIND_CGROUP_TYPE bindType))AscendHalDlsym(gHalApiHandle, "halBindCgroup");

    gHalOps.dlHalNotifyGetInfo = (int (*)(uint32_t devId, uint32_t tsId, uint32_t type,
        uint32_t *val))AscendHalDlsym(gHalApiHandle, "halNotifyGetInfo");

    gHalOps.dlHalGrpQuery = (int (*)(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff,
        unsigned int *outLen))AscendHalDlsym(gHalApiHandle, "halGrpQuery");

    gHalOps.dlHalMemRegUbSegment = (drvError_t(*)(uint32_t devId, uint64_t va,
        uint64_t size))AscendHalDlsym(gHalApiHandle, "halMemRegUbSegment");

    gHalOps.dlHalMemUnRegUbSegment = (drvError_t(*)(uint32_t devId, uint64_t va))AscendHalDlsym(gHalApiHandle,
        "halMemUnRegUbSegment");

    gHalOps.dlDrvMemGetAttribute = (DVresult(*)(DVdeviceptr vptr, struct DVattribute * attr))
        AscendHalDlsym(gHalApiHandle, "drvMemGetAttribute");
}

static void DlHalApiSensorInit(void)
{
    gHalOps.dlHalSensorNodeRegister = (drvError_t(*)(uint32_t devid, struct halSensorNodeCfg * cfg, uint64_t * handle))
        AscendHalDlsym(gHalApiHandle, "halSensorNodeRegister");

    gHalOps.dlHalSensorNodeUnregister = (drvError_t(*)(uint32_t devid, uint64_t handle))AscendHalDlsym(gHalApiHandle,
        "halSensorNodeUnregister");

    gHalOps.dlHalSensorNodeUpdateState = (drvError_t(*)(uint32_t devid, uint64_t handle, int val,
        halGeneralEventType_t assertion))AscendHalDlsym(gHalApiHandle, "halSensorNodeUpdateState");
}

static void DlHalApiEschedInit(void)
{
    gHalOps.dlHalEschedSubmitEvent = (int (*)(uint32_t devId, struct event_summary *event))AscendHalDlsym(gHalApiHandle,
        "halEschedSubmitEvent");

    gHalOps.dlHalEschedAttachDevice = (int (*)(uint32_t devId))AscendHalDlsym(gHalApiHandle, "halEschedAttachDevice");

    gHalOps.dlHalEschedCreateGrp = (int (*)(uint32_t devId, uint32_t grpId,
        GROUP_TYPE type))AscendHalDlsym(gHalApiHandle, "halEschedCreateGrp");

    gHalOps.dlHalEschedSubscribeEvent = (int (*)(uint32_t devId, uint32_t grpId, uint32_t threadId,
        uint64_t eventBitmap))AscendHalDlsym(gHalApiHandle, "halEschedSubscribeEvent");

    gHalOps.dlHalEschedWaitEvent = (int (*)(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout,
        struct event_info *event))AscendHalDlsym(gHalApiHandle, "halEschedWaitEvent");
}

static void DlHalApiResInit(void)
{
    gHalOps.dlHalResAddrMapV2 = (drvError_t(*)(unsigned int devId, struct res_map_info_in *resInfoIn,
        struct res_map_info_out *resInfoOut))AscendHalDlsym(gHalApiHandle, "halResAddrMapV2");

    gHalOps.dlHalResAddrUnmapV2 = (drvError_t(*)(unsigned int devId,
        struct res_map_info_in *resInfoIn))AscendHalDlsym(gHalApiHandle, "halResAddrUnmapV2");
}

static void DlHalApiHostInit(void)
{
    gHalOps.dlHalHostRegister = (drvError_t(*)(void *srcPtr, uint64_t size, uint32_t flag, uint32_t devId,
        void **dstPtr))AscendHalDlsym(gHalApiHandle, "halHostRegister");

    gHalOps.dlHalHostUnregister = (drvError_t(*)(void *src_ptr, uint32_t devid))AscendHalDlsym(gHalApiHandle,
        "halHostUnregister");

    gHalOps.dlHalHostUnregisterEx = (drvError_t(*)(void *src_ptr, uint32_t devid,
        uint32_t flag))AscendHalDlsym(gHalApiHandle, "halHostUnregisterEx");
    return;
}

static void DlHalApiInit(void)
{
    DlHalApiDevInit();

    DlHalApiHdcInit();

    DlHalApiMemInit();

    DlHalApiSensorInit();

    DlHalApiEschedInit();

    DlHalApiResInit();

    DlHalApiHostInit();

    return;
}

void DlHalDeinit(void)
{
    pthread_mutex_lock(&gHalApiLock);
    if (gHalApiHandle != NULL) {
        gHalApiRefcnt--;
        if (gHalApiRefcnt > 0) {
            pthread_mutex_unlock(&gHalApiLock);
            roce_info("dl_hal_deinit success, no need to dlclose libascend_hal.so!");
            return;
        }

        (void)AscendHalDlclose(gHalApiHandle);
        gHalApiHandle = NULL;
    }

    pthread_mutex_unlock(&gHalApiLock);
    roce_info("dl_hal_deinit success!");
    return;
}

int DlHalInit(void)
{
    pthread_mutex_lock(&gHalApiLock);
    if (gHalApiHandle != NULL) {
        gHalApiRefcnt++;
        pthread_mutex_unlock(&gHalApiLock);
        roce_info("dl_hal_init success, no need to dlopen libascend_hal.so!");
        return 0;
    }

    gHalApiHandle = AscendHalDlopen("libascend_hal.so", RTLD_NOW);
    if (gHalApiHandle == NULL) {
        pthread_mutex_unlock(&gHalApiLock);
        roce_err("dlopen libascend_hal.so failed! error_no=[%d]", errno);
        return -EINVAL;
    }

    DlHalApiInit();
    gHalApiRefcnt++;

    pthread_mutex_unlock(&gHalApiLock);
    roce_info("dl_hal_init success!");
    return 0;
}

int DlDrvGetDevNum(unsigned int *numDev)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvGetDevNum, "dl_drv_get_dev_num");

    return DlRetConvert(gHalOps.dlDrvGetDevNum(numDev));
}

int DlDrvGetLocalDevIdByHostDevId(unsigned int devId, unsigned int *chipId)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvGetLocalDevIdByHostDevId,
        "dl_drv_get_local_dev_id_by_host_dev_id");

    return DlRetConvert(gHalOps.dlDrvGetLocalDevIdByHostDevId(devId, chipId));
}

int DlDrvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t *devIndex)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvDeviceGetIndexByPhyId, "dl_drv_device_get_index_by_phy_id");

    return DlRetConvert(gHalOps.dlDrvDeviceGetIndexByPhyId(phyId, devIndex));
}

int DlDrvGetDevIdByLocalDevId(unsigned int localDevId, unsigned int *devId)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvGetDevIdByLocalDevId, "dl_drv_get_dev_id_by_local_dev_id");

    return DlRetConvert(gHalOps.dlDrvGetDevIdByLocalDevId(localDevId, devId));
}

int DlDrvDeviceGetPhyIdByIndex(unsigned int devIndex, unsigned int *phyId)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvDeviceGetPhyIdByIndex, "dl_drv_device_get_phy_id_by_index");

    return DlRetConvert(gHalOps.dlDrvDeviceGetPhyIdByIndex(devIndex, phyId));
}

int DlHalGetPhyDevIdByudevId(unsigned int udevId, unsigned int *phyDevId)
{
    CHK_PRT_RETURN(gHalApiHandle == NULL, roce_err("gHalApiHandle is NULL"), -EINVAL);
    CHK_PRT_RETURN(phyDevId == NULL, roce_err("phyDevId is NULL"), -EINVAL);
    // use udevId as phyDevId for API compatibility issue(abnormal in mdev scenario)
    *phyDevId = udevId;
    CHK_PRT_RETURN(gHalOps.dlHalGetPhyDevIdByudevId == NULL, roce_warn("dlHalGetPhyDevIdByudevId is NULL"), 0);
    return DlRetConvert(gHalOps.dlHalGetPhyDevIdByudevId(udevId, phyDevId));
}

drvError_t DlHalQueryDevPid(struct halQueryDevpidInfo info, pid_t *devPid)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalQueryDevPid, "dl_hal_query_dev_pid");

    return DlRetConvert(gHalOps.dlHalQueryDevPid(info, devPid));
}

drvError_t DlHalMemBindSibling(int hostPid, int aicpuPid, unsigned int vfid, unsigned int devId, unsigned int flag)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemBindSibling, "dl_hal_mem_bind_sibling");

    return DlRetConvert(gHalOps.dlHalMemBindSibling(hostPid, aicpuPid, vfid, devId, flag));
}

drvError_t DlDrvQueryProcessHostPid(int pid, unsigned int *chipId, unsigned int *vfid, unsigned int *hostPid,
    unsigned int *cpType)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvQueryProcessHostPid, "dl_drv_query_process_host_pid");

    return DlRetConvert(gHalOps.dlDrvQueryProcessHostPid(pid, chipId, vfid, hostPid, cpType));
}

drvError_t DlHalMemGetInfoEx(unsigned int devId, unsigned int type, struct MemInfo *info)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemGetInfoEx, "dl_hal_mem_get_info_ex");

    return DlRetConvert(gHalOps.dlHalMemGetInfoEx(devId, type, info));
}

int DlHalGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff, unsigned int *outLen)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalGrpQuery, "dl_hal_grp_query");

    return DlRetConvert(gHalOps.dlHalGrpQuery(cmd, inBuff, inLen, outBuff, outLen));
}

int DlHalHdcGetSessionAttr(HDC_SESSION session, int attr, int *value)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHdcGetSessionAttr, "dl_hal_hdc_get_session_attr");

    return DlRetConvert(gHalOps.dlHalHdcGetSessionAttr(session, attr, value));
}

hdcError_t DlDrvHdcGetCapacity(struct drvHdcCapacity *capacity)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcGetCapacity, "dl_drv_hdc_get_capacity");

    return DlRetConvert(gHalOps.dlDrvHdcGetCapacity(capacity));
}

hdcError_t DlDrvHdcClientCreate(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcClientCreate, "dl_drv_hdc_client_create");

    return DlRetConvert(gHalOps.dlDrvHdcClientCreate(client, maxSessionNum, serviceType, flag));
}

hdcError_t DlDrvHdcClientDestroy(HDC_CLIENT client)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcClientDestroy, "dl_drv_hdc_client_destroy");

    return DlRetConvert(gHalOps.dlDrvHdcClientDestroy(client));
}

hdcError_t DlDrvHdcSessionConnect(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION *session)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcSessionConnect, "dl_drv_hdc_session_connect");

    return DlRetConvert(gHalOps.dlDrvHdcSessionConnect(peerNode, peerDevid, client, session));
}

hdcError_t DlHalHdcSessionConnectEx(int peerNode, int peerDevid, int peerPid, HDC_CLIENT client, HDC_SESSION *pSession)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHdcSessionConnectEx, "dl_hal_hdc_session_connect_ex");

    return DlRetConvert(gHalOps.dlHalHdcSessionConnectEx(peerNode, peerDevid, peerPid, client, pSession));
}

hdcError_t DlDrvHdcServerCreate(int devid, int serviceType, HDC_SERVER *pServer)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcServerCreate, "dl_drv_hdc_server_create");

    return DlRetConvert(gHalOps.dlDrvHdcServerCreate(devid, serviceType, pServer));
}

hdcError_t DlDrvHdcServerDestroy(HDC_SERVER server)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcServerDestroy, "dl_drv_hdc_server_destroy");

    return DlRetConvert(gHalOps.dlDrvHdcServerDestroy(server));
}

hdcError_t DlDrvHdcSessionAccept(HDC_SERVER server, HDC_SESSION *session)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcSessionAccept, "dl_drv_hdc_session_accept");

    return DlRetConvert(gHalOps.dlDrvHdcSessionAccept(server, session));
}

hdcError_t DlDrvHdcSessionClose(HDC_SESSION session)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcSessionClose, "dl_drv_hdc_session_close");

    return DlRetConvert(gHalOps.dlDrvHdcSessionClose(session));
}

hdcError_t DlDrvHdcFreeMsg(struct drvHdcMsg *msg)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcFreeMsg, "dl_drv_hdc_free_msg");

    return DlRetConvert(gHalOps.dlDrvHdcFreeMsg(msg));
}

hdcError_t DlDrvHdcReuseMsg(struct drvHdcMsg *msg)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcReuseMsg, "dl_drv_hdc_reuse_msg");

    return DlRetConvert(gHalOps.dlDrvHdcReuseMsg(msg));
}

hdcError_t DlDrvHdcAddMsgBuffer(struct drvHdcMsg *msg, char *pBuf, int len)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcAddMsgBuffer, "dl_drv_hdc_add_msg_buffer");

    return DlRetConvert(gHalOps.dlDrvHdcAddMsgBuffer(msg, pBuf, len));
}

hdcError_t DlDrvHdcGetMsgBuffer(struct drvHdcMsg *msg, int index, char **pBuf, int *pLen)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcGetMsgBuffer, "dl_drv_hdc_get_msg_buffer");

    return DlRetConvert(gHalOps.dlDrvHdcGetMsgBuffer(msg, index, pBuf, pLen));
}

hdcError_t DlHalHdcRecv(HDC_SESSION session, struct drvHdcMsg *pMsg, int bufLen, UINT64 flag, int *recvBufCount,
    UINT32 timeout)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHdcRecv, "dl_hal_hdc_recv");

    return DlRetConvert(gHalOps.dlHalHdcRecv(session, pMsg, bufLen, flag, recvBufCount, timeout));
}

hdcError_t DlHalHdcSend(HDC_SESSION session, struct drvHdcMsg *pMsg, UINT64 flag, UINT32 timeout)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHdcSend, "dl_hal_hdc_send");

    return DlRetConvert(gHalOps.dlHalHdcSend(session, pMsg, flag, timeout));
}

hdcError_t DlDrvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg **ppMsg, int count)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcAllocMsg, "dl_drv_hdc_alloc_msg");

    return DlRetConvert(gHalOps.dlDrvHdcAllocMsg(session, ppMsg, count));
}

hdcError_t DlDrvHdcSetSessionReference(HDC_SESSION session)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvHdcSetSessionReference, "dl_drv_hdc_set_session_reference");

    return DlRetConvert(gHalOps.dlDrvHdcSetSessionReference(session));
}

int DlDrvGetProcessSign(struct process_sign *sign)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvGetProcessSign, "dl_drv_get_process_sign");

    return DlRetConvert(gHalOps.dlDrvGetProcessSign(sign));
}

pid_t DlDrvDeviceGetBareTgid(void)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvDeviceGetBareTgid, "dl_drv_device_get_bare_tgid");

    return gHalOps.dlDrvDeviceGetBareTgid();
}

int DlHalNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t *val)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalNotifyGetInfo, "dl_hal_notify_get_info");

    return DlRetConvert(gHalOps.dlHalNotifyGetInfo(devId, tsId, type, val));
}

int DlHalMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemAlloc, "dl_hal_mem_alloc");

    return DlRetConvert(gHalOps.dlHalMemAlloc(pp, size, flag));
}

int DlHalMemFree(void *pp)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemFree, "dl_hal_mem_free");

    return DlRetConvert(gHalOps.dlHalMemFree(pp));
}

int DlHalEschedSubmitEvent(uint32_t devId, struct event_summary *event)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalEschedSubmitEvent, "dl_hal_esched_submit_event");

    return DlRetConvert(gHalOps.dlHalEschedSubmitEvent(devId, event));
}

int DlHalMemCtl(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemCtl, "dl_hal_mem_ctl");

    return DlRetConvert(gHalOps.dlHalMemCtl(type, paramValue, paramValueSize, outValue, outSizeRet));
}

int DlHalBuffAllocAlignEx(uint64_t size, unsigned int align, unsigned long flag, int grpId, void **buff)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalBuffAllocAlignEx, "dl_hal_buff_alloc_align_ex");

    return DlRetConvert(gHalOps.dlHalBuffAllocAlignEx(size, align, flag, grpId, buff));
}

int DlHalBuffFree(void *buff)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalBuffFree, "dl_hal_buff_free");

    return DlRetConvert(gHalOps.dlHalBuffFree(buff));
}

int DlHalBindCgroup(BIND_CGROUP_TYPE bindType)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalBindCgroup, "dl_hal_bind_cgroup");

    return DlRetConvert(gHalOps.dlHalBindCgroup(bindType));
}

int DlDrvGetPlatformInfo(uint32_t *info)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvGetPlatformInfo, "dl_drv_get_platform_info");

    return DlRetConvert(gHalOps.dlDrvGetPlatformInfo(info));
}

int DlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalGetDeviceInfo, "dl_hal_get_device_info");

    return DlRetConvert(gHalOps.dlHalGetDeviceInfo(devId, moduleType, infoType, value));
}

int DlHalGetChipInfo(unsigned int devId, halChipInfo *chipInfo)
{
    if (gHalApiHandle == NULL) {
        roce_err("g_hal_api_handle is NULL!");
        return -EINVAL;
    }

    if (gHalOps.dlHalGetChipInfo == NULL) {
        roce_warn("dl_hal_get_chip_info is NULL!");
        return -EINVAL;
    }

    return DlRetConvert(gHalOps.dlHalGetChipInfo(devId, chipInfo));
}

int DlHalSensorNodeRegister(uint32_t devid, struct halSensorNodeCfg *cfg, uint64_t *handle)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalSensorNodeRegister, "dl_hal_sensor_node_register");

    return DlRetConvert(gHalOps.dlHalSensorNodeRegister(devid, cfg, handle));
}

int DlHalSensorNodeUnregister(uint32_t devid, uint64_t handle)
{
    /* sensor may not support, handle is 0 */
    if (handle == 0) {
        return 0;
    }

    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalSensorNodeUnregister, "dl_hal_sensor_node_unregister");

    return DlRetConvert(gHalOps.dlHalSensorNodeUnregister(devid, handle));
}

int DlHalSensorNodeUpdateState(uint32_t devid, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalSensorNodeUpdateState, "dl_hal_sensor_node_update_state");

    return DlRetConvert(gHalOps.dlHalSensorNodeUpdateState(devid, handle, val, assertion));
}

int DlHalEschedAttachDevice(uint32_t devId)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalEschedAttachDevice, "dl_hal_esched_attach_device");

    return DlRetConvert(gHalOps.dlHalEschedAttachDevice(devId));
}

int DlHalEschedCreateGrp(uint32_t devId, uint32_t grpId, GROUP_TYPE type)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalEschedCreateGrp, "dl_hal_esched_create_grp");

    return DlRetConvert(gHalOps.dlHalEschedCreateGrp(devId, grpId, type));
}

int DlHalEschedSubscribeEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, uint64_t eventBitmap)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalEschedSubscribeEvent, "dl_hal_esched_subscribe_event");

    return DlRetConvert(gHalOps.dlHalEschedSubscribeEvent(devId, grpId, threadId, eventBitmap));
}

int DlHalEschedWaitEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout, struct event_info *event)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalEschedWaitEvent, "dl_hal_esched_wait_event");

    return DlRetConvert(gHalOps.dlHalEschedWaitEvent(devId, grpId, threadId, timeout, event));
}

int DlHalResAddrMapV2(unsigned int devId, struct res_map_info_in *resInfoIn, struct res_map_info_out *resInfoOut)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalResAddrMapV2, "dlHalResAddrMapV2");

    return DlRetConvert(gHalOps.dlHalResAddrMapV2(devId, resInfoIn, resInfoOut));
}

int DlHalResAddrUnmapV2(unsigned int devId, struct res_map_info_in *resInfoIn)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalResAddrUnmapV2, "dlHalResAddrUnmapV2");

    return DlRetConvert(gHalOps.dlHalResAddrUnmapV2(devId, resInfoIn));
}

int DlHalMemRegUbSegment(uint32_t devId, uint64_t va, uint64_t size)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemRegUbSegment, "dlHalMemRegUbSegment");

    return DlRetConvert(gHalOps.dlHalMemRegUbSegment(devId, va, size));
}

int DlHalMemUnRegUbSegment(uint32_t devId, uint64_t va)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalMemUnRegUbSegment, "dlHalMemUnRegUbSegment");

    return DlRetConvert(gHalOps.dlHalMemUnRegUbSegment(devId, va));
}

int DlDrvMemGetAttribute(DVdeviceptr vptr, struct DVattribute *attr)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlDrvMemGetAttribute, "dlDrvMemGetAttribute");

    return DlRetConvert(gHalOps.dlDrvMemGetAttribute(vptr, attr));
}

int DlHalHostRegister(void *srcPtr, uint64_t size, uint32_t flag, uint32_t devId, void **dstPtr)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHostRegister, "dlHalHostRegister");

    return DlRetConvert(gHalOps.dlHalHostRegister(srcPtr, size, flag, devId, dstPtr));
}

int DlHalHostUnRegister(void *srcPtr, uint32_t devId)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHostUnregister, "dlHalHostUnregister");

    return DlRetConvert(gHalOps.dlHalHostUnregister(srcPtr, devId));
}

int DlHalHostUnRegisterEx(void *srcPtr, uint32_t devId, uint32_t flag)
{
    DL_API_IS_NULL_CHECK(gHalApiHandle, gHalOps.dlHalHostUnregisterEx, "dlHalHostUnregisterEx");

    return DlRetConvert(gHalOps.dlHalHostUnregisterEx(srcPtr, devId, flag));
}
