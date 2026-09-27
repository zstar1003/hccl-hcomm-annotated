/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DL_HAL_FUNCTION_H__
#define __DL_HAL_FUNCTION_H__

#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>
#include "user_log.h"
#include "ascend_hal.h"
#include "ascend_hal_error.h"
#include "ascend_hal_define.h"

// device info, see hardware_version
#define VER_BIN5 5
#define VER_BIN8 8
#define GET_CHIP_OFFSET 8
#define CHIP_TYPE_910A 1
#define CHIP_TYPE_310P 4
#define CHIP_TYPE_910B_910_93 5

#define RDMA_CQE_ERR_SENSOR_TYPE 0xC3
#define RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE 0x8
/* bit position is 1:event enable; bit position is 0:event disable */
#define RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_MASK (1U << RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE)
/* bit position is 1:fault event;  bit position is 0:notify event(one time) */
#define RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE_MASK (0U << RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE)

struct DlHalOps {
    int (*dlDrvGetDevNum)(unsigned int *numDev);
    int (*dlDrvGetLocalDevIdByHostDevId)(unsigned int devId, unsigned int *chipId);
    int (*dlDrvGetDevIdByLocalDevId)(unsigned int localDevId, unsigned int *devId);
    int (*dlDrvDeviceGetIndexByPhyId)(uint32_t phyId, uint32_t *devIndex);
    int (*dlDrvDeviceGetPhyIdByIndex)(unsigned int devIndex, unsigned int *phyId);
    int (*dlHalGetPhyDevIdByudevId)(unsigned int udevId, unsigned int *phyDevId);
    drvError_t (*dlHalQueryDevPid)(struct halQueryDevpidInfo info, pid_t *devPid);
    drvError_t (*dlHalMemBindSibling)(int hostPid, int aicpuPid, unsigned int vfid, unsigned int devId,
        unsigned int flag);
    drvError_t (*dlDrvQueryProcessHostPid)(int pid, unsigned int *chipId, unsigned int *vfid, unsigned int *hostPid,
        unsigned int *cpType);
    drvError_t (*dlHalMemGetInfoEx)(unsigned int devId, unsigned int type, struct MemInfo *info);
    int (*dlHalGrpQuery)(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff, unsigned int *outLen);
    // HDC
    int (*dlHalHdcGetSessionAttr)(HDC_SESSION session, int attr, int *value);
    hdcError_t (*dlDrvHdcGetCapacity)(struct drvHdcCapacity *capacity);
    hdcError_t (*dlDrvHdcClientCreate)(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag);
    hdcError_t (*dlDrvHdcClientDestroy)(HDC_CLIENT client);
    hdcError_t (*dlDrvHdcSessionConnect)(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION *session);
    hdcError_t (*dlHalHdcSessionConnectEx)(int peerNode, int peerDevid, int peerPid, HDC_CLIENT client,
        HDC_SESSION *session);
    hdcError_t (*dlDrvHdcServerCreate)(int devid, int serviceType, HDC_SERVER *pServer);
    hdcError_t (*dlDrvHdcServerDestroy)(HDC_SERVER server);
    hdcError_t (*dlDrvHdcSessionAccept)(HDC_SERVER server, HDC_SESSION *session);
    hdcError_t (*dlDrvHdcSessionClose)(HDC_SESSION session);
    hdcError_t (*dlDrvHdcFreeMsg)(struct drvHdcMsg *msg);
    hdcError_t (*dlDrvHdcReuseMsg)(struct drvHdcMsg *msg);
    hdcError_t (*dlDrvHdcAddMsgBuffer)(struct drvHdcMsg *msg, char *pBuf, int len);
    hdcError_t (*dlDrvHdcGetMsgBuffer)(struct drvHdcMsg *msg, int index, char **pBuf, int *pLen);
    hdcError_t (*dlHalHdcRecv)(HDC_SESSION session, struct drvHdcMsg *pMsg, int bufLen, UINT64 flag, int *recvBufCount,
        UINT32 timeout);
    hdcError_t (*dlHalHdcSend)(HDC_SESSION session, struct drvHdcMsg *pMsg, UINT64 flag, UINT32 timeout);
    hdcError_t (*dlDrvHdcAllocMsg)(HDC_SESSION session, struct drvHdcMsg **ppMsg, int count);
    hdcError_t (*dlDrvHdcSetSessionReference)(HDC_SESSION session);

    int (*dlDrvGetProcessSign)(struct process_sign *sign);

    pid_t (*dlDrvDeviceGetBareTgid)(void);

    int (*dlHalNotifyGetInfo)(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t *val);
    int (*dlHalMemAlloc)(void **pp, unsigned long long size, unsigned long long flag);
    int (*dlHalMemFree)(void *pp);
    int (*dlHalEschedSubmitEvent)(uint32_t devId, struct event_summary *event);

    int (*dlHalGetDeviceInfo)(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value);

    int (*dlHalMemCtl)(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet);

    int (*dlHalBindCgroup)(BIND_CGROUP_TYPE bindType);
    int (*dlDrvGetPlatformInfo)(uint32_t *info);

    int (*dlHalGetChipInfo)(unsigned int devId, halChipInfo *chipInfo);

    drvError_t (*dlHalSensorNodeRegister)(uint32_t devid, struct halSensorNodeCfg *cfg, uint64_t *handle);
    drvError_t (*dlHalSensorNodeUnregister)(uint32_t devid, uint64_t handle);
    drvError_t (*dlHalSensorNodeUpdateState)(uint32_t devid, uint64_t handle, int val, halGeneralEventType_t assertion);

    int (*dlHalBuffAllocAlignEx)(uint64_t size, unsigned int align, unsigned long flag, int grpId, void **buff);
    int (*dlHalBuffFree)(void *buff);

    int (*dlHalEschedAttachDevice)(uint32_t devId);
    int (*dlHalEschedCreateGrp)(uint32_t devId, uint32_t grpId, GROUP_TYPE type);
    int (*dlHalEschedSubscribeEvent)(uint32_t devId, uint32_t grpId, uint32_t threadId, uint64_t eventBitmap);
    int (*dlHalEschedWaitEvent)(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout,
        struct event_info *event);
    drvError_t (*dlHalResAddrMapV2)(unsigned int devId, struct res_map_info_in *resInfoIn,
        struct res_map_info_out *resInfoOut);
    drvError_t (*dlHalResAddrUnmapV2)(unsigned int devId, struct res_map_info_in *resInfoIn);
    drvError_t (*dlHalMemRegUbSegment)(uint32_t devId, uint64_t va, uint64_t size);
    drvError_t (*dlHalMemUnRegUbSegment)(uint32_t devid, uint64_t va);
    DVresult (*dlDrvMemGetAttribute)(DVdeviceptr vptr, struct DVattribute *attr);
    drvError_t (*dlHalHostRegister)(void *srcPtr, uint64_t size, uint32_t flag, uint32_t devId, void **dstPtr);
    drvError_t (*dlHalHostUnregister)(void *src_ptr, uint32_t devid);
    drvError_t (*dlHalHostUnregisterEx)(void *src_ptr, uint32_t devid, uint32_t flag);
};

int DlHalInit(void);
void DlHalDeinit(void);

int DlDrvGetDevNum(unsigned int *numDev);
int DlDrvGetLocalDevIdByHostDevId(unsigned int devId, unsigned int *chipId);
int DlDrvGetDevIdByLocalDevId(unsigned int localDevId, unsigned int *devId);
int DlDrvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t *devIndex);
int DlDrvDeviceGetPhyIdByIndex(unsigned int devIndex, unsigned int *phyId);
int DlHalGetPhyDevIdByudevId(unsigned int udevId, unsigned int *phyDevId);
drvError_t DlHalQueryDevPid(struct halQueryDevpidInfo info, pid_t *devPid);
drvError_t DlHalMemBindSibling(int hostPid, int aicpuPid, unsigned int vfid, unsigned int devId, unsigned int flag);
drvError_t DlDrvQueryProcessHostPid(int pid, unsigned int *chipId, unsigned int *vfid, unsigned int *hostPid,
    unsigned int *cpType);
drvError_t DlHalMemGetInfoEx(unsigned int devId, unsigned int type, struct MemInfo *info);
int DlHalGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff, unsigned int *outLen);
// HDC
int DlHalHdcGetSessionAttr(HDC_SESSION session, int attr, int *value);
hdcError_t DlDrvHdcGetCapacity(struct drvHdcCapacity *capacity);
hdcError_t DlDrvHdcClientCreate(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag);
hdcError_t DlDrvHdcClientDestroy(HDC_CLIENT client);
hdcError_t DlDrvHdcSessionConnect(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION *session);
hdcError_t DlHalHdcSessionConnectEx(int peerNode, int peerDevid, int peerPid, HDC_CLIENT client, HDC_SESSION *pSession);
hdcError_t DlDrvHdcServerCreate(int devid, int serviceType, HDC_SERVER *pServer);
hdcError_t DlDrvHdcServerDestroy(HDC_SERVER server);
hdcError_t DlDrvHdcSessionAccept(HDC_SERVER server, HDC_SESSION *session);
hdcError_t DlDrvHdcSessionClose(HDC_SESSION session);
hdcError_t DlDrvHdcFreeMsg(struct drvHdcMsg *msg);
hdcError_t DlDrvHdcReuseMsg(struct drvHdcMsg *msg);
hdcError_t DlDrvHdcAddMsgBuffer(struct drvHdcMsg *msg, char *pBuf, int len);
hdcError_t DlDrvHdcGetMsgBuffer(struct drvHdcMsg *msg, int index, char **pBuf, int *pLen);
hdcError_t DlHalHdcRecv(HDC_SESSION session, struct drvHdcMsg *pMsg, int bufLen, UINT64 flag, int *recvBufCount,
    UINT32 timeout);
hdcError_t DlHalHdcSend(HDC_SESSION session, struct drvHdcMsg *pMsg, UINT64 flag, UINT32 timeout);
hdcError_t DlDrvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg **ppMsg, int count);
hdcError_t DlDrvHdcSetSessionReference(HDC_SESSION session);

int DlDrvGetProcessSign(struct process_sign *sign);

pid_t DlDrvDeviceGetBareTgid(void);

int DlHalNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t *val);
int DlHalMemAlloc(void **pp, unsigned long long size, unsigned long long flag);
int DlHalMemFree(void *pp);
int DlHalEschedSubmitEvent(uint32_t devId, struct event_summary *event);

// device info, see hardware_version
int DlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value);

static inline uint32_t DlHalPlatGetVer(uint64_t deviceInfo)
{
    return (uint32_t)(deviceInfo & 0xff);
}

static inline uint32_t DlHalPlatGetChip(uint64_t deviceInfo)
{
    return (uint32_t)((deviceInfo >> GET_CHIP_OFFSET) & 0xff);
}

int DlHalBindCgroup(BIND_CGROUP_TYPE bindType);
int DlDrvGetPlatformInfo(uint32_t *info);

int DlHalMemCtl(int type, void *paramValue, size_t paramValueSize, void *outValue, size_t *outSizeRet);
int DlHalBuffAllocAlignEx(uint64_t size, unsigned int align, unsigned long flag, int grpId, void **buff);
int DlHalBuffFree(void *buff);

int DlHalGetChipInfo(unsigned int devId, halChipInfo *chipInfo);

int DlHalSensorNodeRegister(uint32_t devid, struct halSensorNodeCfg *cfg, uint64_t *handle);
int DlHalSensorNodeUnregister(uint32_t devid, uint64_t handle);
int DlHalSensorNodeUpdateState(uint32_t devid, uint64_t handle, int val, halGeneralEventType_t assertion);

int DlHalEschedAttachDevice(uint32_t devId);
int DlHalEschedCreateGrp(uint32_t devId, uint32_t grpId, GROUP_TYPE type);
int DlHalEschedSubscribeEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, uint64_t eventBitmap);
int DlHalEschedWaitEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout, struct event_info *event);
int DlHalResAddrMapV2(unsigned int devId, struct res_map_info_in *resInfoIn, struct res_map_info_out *resInfoOut);
int DlHalResAddrUnmapV2(unsigned int devId, struct res_map_info_in *resInfoIn);
int DlHalMemRegUbSegment(uint32_t devId, uint64_t va, uint64_t size);
int DlHalMemUnRegUbSegment(uint32_t devId, uint64_t va);
int DlDrvMemGetAttribute(DVdeviceptr vptr, struct DVattribute *attr);
int DlHalHostRegister(void *srcPtr, uint64_t size, uint32_t flag, uint32_t devId, void **dstPtr);
int DlHalHostUnRegister(void *srcPtr, uint32_t devId);
int DlHalHostUnRegisterEx(void *srcPtr, uint32_t devId, uint32_t flag);

#endif // __DL_HAL_FUNCTION_H__
