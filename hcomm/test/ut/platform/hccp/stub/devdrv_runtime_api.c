/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dl_hal_function.h"
#include "ascend_hal_error.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#define LLT_MAX_HDC_DATA 65536
static char tc_host_hdc_memory[LLT_MAX_HDC_DATA] = {0};
static int tc_host_hdc_len = 0;

static char tc_device_hdc_memory[LLT_MAX_HDC_DATA] = {0};
static int tc_device_hdc_len = 0;

static HDC_SESSION tc_host_hdc_session = (HDC_SESSION)(uintptr_t)2;
static HDC_SESSION tc_device_hdc_session = (HDC_SESSION)(uintptr_t)1;

static int tc_host_hdc_flag = 0;
static int tc_device_hdc_flag = 0;

static int tc_host_recv_flag = 0;
static int tc_device_recv_flag = 0;

static int tc_hdc_get_msg_error_flag = 0;

#define RS_DEVICE_NUM 0x3
#define RS_HOSTID2DEVID(dev_id) ((dev_id) & RS_DEVICE_NUM)

DLLEXPORT drvError_t drvDeviceGetPhyIdByIndex(uint32_t devIndex, uint32_t* phyId)
{
    *phyId = devIndex;
    return 0;
}

DLLEXPORT pid_t drvDeviceGetBarePid() { return 0; }

DLLEXPORT pid_t drvDeviceGetBareTgid(void) { return 0; }

DLLEXPORT drvError_t drvGetProcessSign(struct process_sign* sign)
{
    memset(sign, 0, sizeof(struct process_sign));
    return 0;
}

DLLEXPORT drvError_t drvGetDevNum(unsigned int* num_dev)
{
    *num_dev = 4;
    return 0;
}

DLLEXPORT drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 0;
    return 0;
}

DLLEXPORT hdcError_t drvHdcGetCapacity(struct drvHdcCapacity* capacity) { return DRV_ERROR_NONE; };

DLLEXPORT hdcError_t drvHdcServerCreate(int devid, int serviceType, HDC_SERVER* pServer) { return DRV_ERROR_NONE; }

DLLEXPORT hdcError_t drvHdcClientCreate(HDC_CLIENT* client, int maxSessionNum, int serviceType, int flag)
{
    return DRV_ERROR_NONE;
}

DLLEXPORT hdcError_t drvHdcClientCreatePlus(HDC_CLIENT* client, int maxSessionNum, int serviceType, int flag)
{
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcClientDestroy(HDC_CLIENT client) { return DRV_ERROR_NONE; };

DLLEXPORT hdcError_t drvHdcSessionConnect(int peer_node, int peer_devid, HDC_CLIENT client, HDC_SESSION* session)
{
    do {
        *session = tc_host_hdc_session;
    } while (tc_host_hdc_flag);

    tc_host_hdc_flag = 1;

    return DRV_ERROR_NONE;
}

DLLEXPORT hdcError_t drvHdcSessionDestroy(HDC_SESSION session) { return DRV_ERROR_NONE; }

DLLEXPORT hdcError_t drvHdcServerDestroy(HDC_SERVER server) { return DRV_ERROR_NONE; };

DLLEXPORT hdcError_t drvHdcSessionAccept(HDC_SERVER server, HDC_SESSION* session)
{
    do {
        *session = tc_device_hdc_session;
        if (tc_device_hdc_flag)
            sleep(5);
    } while (tc_device_hdc_flag);
    tc_device_hdc_flag = 1;
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcSessionClose(HDC_SESSION session) { return DRV_ERROR_NONE; };

DLLEXPORT hdcError_t drvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg** ppMsg, int count)
{
    *ppMsg = malloc(sizeof(struct drvHdcMsg) + count * sizeof(struct drvHdcMsgBuf));
    (*ppMsg)->count = count;

    ((struct drvHdcMsgBuf*)(*ppMsg + 1))->pBuf = malloc(LLT_MAX_HDC_DATA);

    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcFreeMsg(struct drvHdcMsg* msg)
{
    free(((struct drvHdcMsgBuf*)(msg + 1))->pBuf);
    free(msg);
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcReuseMsg(struct drvHdcMsg* msg) { return DRV_ERROR_NONE; };

DLLEXPORT hdcError_t drvHdcAddMsgBuffer(struct drvHdcMsg* msg, char* pBuf, int len)
{
    memcpy(((struct drvHdcMsgBuf*)(msg + 1))->pBuf, pBuf, len);
    ((struct drvHdcMsgBuf*)(msg + 1))->len = len;
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcGetMsgBuffer(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    int len = -100;
    if (tc_hdc_get_msg_error_flag) {
        *pBuf = NULL;
        *pLen = len;
        return -5;
    } else {
        *pBuf = ((struct drvHdcMsgBuf*)(msg + 1))->pBuf;
        *pLen = ((struct drvHdcMsgBuf*)(msg + 1))->len;
        return DRV_ERROR_NONE;
    }
};

DLLEXPORT hdcError_t halHdcRecv(
    HDC_SESSION session, struct drvHdcMsg* msg, int bufLen, unsigned long long flag, int* recvBufCount,
    unsigned int timeout)
{
    if (session == tc_host_hdc_session) {
        while (!tc_host_recv_flag) {
            usleep(100000);
        }

        ((struct drvHdcMsgBuf*)(msg + 1))->len = tc_host_hdc_len;
        memcpy(((struct drvHdcMsgBuf*)(msg + 1))->pBuf, tc_host_hdc_memory, tc_host_hdc_len);
        *recvBufCount = 1;

        tc_host_recv_flag = 0;
    } else if (session == tc_device_hdc_session) {
        while (!tc_device_recv_flag) {
            usleep(100000);
        }

        ((struct drvHdcMsgBuf*)(msg + 1))->len = tc_device_hdc_len;
        memcpy(((struct drvHdcMsgBuf*)(msg + 1))->pBuf, tc_device_hdc_memory, tc_device_hdc_len);
        *recvBufCount = 1;

        tc_device_recv_flag = 0;
    } else {
        fprintf(stderr, "session[%d] drvHdcRecv fail\n", session);
        return DRV_ERROR_RESERVED;
    }
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t
halHdcSend(HDC_SESSION session, struct drvHdcMsg* msg, unsigned long long flag, unsigned int timeout)
{
    if (session == tc_host_hdc_session) {
        tc_device_hdc_len = ((struct drvHdcMsgBuf*)(msg + 1))->len;
        memcpy(tc_device_hdc_memory, ((struct drvHdcMsgBuf*)(msg + 1))->pBuf, tc_device_hdc_len);
        tc_device_recv_flag = 1;

        fprintf(
            stderr, "tc_host_hdc_session send tc_device_recv_flag[%d] len[%d]\n", tc_device_recv_flag,
            tc_device_hdc_len);
    } else if (session == tc_device_hdc_session) {
        tc_host_hdc_len = ((struct drvHdcMsgBuf*)(msg + 1))->len;
        memcpy(tc_host_hdc_memory, ((struct drvHdcMsgBuf*)(msg + 1))->pBuf, tc_host_hdc_len);
        tc_host_recv_flag = 1;

        fprintf(
            stderr, "tc_device_hdc_session send tc_host_recv_flag[%d] len[%d]\n", tc_host_recv_flag, tc_host_hdc_len);
    } else {
        fprintf(stderr, "session[%d] drvHdcSend fail\n", session);
        return DRV_ERROR_RESERVED;
    }
    return DRV_ERROR_NONE;
};

DLLEXPORT hdcError_t drvHdcSetSessionReference(HDC_SESSION session) { return DRV_ERROR_NONE; };

drvError_t halNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t* val) { return 0; }

drvError_t halMemAlloc(void** pp, unsigned long long size, unsigned long long flag) { return 0; }

drvError_t halMemFree(void* pp) { return 0; }

drvError_t drvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t* devIndex) { return 0; }

drvError_t halHdcGetSessionAttr(HDC_SESSION session, int attr, int* value)
{
    if (value == NULL) {
        return -1;
    }

    *value = 0;
    return 0;
}

drvError_t drvGetLocalDevIDByHostDevID(uint32_t dev_id, uint32_t* chip_id)
{
    *chip_id = RS_HOSTID2DEVID((unsigned int)dev_id);
    return 0;
}

drvError_t drvGetDevIDByLocalDevID(uint32_t localDevId, uint32_t* devId)
{
    *devId = localDevId;
    return 0;
}

drvError_t halGetPhyDevIdByudevId(uint32_t udevId, uint32_t* phyDevId)
{
    *phyDevId = udevId;
    return 0;
}

DV_ONLINE DVresult
halMemBindSibling(int hostPid, int aicpuPid, unsigned int vfid, unsigned int dev_id, unsigned int flag)
{
    return 0;
}

DLLEXPORT drvError_t drvQueryProcessHostPid(
    int pid, unsigned int* chip_id, unsigned int* vfid, unsigned int* host_pid, unsigned int* cp_type)
{
    if (chip_id != NULL) {
        *chip_id = 0;
    }
    if (vfid != NULL) {
        *vfid = 0;
    }
    if (host_pid != NULL) {
        *host_pid = getpid();
    }
    if (cp_type != NULL) {
        *cp_type = 0;
    }
    return 0;
}

DLLEXPORT DVresult halMemGetInfoEx(DVdevice device, unsigned int type, struct MemInfo* info) { return 0; }

int halGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen)
{
    return 0;
}

int AscendHalDlclose(void* handle) { return 0; }
int halSetUserConfig(unsigned int dev_id, const char* name, unsigned char* buf, unsigned int buf_size) { return 0; }
int halClearUserConfig(unsigned int devid, const char* name) { return 0; }
void* hal_api_handle = (void*)(uintptr_t)0xabcd;

struct {
    const char* name;
    void* func;
} func_map[]
    = {{"drvGetDevNum", drvGetDevNum},
       {"drvGetLocalDevIDByHostDevID", drvGetLocalDevIDByHostDevID},
       {"drvGetDevIDByLocalDevID", drvGetDevIDByLocalDevID},
       {"drvDeviceGetIndexByPhyId", drvDeviceGetIndexByPhyId},
       {"drvDeviceGetPhyIdByIndex", drvDeviceGetPhyIdByIndex},
       {"halGetPhyDevIdByudevId", halGetPhyDevIdByudevId},
       {"halHdcGetSessionAttr", halHdcGetSessionAttr},
       {"drvHdcGetCapacity", drvHdcGetCapacity},
       {"drvHdcClientCreate", drvHdcClientCreate},
       {"drvHdcClientDestroy", drvHdcClientDestroy},
       {"drvHdcSessionConnect", drvHdcSessionConnect},
       {"drvHdcServerCreate", drvHdcServerCreate},
       {"drvHdcServerDestroy", drvHdcServerDestroy},
       {"drvHdcSessionAccept", drvHdcSessionAccept},
       {"drvHdcSessionClose", drvHdcSessionClose},
       {"drvHdcFreeMsg", drvHdcFreeMsg},
       {"drvHdcReuseMsg", drvHdcReuseMsg},
       {"drvHdcAddMsgBuffer", drvHdcAddMsgBuffer},
       {"drvHdcGetMsgBuffer", drvHdcGetMsgBuffer},
       {"halHdcRecv", halHdcRecv},
       {"halHdcSend", halHdcSend},
       {"drvHdcAllocMsg", drvHdcAllocMsg},
       {"drvHdcSetSessionReference", drvHdcSetSessionReference},
       {"drvGetProcessSign", drvGetProcessSign},
       {"drvDeviceGetBareTgid", drvDeviceGetBareTgid},
       {"halNotifyGetInfo", halNotifyGetInfo},
       {"halMemAlloc", halMemAlloc},
       {"halMemFree", halMemFree},
       {"halEschedSubmitEvent", halEschedSubmitEvent},
       {"halSetUserConfig", halSetUserConfig},
       {"halClearUserConfig", halClearUserConfig},
       {"halGetDeviceInfo", halGetDeviceInfo},
       {"halBindCgroup", halBindCgroup},
       {"drvGetPlatformInfo", drvGetPlatformInfo},
       {"halGetChipInfo", halGetChipInfo},
       {"halQueryDevpid", halQueryDevpid},
       {"halHdcSessionConnectEx", halHdcSessionConnectEx},
       {"halMemBindSibling", halMemBindSibling},
       {"drvQueryProcessHostPid", drvQueryProcessHostPid},
       {"halMemGetInfoEx", halMemGetInfoEx},
       {"halGrpQuery", halGrpQuery},
       {"halSensorNodeRegister", halSensorNodeRegister},
       {"halSensorNodeUnregister", halSensorNodeUnregister},
       {"halSensorNodeUpdateState", halSensorNodeUpdateState},
       {"halBuffAllocAlignEx", halBuffAllocAlignEx},
       {"halBuffFree", halBuffFree},
       {"halEschedAttachDevice", halEschedAttachDevice},
       {"halEschedCreateGrp", halEschedCreateGrp},
       {"halEschedSubscribeEvent", halEschedSubscribeEvent},
       {"halEschedWaitEvent", halEschedWaitEvent},
       {NULL, NULL}};

static void* find_stub_func_by_name(const char* func_name)
{
    void* ret = NULL;
    int index = 0;

    for (; func_map[index].name != NULL; index++) {
        if (strcmp(func_map[index].name, func_name) == 0) {
            ret = func_map[index].func;
            break;
        }
    }

    return ret;
}

void* AscendHalDlsym(void* handle, const char* funcName)
{
    if (handle == hal_api_handle) {
        return find_stub_func_by_name(funcName);
    } else {
        return dlsym(handle, funcName);
    }
}

void* AscendHalDlopen(const char* libName, int mode)
{
    if (strcmp(libName, "libascend_hal.so") == 0) {
        return hal_api_handle;
    } else {
        return dlopen(libName, mode);
    }
}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType) { return DRV_ERROR_NONE; }

drvError_t drvGetPlatformInfo(uint32_t* info) { return DRV_ERROR_NONE; }

drvError_t halGetChipInfo(unsigned int devId, halChipInfo* chipInfo)
{
    strcpy(chipInfo->name, "950");
    return DRV_ERROR_NONE;
}

drvError_t halQueryDevpid(struct halQueryDevpidInfo info, pid_t* dev_pid) { return DRV_ERROR_NOT_SUPPORT; }

hdcError_t halHdcSessionConnectEx(int peer_node, int peer_devid, int peer_pid, HDC_CLIENT client, HDC_SESSION* pSession)
{
    return DRV_ERROR_NOT_SUPPORT;
}

drvError_t halSensorNodeRegister(uint32_t devId, struct halSensorNodeCfg* cfg, uint64_t* handle)
{
    *handle = 1;
    return 0;
}

drvError_t halSensorNodeUnregister(uint32_t devId, uint64_t handle) { return 0; }

drvError_t halSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    return 0;
}

int halBuffAllocAlignEx(uint64_t size, unsigned int align, unsigned long flag, int grp_id, void** buff) { return 0; }

int halBuffFree(void* buff) { return 0; }

drvError_t halEschedAttachDevice(uint32_t devId) { return 0; }

drvError_t halEschedCreateGrp(uint32_t devId, uint32_t grpId, GROUP_TYPE type) { return 0; }

drvError_t
halEschedSubscribeEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap)
{
    return 0;
}

drvError_t
halEschedWaitEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout, struct event_info* event)
{
    return DRV_ERROR_NO_EVENT;
}
