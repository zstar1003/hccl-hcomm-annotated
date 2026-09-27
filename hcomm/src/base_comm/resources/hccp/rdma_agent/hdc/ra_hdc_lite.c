/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/prctl.h>
#include <pthread.h>
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc.h"
#include "securec.h"
#include "ra.h"
#include "ra_rdma_lite.h"
#include "dl_hal_function.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_hdc_lite.h"

struct RaCqeErrInfo gRaCqeErr[RA_MAX_PHY_ID_NUM];

void RaHdcLiteStoreTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, struct rdma_lite_cq *liteCq,
    struct rdma_lite_device_cq_attr *deviceCqAttr)
{
    struct TypicalLiteCqEntry *entry;
    struct TypicalLiteCqEntry *tmp;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->typicalLiteCqMutex);

    RA_LIST_GET_HEAD_ENTRY(tmp, entry, &rdmaHandle->typicalLiteCqList, list, struct TypicalLiteCqEntry);
    for (; &tmp->list != &rdmaHandle->typicalLiteCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct TypicalLiteCqEntry, list)) {
        if (tmp->cqn == cqn) {
            tmp->liteCq = liteCq;
            tmp->deviceCqAttr = *deviceCqAttr;
            RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
            hccp_info_rma("[store][ra_hdc_lite]updated typical lite cq, phyId[%u] cqn[%u]", phyId, cqn);
            return;
        }
    }

    entry = calloc(1, sizeof(struct TypicalLiteCqEntry));
    if (entry == NULL) {
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
        hccp_err("[store][ra_hdc_lite]calloc typical lite cq entry failed, cqn[%u]", cqn);
        return;
    }
    entry->phyId = phyId;
    entry->cqn = cqn;
    entry->liteCq = liteCq;
    entry->deviceCqAttr = *deviceCqAttr;
    RaListAddTail(&entry->list, &rdmaHandle->typicalLiteCqList);
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
    hccp_info_rma("[store][ra_hdc_lite]stored typical lite cq, phyId[%u] cqn[%u]", phyId, cqn);
}

struct rdma_lite_cq *RaHdcLiteFindTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn)
{
    struct TypicalLiteCqEntry *entry;
    struct TypicalLiteCqEntry *tmp;
    struct rdma_lite_cq *liteCq = NULL;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->typicalLiteCqMutex);
    if (RaListEmpty(&rdmaHandle->typicalLiteCqList)) {
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
        hccp_err("[find][ra_hdc_lite]typical lite cq list is empty, phyId[%u] cqn[%u]", phyId, cqn);
        return NULL;
    }

    RA_LIST_GET_HEAD_ENTRY(tmp, entry, &rdmaHandle->typicalLiteCqList, list, struct TypicalLiteCqEntry);
    for (; &tmp->list != &rdmaHandle->typicalLiteCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct TypicalLiteCqEntry, list)) {
        if (tmp->cqn == cqn) {
            liteCq = tmp->liteCq;
            break;
        }
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);

    if (liteCq != NULL) {
        hccp_info_rma("[find][ra_hdc_lite]found typical lite cq, phyId[%u] cqn[%u]", phyId, cqn);
    } else {
        hccp_err("[find][ra_hdc_lite]typical lite cq not found, phyId[%u] cqn[%u]", phyId, cqn);
    }
    return liteCq;
}

int RaHdcLiteFindTypicalCqAttr(struct RaRdmaHandle *rdmaHandle, unsigned int cqn,
    struct rdma_lite_device_cq_attr **deviceCqAttr)
{
    struct TypicalLiteCqEntry *entry;
    struct TypicalLiteCqEntry *tmp;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret = -EINVAL;

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->typicalLiteCqMutex);
    if (RaListEmpty(&rdmaHandle->typicalLiteCqList)) {
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
        hccp_err("[find][ra_hdc_lite]typical lite cq list is empty, phyId[%u] cqn[%u]", phyId, cqn);
        return -EINVAL;
    }

    RA_LIST_GET_HEAD_ENTRY(tmp, entry, &rdmaHandle->typicalLiteCqList, list, struct TypicalLiteCqEntry);
    for (; &tmp->list != &rdmaHandle->typicalLiteCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct TypicalLiteCqEntry, list)) {
        if (tmp->cqn == cqn) {
            *deviceCqAttr = &tmp->deviceCqAttr;
            ret = 0;
            break;
        }
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);

    if (ret != 0) {
        hccp_err("[find][ra_hdc_lite]typical lite cq attr not found, phyId[%u] cqn[%u]", phyId, cqn);
    }
    return ret;
}

void RaHdcLiteRemoveTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct TypicalLiteCqEntry *entry;
    struct TypicalLiteCqEntry *tmp;

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->typicalLiteCqMutex);
    if (RaListEmpty(&rdmaHandle->typicalLiteCqList)) {
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
        return;
    }

    RA_LIST_GET_HEAD_ENTRY(tmp, entry, &rdmaHandle->typicalLiteCqList, list, struct TypicalLiteCqEntry);
    for (; &tmp->list != &rdmaHandle->typicalLiteCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct TypicalLiteCqEntry, list)) {
        if (tmp->cqn == cqn) {
            RaListDel(&tmp->list);
            free(tmp);
            RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
            hccp_info_rma("[remove][ra_hdc_lite]removed typical lite cq, phyId[%u] cqn[%u]", phyId, cqn);
            return;
        }
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->typicalLiteCqMutex);
    hccp_warn_rma("[remove][ra_hdc_lite]typical lite cq not found for removal, phyId[%u] cqn[%u]", phyId, cqn);
}

STATIC void *RaHdcLitePthread(void *arg);

STATIC int RaHdcGetDrvLiteSupport(unsigned int phyId, bool enabled910aLite, unsigned int *support)
{
    int ret;
    size_t outLen = 0;
    unsigned int logicId;
    int64_t deviceInfo = 0;
    struct supportFeaturePara inPara = {0};
    struct supportFeaturePara outPara = {0};

    ret = DlDrvDeviceGetIndexByPhyId(phyId, &logicId);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_lite]dl_drv_device_get_index_by_phy_id failed, ret(%d), phyId(%u)", ret, phyId), ret);

    // enabled_910a_lite not explicitly set to true, disabled lite if chip_type is 910A due to memory limits
    if (!enabled910aLite) {
        ret = DlHalGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_VERSION, &deviceInfo);
        CHK_PRT_RETURN(ret,
            hccp_err("[init][ra_hdc_lite]dl_hal_get_device_info failed, ret(%d), phyId(%u)", ret, phyId), ret);
        if (DlHalPlatGetChip((uint64_t)deviceInfo) == CHIP_TYPE_910A) { // Memory Limits
            *support = 0;
            hccp_info_init("[init][ra_hdc_lite]device_info:0x%llx not support, phyId(%u)", deviceInfo, phyId);
            return 0;
        }
    }

    inPara.support_feature = CTRL_SUPPORT_DEV_MEM_REGISTER_MASK | CTRL_SUPPORT_PCIE_BAR_HUGE_MEM_MASK;
    inPara.devid = logicId;
    ret = DlHalMemCtl(CTRL_TYPE_SUPPORT_FEATURE, &inPara, sizeof(struct supportFeaturePara), &outPara, &outLen);
    if ((ret != 0) || (((outPara.support_feature & CTRL_SUPPORT_DEV_MEM_REGISTER_MASK) == 0) &&
                          ((outPara.support_feature & CTRL_SUPPORT_PCIE_BAR_HUGE_MEM_MASK) == 0))) {
        *support = 0;
        return 0;
    }

    if ((outPara.support_feature & CTRL_SUPPORT_DEV_MEM_REGISTER_MASK) != 0) {
        *support = LITE_SUPPORT_DEV_MEM_REGISTER;
    }
    if ((outPara.support_feature & CTRL_SUPPORT_PCIE_BAR_HUGE_MEM_MASK) != 0) {
        *support |= LITE_SUPPORT_PCIE_BAR_HUGE_MEM;
    }

    return 0;
}

STATIC void RaHdcGetOpcodeLiteSupport(unsigned int phyId, unsigned int supportFeature, int *support)
{
    int ret;
    unsigned int interfaceVersion = 0;

    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_GET_LITE_SUPPORT, &interfaceVersion);
    // get version failed or opcode interface_version is 0: opcode not support lite
    if (ret != 0 || interfaceVersion == 0) {
        hccp_info_init("[init][ra_hdc_lite]get opcode not support, ret[%d] != 0 or interfaceVersion is 0", ret);
        *support = LITE_NOT_SUPPORT;
        return;
    }

    // at least driver&host support lite 4KB page_size align
    if ((supportFeature & LITE_SUPPORT_DEV_MEM_REGISTER) != 0) {
        *support = LITE_ALIGN_4KB;
        return;
    }

    // driver&host support lite 2MB page_size align
    if ((interfaceVersion == LITE_VERSION_V2) && ((supportFeature & LITE_SUPPORT_PCIE_BAR_HUGE_MEM) != 0)) {
        *support = LITE_ALIGN_2MB;
        return;
    }

    // none of 4KB page_size align & 2MB page_size align lite support
    hccp_info_init("[init][ra_hdc_lite]get opcode not support, interfaceVersion[%u] supportFeature[0x%x]",
        interfaceVersion, supportFeature);
    *support = LITE_NOT_SUPPORT;
    return;
}

STATIC int RaHdcGetRdmaLiteSupport(struct RaRdmaHandle *rdmaHandle, unsigned int supportFeature, int *support)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    unsigned int rdevIndex = rdmaHandle->rdevIndex;
    union OpLiteSupportData liteSupportData;
    int supportLite = 0;
    int ret;

    // get opcode support lite with support_feature
    RaHdcGetOpcodeLiteSupport(phyId, supportFeature, &supportLite);
    if (supportLite == LITE_NOT_SUPPORT) {
        *support = LITE_NOT_SUPPORT;
        return 0;
    }

    // no need to support lite if enabled_2mb_lite not enabled
    if (supportLite == LITE_ALIGN_2MB && !rdmaHandle->enabled2mbLite) {
        hccp_run_info("[init][ra_hdc_lite]rdma_handle->enabled_2mb_lite=%d, no need to support LITE_ALIGN_2MB",
            rdmaHandle->enabled2mbLite);
        *support = LITE_NOT_SUPPORT;
        return 0;
    }

    // RA_RS_GET_LITE_SUPPORT will set rdev_cb->support_lite = 1
    (void)memset_s(&liteSupportData, sizeof(liteSupportData), 0, sizeof(liteSupportData));
    liteSupportData.txData.phyId = phyId;
    liteSupportData.txData.rdevIndex = rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_GET_LITE_SUPPORT, phyId, (char *)&liteSupportData, sizeof(union OpLiteSupportData));
    if (ret != 0) {
        if (ret == -EPROTONOSUPPORT) {
            *support = LITE_NOT_SUPPORT;
            ret = 0;
        } else {
            hccp_err("[init][ra_hdc_lite]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        }
        return ret;
    }

    *support = supportLite;
    return 0;
}

STATIC int RaHdcGetLiteSupport(struct RaRdmaHandle *rdmaHandle, unsigned int phyId)
{
    int ret;
    unsigned int supportFeature = 0;

#ifndef HNS_ROCE_LLT
    ret = RaHdcGetDrvLiteSupport(phyId, rdmaHandle->enabled910aLite, &supportFeature);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_lite]ra_hdc_get_drv_lite_support failed, ret(%d), phyId(%u)", ret, phyId), ret);
#else
    supportFeature = 1;
#endif
    if (supportFeature != 0) {
        ret = RaHdcGetRdmaLiteSupport(rdmaHandle, supportFeature, &rdmaHandle->supportLite);
        CHK_PRT_RETURN(ret,
            hccp_err("[init][ra_hdc_lite]ra_hdc_get_rdma_lite_support failed, ret(%d), phyId(%u)", ret, phyId), ret);

        if (rdmaHandle->supportLite) {
            hccp_run_info("[init][ra_hdc_lite]support_feature:0x%x, supportLite:%u", supportFeature,
                rdmaHandle->supportLite);
            ret = RaHdcRdmaLiteApiInit();
            CHK_PRT_RETURN(ret,
                hccp_err("[init][ra_hdc_lite]ra_hdc_rdma_lite_api_init failed, ret(%d), phyId(%u)", ret, phyId), ret);
        }
    }

    return 0;
}

STATIC int RaSensorNodeRegister(unsigned int phyId, struct RaRdmaHandle *rdmaHandle)
{
    struct halSensorNodeCfg cfg = {0};
    unsigned int interfaceVersion = 0;
    int ret;

    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_RDEV_INIT, &interfaceVersion);
    if ((ret != 0) || (interfaceVersion <= RA_RS_OPCODE_BASE_VERSION)) {
        /* unknown or old version, not support sensor */
        rdmaHandle->sensorHandle = 0;
        hccp_warn_init("[init][ra_hdc_lite]not support sensor, ret:%d, phyId:%u, interfaceVersion:%u", ret, phyId,
            interfaceVersion);
        return 0;
    }

    rdmaHandle->sensorUpdateCnt = 0;
    ret = sprintf_s(cfg.name, sizeof(cfg.name), "roce_ra_%d", getpid());
    CHK_PRT_RETURN(ret <= 0, hccp_err("[init][ra_hdc_lite]sprintf_s name err, ret:%d, phyId:%u", ret, phyId),
        -ESAFEFUNC);

    cfg.NodeType = HAL_DMS_DEV_TYPE_HCCP;
    cfg.SensorType = RDMA_CQE_ERR_SENSOR_TYPE;
    cfg.AssertEventMask = RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_MASK;
    cfg.DeassertEventMask = RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE_MASK;
    ret = DlHalSensorNodeRegister(rdmaHandle->logicDevid, &cfg, &rdmaHandle->sensorHandle);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_lite]dl_hal_sensor_node_register failed, ret(%d)", ret), ret);

    return 0;
}

STATIC int RaHdcLiteGetRdevCap(struct RaRdmaHandle *rdmaHandle, unsigned int phyId, unsigned int rdevIndex,
    union OpLiteRdevCapData *liteRdevCapData)
{
#define PAGE_ALIGN_2MB (2 * 1024 * 1024)
    int ret;

    liteRdevCapData->txData.phyId = phyId;
    liteRdevCapData->txData.rdevIndex = rdevIndex;
    ret = RaHdcProcessMsg(RA_RS_GET_LITE_RDEV_CAP, phyId, (char *)liteRdevCapData, sizeof(union OpLiteRdevCapData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_lite_ctx]hdc get lite rdev cap failed, ret(%d), phyId(%u)", ret, phyId), ret);

    // should change page_size to 2MB
    if (rdmaHandle->supportLite == LITE_ALIGN_2MB) {
        liteRdevCapData->rxData.resp.cap.page_size = PAGE_ALIGN_2MB;
    }
    return 0;
}

STATIC int RaHdcLiteMutexInit(struct RaRdmaHandle *rdmaHandle, unsigned int phyId)
{
    int ret;

    ret = pthread_mutex_init(&rdmaHandle->rdevMutex, NULL);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_lite_ctx]pthread_mutex_init rdev_mutex failed ret(%d) phyId(%u)", ret, phyId);
        return -ESYSFUNC;
    }

    ret = pthread_mutex_init(&rdmaHandle->cqeErrCntMutex, NULL);
    if (ret != 0) {
        (void)pthread_mutex_destroy(&rdmaHandle->rdevMutex);
        hccp_err("[init][ra_hdc_lite_ctx]pthread_mutex_init cqe_err_cnt_mutex failed ret(%d) phyId(%u)", ret, phyId);
        return -ESYSFUNC;
    }

    ret = pthread_mutex_init(&rdmaHandle->typicalLiteCqMutex, NULL);
    if (ret != 0) {
        (void)pthread_mutex_destroy(&rdmaHandle->cqeErrCntMutex);
        (void)pthread_mutex_destroy(&rdmaHandle->rdevMutex);
        hccp_err("[init][ra_hdc_lite_ctx]pthread_mutex_init typical_lite_cq_mutex failed ret(%d) phyId(%u)", ret,
            phyId);
        return -ESYSFUNC;
    }

    return 0;
}

STATIC void RaHdcLiteMutexDeinit(struct RaRdmaHandle *rdmaHandle)
{
    (void)pthread_mutex_destroy(&rdmaHandle->typicalLiteCqMutex);
    (void)pthread_mutex_destroy(&rdmaHandle->cqeErrCntMutex);
    (void)pthread_mutex_destroy(&rdmaHandle->rdevMutex);
}

STATIC int RaHdcLiteCtxInit(struct RaRdmaHandle *rdmaHandle, unsigned int phyId, unsigned int rdevIndex)
{
    union OpLiteRdevCapData liteRdevCapData = {0};
    int ret = 0;

    if (rdmaHandle->supportLite == 0) {
        return 0;
    }

    // register sensor node
    ret = DlDrvDeviceGetIndexByPhyId(phyId, &rdmaHandle->logicDevid);
    CHK_PRT_RETURN(ret,
        hccp_err("[init][ra_hdc_lite_ctx]dl_drv_device_get_index_by_phy_id failed, ret(%d) phyId(%u)", ret, phyId),
        ret);
    ret = RaSensorNodeRegister(phyId, rdmaHandle);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_lite_ctx]ra_sensor_node_register failed, ret(%d) phyId(%u)", ret, phyId), ret);

    // alloc ctx
    ret = RaHdcLiteGetRdevCap(rdmaHandle, phyId, rdevIndex, &liteRdevCapData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[init][ra_hdc_lite_ctx]ra_hdc_lite_get_rdev_cap failed, ret(%d) phyId(%u)", ret, phyId), ret);
    rdmaHandle->liteCtx = RaRdmaLiteAllocCtx(phyId, &liteRdevCapData.rxData.resp.cap);
    if (rdmaHandle->liteCtx == NULL) {
        hccp_err("[init][ra_hdc_lite_ctx]ra_rdma_lite_alloc_ctx errno(%d) phyId(%u)", errno, phyId);
        ret = -EFAULT;
        goto unreg_sensor;
    }

    RA_INIT_LIST_HEAD(&rdmaHandle->qpList);
    RA_INIT_LIST_HEAD(&rdmaHandle->typicalLiteCqList);

    ret = RaHdcLiteMutexInit(rdmaHandle, phyId);
    if (ret != 0) {
        goto free_ctx;
    }

    if (rdmaHandle->disabledLiteThread) {
        hccp_run_info("lite thread disabled");
        return 0;
    }

    rdmaHandle->threadStatus = LITE_THREAD_STATUS_RUNNING;
    ret = pthread_create(&rdmaHandle->tid, NULL, RaHdcLitePthread, (void *)rdmaHandle);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_lite_ctx]pthread_create failed, ret:%d, phyId:%u errno:%d", ret, phyId, errno);
        rdmaHandle->threadStatus = LITE_THREAD_STATUS_DESTROY;
        ret = -ESYSFUNC;
        goto mutex_deinit;
    }

    return 0;

mutex_deinit:
    RaHdcLiteMutexDeinit(rdmaHandle);
free_ctx:
    RaRdmaLiteFreeCtx(rdmaHandle->liteCtx);
unreg_sensor:
    (void)DlHalSensorNodeUnregister(rdmaHandle->logicDevid, rdmaHandle->sensorHandle);
    return ret;
}

int RaHdcLiteInit(struct RaRdmaHandle *rdmaHandle, unsigned int phyId, unsigned int rdevIndex)
{
    int ret;

    ret = RaHdcGetLiteSupport(rdmaHandle, phyId);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_rdev]ra_hdc_get_lite_support failed ret(%d) phyId(%u)", ret, phyId);
        return ret;
    }

    ret = RaHdcLiteCtxInit(rdmaHandle, phyId, rdevIndex);
    if (ret != 0) {
        hccp_err("[init][ra_hdc_rdev]ra_hdc_lite_ctx_init failed ret(%d) phyId(%u)", ret, phyId);
        goto free_lite_api;
    }

    return 0;

free_lite_api:
    RaHdcRdmaLiteApiDeinit();
    return ret;
}

void RaHdcLiteDeinit(struct RaRdmaHandle *rdmaHandle)
{
#define FINISH_RUNNING 2
#define THREAD_STATUS_CHANGE_TIMEOUT 100
    int i;

    if (rdmaHandle->supportLite) {
        if (rdmaHandle->disabledLiteThread) {
            goto disabled_thread_out;
        }

        rdmaHandle->threadStatus = LITE_THREAD_STATUS_DESTROY;
        // wait thread change to finish running status(2), wait 100 times(total cost: 1s) until timeout
        for (i = 0; i < THREAD_STATUS_CHANGE_TIMEOUT && rdmaHandle->threadStatus != FINISH_RUNNING; i++) {
            usleep(RA_LITE_POLL_CQE_PERIOD_TIME);
        }
        // thread not in finish running status(2), report timeout
        if (rdmaHandle->threadStatus != FINISH_RUNNING) {
            hccp_run_info("hdc wait thread tid:%lu finish running timeout, thread status:%d", rdmaHandle->tid,
                rdmaHandle->threadStatus);
        }

    disabled_thread_out:
        if (!RaListEmpty(&rdmaHandle->typicalLiteCqList)) {
            struct TypicalLiteCqEntry *entry;
            struct TypicalLiteCqEntry *tmp;
            RA_LIST_GET_HEAD_ENTRY(tmp, entry, &rdmaHandle->typicalLiteCqList, list, struct TypicalLiteCqEntry);
            while (&tmp->list != &rdmaHandle->typicalLiteCqList) {
                entry = list_entry(tmp->list.next, struct TypicalLiteCqEntry, list);
                RaListDel(&tmp->list);
                free(tmp);
                tmp = entry;
            }
        }
        RaHdcLiteMutexDeinit(rdmaHandle);
        RaRdmaLiteFreeCtx(rdmaHandle->liteCtx);
        RaHdcRdmaLiteApiDeinit();
        (void)DlHalSensorNodeUnregister(rdmaHandle->logicDevid, rdmaHandle->sensorHandle);
    }
}

int RaHdcLiteGetCqAttr(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, struct rdma_lite_device_cq_attr *deviceCqAttr)
{
    union OpGetLiteCqAttrData getLiteCqAttrData = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    int ret;

    getLiteCqAttrData.txData.phyId = phyId;
    getLiteCqAttrData.txData.rdevIndex = rdmaHandle->rdevIndex;
    getLiteCqAttrData.txData.cqn = cqn;

    ret = RaHdcProcessMsg(RA_RS_GET_LITE_CQ_ATTR, phyId, (char *)&getLiteCqAttrData, sizeof(union OpGetLiteCqAttrData));
    if (ret) {
        hccp_err("[create][ra_hdc_lite_get_cq_attr]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
        return ret;
    }

    *deviceCqAttr = getLiteCqAttrData.rxData.deviceCqAttr;
    hccp_info_rma("[create][ra_hdc_lite_get_cq_attr]get lite cq attr success, cqn[%u] depth[%u]", cqn,
        deviceCqAttr->depth);
    return 0;
}

int RaHdcLiteCqCreate(struct RaRdmaHandle *rdmaHandle, unsigned int cqDepth, union OpTypicalCqCreateData *cqData,
    struct rdma_lite_cq **liteCq)
{
    struct rdma_lite_cq_attr liteCqAttr = {0};
    struct rdma_lite_device_cq_attr deviceCqAttr = {0};
    int ret;

    if (rdmaHandle->supportLite == 0) {
        hccp_warn_rma("[RaHdcLiteCqCreate]rdmaHandle->supportLite = 0");
        return 0;
    }

    ret = RaHdcLiteGetCqAttr(rdmaHandle, cqData->rxData.cqn, &deviceCqAttr);
    if (ret) {
        hccp_err("[create][ra_hdc_lite_cq]get lite cq attr failed, cqn[%u] ret[%d]", cqData->rxData.cqn, ret);
        return ret;
    }

    liteCqAttr.device_cq_attr = deviceCqAttr;
    liteCqAttr.mem_idx = 0;

    if (rdmaHandle->supportLite == LITE_ALIGN_2MB) {
        liteCqAttr.mem_idx = 0;
    }

    *liteCq = RaRdmaLiteCreateCq(rdmaHandle->liteCtx, &liteCqAttr);
    if (*liteCq == NULL) {
        hccp_err("[create][ra_hdc_lite_cq]create lite cq failed, errno(%d) cqDepth(%u)", errno, cqDepth);
        return -EFAULT;
    }

    RaHdcLiteStoreTypicalCq(rdmaHandle, cqData->rxData.cqn, *liteCq, &deviceCqAttr);

    hccp_info_rma("[create][ra_hdc_lite_cq]lite cq created successfully, cqDepth(%u), liteCq[%p]", cqDepth, *liteCq);

    return 0;
}

STATIC void RaHdcLiteQpAttrInit(struct RaQpHandle *qpHdc, struct rdma_lite_qp_attr *liteQpAttr,
    struct rdma_lite_qp_cap *cap)
{
    liteQpAttr->send_cq = qpHdc->sendLiteCq;
    liteQpAttr->recv_cq = qpHdc->recvLiteCq;
    liteQpAttr->qp_mode = qpHdc->qpMode;
    liteQpAttr->qp_type = RDMA_LITE_QPT_RC;
    liteQpAttr->cap.max_inline_data = cap->max_inline_data;
    liteQpAttr->cap.max_send_sge = cap->max_send_sge;
    liteQpAttr->cap.max_recv_sge = cap->max_recv_sge;
    liteQpAttr->cap.max_send_wr = cap->max_send_wr;
    liteQpAttr->cap.max_recv_wr = cap->max_recv_wr;
}

STATIC int RaHdcLiteInitMemPool(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc,
    struct rdma_lite_cq_attr *liteSendCqAttr, struct rdma_lite_cq_attr *liteRecvCqAttr,
    struct rdma_lite_qp_attr *liteQpAttr)
{
    union OpLiteMemAttrData liteMemAttrData = {0};
    unsigned int phyId = qpHdc->phyId;
    int ret;

    if (rdmaHandle->supportLite != LITE_ALIGN_2MB) {
        return 0;
    }

    liteMemAttrData.txData.phyId = phyId;
    liteMemAttrData.txData.rdevIndex = rdmaHandle->rdevIndex;
    liteMemAttrData.txData.qpn = qpHdc->qpn;
    ret = RaHdcProcessMsg(RA_RS_GET_LITE_MEM_ATTR, phyId, (char *)&liteMemAttrData, sizeof(union OpLiteMemAttrData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RaRdmaLiteInitMemPool(rdmaHandle->liteCtx, (struct rdma_lite_mem_attr *)&liteMemAttrData.rxData.resp.memData);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra_rdma_lite_init_mem_pool failed ret(%d) phyId(%u)", ret, phyId), ret);

    qpHdc->memIdx = liteMemAttrData.rxData.resp.memData.mem_idx;
    liteSendCqAttr->mem_idx = qpHdc->memIdx;
    liteRecvCqAttr->mem_idx = qpHdc->memIdx;
    liteQpAttr->mem_idx = qpHdc->memIdx;
    return 0;
}

STATIC void RaHdcLiteDeinitMemPool(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc)
{
    unsigned int phyId = qpHdc->phyId;
    int ret;

    if (rdmaHandle->supportLite != LITE_ALIGN_2MB) {
        return;
    }

    ret = RaRdmaLiteDeinitMemPool(rdmaHandle->liteCtx, qpHdc->memIdx);
    if (ret != 0) {
        hccp_err("[create][ra_hdc_lite_qp]ra_rdma_lite_deinit_mem_pool failed ret(%d) phyId(%u)", ret, phyId);
    }
    return;
}

STATIC int RaHdcLiteGetCqQpAttr(struct RaQpHandle *qpHdc, struct rdma_lite_cq_attr *liteSendCqAttr,
    struct rdma_lite_cq_attr *liteRecvCqAttr, struct rdma_lite_qp_attr *liteQpAttr)
{
    union OpLiteQpCqAttrData liteQpCqAttrData = {0};
    unsigned int phyId = qpHdc->phyId;
    int ret;

    liteQpCqAttrData.txData.phyId = phyId;
    liteQpCqAttrData.txData.rdevIndex = qpHdc->rdevIndex;
    liteQpCqAttrData.txData.qpn = qpHdc->qpn;
    ret = RaHdcProcessMsg(RA_RS_GET_LITE_QP_CQ_ATTR, phyId, (char *)&liteQpCqAttrData,
        sizeof(union OpLiteQpCqAttrData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    qpHdc->dbIndex = liteQpCqAttrData.rxData.resp.qpData.qp_info;
    liteSendCqAttr->device_cq_attr = liteQpCqAttrData.rxData.resp.sendCqData;
    liteRecvCqAttr->device_cq_attr = liteQpCqAttrData.rxData.resp.recvCqData;
    ret = memcpy_s((void *)&(liteQpAttr->device_qp_attr), sizeof(liteQpAttr->device_qp_attr),
        (void *)&liteQpCqAttrData.rxData.resp.qpData, sizeof(liteQpCqAttrData.rxData.resp.qpData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[create][ra_hdc_lite_qp]memcpy_s failed ret(%d) phyId(%u)", ret, phyId), ret);

    return 0;
}

int RaHdcLiteGetQpAttr(struct RaQpHandle *qpHdc, struct rdma_lite_qp_attr *liteQpAttr)
{
    union OpLiteQpAttrData liteQpAttrData = {0};
    unsigned int phyId = qpHdc->phyId;
    int ret;

    liteQpAttrData.txData.phyId = phyId;
    liteQpAttrData.txData.rdevIndex = qpHdc->rdevIndex;
    liteQpAttrData.txData.qpn = qpHdc->qpn;
    ret = RaHdcProcessMsg(RA_RS_GET_LITE_QP_ATTR, phyId, (char *)&liteQpAttrData, sizeof(union OpLiteQpAttrData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId), ret);

    qpHdc->dbIndex = liteQpAttrData.rxData.resp.qpData.qp_info;
    ret = memcpy_s((void *)&(liteQpAttr->device_qp_attr), sizeof(liteQpAttr->device_qp_attr),
        (void *)&liteQpAttrData.rxData.resp.qpData, sizeof(liteQpAttrData.rxData.resp.qpData));
    CHK_PRT_RETURN(ret != 0, hccp_err("[create][ra_hdc_lite_qp]memcpy_s failed ret(%d) phyId(%u)", ret, phyId), ret);

    return 0;
}

int RaHdcLiteQpCreate(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc, struct rdma_lite_qp_cap *cap)
{
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct rdma_lite_cq_attr liteSendCqAttr = {0};
    struct rdma_lite_cq_attr liteRecvCqAttr = {0};
    struct rdma_lite_qp_attr liteQpAttr = {0};
    int ret;

    // not support rdma lite or not op mode qp
    if (rdmaHandle->supportLite == 0 || (qpHdc->qpMode != RA_RS_OP_QP_MODE && qpHdc->qpMode != RA_RS_OP_QP_MODE_EXT)) {
        return 0;
    }

    ret = RaHdcLiteGetCqQpAttr(qpHdc, &liteSendCqAttr, &liteRecvCqAttr, &liteQpAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra_hdc_lite_get_cq_qp_attr failed ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RaHdcLiteInitMemPool(rdmaHandle, qpHdc, &liteSendCqAttr, &liteRecvCqAttr, &liteQpAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp]ra_hdc_lite_init_mem_pool failed ret(%d) phyId(%u)", ret, phyId), -EFAULT);

    qpHdc->sendLiteCq = RaRdmaLiteCreateCq(rdmaHandle->liteCtx, &liteSendCqAttr);
    if (qpHdc->sendLiteCq == NULL) {
        hccp_err("[create][ra_hdc_lite_qp]create send_lite_cq failed, errno(%d) phyId(%u)", errno, phyId);
        ret = -EFAULT;
        goto free_mem_pool;
    }

    qpHdc->recvLiteCq = RaRdmaLiteCreateCq(rdmaHandle->liteCtx, &liteRecvCqAttr);
    if (qpHdc->recvLiteCq == NULL) {
        hccp_err("[create][ra_hdc_lite_qp]create recv_lite_cq failed, errno(%d) phyId(%u)", errno, phyId);
        ret = -EFAULT;
        goto free_send_lite_cq;
    }

    RaHdcLiteQpAttrInit(qpHdc, &liteQpAttr, cap);
    qpHdc->liteQp = RaRdmaLiteCreateQp(rdmaHandle->liteCtx, &liteQpAttr);
    if (qpHdc->liteQp == NULL) {
        hccp_err("[create][ra_hdc_lite_qp]ra_rdma_lite_create_qp failed, errno(%d) phyId(%u)", errno, phyId);
        ret = -EFAULT;
        goto free_recv_lite_cq;
    }

    ret = pthread_mutex_init(&qpHdc->qpMutex, NULL);
    if (ret != 0) {
        hccp_err("[create][ra_hdc_lite_qp]pthread_mutex_init failed ret(%d) phyId(%u)", ret, phyId);
        goto free_lite_qp;
    }

    ret = pthread_mutex_init(&qpHdc->cqeErrInfo.mutex, NULL);
    if (ret != 0) {
        hccp_err("[create][ra_hdc_lite_qp]pthread_mutex_init failed ret(%d) phyId(%u)", ret, phyId);
        (void)pthread_mutex_destroy(&qpHdc->qpMutex);
        goto free_lite_qp;
    }

    qpHdc->liteWc = calloc(MAX_POLL_CQE_NUM, sizeof(struct rdma_lite_wc));
    if (qpHdc->liteWc == NULL) {
        ret = -ENOMEM;
        (void)pthread_mutex_destroy(&qpHdc->qpMutex);
        (void)pthread_mutex_destroy(&qpHdc->cqeErrInfo.mutex);
        hccp_err("[create][ra_hdc_lite_qp]lite_wc calloc failed phyId(%u)", phyId);
        goto free_lite_qp;
    }

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
    RaListAddTail(&qpHdc->list, &rdmaHandle->qpList);
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);

    qpHdc->supportLite = rdmaHandle->supportLite;

    return 0;

free_lite_qp:
    (void)RaRdmaLiteDestroyQp(qpHdc->liteQp);
free_recv_lite_cq:
    (void)RaRdmaLiteDestroyCq(qpHdc->recvLiteCq);
free_send_lite_cq:
    (void)RaRdmaLiteDestroyCq(qpHdc->sendLiteCq);
free_mem_pool:
    RaHdcLiteDeinitMemPool(rdmaHandle, qpHdc);
    return ret;
}

void RaHdcLiteQpDestroy(struct RaQpHandle *qpHdc)
{
    if ((qpHdc->supportLite != LITE_NOT_SUPPORT) &&
        (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT)) {
        RA_PTHREAD_MUTEX_LOCK(&qpHdc->rdmaHandle->rdevMutex);
        RaListDel(&qpHdc->list);
        RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->rdmaHandle->rdevMutex);

        free(qpHdc->liteWc);
        qpHdc->liteWc = NULL;
        (void)pthread_mutex_destroy(&qpHdc->qpMutex);
        (void)pthread_mutex_destroy(&qpHdc->cqeErrInfo.mutex);
        (void)RaRdmaLiteDestroyQp(qpHdc->liteQp);
        qpHdc->liteQp = NULL;
        (void)RaRdmaLiteDestroyCq(qpHdc->sendLiteCq);
        qpHdc->sendLiteCq = NULL;
        (void)RaRdmaLiteDestroyCq(qpHdc->recvLiteCq);
        qpHdc->recvLiteCq = NULL;
        if (qpHdc->supportLite == LITE_ALIGN_2MB) {
            (void)RaRdmaLiteDeinitMemPool(qpHdc->rdmaHandle->liteCtx, qpHdc->memIdx);
        }
    }
}

void RaHdcLiteQpDestroyWithoutCQ(struct RaQpHandle *verbsQpHdc)
{
    if ((verbsQpHdc->supportLite != LITE_NOT_SUPPORT) &&
        (verbsQpHdc->qpMode == RA_RS_OP_QP_MODE || verbsQpHdc->qpMode == RA_RS_OP_QP_MODE_EXT)) {
        RA_PTHREAD_MUTEX_LOCK(&verbsQpHdc->rdmaHandle->rdevMutex);
        RaListDel(&verbsQpHdc->list);
        RA_PTHREAD_MUTEX_UNLOCK(&verbsQpHdc->rdmaHandle->rdevMutex);

        free(verbsQpHdc->liteWc);
        verbsQpHdc->liteWc = NULL;
        (void)pthread_mutex_destroy(&verbsQpHdc->qpMutex);
        (void)pthread_mutex_destroy(&verbsQpHdc->cqeErrInfo.mutex);
        (void)RaRdmaLiteDestroyQp(verbsQpHdc->liteQp);
        verbsQpHdc->liteQp = NULL;
        if (verbsQpHdc->supportLite == LITE_ALIGN_2MB) {
            (void)RaRdmaLiteDeinitMemPool(verbsQpHdc->rdmaHandle->liteCtx, verbsQpHdc->memIdx);
        }
    }
}

int RaHdcLiteQpCreateWithCQ(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc, struct rdma_lite_qp_cap *cap,
    struct rdma_lite_cq *sendLiteCq, struct rdma_lite_cq *recvLiteCq, unsigned int sendCqn, unsigned int recvCqn)
{
    struct rdma_lite_device_cq_attr *storedSendCqAttr = NULL;
    struct rdma_lite_device_cq_attr *storedRecvCqAttr = NULL;
    struct rdma_lite_cq_attr liteSendCqAttr = {0};
    struct rdma_lite_cq_attr liteRecvCqAttr = {0};
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;
    struct rdma_lite_qp_attr liteQpAttr = {0};
    int ret;

    if (rdmaHandle->supportLite == 0 || (qpHdc->qpMode != RA_RS_OP_QP_MODE && qpHdc->qpMode != RA_RS_OP_QP_MODE_EXT)) {
        return 0;
    }

    hccp_info_rma(
        "[QP_PATH][ra_hdc_lite_qp_with_cq]RaHdcLiteQpCreateWithCQ NEW path, phyId[%u] sendCqn[%u] recvCqn[%u]", phyId,
        sendCqn, recvCqn);

    ret = RaHdcLiteGetQpAttr(qpHdc, &liteQpAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp_with_cq]get qp attr failed ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RaHdcLiteFindTypicalCqAttr(rdmaHandle, sendCqn, &storedSendCqAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp_with_cq]find send cq attr failed phyId(%u) cqn(%u)", phyId, sendCqn), ret);
    ret = RaHdcLiteFindTypicalCqAttr(rdmaHandle, recvCqn, &storedRecvCqAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp_with_cq]find recv cq attr failed phyId(%u) cqn(%u)", phyId, recvCqn), ret);
    liteSendCqAttr.device_cq_attr = *storedSendCqAttr;
    liteRecvCqAttr.device_cq_attr = *storedRecvCqAttr;

    ret = RaHdcLiteInitMemPool(rdmaHandle, qpHdc, &liteSendCqAttr, &liteRecvCqAttr, &liteQpAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[create][ra_hdc_lite_qp_with_cq]ra_hdc_lite_init_mem_pool failed ret(%d) phyId(%u)", ret, phyId),
        -EFAULT);

    qpHdc->sendLiteCq = sendLiteCq;
    qpHdc->recvLiteCq = recvLiteCq;

    RaHdcLiteQpAttrInit(qpHdc, &liteQpAttr, cap);
    qpHdc->liteQp = RaRdmaLiteCreateQp(rdmaHandle->liteCtx, &liteQpAttr);
    if (qpHdc->liteQp == NULL) {
        hccp_err("[create][ra_hdc_lite_qp_with_cq]ra_rdma_lite_create_qp failed, errno(%d) phyId(%u)", errno, phyId);
        ret = -EFAULT;
        goto free_mem_pool;
    }

    ret = pthread_mutex_init(&qpHdc->qpMutex, NULL);
    if (ret != 0) {
        hccp_err("[create][ra_hdc_lite_qp_with_cq]pthread_mutex_init failed ret(%d) phyId(%u)", ret, phyId);
        goto free_lite_qp;
    }

    ret = pthread_mutex_init(&qpHdc->cqeErrInfo.mutex, NULL);
    if (ret != 0) {
        hccp_err("[create][ra_hdc_lite_qp_with_cq]pthread_mutex_init failed ret(%d) phyId(%u)", ret, phyId);
        (void)pthread_mutex_destroy(&qpHdc->qpMutex);
        goto free_lite_qp;
    }

    qpHdc->liteWc = calloc(MAX_POLL_CQE_NUM, sizeof(struct rdma_lite_wc));
    if (qpHdc->liteWc == NULL) {
        ret = -ENOMEM;
        (void)pthread_mutex_destroy(&qpHdc->qpMutex);
        (void)pthread_mutex_destroy(&qpHdc->cqeErrInfo.mutex);
        hccp_err("[create][ra_hdc_lite_qp_with_cq]lite_wc calloc failed phyId(%u)", phyId);
        goto free_lite_qp;
    }

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
    RaListAddTail(&qpHdc->list, &rdmaHandle->qpList);
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);

    qpHdc->supportLite = rdmaHandle->supportLite;

    return 0;

free_lite_qp:
    (void)RaRdmaLiteDestroyQp(qpHdc->liteQp);
free_mem_pool:
    RaHdcLiteDeinitMemPool(rdmaHandle, qpHdc);
    return ret;
}

int RaHdcLiteGetConnectedInfo(struct RaQpHandle *qpHdc)
{
    int ret;
    union OpLiteConnectedInfoData liteConnectedInfoData = {{0}};

    if ((qpHdc->supportLite != LITE_NOT_SUPPORT) &&
        (qpHdc->qpMode == RA_RS_OP_QP_MODE || qpHdc->qpMode == RA_RS_OP_QP_MODE_EXT)) {
        liteConnectedInfoData.txData.phyId = qpHdc->phyId;
        liteConnectedInfoData.txData.rdevIndex = qpHdc->rdevIndex;
        liteConnectedInfoData.txData.qpn = qpHdc->qpn;
        ret = RaHdcProcessMsg(RA_RS_GET_LITE_CONNECTED_INFO, qpHdc->phyId, (char *)&liteConnectedInfoData,
            sizeof(union OpLiteConnectedInfoData));
        CHK_PRT_RETURN(ret,
            hccp_err("[get][ra_hdc_lite_connect]ra hdc message process failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
            ret);

        ret = memcpy_s((void *)&qpHdc->localMr[0], sizeof(qpHdc->localMr),
            (void *)&liteConnectedInfoData.rxData.resp.localMr[0], sizeof(liteConnectedInfoData.rxData.resp.localMr));
        CHK_PRT_RETURN(ret,
            hccp_err("[recv][ra_hdc_lite_connect]memcpy_s local_mr failed, ret(%d) phyId(%u)", ret, qpHdc->phyId),
            -ESAFEFUNC);

        ret = memcpy_s((void *)&qpHdc->remMr[0], sizeof(qpHdc->remMr),
            (void *)&liteConnectedInfoData.rxData.resp.remMr[0], sizeof(liteConnectedInfoData.rxData.resp.remMr));
        CHK_PRT_RETURN(ret,
            hccp_err("[recv][ra_hdc_lite_connect]memcpy_s rem_mr failed, ret(%d) phyId(%u)", ret, qpHdc->phyId),
            -ESAFEFUNC);

        ret = RaRdmaLiteSetQpSl(qpHdc->liteQp, liteConnectedInfoData.rxData.resp.qosAttr.sl);
        CHK_PRT_RETURN(ret,
            hccp_err("[get][ra_hdc_lite_connect]ra_rdma_lite_set_qp_sl failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
            ret);
    }

    return 0;
}

void RaHdcLiteGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info)
{
    struct RaCqeErrInfo *errInfo = &gRaCqeErr[phyId];
    struct CqeErrInfo *tempInfo = &errInfo->info;

    RA_PTHREAD_MUTEX_LOCK(&errInfo->mutex);
    info->qpn = tempInfo->qpn;
    info->status = tempInfo->status;
    info->time = tempInfo->time;
    (void)memset_s(&errInfo->info, sizeof(struct CqeErrInfo), 0, sizeof(struct CqeErrInfo));
    RA_PTHREAD_MUTEX_UNLOCK(&errInfo->mutex);
}

int RaHdcLiteGetCqeErrInfoList(struct RaRdmaHandle *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num)
{
    struct RaQpHandle *qpHdcTmp1 = NULL;
    struct RaQpHandle *qpHdcTmp = NULL;
    unsigned int cqeErrIdx = 0;
    unsigned int numTmp = 0;

    // not support lite
    if (rdmaHandle->supportLite == 0) {
        *num = 0;
        return 0;
    }

    // no cqe err or no qp
    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->cqeErrCntMutex);
    if (rdmaHandle->cqeErrCnt == 0) {
        *num = 0;
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->cqeErrCntMutex);
        return 0;
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->cqeErrCntMutex);

    RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
    if (RaListEmpty(&rdmaHandle->qpList)) {
        *num = 0;
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);
        return 0;
    }

    // get & clear cqe err info from qp
    numTmp = *num;
    RA_LIST_GET_HEAD_ENTRY(qpHdcTmp, qpHdcTmp1, &rdmaHandle->qpList, list, struct RaQpHandle);
    for (; (&qpHdcTmp->list) != &rdmaHandle->qpList;
         qpHdcTmp = qpHdcTmp1, qpHdcTmp1 = list_entry(qpHdcTmp1->list.next, struct RaQpHandle, list)) {
        RA_PTHREAD_MUTEX_LOCK(&qpHdcTmp->cqeErrInfo.mutex);
        if (qpHdcTmp->cqeErrInfo.info.status == 0) {
            RA_PTHREAD_MUTEX_UNLOCK(&qpHdcTmp->cqeErrInfo.mutex);
            continue;
        }
        infoList[cqeErrIdx].status = qpHdcTmp->cqeErrInfo.info.status;
        infoList[cqeErrIdx].qpn = qpHdcTmp->cqeErrInfo.info.qpn;
        infoList[cqeErrIdx].time = qpHdcTmp->cqeErrInfo.info.time;
        qpHdcTmp->cqeErrInfo.info.status = 0;
        RA_PTHREAD_MUTEX_UNLOCK(&qpHdcTmp->cqeErrInfo.mutex);

        RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->cqeErrCntMutex);
        rdmaHandle->cqeErrCnt--;
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->cqeErrCntMutex);
        cqeErrIdx++;
        if (cqeErrIdx >= numTmp) {
            break;
        }
    }
    RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);

    *num = cqeErrIdx;
    return 0;
}

STATIC void RaHdcLiteSaveCqeErrInfo(struct RaQpHandle *qpHdc, unsigned int status)
{
    unsigned int phyId = qpHdc->phyId;
    struct RaCqeErrInfo *errInfo = &gRaCqeErr[phyId];
    struct CqeErrInfo *tempInfo = &errInfo->info;

    RA_PTHREAD_MUTEX_LOCK(&errInfo->mutex);
    if (tempInfo->status != 0) {
        hccp_run_info("over status=[0x%x], drop qpn[0x%x] err cqe status[0x%x]", tempInfo->status, qpHdc->qpn, status);
        RA_PTHREAD_MUTEX_UNLOCK(&errInfo->mutex);
        return;
    }
    tempInfo->status = status;
    tempInfo->qpn = qpHdc->qpn;
    (void)gettimeofday(&tempInfo->time, NULL);
    RA_PTHREAD_MUTEX_UNLOCK(&errInfo->mutex);
}

STATIC void RaHdcLiteSaveQpCqeErrInfo(struct RaQpHandle *qpHdc, unsigned int status)
{
    RA_PTHREAD_MUTEX_LOCK(&qpHdc->cqeErrInfo.mutex);
    if (qpHdc->cqeErrInfo.info.status != 0) {
        RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->cqeErrInfo.mutex);
        return;
    }
    qpHdc->cqeErrInfo.info.status = status;
    qpHdc->cqeErrInfo.info.qpn = (uint32_t)qpHdc->qpn;
    (void)gettimeofday(&qpHdc->cqeErrInfo.info.time, NULL);
    RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->cqeErrInfo.mutex);

    RA_PTHREAD_MUTEX_LOCK(&qpHdc->rdmaHandle->cqeErrCntMutex);
    qpHdc->rdmaHandle->cqeErrCnt++;
    RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->rdmaHandle->cqeErrCntMutex);
    return;
}

int RaHdcLiteInitCqeErrInfo(unsigned int phyId)
{
    int ret;
    struct RaCqeErrInfo *errInfo = &gRaCqeErr[phyId];

    ret = pthread_mutex_init(&errInfo->mutex, NULL);
    CHK_PRT_RETURN(ret, hccp_err("cqe err mutex_init failed ret %d!, normal ret 0", ret), -ESYSFUNC);

    (void)memset_s(&errInfo->info, sizeof(struct CqeErrInfo), 0, sizeof(struct CqeErrInfo));

    return 0;
}

void RaHdcLiteDeinitCqeErrInfo(unsigned int phyId)
{
    struct RaCqeErrInfo *errInfo = &gRaCqeErr[phyId];

    (void)pthread_mutex_destroy(&errInfo->mutex);
}

STATIC void RaRetryTimeoutExceptionCheck(struct RaRdmaHandle *rdmaHandle, struct rdma_lite_wc *wc)
{
    int ret = 0;

    if (rdmaHandle->sensorHandle == 0) {
        return;
    }

    if (wc->status != RDMA_LITE_WC_RETRY_EXC_ERR) {
        return;
    }

    /* The notification alarm framework does not filter alarms. In this example, only one notification
       alarm is reported by a single process, which does not need to be accurate. Therefore, no lock is used. */
    if (rdmaHandle->sensorUpdateCnt == 0) {
        ret = DlHalSensorNodeUpdateState(rdmaHandle->logicDevid, rdmaHandle->sensorHandle,
            RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE, GENERAL_EVENT_TYPE_ONE_TIME);
        if (ret == 0) {
            rdmaHandle->sensorUpdateCnt++;
        }
    }

    hccp_warn_rma("update sensor state logic_devid(%u), qpn(%u), sensorUpdateCnt(%d), ret(%d)\n",
        rdmaHandle->logicDevid, wc->qp_num, rdmaHandle->sensorUpdateCnt, ret);
}

STATIC void RaHdcLitePeriodPollCqe(struct RaRdmaHandle *rdmaHandle)
{
    int i;
    int ret = 0;
    struct rdma_lite_wc *liteWc;
    unsigned int sentWr, pollCqe;
    struct RaQpHandle *qpHdcTmp = NULL;
    struct RaQpHandle *qpHdcTmp1 = NULL;

    RA_LIST_GET_HEAD_ENTRY(qpHdcTmp, qpHdcTmp1, &rdmaHandle->qpList, list, struct RaQpHandle);
    for (; (&qpHdcTmp->list) != &rdmaHandle->qpList;
         qpHdcTmp = qpHdcTmp1, qpHdcTmp1 = list_entry(qpHdcTmp1->list.next, struct RaQpHandle, list)) {
        sentWr = qpHdcTmp->sendWrNum;
        pollCqe = sentWr - qpHdcTmp->pollCqeNum;

        if (pollCqe == 0) {
            continue;
        }

        liteWc = calloc(pollCqe, sizeof(struct rdma_lite_wc));
        if (liteWc == NULL) {
            hccp_err("[create][ra_hdc_period_poll]lite_wc calloc failed phyId(%u)", qpHdcTmp->phyId);
            break;
        }

        ret = RaRdmaLitePollCq(qpHdcTmp->sendLiteCq, pollCqe, liteWc);
        if (ret < 0) {
            hccp_err("ra_rdma_lite_poll_cq failed ret %d", ret);
            goto poll_cq_err;
        }

        for (i = 0; i < ret; i++) {
            if (liteWc[i].status != RDMA_LITE_WC_SUCCESS && liteWc[i].status != RDMA_LITE_WC_WR_FLUSH_ERR) {
                hccp_err("[create][ra_hdc_period_poll]failed CQE status[%u], wr[%llu]", liteWc[i].status,
                    liteWc[i].wr_id);
                RaHdcLiteSaveCqeErrInfo(qpHdcTmp, liteWc[i].status);
                RaHdcLiteSaveQpCqeErrInfo(qpHdcTmp, liteWc[i].status);
                RaRetryTimeoutExceptionCheck(rdmaHandle, &liteWc[i]);
                qpHdcTmp->liteQpState = LITE_QP_STATE_ERR;
            }
        }

        qpHdcTmp->pollCqeNum += (unsigned int)ret;

    poll_cq_err:
        free(liteWc);
        liteWc = NULL;
    }
}

STATIC void *RaHdcLitePthread(void *arg)
{
    struct RaRdmaHandle *rdmaHandle = (struct RaRdmaHandle *)arg;
    unsigned int phyId = rdmaHandle->rdevInfo.phyId;

    hccp_run_info("lite thread begin! thread_id:%lu, pid:%d, ppid:%d, phyId:%u", pthread_self(), getpid(), getppid(),
        phyId);
    CHK_PRT_RETURN(pthread_detach(pthread_self()),
        hccp_err("pthread_detach failed! thread_id:%lu, errno:%d, phyId:%u", pthread_self(), errno, phyId), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_hdc_lite", 0, 0, 0);

    while (1) {
        if (rdmaHandle->threadStatus == LITE_THREAD_STATUS_DESTROY) {
            break;
        }
        RA_PTHREAD_MUTEX_LOCK(&rdmaHandle->rdevMutex);
        if (rdmaHandle->threadStatus == LITE_THREAD_STATUS_SUSPEND) {
            RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);
            usleep(THREAD_SLEEP_TIME);
            continue;
        }
        RaHdcLitePeriodPollCqe(rdmaHandle);
        RA_PTHREAD_MUTEX_UNLOCK(&rdmaHandle->rdevMutex);
        usleep(RA_LITE_POLL_CQE_PERIOD_TIME);
    }

    // thread quit, change status to finish running status(2)
    rdmaHandle->threadStatus = LITE_THREAD_STATUS_FINISH_RUNNING;
    hccp_run_info("lite QUIT thread_id:%lu, pid:%d, phyId:%u", pthread_self(), getpid(), phyId);

    return NULL;
}

int RaHdcLitePollCq(struct RaQpHandle *qpHdc, bool isSendCq, unsigned int numEntries, struct rdma_lite_wc_v2 *liteWc)
{
    int ret = 0;
    struct rdma_lite_cq *cq = isSendCq ? qpHdc->liteQp->send_cq : qpHdc->liteQp->recv_cq;
    unsigned int *pollCqeNum = isSendCq ? &qpHdc->pollCqeNum : &qpHdc->pollRecvCqeNum;
    unsigned int wrNum = isSendCq ? qpHdc->sendWrNum : qpHdc->recvWrNum;
    int i;

    // no need to poll
    if ((wrNum - *pollCqeNum) == 0) {
        return 0;
    }

    ret = RaRdmaLitePollCqV2(cq, (int)numEntries, liteWc);
    CHK_PRT_RETURN(ret < 0, hccp_err("ra_rdma_lite_poll_cq_v2 failed, ret %d", ret), ret);
    CHK_PRT_RETURN(ret > (int)numEntries,
        hccp_err("ra_rdma_lite_poll_cq_v2 failed, expect maximum numEntries:%u but got %d", numEntries, ret), -EIO);

    for (i = 0; i < ret; i++) {
        RaRetryTimeoutExceptionCheck(qpHdc->rdmaHandle, &liteWc[i].wc);
    }

    *pollCqeNum += (unsigned int)ret;
    return ret;
}

STATIC int RaHdcLitePostSend(struct RaQpHandle *qpHdc, struct LiteMrInfo *localMr, struct LiteMrInfo *remMr,
    struct LiteSendWr *wr, struct SendWrRsp *wrRsp, u64 wrId)
{
    int i;
    int ret;
    struct rdma_lite_sge list[RA_SGLIST_MAX];
    struct rdma_lite_send_wr liteWr = {
        .sg_list = list,
        .opcode = wr->wr.op,
        .send_flags = wr->wr.sendFlag,
    };
    struct rdma_lite_send_wr *badWr = NULL;
    struct rdma_lite_post_send_resp resp = {0};
    struct rdma_lite_post_send_attr attr = {0};

    for (i = 0; i < wr->wr.bufNum && i < RA_SGLIST_MAX; i++) {
        list[i].addr = (uintptr_t)wr->wr.bufList[i].addr;
        list[i].length = wr->wr.bufList[i].len;
        list[i].lkey = localMr->key;
    }

    if (liteWr.opcode == RDMA_LITE_WR_WRITE_WITH_NOTIFY || liteWr.opcode == RDMA_LITE_WR_REDUCE_WRITE ||
        liteWr.opcode == RDMA_LITE_WR_REDUCE_WRITE_NOTIFY) {
        liteWr.imm_data = htobe32((wr->aux.notifyOffset & WRITE_NOTIFY_OFFSET_MASK) | WRITE_NOTIFY_VALUE_RECORD);
        attr.reduce_op = wr->aux.reduceType;
        attr.reduce_type = wr->aux.dataType;
    }

    if (liteWr.opcode == RDMA_LITE_WR_RDMA_WRITE_WITH_IMM || liteWr.opcode == RDMA_LITE_WR_SEND_WITH_IMM ||
        liteWr.opcode == RDMA_LITE_WR_ATOMIC_WRITE) {
        liteWr.imm_data = htobe32(wr->ext.immData);
    }

    liteWr.num_sge = i;
    liteWr.wr_id = wrId;
    // send op has no rem_mr, no need to assign
    if (wr->wr.op != RA_WR_SEND && wr->wr.op != RA_WR_SEND_WITH_IMM) {
        liteWr.rkey = remMr->key;
        liteWr.remote_addr = wr->wr.dstAddr;
    }

    ret = RaRdmaLitePostSend(qpHdc->liteQp, &liteWr, &badWr, &attr, &resp);
    if (ret) {
        return ret;
    }

    wrRsp->db.dbIndex = (unsigned int)qpHdc->dbIndex;
    wrRsp->db.dbInfo = resp.db.lite_db_info;

    return 0;
}

static int RaHdcLiteGetMr(struct RaQpHandle *qpHdc, unsigned long long addr, struct LiteMrInfo **mr,
    struct LiteMrInfo *srcMr, unsigned int mrNum)
{
    unsigned int i;

    RA_PTHREAD_MUTEX_LOCK(&qpHdc->qpMutex);

    for (i = 0; i < mrNum; i++) {
        if ((srcMr[i].addr <= addr) && (addr < srcMr[i].addr + srcMr[i].len)) {
            *mr = &srcMr[i];
            RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->qpMutex);
            return 0;
        }
    }

    RA_PTHREAD_MUTEX_UNLOCK(&qpHdc->qpMutex);

    return -EINVAL;
}

STATIC int RaHdcLiteHandleBp(struct RaQpHandle *qpHdc)
{
    u32 sendWr;

    if (qpHdc->sendWrNum >= qpHdc->pollCqeNum) {
        sendWr = qpHdc->sendWrNum - qpHdc->pollCqeNum;
    } else {
        sendWr = qpHdc->sendWrNum + (0xFFFFFFFF - qpHdc->pollCqeNum);
    }

    /*
     * Due to driver limitations, the software pointer updates before the hardware pointer.
     * The software must reserve sq_depth(2^x - 2) to prevent the backpressure mechanism from failing.
     */
    if (sendWr < (qpHdc->sqDepth - 2U)) {
        if (qpHdc->bpCnt != 0) {
            hccp_run_info("qpn:%u send_wr_num:%u poll_cqe_num:%u send_wr:%u sq_depth:%u "
                          "bp_cnt:%u, back pressure relieved",
                qpHdc->qpn, qpHdc->sendWrNum, qpHdc->pollCqeNum, sendWr, qpHdc->sqDepth, qpHdc->bpCnt);
            qpHdc->bpCnt = 0;
        }
        return 0;
    }

    // first time back pressure occurred
    if (qpHdc->bpCnt == 0) {
        hccp_run_warn("qpn:%u send_wr_num:%u poll_cqe_num:%u send_wr:%u sq_depth:%u, back pressure occurred",
            qpHdc->qpn, qpHdc->sendWrNum, qpHdc->pollCqeNum, sendWr, qpHdc->sqDepth);
    } else {
        hccp_warn_rma("qpn:%u send_wr_num:%u poll_cqe_num:%u send_wr:%u sq_depth:%u, back pressure continues bpCnt:%u",
            qpHdc->qpn, qpHdc->sendWrNum, qpHdc->pollCqeNum, sendWr, qpHdc->sqDepth, qpHdc->bpCnt);
    }

    qpHdc->bpCnt++;
    return -ENOMEM;
}

int RaHdcLiteTypicalSendWr(struct RaQpHandle *qpHdc, struct LiteSendWr *wr, struct SendWrRsp *opRsp,
    unsigned long long wrId)
{
    struct rdma_lite_post_send_resp resp = {0};
    struct rdma_lite_post_send_attr attr = {0};
    struct rdma_lite_sge list[RA_SGLIST_MAX];
    struct rdma_lite_send_wr liteWr = {
        .sg_list = list,
        .opcode = wr->wr.op,
        .send_flags = wr->wr.sendFlag,
    };
    struct rdma_lite_send_wr *badWr = NULL;
    int ret;
    int i;

    CHK_PRT_RETURN(qpHdc->liteQpState == LITE_QP_STATE_ERR,
        hccp_err("invalid liteQpState:%u qpn:%u phyId:%u", qpHdc->liteQpState, qpHdc->qpn, qpHdc->phyId), -EINVAL);

    ret = RaHdcLiteHandleBp(qpHdc);
    if (ret != 0) {
        return ret;
    }

    for (i = 0; i < wr->wr.bufNum && i < RA_SGLIST_MAX; i++) {
        list[i].addr = (uintptr_t)wr->wr.bufList[i].addr;
        list[i].length = wr->wr.bufList[i].len;
        list[i].lkey = wr->wr.bufList[i].lkey;
    }

    liteWr.num_sge = i;
    liteWr.wr_id = wrId;
    liteWr.rkey = wr->wr.rkey;
    liteWr.remote_addr = wr->wr.dstAddr;
    if (liteWr.opcode == RDMA_LITE_WR_WRITE_WITH_NOTIFY || liteWr.opcode == RDMA_LITE_WR_REDUCE_WRITE ||
        liteWr.opcode == RDMA_LITE_WR_REDUCE_WRITE_NOTIFY) {
        liteWr.imm_data = htobe32((wr->aux.notifyOffset & WRITE_NOTIFY_OFFSET_MASK) | WRITE_NOTIFY_VALUE_RECORD);
        attr.reduce_op = wr->aux.reduceType;
        attr.reduce_type = wr->aux.dataType;
    }
    liteWr.imm_data = htobe32(wr->ext.immData);

    ret = RaRdmaLitePostSend(qpHdc->liteQp, &liteWr, &badWr, &attr, &resp);
    if (ret) {
        if (ret == -ENOMEM) {
            hccp_warn_rma("[send][ra_hdc_wr]ra hdc post send unsuccessful, ret(%d) phyId(%u)", ret, qpHdc->phyId);
        } else {
            hccp_err("[send][ra_hdc_wr]ra hdc post send failed ret(%d) phyId(%u)", ret, qpHdc->phyId);
        }

        return ret;
    }

    opRsp->db.dbIndex = (unsigned int)qpHdc->dbIndex;
    opRsp->db.dbInfo = resp.db.lite_db_info;

    // user specify wr send_signal flag or user specify qp sq_sig_all flag
    if ((((uint32_t)wr->wr.sendFlag & RA_SEND_SIGNALED) != 0) || (qpHdc->sqSigAll != 0)) {
        qpHdc->sendWrNum++;
    }

    return 0;
}

int RaHdcLiteSendWr(struct RaQpHandle *qpHdc, struct LiteSendWr *wr, struct SendWrRsp *opRsp, unsigned long long wrId)
{
    struct LiteMrInfo *localMr = NULL;
    struct LiteMrInfo *remMr = NULL;
    int ret;

    ret = RaHdcLiteGetMr(qpHdc, wr->wr.bufList[0].addr, &localMr, qpHdc->localMr, RA_MR_MAX_NUM);
    CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_wr]ra hdc get local_mr failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
        ret);

    // send op no need to check & get remote mr
    if (wr->wr.op != RA_WR_SEND && wr->wr.op != RA_WR_SEND_WITH_IMM) {
        ret = RaHdcLiteGetMr(qpHdc, wr->wr.dstAddr, &remMr, qpHdc->remMr, RA_MR_MAX_NUM);
        CHK_PRT_RETURN(ret, hccp_err("[send][ra_hdc_wr]ra hdc get rem_mr failed ret(%d) phyId(%u)", ret, qpHdc->phyId),
            ret);
    }

    CHK_PRT_RETURN(qpHdc->liteQpState == LITE_QP_STATE_ERR,
        hccp_err("invalid liteQpState:%u qpn:%u phyId:%u", qpHdc->liteQpState, qpHdc->qpn, qpHdc->phyId), -EINVAL);

    ret = RaHdcLiteHandleBp(qpHdc);
    if (ret != 0) {
        return ret;
    }

    ret = RaHdcLitePostSend(qpHdc, localMr, remMr, wr, opRsp, wrId);
    if (ret) {
        if (ret == -ENOMEM) {
            hccp_warn_rma("[send][ra_hdc_wr]ra hdc post send unsuccessful, ret(%d) phyId(%u)", ret, qpHdc->phyId);
        } else {
            hccp_err("[send][ra_hdc_wr]ra hdc post send failed, ret(%d) phyId(%u)", ret, qpHdc->phyId);
        }

        return ret;
    }

    // user specify wr send_signal flag or user specify qp sq_sig_all flag
    if ((((uint32_t)wr->wr.sendFlag & RA_SEND_SIGNALED) != 0) || (qpHdc->sqSigAll != 0)) {
        qpHdc->sendWrNum++;
    }

    return 0;
}

int RaHdcLiteSendWrlist(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret;
    unsigned int i = 0;
    struct LiteSendWr normalWr = {0};

    while (i < wrlistNum.sendNum) {
        normalWr.wr.bufList = &(wr[i].memList);
        normalWr.wr.bufNum = 1;
        normalWr.wr.dstAddr = wr[i].dstAddr;
        normalWr.wr.op = wr[i].op;
        normalWr.wr.sendFlag = wr[i].sendFlags;
        ret = RaHdcLiteSendWr(qpHdc, &normalWr, &opRsp[i], HDC_LITE_DEFAULT_WR_ID);
        if (ret) {
            if (ret == -ENOMEM) {
                hccp_warn_rma("[send][ra_hdc_lite_wrlist]ra_hdc_lite_send_wr unsuccessful, ret(%d) phyId(%u) "
                              "send_index(%u)",
                    ret, qpHdc->phyId, i);
            } else {
                hccp_err("[send][ra_hdc_lite_wrlist]ra_hdc_lite_send_wr failed, ret(%d) phyId(%u) send_index(%u)", ret,
                    qpHdc->phyId, i);
            }

            *(wrlistNum.completeNum) = i;
            return ret;
        }

        i++;
    }

    *(wrlistNum.completeNum) = i;

    return 0;
}

int RaHdcLiteSendWrlistExt(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    int ret;
    unsigned int i = 0;
    struct LiteSendWr normalWr = {0};

    while (i < wrlistNum.sendNum) {
        normalWr.wr.bufList = &(wr[i].memList);
        normalWr.wr.bufNum = 1;
        normalWr.wr.dstAddr = wr[i].dstAddr;
        normalWr.wr.op = wr[i].op;
        normalWr.wr.sendFlag = wr[i].sendFlags;
        normalWr.aux = wr[i].aux;
        normalWr.ext = wr[i].ext;
        ret = RaHdcLiteSendWr(qpHdc, &normalWr, &opRsp[i], HDC_LITE_DEFAULT_WR_ID);
        if (ret) {
            if (ret == -ENOMEM) {
                hccp_warn_rma("[send][ra_hdc_lite_send_wrlist_ext]ra_hdc_lite_send_wr unsuccessful, ret(%d) phyId(%u) "
                              "send_index(%u)",
                    ret, qpHdc->phyId, i);
            } else {
                hccp_err("[send][ra_hdc_lite_send_wrlist_ext]ra_hdc_lite_send_wr failed, ret(%d) phyId(%u) "
                         "send_index(%u)",
                    ret, qpHdc->phyId, i);
            }

            *(wrlistNum.completeNum) = i;
            return ret;
        }

        i++;
    }

    *(wrlistNum.completeNum) = i;

    return 0;
}

int RaHdcLiteSendNormalWrlist(struct RaQpHandle *qpHdc, struct WrInfo wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum)
{
    struct LiteSendWr normalWr = {0};
    unsigned int i = 0;
    int ret = 0;

    while (i < wrlistNum.sendNum) {
        normalWr.wr.sendFlag = wr[i].sendFlags;
        normalWr.wr.rkey = wr[i].rkey;
        normalWr.wr.op = wr[i].op;
        normalWr.wr.dstAddr = wr[i].dstAddr;
        normalWr.wr.bufList = &(wr[i].memList);
        normalWr.wr.bufNum = 1;
        normalWr.aux = wr[i].aux;
        if (wr[i].op == RDMA_LITE_WR_RDMA_WRITE_WITH_IMM || wr[i].op == RDMA_LITE_WR_SEND_WITH_IMM ||
            wr[i].op == RDMA_LITE_WR_ATOMIC_WRITE) {
            normalWr.ext.immData = wr[i].immData;
        }
        ret = RaHdcLiteTypicalSendWr(qpHdc, &normalWr, &opRsp[i], wr[i].wrId);
        if (ret != 0) {
            if (ret == -ENOMEM) {
                hccp_warn_rma("[send][send_wrlist]ra_hdc_lite_send_wr unsuccessful, ret(%d) phyId(%u) send_index(%u)",
                    ret, qpHdc->phyId, i);
            } else {
                hccp_err("[send][send_wrlist]ra_hdc_lite_send_wr failed, ret(%d) phyId(%u) send_index(%u)", ret,
                    qpHdc->phyId, i);
            }

            break;
        }

        i++;
    }

    *(wrlistNum.completeNum) = i;

    return ret;
}

STATIC void RaHdcLiteBuildRecvWr(struct RecvWrlistData *wr, struct rdma_lite_sge *list,
    struct rdma_lite_recv_wr *liteWr)
{
    list->addr = (uintptr_t)wr->memList.addr;
    list->length = wr->memList.len;
    list->lkey = wr->memList.lkey;

    liteWr->sg_list = list;
    liteWr->wr_id = wr->wrId;
    liteWr->num_sge = 1; /* only support one sge */
}

int RaHdcLiteRecvWrlist(struct RaQpHandle *qpHdc, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    struct rdma_lite_recv_wr *liteWr = NULL;
    struct rdma_lite_recv_wr *badWr = NULL;
    struct rdma_lite_sge *list = NULL;
    unsigned int index;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(recvNum == 0, hccp_err("lite recv_num[%u] is invalid!", recvNum), -EINVAL);

    liteWr = (struct rdma_lite_recv_wr *)calloc(recvNum, sizeof(struct rdma_lite_recv_wr));
    CHK_PRT_RETURN(liteWr == NULL, hccp_err("lite calloc lite_wr failed!"), -ENOSPC);

    list = (struct rdma_lite_sge *)calloc(recvNum, sizeof(struct rdma_lite_sge));
    if (list == NULL) {
        hccp_err("lite calloc list failed!");
        ret = -ENOSPC;
        goto alloc_sge_fail;
    }

    // build up recv lite wr
    for (i = 0; i < recvNum; i++) {
        RaHdcLiteBuildRecvWr(&wr[i], &list[i], &liteWr[i]);
        index = i + 1;
        liteWr[i].next = (i < recvNum - 1) ? &(liteWr[index]) : NULL;
    }

    ret = RaRdmaLitePostRecv(qpHdc->liteQp, liteWr, &badWr);
    if (ret == 0) {
        *completeNum = recvNum;
    } else if (ret == -ENOMEM) {
        *completeNum = (unsigned int)((void *)badWr - (void *)liteWr) / sizeof(struct rdma_lite_recv_wr);
        hccp_dbg("ra_rdma_lite_post_recv wqe overflow, completeNum[%d]", *completeNum);
    } else {
        *completeNum = 0;
        hccp_err("ra_rdma_lite_post_recv failed, ret[%d]", ret);
    }

    qpHdc->recvWrNum += *completeNum;

    free(list);
    list = NULL;

alloc_sge_fail:
    free(liteWr);
    liteWr = NULL;
    return (ret == -ENOMEM) ? 0 : ret;
}
