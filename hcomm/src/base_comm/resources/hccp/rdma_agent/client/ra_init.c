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
#include <dlfcn.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "hccp.h"
#include "hccp_common.h"
#include "ra.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_hdc.h"
#include "ra_peer.h"
#include "ra_hdc_async.h"
#include "ra_init.h"
#include "config_log.h"

static unsigned int gSendWrNum = 0;
static void *gRaRdevHandle[RA_MAX_PHY_ID_NUM] = {0};
static RaInstance gRefInstances[RA_MAX_INSTANCES] = {{0, PTHREAD_MUTEX_INITIALIZER}};

HCCP_ATTRI_VISI_DEF int RaIsFirstUsed(int insId)
{
    int isFirst = 0;

    CHK_PRT_RETURN(insId < 0 || insId >= RA_MAX_INSTANCES,
        hccp_err("[ra]ins_id(%d) must be in [0, %u)", insId, RA_MAX_INSTANCES), -EINVAL);

    pthread_mutex_lock(&gRefInstances[insId].mutex);
    if (gRefInstances[insId].refCount == 0) {
        isFirst++;
        hccp_run_info("[ra]ins_id(%d) is first used", insId);
    }

    gRefInstances[insId].refCount++;
    hccp_info_others("[ra]ins_id[%d] is %d", insId, gRefInstances[insId].refCount);
    pthread_mutex_unlock(&gRefInstances[insId].mutex);

    return isFirst;
}

HCCP_ATTRI_VISI_DEF int RaIsLastUsed(int insId)
{
    int isLast = 0;

    CHK_PRT_RETURN(insId < 0 || insId >= RA_MAX_INSTANCES,
        hccp_err("[ra]ins_id(%d) must be in [0, %u)", insId, RA_MAX_INSTANCES), -EINVAL);

    pthread_mutex_lock(&gRefInstances[insId].mutex);
    if (gRefInstances[insId].refCount == 0) {
        hccp_err("[ra]ins_id %d has not been used", insId);
        pthread_mutex_unlock(&gRefInstances[insId].mutex);
        return -EINVAL;
    }

    if (gRefInstances[insId].refCount == 1) {
        isLast++;
        hccp_run_info("[ra]ins_id(%d) is last used", insId);
    }

    hccp_info_others("[ra]ins_id[%d] is %d", insId, gRefInstances[insId].refCount);
    gRefInstances[insId].refCount--;
    pthread_mutex_unlock(&gRefInstances[insId].mutex);

    return isLast;
}

HCCP_ATTRI_VISI_DEF int RaRdevGetHandle(unsigned int phyId, void **rdmaHandle)
{
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_rdev]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM), -EINVAL);
    CHK_PRT_RETURN(rdmaHandle == NULL, hccp_err("[get][ra_rdev]rdma_handle is NULL, phyId(%u)", phyId), -EINVAL);
    CHK_PRT_RETURN(gRaRdevHandle[phyId] == NULL, hccp_run_info("[get][ra_rdev]handle is NULL, phyId(%u)", phyId),
        -ENODEV);

    *rdmaHandle = gRaRdevHandle[phyId];
    return 0;
}

void RaRdevSetHandle(unsigned int phyId, void *rdmaHandle)
{
    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_warn_rma("[set][ra_rdev]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM);
        return;
    }

    gRaRdevHandle[phyId] = rdmaHandle;
}

void RaRdevIncSendWrNum(void)
{
    gSendWrNum++;
}

STATIC int RaInitHdc(struct RaInitConfig *config)
{
    struct ProcessRaSign pRaSign = {0};
    unsigned int phyId = config->phyId;
    struct process_sign psign = {0};
    int ret;

    ret = DlDrvGetProcessSign(&psign);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]Get process sign failed, ret(%d) phyId(%u)", ret, phyId), ret);

    pRaSign.tgid = psign.tgid;
    ret = strcpy_s(pRaSign.sign, PROCESS_RA_SIGN_LENGTH, psign.sign);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]Invalid pid sign, ret(%d) phyId(%u)", ret, phyId), -ESAFEFUNC);

    ret = RaHdcInit(config, pRaSign);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]ra hdc init failed, ret(%d) phyId(%u)", ret, phyId), ret);

    RaHdcGetAllOpcodeVersion(phyId);

    // init async hdc session via sync hdc session
    ret = RaHdcInitAsync(config);
    if (ret != 0) {
        hccp_err("[init][ra]ra_hdc_init_async failed, ret(%d) phyId(%u)", ret, phyId);
        (void)RaHdcDeinit(config);
    }

    return ret;
}

STATIC int RaInitPeer(struct RaInitConfig *config)
{
    unsigned int phyId = config->phyId;
    unsigned int whiteListSwitch = 0;
    int ret;

    ret = RaSocketGetWhiteListStatus(&whiteListSwitch);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]get white_list_status failed, ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RaPeerInit(config, whiteListSwitch);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]ra_peer_init failed, ret(%d) phyId(%u)", ret, phyId), ret);

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaInit(struct RaInitConfig *config)
{
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(config == NULL, hccp_err("[init][ra]config is NULL"), ConverReturnCode(HCCP_INIT, -EINVAL));

    phyId = config->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[init][ra]phyId(%u) is invalid! it must be greater than or "
                 "equal to 0 and less than %d!",
            phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    if (config->hdcType != HDC_SERVICE_TYPE_RDMA && config->hdcType != HDC_SERVICE_TYPE_RDMA_V2) {
        hccp_warn_init("[init][ra]hdc_type(%d) is invalid, set it to default hdcType(%d)", config->hdcType,
            HDC_SERVICE_TYPE_RDMA);
        config->hdcType = HDC_SERVICE_TYPE_RDMA;
    }

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%u] hdcType:[%d] enableHdcAsync[%d]", phyId,
        config->nicPosition, config->hdcType, config->enableHdcAsync);
    ret = DlHalInit();
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra]dl_hal_init failed, ret(%d) phyId(%u)", ret, phyId), ret);

    if (config->nicPosition == NETWORK_OFFLINE) {
        ret = RaInitHdc(config);
        if (ret != 0) {
            hccp_err("[init][ra]ra_init_hdc failed, ret(%d) phyId(%u)", ret, phyId);
            goto err;
        }
    } else if (config->nicPosition == NETWORK_PEER_ONLINE) {
        ret = RaInitPeer(config);
        if (ret != 0) {
            hccp_err("[init][ra]ra_init_peer failed, ret(%d) phyId(%u)", ret, phyId);
            goto err;
        }
    } else {
        hccp_err("[init][ra]do not support nic_position(%u) phyId(%u)", config->nicPosition, phyId);
        ret = -EPROTONOSUPPORT;
        goto err;
    }

    return 0;

err:
    DlHalDeinit();
    return ConverReturnCode(HCCP_INIT, ret);
}

STATIC int RaDeinitHdc(struct RaInitConfig *config)
{
    int ret;

    ret = RaHdcDeinitAsync(config->phyId);
    CHK_PRT_RETURN(ret != 0 && ret != -ENODEV, hccp_err("[deinit][ra]ra_hdc_deinit_async failed, ret(%d)", ret), ret);

    ret = RaHdcDeinit(config);
    CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra]ra hdc deinit failed, ret(%d), gSendWrNum(%u)", ret, gSendWrNum),
        ret);
    return ret;
}

HCCP_ATTRI_VISI_DEF int RaDeinit(struct RaInitConfig *config)
{
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(config == NULL, hccp_err("[deinit][ra]config is NULL, invalid"),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    phyId = config->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[deinit][ra]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%u]", phyId, config->nicPosition);

    if (config->nicPosition == NETWORK_OFFLINE) {
        ret = RaDeinitHdc(config);
        CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra]ra_deinit_hdc failed, ret(%d) phyId(%u)", ret, phyId),
            ConverReturnCode(HCCP_INIT, ret));
    } else if (config->nicPosition == NETWORK_PEER_ONLINE) {
        ret = RaPeerDeinit(config);
        CHK_PRT_RETURN(ret == -EAGAIN,
            hccp_warn_init("[deinit][ra]ra_peer_deinit unsuccessful, ret(%d) phyId(%u)", ret, phyId),
            ConverReturnCode(HCCP_INIT, ret));
        CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra]ra_peer_deinit failed, ret(%d) phyId(%u)", ret, phyId),
            ConverReturnCode(HCCP_INIT, ret));
    } else {
        hccp_err("[deinit][ra]do not support nic_position(%u) phyId(%u)", config->nicPosition, phyId);
        return ConverReturnCode(HCCP_INIT, -EPROTONOSUPPORT);
    }

    RaSocketSetWhiteListStatus(WHITE_LIST_DISABLE);
    gSendWrNum = 0;
    DlHalDeinit();
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSetDebugConfig(struct RaInfo *info, uint64_t debugConfig)
{
    CHK_PRT_RETURN(info == NULL, hccp_err("[set][debug_config]info is NULL"), ConverReturnCode(HCCP_INIT, -EINVAL));

    CHK_PRT_RETURN(info->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[set][debug_config]phyId(%u) must be smaller than %u", info->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%d], debugConfig[0x%llx]", info->phyId, info->mode,
        (unsigned long long)debugConfig);

    HccpSetDebugConfig(debugConfig);
    if (info->mode == NETWORK_PEER_ONLINE) {
        RaPeerSetDebugConfig(debugConfig);
        return 0;
    }

    if (info->mode == NETWORK_OFFLINE) {
        return ConverReturnCode(HCCP_INIT, RaHdcSetDebugConfig(info->phyId, debugConfig));
    }

    hccp_err("[set][debug_config]do not support mode(%u) phyId(%u)", info->mode, info->phyId);
    return ConverReturnCode(HCCP_INIT, -EPROTONOSUPPORT);
}

STATIC void __attribute__((destructor)) RaUninitAsync(void)
{
    RaHdcDeinitAsyncAll();
}
