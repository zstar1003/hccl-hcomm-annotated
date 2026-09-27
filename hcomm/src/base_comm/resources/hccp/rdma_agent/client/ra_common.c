/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdbool.h>
#include <errno.h>
#include "user_log.h"
#include "config_log.h"
#include "hccp.h"
#include "hccp_common.h"
#include "ra_client_host.h"
#include "ra.h"
#include "ra_peer.h"
#include "ra_hdc.h"
#include "ra_hdc_rdma.h"
#include "ra_rs_err.h"
#include "ra_rs_comm.h"

/* rdma ops for ra_restore_snapshot: The use of RDMA-lite related interfaces is prohibited. */
static struct RaRdmaOps gRaRestoreRdmaOps = {
    .raRdevDeinit = RaHdcRdevRestoreDeinit,
    .raQpDestroy = RaHdcQpDestroy,
    .raDeregisterMr = RaHdcTypicalMrDereg,
};

HCCP_ATTRI_VISI_DEF int RaGetTlsEnable(struct RaInfo *info, bool *tlsEnable)
{
    int ret;

    CHK_PRT_RETURN(info == NULL || tlsEnable == NULL, hccp_err("[get][tls_enable]info or tls_enable is NULL"),
        ConverReturnCode(OTHERS, -EINVAL));
    CHK_PRT_RETURN(info->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][tls_enable]phyId(%u) must be smaller than %u", info->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%d]", info->phyId, info->mode);
    if (info->mode == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetTlsEnable(info->phyId, tlsEnable);
    } else if (info->mode == NETWORK_OFFLINE) {
        ret = RaHdcGetTlsEnable(info->phyId, tlsEnable);
    } else {
        hccp_err("[get][tls_enable]do not support mode(%d) phyId(%u)", info->mode, info->phyId);
        return ConverReturnCode(OTHERS, -ENOTSUPP);
    }
    return ConverReturnCode(OTHERS, ret);
}

HCCP_ATTRI_VISI_DEF int RaSaveSnapshot(struct RaInfo *info, enum SaveSnapshotAction action)
{
    struct RaRdmaHandle *rdmaHandle = NULL;
    int ret;

    CHK_PRT_RETURN(info == NULL, hccp_err("[save][snapshot]info is NULL"), ConverReturnCode(OTHERS, -EINVAL));
    CHK_PRT_RETURN(action < SAVE_SNAPSHOT_ACTION_PRE_PROCESSING || action >= SAVE_SNAPSHOT_ACTION_MAX,
        hccp_err("[save][snapshot]invalid action(%d)", action), ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%d], action:[%d]", info->phyId, info->mode, action);
    if (info->mode == NETWORK_PEER_ONLINE) {
        return 0;
    } else if (info->mode == NETWORK_OFFLINE) {
        ret = RaRdevGetHandle(info->phyId, (void **)&rdmaHandle);
        CHK_PRT_RETURN(ret != 0 && ret != -ENODEV, hccp_err("[save][snapshot]ra_rdev_get_handle failed, ret[%d]", ret),
            ConverReturnCode(OTHERS, ret));
        ret = RaHdcRdmaSaveSnapshot(rdmaHandle, action);
        CHK_PRT_RETURN(ret != 0, hccp_err("[save][snapshot]ra_hdc_rdma_save_snapshot failed, ret[%d]", ret),
            ConverReturnCode(OTHERS, ret));

        ret = RaHdcSaveSnapshot(info->phyId, action);
        CHK_PRT_RETURN(ret != 0, hccp_err("[save][snapshot]ra_hdc_save_snapshot failed, ret[%d]", ret),
            ConverReturnCode(OTHERS, ret));
    } else {
        hccp_err("[save][snapshot]do not support mode[%d] phyId[%u]", info->mode, info->phyId);
        return ConverReturnCode(OTHERS, -ENOTSUPP);
    }

    return ConverReturnCode(OTHERS, ret);
}

HCCP_ATTRI_VISI_DEF int RaRestoreSnapshot(struct RaInfo *info)
{
    struct RaRdmaHandle *rdmaHandle = NULL;
    int ret = 0;

    CHK_PRT_RETURN(info == NULL, hccp_err("[restore][snapshot]info is NULL"), ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%d]", info->phyId, info->mode);
    if (info->mode == NETWORK_PEER_ONLINE) {
        return 0;
    } else if (info->mode == NETWORK_OFFLINE) {
        ret = RaRdevGetHandle(info->phyId, (void **)&rdmaHandle);
        CHK_PRT_RETURN(ret != 0 && ret != -ENODEV,
            hccp_err("[restore][snapshot]ra_rdev_get_handle failed, ret[%d]", ret), ConverReturnCode(OTHERS, ret));
        ret = RaHdcRdmaRestoreSnapshot(rdmaHandle, &gRaRestoreRdmaOps);
        CHK_PRT_RETURN(ret != 0, hccp_err("[restore][snapshot]ra_hdc_rdma_restore_snapshot failed, ret[%d]", ret),
            ConverReturnCode(OTHERS, ret));

        ret = RaHdcRestoreSnapshot(info->phyId);
        CHK_PRT_RETURN(ret != 0, hccp_err("[restore][snapshot]ra_hdc_restore_snapshot failed, ret[%d]", ret),
            ConverReturnCode(OTHERS, ret));
    } else {
        hccp_err("[restore][snapshot]do not support mode[%d] phyId[%u]", info->mode, info->phyId);
        return ConverReturnCode(OTHERS, -ENOTSUPP);
    }

    return ConverReturnCode(OTHERS, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    int ret;

    CHK_PRT_RETURN(info == NULL || value == NULL || valueLen == NULL,
        hccp_err("[get][hccn_cfg]info or value or value_len is NULL"), ConverReturnCode(OTHERS, -EINVAL));
    CHK_PRT_RETURN(*valueLen < HCCN_CFG_MSG_DATA_LEN,
        hccp_err("[get][hccn_cfg] failed, valueLen[%d] < min_len[%d]", *valueLen, HCCN_CFG_MSG_DATA_LEN),
        ConverReturnCode(OTHERS, -EINVAL));
    CHK_PRT_RETURN(info->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][hccn_cfg]phyId(%u) must be smaller than %u", info->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%d], hccn_cfg_key[%d]", info->phyId, info->mode, key);
    if (info->mode == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetHccnCfg(info, key, value, valueLen);
    } else if (info->mode == NETWORK_OFFLINE) {
        ret = RaHdcGetHccnCfg(info->phyId, key, value, valueLen);
    } else {
        hccp_err("[get][hccn_cfg]do not support mode(%d) phyId(%u)", info->mode, info->phyId);
        return ConverReturnCode(OTHERS, -ENOTSUPP);
    }
    if (ret != 0) {
        hccp_err("[get][hccn_cfg] failed, mode[%d], phyId[%u], ret[%d]", info->mode, info->phyId, ret);
    }

    return ConverReturnCode(OTHERS, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetSecRandom(struct RaInfo *info, u32 *value)
{
    int ret;

    CHK_PRT_RETURN(info == NULL || value == NULL, hccp_err("[get][sec_random]info or value is NULL"),
        ConverReturnCode(OTHERS, -EINVAL));
    hccp_run_info("Input parameters: phy_id[%u], nic_position:[%d]", info->phyId, info->mode);

    ret = RaPeerGetSecRandom(value);
    if (ret != 0 && info->mode == NETWORK_OFFLINE) {
        CHK_PRT_RETURN(info->phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[get][sec_random]phy_id(%u) must be smaller than %u", info->phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(OTHERS, -EINVAL));
        ret = RaHdcGetSecRandom(info->phyId, value);
    } else if (ret != 0 && info->mode != NETWORK_OFFLINE) {
        hccp_err("[get][sec_random] failed, mode[%u], ret[%d]", info->mode, ret);
    }

    return ConverReturnCode(OTHERS, ret);
}

HCCP_ATTRI_VISI_DEF bool RaHasCapability(struct RaInfo *info, unsigned int capability)
{
    CHK_PRT_RETURN(info == NULL, hccp_warn_rma("info is NULL"), false);
    CHK_PRT_RETURN(info->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_warn_rma("phy_id(%u) must be smaller than %u", info->phyId, RA_MAX_PHY_ID_NUM), false);
    CHK_PRT_RETURN(info->mode != NETWORK_OFFLINE, hccp_warn_rma("mode:%u not support", info->mode), false);

    return RaHdcHasCapability(info->phyId, capability);
}
