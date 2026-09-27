/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "ra_rs_err.h"
#include "ra_hdc_ping.h"

int RaHdcPingInit(struct RaPingHandle *pingHandle, struct PingInitAttr *initAttr, struct PingInitInfo *initInfo)
{
    unsigned int phyId = pingHandle->phyId;
    union OpPingInitData pingData = {0};
    int ret;

    ret = memcpy_s(&(pingData.txData.attr), sizeof(struct PingInitAttr), initAttr, sizeof(struct PingInitAttr));
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_ping]memcpy_s init_attr failed, ret(%d) phyId(%u)", ret, phyId),
        -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_PING_INIT, phyId, (char *)&pingData, sizeof(union OpPingInitData));
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    ret = memcpy_s(initInfo, sizeof(struct PingInitInfo), &(pingData.rxData.info), sizeof(struct PingInitInfo));
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_ping]memcpy_s init_info failed, ret(%d) phyId(%u)", ret, phyId),
        -ESAFEFUNC);
    ret = memcpy_s(&(pingHandle->dev), sizeof(union PingDev), &(initAttr->dev), sizeof(union PingDev));
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_hdc_ping]memcpy_s dev info failed, ret(%d) phyId(%u)", ret, phyId),
        -ESAFEFUNC);
    pingHandle->devIndex = pingData.rxData.devIndex;

    return 0;
}

STATIC void RaHdcPingInitRdev(struct RaRsDevInfo *rdev, unsigned int phyId, unsigned int devIndex)
{
    rdev->phyId = phyId;
    rdev->devIndex = devIndex;
}

int RaHdcPingTargetAdd(struct RaPingHandle *pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    unsigned int phyId = pingHandle->phyId;
    union OpPingAddData pingData;
    unsigned int i;
    int ret;

    for (i = 0; i < num; i++) {
        if (pingHandle->protocol == PROTOCOL_RDMA) {
            CHK_PRT_RETURN(target[i].localInfo.rdma.udpSport > MAX_PORT_NUM,
                hccp_err("[add][ra_hdc_ping]udp_sport(%u) invalid, i(%u), phyId(%u)", target[i].localInfo.rdma.udpSport,
                    i, phyId),
                -EINVAL);
        }

        (void)memset_s(&pingData, sizeof(pingData), 0, sizeof(pingData));
        RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);
        ret = memcpy_s(&(pingData.txData.target), sizeof(struct PingTargetInfo), &(target[i]),
            sizeof(struct PingTargetInfo));
        CHK_PRT_RETURN(ret,
            hccp_err("[add][ra_hdc_ping]memcpy_s target failed, ret(%d) i(%u) phyId(%u)", ret, i, phyId), -ESAFEFUNC);
        ret = RaHdcProcessMsg(RA_RS_PING_ADD, phyId, (char *)&pingData, sizeof(union OpPingAddData));
        CHK_PRT_RETURN(ret,
            hccp_err("[add][ra_hdc_ping]ra hdc message process failed ret(%d) i(%u) phyId(%u)", ret, i, phyId), ret);
    }

    return 0;
}

int RaHdcPingTaskStart(struct RaPingHandle *pingHandle, struct PingTaskAttr *attr)
{
    union OpPingStartData pingData = {0};
    unsigned int phyId = pingHandle->phyId;
    int ret;

    RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);
    ret = memcpy_s(&(pingData.txData.attr), sizeof(struct PingTaskAttr), attr, sizeof(struct PingTaskAttr));
    CHK_PRT_RETURN(ret, hccp_err("[start][ra_hdc_ping]memcpy_s attr failed, ret(%d), phyId(%u)", ret, phyId),
        -ESAFEFUNC);

    ret = RaHdcProcessMsg(RA_RS_PING_START, phyId, (char *)&pingData, sizeof(union OpPingStartData));
    CHK_PRT_RETURN(ret, hccp_err("[start][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);
    return 0;
}

int RaHdcPingGetResults(struct RaPingHandle *pingHandle, struct PingTargetResult target[], uint32_t *num)
{
    unsigned int phyId = pingHandle->phyId;
    union OpPingResultsData pingData;
    unsigned int totalNum = *num;
    unsigned int completeCnt = 0;
    unsigned int sendNum = 0;
    unsigned int i = 0;
    unsigned int j = 0;
    int ret = 0;

    while (completeCnt < totalNum) {
        (void)memset_s(&pingData, sizeof(pingData), 0, sizeof(pingData));
        RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);
        sendNum = ((totalNum - completeCnt) >= RA_MAX_PING_TARGET_NUM) ? RA_MAX_PING_TARGET_NUM
                                                                       : (totalNum - completeCnt);

        // prepare tx data target
        for (i = 0; i < sendNum; i++) {
            j = i + completeCnt;
            ret = memcpy_s(&(pingData.txData.target[i]), sizeof(struct PingTargetCommInfo), &(target[j].remoteInfo),
                sizeof(struct PingTargetCommInfo));
            if (ret) {
                hccp_err("[get][ra_hdc_ping]memcpy_s remote_info failed, ret(%d), i(%u), j(%u), phyId(%u)", ret, i, j,
                    phyId);
                goto out;
            }
        }
        pingData.txData.num = sendNum;

        ret = RaHdcProcessMsg(RA_RS_PING_GET_RESULTS, phyId, (char *)&pingData, sizeof(union OpPingResultsData));
        // caller needs to retry, degrade log level
        if (ret == -EAGAIN) {
            hccp_warn_rma("[get][ra_hdc_ping]ra hdc message process unsuccessful, ret(%d) phyId(%u)", ret, phyId);
            goto out;
        }

        if (pingData.rxData.num > sendNum) {
            hccp_err("[get][ra_hdc_ping]rx_data.num[%u] is larger than send_num[%u], ret(%d) phyId(%u)",
                pingData.rxData.num, sendNum, ret, phyId);
            ret = -EINVAL;
            goto out;
        }

        // prepare rx data target
        for (i = 0; i < pingData.rxData.num; i++) {
            j = i + completeCnt;
            ret = memcpy_s(&(target[j].result), sizeof(struct PingResultInfo), &(pingData.rxData.target[i]),
                sizeof(struct PingResultInfo));
            if (ret) {
                hccp_err("[get][ra_hdc_ping]memcpy_s result failed, ret(%d), i(%u), j(%u), phyId(%u)", ret, i, j,
                    phyId);
                ret = -ESAFEFUNC;
                goto out;
            }
        }

        completeCnt += pingData.rxData.num;
        if (ret) {
            hccp_err("[get][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId);
            goto out;
        }
    }

out:
    *num = completeCnt;

    return ret;
}

int RaHdcPingTargetDel(struct RaPingHandle *pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    unsigned int phyId = pingHandle->phyId;
    union OpPingDelData pingData;
    unsigned int completeCnt = 0;
    unsigned int sendNum = 0;
    unsigned int i = 0;
    unsigned int j = 0;
    int ret;

    while (completeCnt < num) {
        (void)memset_s(&pingData, sizeof(pingData), 0, sizeof(pingData));
        RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);
        sendNum = ((num - completeCnt) >= RA_MAX_PING_TARGET_NUM) ? RA_MAX_PING_TARGET_NUM : (num - completeCnt);

        // prepare tx data target
        for (i = 0; i < sendNum; i++) {
            j = i + completeCnt;
            ret = memcpy_s(&(pingData.txData.target[i]), sizeof(struct PingTargetCommInfo), &(target[j]),
                sizeof(struct PingTargetCommInfo));
            CHK_PRT_RETURN(ret,
                hccp_err("[del][ra_hdc_ping]memcpy_s target failed, ret(%d), i(%u) j(%u), phyId(%u)", ret, i, j, phyId),
                -ESAFEFUNC);
        }
        pingData.txData.num = sendNum;

        ret = RaHdcProcessMsg(RA_RS_PING_DEL, phyId, (char *)&pingData, sizeof(union OpPingDelData));
        CHK_PRT_RETURN(ret, hccp_err("[del][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
            ret);
        completeCnt += sendNum;
    }

    return 0;
}

int RaHdcPingTaskStop(struct RaPingHandle *pingHandle)
{
    union OpPingStopData pingData = {0};
    unsigned int phyId = pingHandle->phyId;
    int ret;

    RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);

    ret = RaHdcProcessMsg(RA_RS_PING_STOP, phyId, (char *)&pingData, sizeof(union OpPingStopData));
    CHK_PRT_RETURN(ret, hccp_err("[stop][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    return 0;
}

int RaHdcPingDeinit(struct RaPingHandle *pingHandle)
{
    union OpPingDeinitData pingData = {0};
    unsigned int phyId = pingHandle->phyId;
    int ret;

    RaHdcPingInitRdev(&pingData.txData.rdev, phyId, pingHandle->devIndex);

    ret = RaHdcProcessMsg(RA_RS_PING_DEINIT, phyId, (char *)&pingData, sizeof(union OpPingDeinitData));
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_hdc_ping]ra hdc message process failed ret(%d) phyId(%u)", ret, phyId),
        ret);

    return 0;
}
