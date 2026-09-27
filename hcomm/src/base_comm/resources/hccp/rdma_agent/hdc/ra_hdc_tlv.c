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
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "hccp_tlv.h"
#include "ra_hdc_tlv.h"

int RaHdcTlvInit(struct RaTlvHandle *tlvHandle)
{
    unsigned int phyId = tlvHandle->initInfo.phyId;
    unsigned int opCode = RA_RS_TLV_INIT_V1;
    union OpTlvInitData tlvData = {0};
    unsigned int interfaceVersion = 0;
    int ret = 0;

    tlvData.txData.phyId = phyId;

    ret = RaHdcGetInterfaceVersion(phyId, RA_RS_TLV_INIT, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_OPCODE_BASE_VERSION) {
        opCode = RA_RS_TLV_INIT;
    } else {
        ret = -ENOTSUPP;
        hccp_warn_init("[init][ra_hdc_tlv]ra tlv init version not support, phy_id(%u)", phyId);
        return ret;
    }

    ret = RaHdcProcessMsg(opCode, phyId, (char *)&tlvData, sizeof(union OpTlvInitData));
    CHK_PRT_RETURN(ret == -ENOTSUPP,
        hccp_warn_init("[init][ra_hdc_tlv]ra hdc message process unsuccessful ret(%d) phy_id(%u)", ret, phyId), ret);
    CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_hdc_tlv]ra hdc message process failed ret(%d) phy_id(%u)", ret, phyId),
        ret);

    tlvHandle->bufferSize = tlvData.rxData.bufferSize;
    return ret;
}

int RaHdcTlvDeinit(struct RaTlvHandle *tlvHandle)
{
    unsigned int phyId = tlvHandle->initInfo.phyId;
    union OpTlvDeinitData tlvData = {0};
    int ret;

    tlvData.txData.phyId = phyId;

    ret = RaHdcProcessMsg(RA_RS_TLV_DEINIT, phyId, (char *)&tlvData, sizeof(union OpTlvDeinitData));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[deinit][ra_hdc_tlv]ra hdc message process failed ret(%d) phy_id(%u)", ret, phyId), ret);

    return 0;
}

STATIC void RaHdcTlvRequestHeadInit(struct RaTlvHandle *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg,
    struct TlvRequestMsgHead *head)
{
    head->moduleType = moduleType;
    head->phyId = tlvHandle->initInfo.phyId;
    head->totalBytes = sendMsg->length;
    head->type = sendMsg->type;
    head->offset = 0;
}

STATIC int RaHdTlvRequestForSendNullMsg(unsigned int phyId, union OpTlvRequestDataCommon opTlvCommon,
    struct TlvRequestMsgHead *head, struct TlvMsg *recvMsg)
{
    void *tlvData = opTlvCommon.txData.tlvData;
    int ret = 0;

    (void)memcpy_s(opTlvCommon.txData.head, sizeof(struct TlvRequestMsgHead), head, sizeof(struct TlvRequestMsgHead));

    ret = RaHdcProcessMsg(opTlvCommon.txData.opcode, phyId, tlvData, opTlvCommon.txData.size);
    CHK_PRT_RETURN(ret == -EUSERS || ret == -ENOTSUPP,
        hccp_warn_others("[request][ra_hdc_tlv]hdc message process unsuccessful ret(%d) phy_id(%u)", ret, phyId), ret);
    CHK_PRT_RETURN(ret != 0, hccp_err("[request][ra_hdc_tlv]hdc message process failed ret(%d) phy_id(%u)", ret, phyId),
        ret);

    TLV_SETUP_RX(opTlvCommon.txData.opcode == RA_RS_TLV_REQUEST, tlvData, opTlvCommon);
    recvMsg->type = head->type;
    recvMsg->length = *opTlvCommon.rxData.recvBytes;
    return ret;
}

STATIC int RaTlvRequestGetTlvMsg(struct TlvRequestMsgHead *head, union OpTlvRequestDataCommon opTlvCommon,
    struct TlvMsg *recvMsg)
{
    void *tlvData = opTlvCommon.txData.tlvData;
    int ret = 0;

    TLV_SETUP_RX(opTlvCommon.txData.opcode == RA_RS_TLV_REQUEST, tlvData, opTlvCommon);

    if (recvMsg->data == NULL || recvMsg->length == 0) {
        *opTlvCommon.rxData.recvBytes = 0;
        goto out;
    }

    CHK_PRT_RETURN(*opTlvCommon.rxData.recvBytes > recvMsg->length,
        hccp_err("[request][ra_hdc_tlv]rxData.recvBytes(%u) > recvLen(%u), phyId(%u)", *opTlvCommon.rxData.recvBytes,
            recvMsg->length, head->phyId),
        -EINVAL);

    ret = memcpy_s(recvMsg->data, recvMsg->length, opTlvCommon.rxData.recvData, *opTlvCommon.rxData.recvBytes);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[request][ra_hdc_tlv]memcpy_s recvData failed, ret(%d) rxData.recvBytes(%u)"
                 " recvLen(%u) phyId(%u)",
            ret, *opTlvCommon.rxData.recvBytes, recvMsg->length, head->phyId),
        -ESAFEFUNC);

out:
    recvMsg->type = head->type;
    recvMsg->length = *opTlvCommon.rxData.recvBytes;
    return ret;
}

int RaHdcTlvRequest(struct RaTlvHandle *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg,
    struct TlvMsg *recvMsg)
{
    unsigned int phyId = tlvHandle->initInfo.phyId;
    union OpTlvRequestDataCommon opTlvCommon = {0};
    struct TlvRequestMsgHead head = {0};
    unsigned int interfaceVersion;
    void *tlvData;
    int ret = 0;

    RaHdcGetInterfaceVersion(phyId, RA_RS_TLV_REQUEST_V2, &interfaceVersion);
    tlvData = calloc(1, (interfaceVersion == 0) ? sizeof(union OpTlvRequestData) : sizeof(union OpTlvRequestDataV2));
    CHK_PRT_RETURN(tlvData == NULL, hccp_err("[ra_hdc_tlv_request]calloc op tlv request data failed! phyId(%u)", phyId),
        -ENOMEM);
    TLV_SETUP_TX(interfaceVersion == 0, tlvData, opTlvCommon);
    RaHdcTlvRequestHeadInit(tlvHandle, moduleType, sendMsg, &head);
    if (sendMsg->length == 0) {
        ret = RaHdTlvRequestForSendNullMsg(phyId, opTlvCommon, &head, recvMsg);
        goto out;
    }

    while (head.offset < sendMsg->length) {
        head.sendBytes = (head.totalBytes - head.offset) >= opTlvCommon.txData.dataLength
                             ? opTlvCommon.txData.dataLength
                             : (head.totalBytes - head.offset);
        (void)memcpy_s(opTlvCommon.txData.head, sizeof(struct TlvRequestMsgHead), &head,
            sizeof(struct TlvRequestMsgHead));

        (void)memset_s(opTlvCommon.txData.data, opTlvCommon.txData.dataLength, 0, opTlvCommon.txData.dataLength);
        ret = memcpy_s(opTlvCommon.txData.data, opTlvCommon.txData.dataLength, (sendMsg->data + head.offset),
            head.sendBytes);
        if (ret != 0) {
            hccp_err("[request][ra_hdc_tlv]memcpy_s data failed ret(%d) phy_id(%u) send_bytes(%u)", ret, phyId,
                head.sendBytes);
            ret = -ESAFEFUNC;
            goto out;
        }

        ret = RaHdcProcessMsg(opTlvCommon.txData.opcode, phyId, opTlvCommon.txData.tlvData, opTlvCommon.txData.size);
        if (ret == -EUSERS || ret == -ENOTSUPP) {
            hccp_warn_others("[request][ra_hdc_tlv]hdc message process unsuccessful ret(%d) phy_id(%u)", ret, phyId);
            ret = -ESAFEFUNC;
            goto out;
        } else if (ret != 0) {
            hccp_err("[request][ra_hdc_tlv]hdc message process failed ret(%d) phy_id(%u)", ret, phyId);
            goto out;
        }
        head.offset += head.sendBytes;
    }

    ret = RaTlvRequestGetTlvMsg(&head, opTlvCommon, recvMsg);
    if (ret != 0) {
        hccp_err("[request][ra_hdc_tlv]RaTlvRequestGetTlvMsg failed ret(%d) phy_id(%u)", ret, phyId);
        goto out;
    }
out:
    free(tlvData);
    tlvData = NULL;
    return ret;
}
