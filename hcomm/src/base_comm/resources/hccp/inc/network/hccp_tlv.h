/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_TLV_H
#define HCCP_TLV_H

#include "hccp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

enum TlvModuleType {
    TLV_MODULE_TYPE_NSLB,
    TLV_MODULE_TYPE_CCU,
    TLV_MODULE_TYPE_MAX,
};

enum TlvCcuMsgType {
    MSG_TYPE_CCU_INIT = 0,
    MSG_TYPE_CCU_UNINIT,
    MSG_TYPE_CCU_GET_MEM_INFO,
    MSG_TYPE_CCU_DISPATCH_CMD,
    MSG_TYPE_CCU_MAX,
};

struct TlvInitInfo {
    int version;
    unsigned int phyId;
    unsigned int nicPosition;
    unsigned int reserved[16U];
};

struct TlvMsg {
    unsigned int type;
    unsigned int length;
    char *data;
};

#define CUSTOM_CHAN_DATA_MAX_SIZE 2048

struct CustomChanInfoIn {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetStart;
    unsigned int op;
};

struct CustomChanInfoOut {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetNext;
    int opRet;
};

struct CcuMemReq {
    unsigned int udieIdx;
    unsigned int reserved;
    unsigned long long memTypeBitmap;
};

/**
 * @ingroup libinit
 * @brief Rdma_agent tlv initialization
 * @param init_info [IN] tlv init info
 * @param buffer_size [OUT] tlv buffer size
 * @param tlv_handle [OUT] tlv handle info
 * @see ra_tlv_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTlvInit(struct TlvInitInfo *initInfo, unsigned int *bufferSize, void **tlvHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent tlv deinitialization
 * @param tlv_handle [IN] tlv handle info
 * @see ra_tlv_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTlvDeinit(void *tlvHandle);

/**
 * @ingroup libcommon
 * @brief Rdma_agent tlv request process
 * @param tlv_handle [IN] tlv handle info
 * @param module_type [IN] tlv module type
 * @param send_msg [IN] tlv message to send
 * @param recv_msg [OUT] tlv message to receive
 * @see ra_tlv_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTlvRequest(void *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg,
    struct TlvMsg *recvMsg);

#ifdef __cplusplus
}
#endif
#endif // HCCP_TLV_H