/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_CTX_DATA_PLANE_H
#define HCCP_CTX_DATA_PLANE_H

#include "hccp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct WrSgeList {
    uint64_t addr;
    uint32_t len;
    void *lmemHandle;
};

enum RaUbOpcode {
    RA_UB_OPC_WRITE = 0x00,
    RA_UB_OPC_WRITE_NOTIFY = 0x02,
    RA_UB_OPC_READ = 0x10,
    RA_UB_OPC_NOP = 0x51,
    RA_UB_OPC_LAST
};

union JfsWrFlag {
    struct {
        uint32_t placeOrder : 2;      /* 0: There is no order with other WR.
                                          1: relax order.
                                          2: strong order.
                                          3: reserve. see urma_order_type_t */
        uint32_t compOrder : 1;       /* 0: There is no completion order with other WR.
                                          1: Completion order with previous WR. */
        uint32_t fence : 1;           /* 0: There is no fence.
                                         1: Fence with previous read and atomic WR. */
        uint32_t solicitedEnable : 1; /* 0: not solicited.
                                          1: Solicited. */
        uint32_t completeEnable : 1;  /* 0: DO not Generate CR for this WR.
                                          1: Generate CR for this WR after the WR is completed. */
        uint32_t inlineFlag : 1;      /* 0: No data.
                                          1: Inline data. */
        uint32_t reserved : 25;
    } bs;
    uint32_t value;
};

struct WrNotifyInfo {
    uint64_t notifyData; /**< notify data */
    uint64_t notifyAddr; /**< remote notify addr */
    void *notifyHandle;  /**< remote notify handle */
};

struct WrReduceInfo {
    bool reduceEn;
    uint8_t reduceOpcode;
    uint8_t reduceDataType;
};

struct SendWrData {
    struct WrSgeList *sges;
    uint32_t numSge; /**< size of segs, not exceeds to MAX_SGE_NUM */

    uint8_t *inlineData;
    uint32_t inlineSize; /**< size of inline_data, see struct dev_base_attr */

    uint64_t remoteAddr;
    void *rmemHandle;

    union {
        struct {
            uint64_t wrId;
            enum RaWrOpcode opcode;
            unsigned int flags;   /**< reference to ra_send_flags */
            struct WrAuxInfo aux; /**< aux info */
        } rdma;

        struct {
            uint64_t userCtx;
            enum RaUbOpcode opcode;         /**< refer to urma_opcode_t */
            union JfsWrFlag flags;          /**< refer to urma_jfs_wr_flag_t */
            void *remQpHandle;              /**< resv for RM use */
            struct WrNotifyInfo notifyInfo; /**< required for opcode RA_UB_OPC_WRITE_NOTIFY */
            struct WrReduceInfo reduceInfo; /**<reduce is enabled when reduce_en is set to true */
        } ub;
    };

    uint32_t immData;

    uint32_t resv[16U];
};

struct UbPostInfo {
    uint16_t funcId : 7;
    uint16_t dieId : 1;
    uint16_t rsv : 8;
    uint16_t jettyId;
    // doorbell value
    uint16_t piVal;
    // direct wqe
    uint8_t dwqe[128U];
    uint16_t dwqeSize; /**< size of dwqe calc by piVal, 64 or 128 */
};

struct SendWrResp {
    union {
        struct WqeInfoT wqeTmp;         /**< wqe template info used for V80 offload */
        struct DbInfo db;               /**< doorbell info used for V71 and V80 opbase */
        struct UbPostInfo doorbellInfo; /**< doorbell info used for UB */
        uint8_t resv[384U];             /**< resv for write value doorbell info */
    };
};

/**
 * @ingroup libudma
 * @brief batch post send wr
 * @param qp_handle [IN] qp handle
 * @param send_wr_data [IN] send wr data
 * @param op_resp [IN/OUT] send wr resp
 * @param num [IN] size of wr_list & op_resp
 * @param complete_num [OUT] number of wr been post send successfully
 * @see ra_ctx_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaBatchSendWr(void *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[],
    unsigned int num, unsigned int *completeNum);

/**
 * @ingroup libudma
 * @brief update ci
 * @param qp_handle [IN] qp handle
 * @param ci [IN] ci
 * @see ra_ctx_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxUpdateCi(void *qpHandle, uint16_t ci);
#ifdef __cplusplus
}
#endif

#endif // HCCP_CTX_DATA_PLANE_H
