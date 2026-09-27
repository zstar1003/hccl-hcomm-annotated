/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_U_API_H
#define CCU_U_API_H

#include "ccu_u_comm.h"

#define MAX_IO_DIE 2
#define CCU_DIE_ATTACHED 1
#define CCU_INIT_OK 1
#define CCU_INIT_NO 0

#define CCU_ATTRI_VISI_DEF __attribute__((visibility("default")))

struct ccu_u_op_handle {
    unsigned int opcode;
    int (*op_handle)(const struct channel_info_in *, struct channel_info_out *);
};

struct ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

struct ccu_tlv_request_info {
    unsigned int ccuMsg;
    char *dataIn;
    char *dataOut;
    unsigned int dataInLength;
    unsigned int dataOutLength;
    unsigned int dataOutMaxLength;
    unsigned int resv[8];
};

CCU_ATTRI_VISI_DEF int ccu_init(void);
CCU_ATTRI_VISI_DEF int ccu_uninit(void);
CCU_ATTRI_VISI_DEF unsigned long long ccu_get_cqe_base_addr(unsigned int die_id);
CCU_ATTRI_VISI_DEF int ccu_custom_channel(const struct channel_info_in *in, struct channel_info_out *out);
CCU_ATTRI_VISI_DEF int ccu_get_mem_info(unsigned int die_id, unsigned long long mem_type_bitmap,
    struct ccu_mem_rsp *rsp);
CCU_ATTRI_VISI_DEF int ccu_tlv_request(struct ccu_tlv_request_info *tlvRequestInfo);
int get_ccu_u_info(unsigned int die_id, struct ccu_u_info *info);
int get_region_by_op(ccu_u_opcode_t op, struct ccu_region **region);

#endif /* CCU_U_API_H */
