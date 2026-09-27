/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IB_USER_IOCTL_VERBS_H
#define IB_USER_IOCTL_VERBS_H

#include <rdma/ib_user_verbs.h>

#define UVERBS_ID_NS_SHIFT 12

#define UVERBS_UDATA_DRIVER_DATA_NS 1
#define UVERBS_UDATA_DRIVER_DATA_FLAG (1UL << UVERBS_ID_NS_SHIFT)

enum uverbs_default_objects {
    UVERBS_OBJECT_DEVICE, /* No instances of DEVICE are allowed */
    UVERBS_OBJECT_PD,
    UVERBS_OBJECT_COMP_CHANNEL,
    UVERBS_OBJECT_CQ,
    UVERBS_OBJECT_QP,
    UVERBS_OBJECT_SRQ,
    UVERBS_OBJECT_AH,
    UVERBS_OBJECT_MR,
    UVERBS_OBJECT_MW,
    UVERBS_OBJECT_FLOW,
    UVERBS_OBJECT_XRCD,
    UVERBS_OBJECT_RWQ_IND_TBL,
    UVERBS_OBJECT_WQ,
    UVERBS_OBJECT_LAST,
};

enum {
    UVERBS_UHW_IN = UVERBS_UDATA_DRIVER_DATA_FLAG,
    UVERBS_UHW_OUT,
};

enum uverbs_create_cq_cmd_attr_ids {
    CREATE_CQ_HANDLE,
    CREATE_CQ_CQE,
    CREATE_CQ_USER_HANDLE,
    CREATE_CQ_COMP_CHANNEL,
    CREATE_CQ_COMP_VECTOR,
    CREATE_CQ_FLAGS,
    CREATE_CQ_RESP_CQE,
};

enum uverbs_destroy_cq_cmd_attr_ids {
    DESTROY_CQ_HANDLE,
    DESTROY_CQ_RESP,
};

enum uverbs_actions_cq_ops {
    UVERBS_CQ_CREATE,
    UVERBS_CQ_DESTROY,
};

enum ib_uverbs_read_counters_flags {
    /* prefer read values from driver cache */
    IB_UVERBS_READ_COUNTERS_PREFER_CACHED = 1 << 0,
};

enum ib_uverbs_advise_mr_advice {
    IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH,
    IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE,
};

enum ib_uverbs_advise_mr_flag {
    IB_UVERBS_ADVISE_MR_FLAG_FLUSH = 1 << 0,
};

struct ib_uverbs_query_port_resp_ex {
    struct ib_uverbs_query_port_resp legacy_resp;
    __u16 port_cap_flags2;
    __u8 reserved[6];
};

#endif
