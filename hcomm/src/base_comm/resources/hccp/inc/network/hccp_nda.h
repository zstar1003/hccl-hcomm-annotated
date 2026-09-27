/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_NDA_H
#define HCCP_NDA_H

#include <infiniband/verbs.h>
#include <stdint.h>
#include "hccp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NdaOps {
    void *(*alloc)(size_t size);
    void (*free)(void *ptr);

    void (*memset_s)(void *dst, int value, size_t count);
    int (*memcpy_s)(void *dst, size_t dstSize, void *src, size_t srcSize, uint32_t direct);
};

enum {
    QBUF_DMA_MODE_DEFAULT = 0,
    QBUF_DMA_MODE_INDEP_UB = 1,
    QBUF_DMA_MODE_MAX = 2,
};

enum {
    MEMCPY_DIRECT_HOST_TO_HOST = 0,
    MEMCPY_DIRECT_HOST_TO_DEVICE,
    MEMCPY_DIRECT_DEVICE_TO_HOST,
    MEMCPY_DIRECT_DEVICE_TO_DEVICE,
};

struct NdaCqInitAttr {
    struct ibv_cq_init_attr_ex attr;

    uint32_t cqCapFlag;
    uint32_t dmaMode;
    struct NdaOps *ops;
    uint64_t resv[8U];
};

struct iovAddrDesc {
    void *iovBase;
    size_t iovLen;
};

struct queueBuf {
    uint64_t base;
    uint32_t entryCnt;
    uint32_t entrySize;
    uint64_t resv[4U];
};

struct queueInfo {
    struct queueBuf qBuf;
    struct iovAddrDesc dbrPiVa;
    struct iovAddrDesc dbrCiVa;
    struct iovAddrDesc dbHwVa;
    uint64_t resv[4U];
};

struct NdaCqInfo {
    struct ibv_cq *cq;
    struct queueInfo cqInfo;
    uint64_t resv[32U];
};

struct NdaQpInitAttr {
    struct ibv_qp_init_attr attr;

    uint32_t qpCapFlag;
    uint32_t dmaMode;
    struct NdaOps *ops;
    uint64_t resv[8U];
};

struct NdaQpInfo {
    struct ibv_qp *qp;
    struct queueInfo sqInfo;
    struct queueInfo rqInfo;
    uint64_t resv[31U];
    uint64_t vendorPrivInfo;
};

enum {
    DIRECT_FLAG_NOTSUPP = 0,
    DIRECT_FLAG_PCIE = 1,
    DIRECT_FLAG_UB = 2,
};

/**
 * @ingroup libinit
 * @brief Get NDA direct flag
 * @param rdmaHandle [IN] rdma handle
 * @param directFlag [OUT] direct flag
 * @see RaRdevInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNdaGetDirectFlag(void *rdmaHandle, int *directFlag);

/**
 * @ingroup librdma
 * @brief Create NDA cq(only one cq handle)
 * @param rdmaHandle [IN] rdma handle
 * @param attr [IN] NDA cq attr
 * @param info [OUT] NDA cq info
 * @param cqHandle [OUT] NDA cq handle
 * @see RaNdaCqDestroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNdaCqCreate(void *rdmaHandle, struct NdaCqInitAttr *attr, struct NdaCqInfo *info,
    void **cqHandle);

/**
 * @ingroup librdma
 * @brief Destroy NDA cq(only one cq handle)
 * @param rdmaHandle [IN] rdma handle
 * @param cqHandle [IN] NDA cq handle
 * @see RaNdaCqCreate
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNdaCqDestroy(void *rdmaHandle, void *cqHandle);

/**
 * @ingroup librdma
 * @brief Create NDA qp handle(only one qp handle)
 * @param rdmaHandle [IN] rdma handle
 * @param attr [IN] NDA qp attr
 * @param info [OUT] NDA qp info
 * @param qpHandle [OUT] NDA qp handle
 * @see RaQpDestroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNdaQpCreate(void *rdmaHandle, struct NdaQpInitAttr *attr, struct NdaQpInfo *info,
    void **qpHandle);
#ifdef __cplusplus
}
#endif
#endif // HCCP_NDA_H
