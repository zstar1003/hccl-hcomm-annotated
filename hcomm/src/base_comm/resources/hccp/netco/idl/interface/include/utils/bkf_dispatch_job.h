/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_JOB_ADP_H
#define BKF_JOB_ADP_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_table.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/**
 * @brief 当app自身job使用静态表分发时，可以选择该文件中的接口实现一个简单的动态job分发
 */

/**
 * @brief job回调接口，适配vrp job，有两个入参
 */
typedef uint32_t (*F_BKF_VRP_JOB_CALLBACK)(void *para1, void *para2);

typedef struct tagBkfDispatchJobInfo {
    uint32_t typeId;
    F_BKF_VRP_JOB_CALLBACK callBack;
} BkfDispatchJobInfo;

BkfTableMng *BkfDispatchJobInit(BkfMemMng *memMng)
{
    BkfTableInitArg arg = {.memMng = memMng,
        .keyLen = sizeof(uint32_t),
        .valLen = sizeof(F_BKF_VRP_JOB_CALLBACK),
        .keyCmp = (F_BKF_CMP)Bkfuint32_tCmp};
    return BkfTableInit(&arg);
}

void BkfDispatchJobUnInit(BkfTableMng *handle)
{
    BkfTableUnInit(handle);
}

uint32_t BkfDispatchJobReg(BkfTableMng *handle, uint32_t jobTypeId, F_BKF_VRP_JOB_CALLBACK callBack)
{
    return BkfTableAdd(handle, &jobTypeId, &callBack, VOS_NULL);
}

uint32_t BkfDispatchJobDispatch(BkfTableMng *handle, uint32_t jobTypeId, void *para1, void *para2)
{
    BkfTableInfo info;
    uint32_t ret = BkfTableFind(handle, &jobTypeId, &info);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }
    F_BKF_VRP_JOB_CALLBACK callBack = *((F_BKF_VRP_JOB_CALLBACK *)(info.val));
    return callBack(para1, para2);
}

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif