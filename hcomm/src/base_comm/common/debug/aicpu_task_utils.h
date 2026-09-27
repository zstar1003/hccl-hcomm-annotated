/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_TASK_UTILS_H
#define HCOMM_AICPU_TASK_UTILS_H

#include <cstdint>

#include "config_plf_log_v2.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"

// 确认ptr应该为空
#define CHK_PTR_NOTNULL(ptr)                                                                                     \
    do {                                                                                                         \
        if (UNLIKELY((ptr) != nullptr)) {                                                                        \
            HCCL_ERROR(                                                                                          \
                "[%s] errNo[0x%016llx] ptr[%s] is 0x%016llx (should be null), return HCCL_E_INTERNAL", __func__, \
                HCCL_ERROR_CODE(HCCL_E_INTERNAL), #ptr, (ptr));                                                  \
            return HCCL_E_INTERNAL;                                                                              \
        }                                                                                                        \
    } while (0)

// 确认ptrPtr不应该为空, 但*ptrPtr应该为空
#define CHK_PTRPTR_NULL(ptrPtr)     \
    do {                            \
        CHK_PTR_NULL(ptrPtr);       \
        CHK_PTR_NOTNULL(*(ptrPtr)); \
    } while (0)

namespace hcomm {

class AicpuTaskUtils {
public:
    // GLOBAL_LOG_LEVEL=0或者HCCL_DEBUG_CONFIG="TASK"时, 打印task内容
    // 注意: 底层调试的关键DFX能力, 例如打印正常展开与cache刷新的task内容并比对, 确保刷新数量与内容正确
    static HcclResult DumpSqeContent(const uint8_t* sqePtr);
    static HcclResult DumpWqeContent(const uint8_t* wqePtr);

private:
    static inline HcclResult DumpUbdmaSqe_(const uint8_t* sqePtr);
    static inline HcclResult DumpNotifySqe_(const uint8_t* sqePtr);
    static inline HcclResult DumpSdmaSqe_(const uint8_t* sqePtr);
    static inline HcclResult DumpReadWriteWqe_(const uint8_t* wqePtr);
    static inline HcclResult DumpWriteWithNotifyWqe_(const uint8_t* wqePtr);
};

} // namespace hcomm

#endif
