/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    TSD_HCCP = 0,       /**< HCCP*/
    TSD_COMPUTE,        /**< Compute_process*/
    TSD_CUSTOM_COMPUTE, /**< Custom Compute_process*/
    TSD_QS,
    TSD_WAITTYPE_MAX /**< Max*/
} TsdWaitType;

/**
 * @ingroup SendStartUpFinishMsg
 * @brief sub process start finish and Wait for the TSD process to issue the shutdown command
 *
 * @par Function
 * Wait for the TSD process to issue the shutdown command
 *
 * @param NA
 * @param deviceId [IN] type #unsigned int. Physical device ID
 * @param waitType [IN] type #TsdWaitType. HCCP or CP
 * @param hostPid [IN] type #unsigned int. Host pid
 * @param vfId [IN] type #unsigned int. Virtual force Id
 * @retval 0 Success
 * @retval OtherValues 0 Fail
 *
 * @par Dependency
 * @li libtsdppc.so: Library to which the interface belongs.
 * @li tsd.h: Header file where the interface declaration is located.
 */
int32_t SendStartUpFinishMsg(const uint32_t deviceId, const TsdWaitType waitType, const uint32_t hostPid,
    const uint32_t vfId)
{
    (void)deviceId;
    (void)waitType;
    (void)hostPid;
    (void)vfId;
    return 0;
}

/**
 * @ingroup tsd_event_client
 * @brief sub process send error code while start/stop period
 * @param [in] deviceId : device id
 * @param [in] waitType : process type
 * @param [in] hostPid :  host pid
 * @param [in] vfId : vf id
 * @param [in] errCode : errCode: errMsg code produced by host
 * @param [in] errLen : errLen: errMsg code length
 * @return TSD_OK: success, other: error code
 */
int32_t ReportProcessStartUpErrorCode(const uint32_t deviceId, const TsdWaitType waitType, const uint32_t hostPid,
    const uint32_t vfId, const char *errCode, const uint32_t errLen)
{
    (void)deviceId;
    (void)waitType;
    (void)hostPid;
    (void)vfId;
    (void)errCode;
    (void)errLen;
    return 0;
}