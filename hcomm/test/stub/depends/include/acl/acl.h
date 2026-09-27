/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_H_
#define INC_EXTERNAL_ACL_ACL_H_

#include "acl_rt.h"
#include "acl_op.h"
#include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_PKG_VERSION_MAX_SIZE 128
#define ACL_PKG_VERSION_PARTS_MAX_SIZE 64

/**
 * @ingroup AscendCL
 * @brief acl initialize
 *
 * @par Restriction
 * The aclInit interface can be called only once in a process
 * @param configPath [IN]    the config path,it can be NULL
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclInit(const char* configPath);

/**
 * @ingroup AscendCL
 * @brief acl finalize
 *
 * @par Restriction
 * Need to call aclFinalize before the process exits.
 * After calling aclFinalize,the services cannot continue to be used normally.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalize();

/**
 * @ingroup AscendCL
 * @brief acl finalize reference
 *
 * @par Restriction
 * This interface decrements the internal reference count each time it is called.
 * Resources are only released when the reference count reaches 0.
 * To get the current reference count, pass a valid pointer to refCount.
 * To ignore the reference count, pass nullptr instead.
 *
 * @param refCount [IN/OUT] Pointer to receive current reference count after calling aclFinalizeReference; can be
 * nullptr.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalizeReference(uint64_t* refCount);

/**
 * @ingroup AscendCL
 * @brief get recent error message
 *
 * @retval null for failed
 * @retval OtherValues success
 */
ACL_FUNC_VISIBILITY const char* aclGetRecentErrMsg();

/**
 * @ingroup AscendCL
 * @brief enum for CANN package name
 */
typedef enum aclCANNPackageName {
    ACL_PKG_NAME_CANN,
    ACL_PKG_NAME_RUNTIME,
    ACL_PKG_NAME_COMPILER,
    ACL_PKG_NAME_HCCL,
    ACL_PKG_NAME_TOOLKIT,
    ACL_PKG_NAME_OPP,
    ACL_PKG_NAME_OPP_KERNEL,
    ACL_PKG_NAME_DRIVER
} aclCANNPackageName;

/**
 * @ingroup AscendCL
 * @brief struct for storaging CANN package version
 */
typedef struct aclCANNPackageVersion {
    char version[ACL_PKG_VERSION_MAX_SIZE];
    char majorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char minorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char releaseVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char patchVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char reserved[ACL_PKG_VERSION_MAX_SIZE];
} aclCANNPackageVersion;

/**
 * @ingroup AscendCL
 * @brief query CANN package version
 *
 * @param name[IN] CANN package name
 * @param version[OUT] CANN package version information
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion* version);

ACL_FUNC_VISIBILITY aclError aclsysGetVersionStr(char* pkgNname, char* versionStr);

ACL_FUNC_VISIBILITY aclError aclsysGetVersionNum(char* pkgNname, int32_t* versionNum);
#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_H_
