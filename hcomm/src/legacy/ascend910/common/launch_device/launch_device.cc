/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string>
#include "launch_device.h"
#include "log.h"
#include "mmpa_api.h"
#include "env_config.h"

using namespace std;

namespace hccl {
HcclResult LoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle)
{
    CHK_PRT_RET(binPath == nullptr, HCCL_ERROR("[Load][Binary]binary path is nullptr"), HCCL_E_PTR);

    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(binPath, realPath) == nullptr,
        HCCL_ERROR("LoadBinaryFromFile: %s is not a valid real path, err[%d]", binPath, errno), HCCL_E_INTERNAL);
    HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);

    aclrtBinaryLoadOptions loadOptions = {};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = optionType;
    option.value.cpuKernelMode = cpuKernelMode;
    aclError aclRet
        = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle); // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx] load binary from file error.", aclRet),
        HCCL_E_OPEN_FILE_FAILURE);

    return HCCL_SUCCESS;
}

} // namespace hccl
