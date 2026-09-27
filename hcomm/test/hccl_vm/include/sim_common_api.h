/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMMON_API_H
#define SIM_COMMON_API_H

#include <string>
#include "hccl/hccl_types.h"

class InstallPath {
public:
    // 获取 hccl-vm-install 根目录的绝对路径
    static const std::string& GetHcclVmInstallAbsPath();

    // 把相对路径解析为基于 install root 的绝对路径
    // @param relPath 相对路径(如 "data/hccl_vm_data.db")
    // @note  - 已是绝对路径(以 '/' 开头)则原样返回
    //        - 以 "./" 开头也原样返回
    //        - 空字符串原样返回
    static std::string ResolveToInstallRoot(const std::string& relPath);
};

std::string GetDataTypeStr(HcclDataType type);

std::string GetReduceOpStr(HcclReduceOp op);

#endif // SIM_COMMON_API_H
