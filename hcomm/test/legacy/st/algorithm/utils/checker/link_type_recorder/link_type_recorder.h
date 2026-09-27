/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_LINK_TYPE_RECORDER_H
#define HCCLV1_LINK_TYPE_RECORDER_H

#include "llt_common.h"
#include <map>
#include "hccl_common.h"
#include "checker_def.h"
namespace checker {

class LinkTypeRecorder {
public:
    static LinkTypeRecorder* Global();
    std::map<CheckerDevType, std::map<u32, std::map<u32, LinkTypeInServer>>> devLinkTypeMap_;

    void SetLinkTypeMap(std::vector<CheckerDevType>& devTypes);
    void SetLinkTypeMapOf910A();
    void SetLinkTypeMapOf910B();
    void SetLinkTypeMapOf310P3V();
    void SetLinkTypeMapOf310P3Dou();
    void SetLinkTypeMapOf910_93();
    void SetIs310P3V(bool is310P3V);

    bool is310P3V_ = false;
};

} // namespace checker

#endif
