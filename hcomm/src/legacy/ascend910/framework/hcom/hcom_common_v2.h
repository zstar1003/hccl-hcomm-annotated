/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_COMMOM_V2_H
#define HCOM_COMMOM_V2_H

#include <string>
#include <vector>
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#ifdef __cplusplus
extern "C" {
#endif
HcclResult __attribute__((weak)) HcomDestroyV2(void);
HcclResult __attribute__((weak))
HcomCreateGroupImplV2(const std::string& group, u32 rankNum, const std::vector<u32>& rankIds);
HcclResult __attribute__((weak)) HcomDestroyGroupImplV2(const std::string& group);
HcclResult __attribute__((weak)) HcomGetWorldRankFromGroupRankV2(const char* group, u32 groupRank, u32* worldRank);
HcclResult __attribute__((weak)) HcomGetGroupRankFromWorldRankV2(u32 worldRank, const char* group, u32* groupRank);
HcclResult __attribute__((weak)) HcomGetRankSizeV2(const char* group, u32* rankSize);
HcclResult __attribute__((weak)) HcomGetCommV2(void** commV2);
HcclResult __attribute__((weak)) HcomGetGroupParamsV2(const char* group, void* groupParams, void** commV2);
HcclResult __attribute__((weak)) HcomInitByFileV2(const char* rankTablePath, const char* identify);
HcclResult __attribute__((weak)) HcomInitByStringV2(const char* rankTableM, const char* identify);
#ifdef __cplusplus
}
#endif
#endif /* HCCL_COMM_PUB_H */
