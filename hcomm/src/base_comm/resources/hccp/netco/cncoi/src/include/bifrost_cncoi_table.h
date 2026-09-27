/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_TABLE_H

#define BIFROST_CNCOI_TABLE_H

#include "bifrost_cncoi_build_multi_layer.h"
#include "bkf_comm.h"

#if __cplusplus
extern "C" {
#endif
#pragma pack(4)

#define BIFROST_CNCOI_TABLE_TYPE_COMMINFO 100
int32_t BifrostCncoiComminfoKeyCmp(const BifrostCncoiComminfoKeyT *key1Input, const BifrostCncoiComminfoKeyT *key2InDs);
char *BifrostCncoiComminfoKeyGetStr(const BifrostCncoiComminfoKeyT *key, uint8_t *buf, int32_t bufLen);

#define BIFROST_CNCOI_TABLE_TYPE_OPERATOR 101
int32_t BifrostCncoiOperatorKeyCmp(const BifrostCncoiOperatorKeyT *key1Input, const BifrostCncoiOperatorKeyT *key2InDs);
char *BifrostCncoiOperatorKeyGetStr(const BifrostCncoiOperatorKeyT *key, uint8_t *buf, int32_t bufLen);

#define BIFROST_CNCOI_TABLE_TYPE_ADJACENCY 102
int32_t BifrostCncoiAdjacencyKeyCmp(const BifrostCncoiAdjacencyKeyT *key1Input,
    const BifrostCncoiAdjacencyKeyT *key2InDs);
char *BifrostCncoiAdjacencyKeyGetStr(const BifrostCncoiAdjacencyKeyT *key, uint8_t *buf, int32_t bufLen);

#define BIFROST_CNCOI_TABLE_TYPE_RANK 103
int32_t BifrostCncoiRankKeyCmp(const BifrostCncoiRankKeyT *key1Input, const BifrostCncoiRankKeyT *key2InDs);
char *BifrostCncoiRankKeyGetStr(const BifrostCncoiRankKeyT *key, uint8_t *buf, int32_t bufLen);

#define BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE 104
int32_t BifrostCncoiRankDistributeKeyCmp(const BifrostCncoiRankDistributeKeyT *key1Input,
    const BifrostCncoiRankDistributeKeyT *key2InDs);
char *BifrostCncoiRankDistributeKeyGetStr(const BifrostCncoiRankDistributeKeyT *key, uint8_t *buf, int32_t bufLen);

#define BIFROST_CNCOI_TABLE_TYPE_ROOT_RANK 105
int32_t BifrostCncoiRootRankKeyCmp(const BifrostCncoiRootRankKeyT *key1Input, const BifrostCncoiRootRankKeyT *key2InDs);
char *BifrostCncoiRootRankKeyGetStr(const BifrostCncoiRootRankKeyT *key, uint8_t *buf, int32_t bufLen);

#pragma pack()

#if __cplusplus
}
#endif

#endif
