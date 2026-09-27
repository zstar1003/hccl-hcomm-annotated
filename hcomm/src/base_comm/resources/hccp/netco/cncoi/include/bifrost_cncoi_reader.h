/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_READER_H
#define BIFROST_CNCOI_READER_H
#include "flatbuffers_common_builder.h"
#ifndef SimpoAlignas
#define SimpoAlignas(t) __attribute__((__aligned__(t)))
#endif
#if defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic push
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BifrostCncoiSliceKey BifrostCncoiSliceKeyT;
typedef struct BifrostCncoiComminfoKey BifrostCncoiComminfoKeyT;
typedef struct BifrostCncoiSendRankInfo BifrostCncoiSendRankInfoT;
typedef struct BifrostCncoiCommMd5sum BifrostCncoiCommMd5sumT;
typedef struct BifrostCncoiCommInfoRankInfo BifrostCncoiCommInfoRankInfoT;
typedef struct BifrostCncoiOperatorKey BifrostCncoiOperatorKeyT;
typedef struct BifrostCncoiAdjacencyKey BifrostCncoiAdjacencyKeyT;
typedef struct BifrostCncoiAdjInfo BifrostCncoiAdjInfoT;
typedef struct BifrostCncoiRankKey BifrostCncoiRankKeyT;
typedef struct BifrostCncoiRankInfo BifrostCncoiRankInfoT;
typedef struct BifrostCncoiRankDistributeKey BifrostCncoiRankDistributeKeyT;
typedef struct BifrostCncoiRootRankKey BifrostCncoiRootRankKeyT;
typedef struct BifrostCncoiRootRankInfo BifrostCncoiRootRankInfoT;

typedef struct BifrostCncoiComminfoEntryStruct BifrostCncoiComminfoEntryT;
typedef struct BifrostCncoiOperatorEntryStruct BifrostCncoiOperatorEntryT;
typedef struct BifrostCncoiAdjacencyEntryStruct BifrostCncoiAdjacencyEntryT;
typedef struct BifrostCncoiRankEntryStruct BifrostCncoiRankEntryT;
typedef struct BifrostCncoiRankDistributeEntryStruct BifrostCncoiRankDistributeEntryT;
typedef struct BifrostCncoiRootRankEntryStruct BifrostCncoiRootRankEntryT;

struct BifrostCncoiSliceKey {
    SimpoAlignas(4) uint32_t sliceKey;
};
struct BifrostCncoiComminfoKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(1) int8_t commDesc[128];
    SimpoAlignas(8) uint64_t commInitTime;
    SimpoAlignas(2) uint16_t packetId;
    SimpoAlignas(2) uint16_t resv[3];
};
struct BifrostCncoiSendRankInfo {
    SimpoAlignas(4) uint32_t sendRankInfo;
};
struct BifrostCncoiCommMd5sum {
    SimpoAlignas(1) uint8_t commMd5sum[16];
};
struct BifrostCncoiCommInfoRankInfo {
    SimpoAlignas(4) uint32_t deviceIp;
    SimpoAlignas(4) uint32_t serverIp;
    SimpoAlignas(2) uint16_t podId;
    SimpoAlignas(2) uint16_t resv;
};
struct BifrostCncoiOperatorKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(1) int8_t commDesc[128];
    SimpoAlignas(8) uint64_t commInitTime;
    SimpoAlignas(1) uint8_t operator;
    SimpoAlignas(1) uint8_t algorithm;
    SimpoAlignas(2) uint16_t rootRank;
    SimpoAlignas(1) uint8_t resv[4];
};
struct BifrostCncoiAdjacencyKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(1) int8_t commDesc[128];
    SimpoAlignas(1) uint8_t commMd5sum[16];
    SimpoAlignas(2) uint16_t srcLocalRankId;
    SimpoAlignas(1) uint8_t operator;
    SimpoAlignas(1) uint8_t algorithm;
    SimpoAlignas(2) uint16_t rootRank;
    SimpoAlignas(2) uint16_t resv;
};
struct BifrostCncoiAdjInfo {
    SimpoAlignas(2) uint16_t dstLocalRankId;
    SimpoAlignas(1) uint8_t phaseId;
    SimpoAlignas(1) uint8_t resv;
};
struct BifrostCncoiRankKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(1) int8_t commDesc[128];
    SimpoAlignas(8) uint64_t commInitTime;
    SimpoAlignas(2) uint16_t packetId;
    SimpoAlignas(2) uint16_t resv[3];
};
struct BifrostCncoiRankInfo {
    SimpoAlignas(4) uint32_t deviceIp;
    SimpoAlignas(4) uint32_t serverIp;
};
struct BifrostCncoiRankDistributeKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(4) uint32_t npuIp;
    SimpoAlignas(4) uint32_t resv;
};
struct BifrostCncoiRootRankKey {
    SimpoAlignas(8) uint64_t taskId;
    SimpoAlignas(1) int8_t commDesc[128];
    SimpoAlignas(8) uint64_t commInitTime;
    SimpoAlignas(1) uint8_t operator;
    SimpoAlignas(1) uint8_t algorithm;
    SimpoAlignas(1) uint8_t resv[6];
};
struct BifrostCncoiRootRankInfo {
    SimpoAlignas(2) uint16_t rootRankInfo;
};

struct BifrostCncoiComminfoEntryStruct {
    uint16_t packetNum;
    uint16_t resv;
    uint32_t sendRankInfo_len;
    BifrostCncoiSendRankInfoT *sendRankInfo;
    BifrostCncoiCommMd5sumT md5sum;
    uint16_t rankTotalNum;
    uint16_t resv2;
    uint32_t rankInfo_len;
    BifrostCncoiCommInfoRankInfoT *rankInfo;
};

struct BifrostCncoiOperatorEntryStruct {
    uint64_t trafficCnt;
    uint16_t l4SPortId;
    uint16_t maskLen;
    uint32_t resv;
};

struct BifrostCncoiAdjacencyEntryStruct {
    uint32_t adjInfo_len;
    BifrostCncoiAdjInfoT *adjInfo;
};

struct BifrostCncoiRankEntryStruct {
    uint16_t packetNum;
    uint16_t resv;
    uint32_t sendRankInfo_len;
    BifrostCncoiSendRankInfoT *sendRankInfo;
    BifrostCncoiCommMd5sumT md5sum;
    uint32_t rankTotalNum;
    uint32_t rankInfo_len;
    BifrostCncoiRankInfoT *rankInfo;
};

struct BifrostCncoiRankDistributeEntryStruct {
    uint32_t serverIp;
    uint32_t nodeId;
    uint8_t localRankNum;
    uint8_t pad1;
    uint16_t pad2;
    uint32_t rankTotalNum;
};

struct BifrostCncoiRootRankEntryStruct {
    uint32_t rootRankInfo_len;
    BifrostCncoiRootRankInfoT *rootRankInfo;
};

#ifdef __cplusplus
}
#endif

#if defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#endif

#endif /* BIFROST_CNCOI_READER_H */
