/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_MSG_H
#define BKF_MSG_H

#include "bkf_comm.h"
#include "bkf_assert.h"
#include "bkf_url.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

/*
消息整体定义
      0                   1                   2                   3
      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                              sign                             |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |           messageId           |            RESERVED           |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                             bodyLen                           |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     //                            TLV0                             //
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     ~                                                               ~
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     //                            TLVn                             //
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

TLV整体定义
      0                   1                   2                   3
      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |           typeId              |     flag      |   RESERVED    |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                             valLen                            |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     //                                                             //
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                                               |      pad      |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*/

/* msg */
#define BKF_MSG_SIGN (0xABCD2021)
#define BKF_MSG_RSV_VAL (0)

typedef struct tagBkfMsgHead {
    uint32_t sign;
    uint16_t msgId;
    uint8_t flag;
    uint8_t reserved;
    uint32_t bodyLen;
} BkfMsgHead;
#define BKF_FLAG_VERIFY ((uint8_t)1 << 0)
#define BKF_FLAG_NEED_TABLE_COMPLETE ((uint8_t)1 << 1)

enum {
    BKF_MSG_RESERVED = 0,
    BKF_MSG_HELLO = 1,
    BKF_MSG_HELLO_ACK = 2,
    BKF_MSG_SUB = 3,
    BKF_MSG_SUB_ACK = 4,
    BKF_MSG_UNSUB = 5,
    BKF_MSG_BATCH_BEGIN = 6,
    BKF_MSG_BATCH_END = 7,
    BKF_MSG_DATA = 8,
    BKF_MSG_DEL_SUB_NTF = 9,
    /* sub/pub模型预留一段消息id */
    BKF_MSG_NOTIFY = 21,
    /* notify预留一段消息id */
    BKF_MSG_RPC = 31,
    /* 51 -88，singe发布订阅 */
    BKF_MSG_SINGLE_SUB = 51,
    BKF_MSG_SINGLE_ACK = 52,
    BKF_MSG_SINGLE_UNSUB = 53,
    BKF_MSG_SINGLE_COND_BATCH_BEGIN = 54,
    BKF_MSG_SINGLE_COND_BATCH_END = 55,
    BKF_MSG_SINGLE_COND_DATA = 56,
    BKF_MSG_SINGLE_RESULT_DATA = 57,
    BKF_MSG_MAX
};
#define BKF_MSG_IS_VALID(msgId) (((msgId) > BKF_MSG_RESERVED) && ((msgId) < BKF_MSG_MAX))
#define BKF_MSG_IS_CONN(msgId) (((msgId) >= BKF_MSG_HELLO) && ((msgId) <= BKF_MSG_HELLO_ACK))
#define BKF_MSG_IS_SESS(msgId) (BKF_MSG_IS_VALID(msgId) && !BKF_MSG_IS_CONN(msgId))

/**
 * @brief 获取消息id诊断信息
 *
 * @param[in] msgId 消息id
 * @return msgId对应的文本诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfMsgGetIdStr(uint16_t msgId);

/* tlv */
typedef struct tagBkfTL {
    uint16_t typeId;
    uint8_t flag;
    uint8_t reserved;
    uint32_t valLen;
} BkfTL;

enum {
    BKF_TLV_RESERVED = 0,
    BKF_TLV_TRANS_NUM = 1,
    BKF_TLV_PROJECT_NAME = 2,
    BKF_TLV_PROJECT_VERSION = 3,
    BKF_TLV_SERVICE_NAME = 4,
    BKF_TLV_PUBER_NAME = 5,
    BKF_TLV_SUBER_NAME = 6,
    BKF_TLV_IDL_VERSION = 7,
    BKF_TLV_REASON_CODE = 8,
    BKF_TLV_SLICE_KEY = 9,
    BKF_TLV_TABLE_TYPE = 10,
    BKF_TLV_TUPLE_IDL_DATA = 11,
    BKF_TLV_RPC = 12,
    BKF_TLV_COND_IDL_DATA = 13,
    BKF_TLV_RESULT_IDL_DATA = 14,
    BKF_TLV_SUBER_LSNURL = 15,
    BKF_TLV_MAX
};
#define BKF_TLV_IS_VALID(typeId) (((typeId) > BKF_TLV_RESERVED) && ((typeId) < BKF_TLV_MAX))

/**
 * @brief 获取tlv type诊断信息
 *
 * @param[in] typeId tlv类型id
 * @return 类型id对应的文本诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfTlvGetTypeStr(uint16_t typeId);

/* reason code */
enum {
    BKF_RC_OK_MIN = 0,
    BKF_RC_OK = 0,
    BKF_RC_OK_MAX,

    BKF_RC_ERR_MIN = 128,
    BKF_RC_ERR = 128,
    BKF_RC_ERR_PROJECT_NAME = 129,
    BKF_RC_ERR_PROJECT_VERSION = 130,
    BKF_RC_ERR_SERVICE_NAME = 131,
    BKF_RC_ERR_SUBER_NAME = 132,
    BKF_RC_ERR_IDL_VERSION = 133,
    BKF_RC_ERR_CODE_MAX
};

/**
 * @brief 获取reasonCode诊断信息
 *
 * @param[in] reasonCode 原因码
 * @return 原因码对应的文本诊断信息
 *   @retval 非空诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfReasonCodeGetStr(uint8_t reasonCode);

/*  各tlv定义 */
typedef struct tagBkfTlvTransNum {
    BkfTL tl;
    uint64_t num;
} BkfTlvTransNum;

#define BKF_TRANS_NUM_INVALID ((uint64_t)0xffffffffffffffff)

typedef struct tagBkfTlvName {
    BkfTL tl;     /* len 长度包括字串结束'\0'字符  */
    char name[0]; /* 占位符 */
} BkfTlvName;

#define BKF_PROJECT_NAME "backfin"
typedef BkfTlvName BkfTlvProjectName;

typedef struct tagBkfTlvProjectVersion {
    BkfTL tl;
    uint8_t version;
    uint8_t reserved[3];
} BkfTlvProjectVersion;
#define BKF_PROJECT_VERSION 1

#define BKF_TLV_NAME_LEN_MAX 63
typedef BkfTlvName BkfTlvServiceName;
typedef BkfTlvName BkfTlvPuberName;
typedef BkfTlvName BkfTlvSuberName;

typedef struct tagBkfTlvIdlVersion {
    BkfTL tl;
    uint16_t major;
    uint16_t minor;
} BkfTlvIdlVersion;

typedef struct tagBkfTlvReasonCode {
    BkfTL tl;
    uint8_t reasonCode;
    uint8_t reserved[3];
} BkfTlvReasonCode;

typedef struct tagBkfTlvSliceKey {
    BkfTL tl;
    uint8_t sliceKey[0]; /* 占位符 */
} BkfTlvSliceKey;

typedef struct tagBkfTlvSuberLsnUrl {
    BkfTL tl;
    BkfUrl lsnUrl;
} BkfTlvSuberLsnUrl;

typedef struct tagBkfTlvTableType {
    BkfTL tl;
    uint16_t tableTypeId;
    uint8_t reserved[2];
} BkfTlvTableType;

enum { BKF_RPC_OPER_REQUEST = 0, BKF_RPC_OPER_RESPONSE = 1, BKF_RPC_OPER_CANCEL = 2, BKF_RPC_OPER_MAX };
typedef struct {
    BkfTL tl;
    uint16_t type;
    uint16_t operCode;
    uint64_t seq;
    uint8_t idlData[0]; /* 占位符 */
} BkfTlvRpc;

/* [0, 65525)的表类型都有效 */
#define BKF_TABLE_TYPE_ID_INVALID ((uint16_t) - 1)

typedef struct tagBkfTlvTupleIdlData {
    BkfTL tl;
    uint64_t seq;
    uint8_t idlData[0]; /* 占位符 */
} BkfTlvTupleIdlData;

typedef struct tagBkfTlvCondIdlData {
    BkfTL tl;
    uint8_t idlData[0]; /* 占位符 */
} BkfTlvCondIdlData;

typedef struct tagBkfTlvResultIdlData {
    BkfTL tl;
    uint8_t idlData[0]; /* 占位符 */
} BkfTlvResultIdlData;

#define BKF_FLAG_TUPLE_UPDATE ((uint8_t)1 << 0)
#define BKF_TLV_GET_TUPLE_IDL_DATA_LEN(tl) ((tl)->valLen - BKF_MBR_SIZE(BkfTlvTupleIdlData, seq))
// 条件update,无此标记即为delete
#define BKF_FLAG_COND_UPDATE ((uint8_t)1 << 0)
// 一次性订阅
#define BKF_FLAG_COND_ONCE ((uint8_t)1 << 1)

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
