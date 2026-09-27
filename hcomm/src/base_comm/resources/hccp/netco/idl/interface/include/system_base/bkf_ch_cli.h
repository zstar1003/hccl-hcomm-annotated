/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_CLI_H
#define BKF_CH_CLI_H

#include "bkf_ch_cli_adef.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* func */
BkfChCliMng *BkfChCliInit(BkfChCliMngInitArg *arg);
void BkfChCliUninit(BkfChCliMng *chMng);

/*
1. 融合调度需要selfCid。app最好不要/不应该显式输入。
2. 先regType后enable。如果逆序，reg返回失败
3. malloc/write/free:
   1) 放弃发送: 直接free
   2) 正常发送，返回长度:
      a) = 请求长度，不要free, **此处要特别小心，不要重复释放**
      b) [0, 请求长度)， 需要free
      c) 其他，异常，排查错误
4. 有些支持拉/read获取数据模式，有些支持推数据模式。
   read接口可以注册为空。但如果为空，app对此类型的url调用read时，会断言。
5. 频繁调用listen/unlisten同一个地址，会导致链路实际状态震荡。当然，最终结果是对的。
6. 在mayAccept和conn回调中，禁止调用close
7. 在disconn回调中及其通知之后，禁止调用close
*/

/* 先regType后enable。如果逆序，reg返回失败 */
uint32_t BkfChCliRegType(BkfChCliMng *chMng, BkfChCliTypeVTbl *vTbl);
uint32_t BkfChCliUnRegType(BkfChCliMng *chMng, uint8_t chType);

uint32_t BkfChCliEnable(BkfChCliMng *chMng, BkfChCliEnableArg *arg);

uint32_t BkfChCliSetSelfCid(BkfChCliMng *chMng, uint32_t selfCid);
uint32_t BkfChCliSetCera(BkfChCliMng *chMng, BkfCera *cera);
uint32_t BkfChCliUnSetCera(BkfChCliMng *chMng);

uint32_t BkfChCliSetSelfUrl(BkfChCliMng *chMng, BkfUrl *selfUrl);

BkfChCliConnId *BkfChCliConn(BkfChCliMng *chMng, BkfUrl *urlServer);
BkfChCliConnId *BkfChCliConnEx(BkfChCliMng *chMng, BkfUrl *urlServer, BkfUrl *urlLocal);

void BkfChCliDisConn(BkfChCliMng *chMng, BkfChCliConnId *connId);

void *BkfChCliMallocDataBuf(BkfChCliMng *chMng, BkfChCliConnId *connId, int32_t dataBufLen);
void BkfChCliFreeDataBuf(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf);

int32_t BkfChCliRead(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf, int32_t bufLen);
int32_t BkfChCliWrite(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf, int32_t dataLen);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
