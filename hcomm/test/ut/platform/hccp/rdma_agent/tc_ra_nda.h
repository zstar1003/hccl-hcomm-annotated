/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TC_RA_NDA_H
#define TC_RA_NDA_H
#ifdef __cplusplus
extern "C" {
#endif
void TcRaNdaGetDirectFlag();
void TcRaPeerNdaGetDirectFlag();
void TcRaNdaQpCreate();
void TcRaPeerNdaQpCreate();
void TcRaPeerGetQpHyperFeature();
#ifdef __cplusplus
}
#endif
#endif // TC_RA_NDA_H
