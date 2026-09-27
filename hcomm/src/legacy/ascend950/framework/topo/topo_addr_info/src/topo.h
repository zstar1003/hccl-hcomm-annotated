/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TOPO_H_
#define __TOPO_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 从文件中解析拓扑文件路径
 */
int GetTopoFilePathFromFile(const char* filePath, char* topoFilePath, size_t bufSize);

/**
 * 根据mainboard id选择对于的拓扑文件
 */
int TopoGetFilePath(unsigned mainboard_id, unsigned int spod_type, char* buf_size, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif
