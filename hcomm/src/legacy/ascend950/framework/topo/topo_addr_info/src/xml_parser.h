/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XML_PARSER_H
#define XML_PARSER_H

#include <stdbool.h>
#include "topo_addr_info_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TAG_NAME_LEN 32
#define MAX_ATTR_NAME_LEN 48
#define MAX_ATTR_VAL_LEN 64
#define MAX_ATTRS_PER_TAG 8

/* ─── 通用键值对属性 ─── */
typedef struct {
    char name[MAX_ATTR_NAME_LEN];
    char value[MAX_ATTR_VAL_LEN];
} TagAttr;

/* ─── 单条标签解析结果（纯语法，不含任何业务语义） ─── */
typedef struct {
    char tagName[MAX_TAG_NAME_LEN]; /* 原始标签名，如 "pci"、"net" */
    bool isSelfClose;
    int depth;
    TagAttr attrs[MAX_ATTRS_PER_TAG];
    int attrCount;
} TagEntry;

#define MAX_TAG_ENTRIES 1024

/**
 * 从标签的属性列表中按 name 查找属性值。
 * @return 指向 value 的指针，未找到返回 NULL
 */
const char* TagFindAttr(const TagEntry* e, const char* name);

/**
 * 解析 XML 文件，提取所有标签到 TagEntry 数组。
 * @param xmlPath  XML 文件路径
 * @param tags    输出数组
 * @param tagCount 输出标签数
 * @param maxTags 数组容量
 * @return HCCL_SUCCESS=成功, 其他=错误码
 */
TopoAddrResult ParseXmlTags(const char* xmlPath, TagEntry* tags, unsigned int* tagCount, unsigned int maxTags);

#ifdef __cplusplus
}
#endif

#endif /* XML_PARSER_H */
