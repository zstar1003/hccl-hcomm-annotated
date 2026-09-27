/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xml_parser.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "securec.h"
#include "topo_addr_info_log.h"

#define MAX_LINE_LEN 4096

/* ─── 工具函数 ─── */

const char* TagFindAttr(const TagEntry* e, const char* name)
{
    for (int i = 0; i < e->attrCount; i++) {
        if (strcmp(e->attrs[i].name, name) == 0) {
            return e->attrs[i].value;
        }
    }
    return NULL;
}

static bool IsSelfClose(const char* tag, const char* tagEnd)
{
    size_t n = (size_t)(tagEnd - tag + 1);
    return n >= 2 && tag[n - 2] == '/';
}

static void GetTagName(const char* tag, char* name, size_t len)
{
    if (name == NULL || len == 0) {
        return;
    }

    const char* p = tag + 1;
    if (*p == '/') {
        p++;
    }
    const char* start = p;
    while (*p != '\0' && *p != ' ' && *p != '>' && *p != '/') {
        p++;
    }
    size_t n = (size_t)(p - start);
    if (n >= len) {
        n = len - 1;
    }
    if (strncpy_s(name, len, start, n) != 0) {
        name[0] = '\0';
        return;
    }
    name[n] = '\0';
}

static void ExtractAttrs(const char* tag, TagEntry* e)
{
    int ac = e->attrCount;
    const char* scan = tag;

    while (ac < MAX_ATTRS_PER_TAG) {
        scan = strchr(scan, ' ');
        if (scan == NULL) {
            break;
        }
        scan++;
        if (*scan == '\0') {
            break;
        }
        const char* eq = strchr(scan, '=');
        if (eq == NULL || eq <= scan) {
            break;
        }
        size_t nlen = (size_t)(eq - scan);
        if (nlen >= MAX_ATTR_NAME_LEN) {
            nlen = MAX_ATTR_NAME_LEN - 1;
        }
        (void)strncpy_s(e->attrs[ac].name, sizeof(e->attrs[ac].name), scan, nlen);

        const char* vq = strchr(eq, '\"');
        if (vq == NULL) {
            break;
        }
        vq++;
        const char* ve = strchr(vq, '\"');
        if (ve == NULL) {
            break;
        }
        size_t vlen = (size_t)(ve - vq);
        if (vlen >= MAX_ATTR_VAL_LEN) {
            vlen = MAX_ATTR_VAL_LEN - 1;
        }
        (void)strncpy_s(e->attrs[ac].value, sizeof(e->attrs[ac].value), vq, vlen);

        ac++;
        scan = ve + 1;
    }
    e->attrCount = ac;
}

/* ─── 内部辅助：处理一个开标签 ─── */
/* 返回 tagEnd+1（下一位置），通过指针更新 cnt / dc */
static char* HandleOpenTag(char* t, char* tagEnd, TagEntry* tags, unsigned int maxTags, unsigned int* cnt, int* dc)
{
    char name[MAX_TAG_NAME_LEN];
    GetTagName(t, name, sizeof(name));
    if (name[0] == '\0') {
        return tagEnd + 1;
    }

    bool selfClose = IsSelfClose(t, tagEnd);
    if (*cnt < maxTags) {
        TagEntry* e = &tags[*cnt];
        (void)memset_s(e, sizeof(*e), 0, sizeof(*e));
        if (strcpy_s(e->tagName, sizeof(e->tagName), name) != 0) {
            TOPO_ERR("HandleOpenTag: strcpy_s failed, name=%s", name);
            return NULL;
        }
        e->isSelfClose = selfClose;
        e->depth = *dc;
        ExtractAttrs(t, e);
        (*cnt)++;
    }

    if (!selfClose) {
        (*dc)++;
    }
    return tagEnd + 1;
}

/* ─── 逐行解析 XML，识别标签 ─── */
/* 逐行解析 XML，识别标签 */
static TopoAddrResult ParseXmlLines(FILE* fp, TagEntry* tags, unsigned int maxTags, unsigned int* cnt)
{
    int dc = 0;
    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char* t = line;
        while (*t != '\0') {
            while (*t != '\0' && isspace((unsigned char)*t)) {
                t++;
            }
            if (*t == '\0' || strncmp(t, "<!--", 4) == 0) {
                break;
            }

            char* tagEnd = strchr(t, '>');
            if (tagEnd == NULL) {
                break;
            }

            if (t[0] == '<' && t[1] == '/') {
                if (dc > 0) {
                    dc--;
                }
                t = tagEnd + 1;
                continue;
            }

            if (t[0] == '<') {
                t = HandleOpenTag(t, tagEnd, tags, maxTags, cnt, &dc);
                if (t == NULL) {
                    return TOPO_ERR_INTERNAL;
                }
                continue;
            }
            break;
        }
    }
    return TOPO_SUCCESS;
}

/* ─── 核心解析函数 ─── */

TopoAddrResult ParseXmlTags(const char* xmlPath, TagEntry* tags, unsigned int* tagCount, unsigned int maxTags)
{
    *tagCount = 0;

    FILE* fp = fopen(xmlPath, "rb");
    if (fp == NULL) {
        TOPO_INFO("ParseXmlTags: cannot open %s", xmlPath);
        return TOPO_ERR_OPEN_FILE;
    }

    struct stat st;
    if (fstat(fileno(fp), &st) != 0 || st.st_size == 0) {
        TOPO_ERR("ParseXmlTags: stat failed or empty file, path=%s", xmlPath);
        fclose(fp);
        return TOPO_ERR_OPEN_FILE;
    }

    unsigned int cnt = 0;
    if (ParseXmlLines(fp, tags, maxTags, &cnt) != TOPO_SUCCESS) {
        fclose(fp);
        return TOPO_ERR_INTERNAL;
    }

    fclose(fp);
    *tagCount = cnt;
    if (cnt == 0) {
        TOPO_ERR("ParseXmlTags: no valid tags found in %s", xmlPath);
        return TOPO_ERR_NOT_FOUND;
    }
    return TOPO_SUCCESS;
}
