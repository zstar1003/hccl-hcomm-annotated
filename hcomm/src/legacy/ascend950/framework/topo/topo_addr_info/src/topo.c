/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <regex.h>
#include "securec.h"
#include "hal.h"

#define MAX_TOPO_FILE_SIZE (40960)

/**
 * @brief 用正则表达式解析JSON文件中的"topo_file_path"字段值（无goto，纯C实现）
 * @param json_file_path 输入的JSON文件路径
 * @return 成功返回动态分配的字符串（需调用者手动free），失败返回NULL
 */
int GetTopoFilePathFromFile(const char* filePath, char* topoFilePath, size_t bufSize)
{
    // 初始化所有资源为NULL/初始值，便于后续释放判断
    regex_t regex;
    regmatch_t match[2]; // match[0]整体匹配，match[1]捕获路径值

    // 1. 打开JSON文件
    FILE* fp = fopen(filePath, "rb");
    if (fp == NULL) {
        return -1;
    }

    struct stat st;
    fstat(fileno(fp), &st);
    size_t fileSize = st.st_size;
    // 避免消耗过大内存
    if (fileSize > MAX_TOPO_FILE_SIZE) {
        fclose(fp);
        return -1;
    }
    // 3. 分配内存并读取文件全部内容
    char* fileBuf = (char*)malloc(fileSize + 1);
    if (fileBuf == NULL) {
        fclose(fp);
        return -1;
    }
    if (memset_s(fileBuf, fileSize + 1, 0, fileSize + 1) != EOK) {
        free(fileBuf);
        fclose(fp);
        return -1;
    }
    size_t readBytes = fread(fileBuf, 1, fileSize, fp);
    if (readBytes != fileSize) {
        free(fileBuf); // 释放已分配的内存
        fclose(fp);    // 释放文件
        return -1;
    }
    fclose(fp); // 文件内容已读取，提前释放文件资源

    // 4. 编译正则表达式
    const char* pattern = "\"topo_file_path\"\\s*:\\s*\"([^\"]*)\"";
    if (regcomp(&regex, pattern, REG_EXTENDED) != 0) {
        free(fileBuf);
        return -1;
    }

    // 5. 执行正则匹配
    if (regexec(&regex, fileBuf, 2, match, 0) != 0) {
        regfree(&regex); // 释放正则编译资源
        free(fileBuf);
        return -1;
    }

    // 6. 提取匹配到的路径值
    int start = match[1].rm_so;
    int len = match[1].rm_eo - match[1].rm_so;
    int ret = strncpy_s(topoFilePath, bufSize, fileBuf + start, len);
    topoFilePath[len] = '\0'; // 确保字符串以'\0'结尾
    // 7. 释放临时资源（仅保留result作为返回值）
    regfree(&regex);
    free(fileBuf);
    return ret;
}

typedef struct closPorts {
    unsigned int mainboardId;
    int dieId;
    int ports[32];
    int portCnt;
} ClosPortMap;

/**
 * 固定写死每种产品类型的端口
 */
static ClosPortMap closPortMap[2] = {
    {MAIN_BOARD_ID_SERVER_TYPE1, 0, {4, 5, 6, 7}, 4},
    {MAIN_BOARD_ID_SERVER_TYPE1, 1, {5, 6}, 2},
};

int TopoGetClosPort(unsigned int mainboardId, int dieId, int* ports, int* portCnt)
{
    int size = sizeof(closPortMap) / sizeof(ClosPortMap);
    for (int i = 0; i < size; ++i) {
        if (mainboardId == closPortMap[i].mainboardId && dieId == closPortMap[i].dieId) {
            if (memcpy_s(ports, (*portCnt), closPortMap[i].ports, closPortMap[i].portCnt) != EOK) {
                return -1;
            }
            *portCnt = closPortMap[i].portCnt;
            return 0;
        }
    }
    return 0;
}

#define MAX_DRIVER_INSTALL_PATH (128)

int TopoGetFilePath(unsigned mainboard_id, unsigned int spod_type, char* buf_size, size_t buf_len)
{
    char driver_install_path[MAX_DRIVER_INSTALL_PATH] = {0};
    if (0 != hal_get_driver_install_path(driver_install_path, MAX_DRIVER_INSTALL_PATH)) {
        return -1;
    }
    int ret = -1;
    switch (mainboard_id) {
        case MAIN_BOARD_ID_CARD_NOMESH:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_350_1.json");
            break;
        case MAIN_BOARD_ID_CARD_2PMESH:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_350_2.json");
            break;
        case MAIN_BOARD_ID_CARD_4PMESH:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_350_3.json");
            break;
        case MAIN_BOARD_ID_POD:
        case MAIN_BOARD_ID_POD_2D:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_950_1.json");
            break;
        case MAIN_BOARD_ID_POD_FLEX:
        case MAIN_BOARD_ID_POD_FLEX_RTP:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_950_2.json");
            break;
        case MAIN_BOARD_ID_SERVER_TYPE1:
        case MAIN_BOARD_ID_SERVER_8PMESH:
        case MAIN_BOARD_ID_SERVER_8PMESH_UBOE:
        case MAIN_BOARD_ID_SERVER_8PMESH_NOSP:
        case MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE:
            if (spod_type == TOPO_TYPE_SERVER_16FM) {
                ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_850_2.json");
            } else {
                ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_850_1.json");
            }
            break;
        case MAIN_BOARD_ID_SERVER_350L:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_850_3.json");
            break;
        case MAIN_BOARD_ID_SERVER_550EL_100:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_550EL_100.json");
            break;
        case MAIN_BOARD_ID_SERVER_550EL_200:
            ret = sprintf_s(buf_size, buf_len, "%s/%s", driver_install_path, "driver/topo/950/atlas_550EL_200.json");
            break;
        default:
            break;
    }
    if (ret < 0) {
        return -1;
    }
    return 0;
}
