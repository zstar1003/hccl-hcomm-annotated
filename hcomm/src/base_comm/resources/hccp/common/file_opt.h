/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _COMMON_FILE_OPT_H_
#define _COMMON_FILE_OPT_H_

#include <unistd.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/file.h>

#define CONF_NAME_MAX_LEN 64
#define CONF_VALUE_MAX_LEN 2048
#define CONLINE_LEN (CONF_NAME_MAX_LEN + CONF_VALUE_MAX_LEN + 1)

#define FILE_EXIST 0

#define FILE_CHECK_RET_WITHOUT_RETURN(ret, fmt, val...)                                                                \
    do {                                                                                                               \
        if (ret) {                                                                                                     \
            roce_warn(fmt, ##val);                                                                                     \
        }                                                                                                              \
    } while (0)

#define FILE_CHECK_PTR_VALID_RETURN_VAL(p, ret)                                                                        \
    do {                                                                                                               \
        if ((p) == NULL) {                                                                                             \
            roce_err("ptr is NULL!");                                                                                  \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

enum {
    FILE_OPT_ERR = 0x3000,
    FILE_OPT_NO_MEM_ERR,
    FILE_OPT_INNER_PARAM_ERR,
    FILE_OPT_SYS_FOPEN_ERR,
    FILE_OPT_SYS_WRITE_FILE_ERR,
    FILE_OPT_SYS_READ_FILE_ERR,
    FILE_OPT_SYS_RD_FILE_NOT_FOUND,
    FILE_OPT_SYS_DELETE_FILE_ERR,
    FILE_OPT_SYS_TIME_OP_ERR,
    FILE_OPT_SYS_CERT_EXPRD_ERR,
    FILE_OPT_SYS_TERMIOS_ERR,
    FILE_OPT_SYS_BUSY_ERR,
    FILE_OPT_SYS_NOT_ACCESS,
    FILE_OPT_OP_NOT_SUPPORT_IPV6_ERR,
    FILE_OPT_OP_NOT_SUPPORT_BOND_ERR,
};

int ReadFileToBuf(const char *path, char *content, unsigned int *contentLen);
int WriteBufToFile(const unsigned char *buf, unsigned int bufLen, const char *file, mode_t mode);
void RemoveFile(const char *file);
int GetFileLock(const char *path, int *lockFd);
int ReleaseFileLock(int lockFd, const char *lockFilePath);
int CheckFilePath(const char *path, mode_t mode);
int FileReadCfg(const char *filePath, int devId, const char *confName, char *confValue, unsigned int len);

#endif
