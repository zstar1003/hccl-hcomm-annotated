/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_opt.h"
#include <limits.h>
#include <errno.h>
#include <unistd.h>
#include "securec.h"
#include "user_log.h"

#ifndef CRYPT_LLT
#define STATIC static
#else
#define STATIC
#endif

STATIC int ReadFileToBufCheckParam(const char *path, const char *content, const unsigned int *contentLen)
{
    if (path == NULL || content == NULL || contentLen == NULL) {
        roce_err("path or conf_path or content_len is NULL, invalid");
        return -EINVAL;
    }

    if (strlen(path) > PATH_MAX) {
        roce_err("path_len[%lu] > [%d](PATH_MAX)", strlen(path), PATH_MAX);
        return -EFAULT;
    }

    return 0;
}

int ReadFileToBuf(const char *path, char *content, unsigned int *contentLen)
{
    size_t len;
    long tempLen;
    int ret, retVal;
    FILE *readFile = NULL;
    char realConfPath[PATH_MAX + 1] = {0}; // lint !e813

    ret = ReadFileToBufCheckParam(path, content, contentLen);
    if (ret) {
        roce_err("read_file_to_buf_check_param failed, ret[%d]", ret);
        return ret;
    }

    if (realpath(path, realConfPath) == NULL) {
        ret = -errno;
        if (ret != -ENOENT) {
            roce_err("conf_path[%s] is invalid, err[%d]", path, ret);
        }
        return ret;
    }

    readFile = fopen(realConfPath, "r");
    if (readFile == NULL) {
        roce_err("read_file is NULL, invalid");
        return -EINVAL;
    }

    ret = fseek(readFile, 0, SEEK_END);
    if (ret < 0) {
        roce_err("fseek failed with error:%d", errno);
        goto out;
    }

    tempLen = ftell(readFile);
    if (tempLen <= 0 || tempLen > (int)*contentLen || tempLen > INT_MAX) {
        ret = -EINVAL;
        roce_err("ftell failed with error:%d, tempLen=%ld, contentLen[%u]", errno, tempLen, *contentLen);
        goto out;
    }

    len = (size_t)tempLen;

    rewind(readFile);
    ret = (int)fread((void *)content, len, 1, readFile); /* read a buf which size is len */
    if (ret != 1) {
        roce_err("fread failed ret:%d, error:%d, ferror(fp):%d", ret, errno, ferror(readFile));
        ret = -EINVAL;
        goto out;
    }
    ret = 0;
    *contentLen = (unsigned int)len;
out:
    retVal = fclose(readFile);
    if (retVal) {
        roce_warn("fclose unsuccessful, retVal:%d, errno:%d", retVal, errno);
    }

    return ret;
}

STATIC void CheckFile(const char *file)
{
    int ret;

    if (access(file, F_OK) == FILE_EXIST) {
        ret = remove(file);
        roce_run_info("file %s exist, should remove it, ret:%d, errno:%d", file, ret, errno);
    } else {
        ret = errno;
        if (ret != ENOENT) {
            roce_run_info("access file %s invalid, errno:%d", file, ret);
        }
    }
}

int WriteBufToFile(const unsigned char *buf, unsigned int bufLen, const char *file, mode_t mode)
{
    int ret, retVal;
    int fd = -1;

    if (buf == NULL || file == NULL) {
        roce_err("buf or file is NULL, invalid!");
        return -EINVAL;
    }

    CheckFile(file);
    fd = creat(file, mode);
    if (fd < 0) {
        ret = -errno;
        roce_err("create file %s failed errno %d", file, ret);
        return ret;
    }

    ret = (int)write(fd, buf, bufLen);
    if (ret != (int)bufLen) {
        roce_err("write file %s failed %d, bufLen:%u", file, ret, bufLen);
        ret = -errno;
        do {
            retVal = close(fd);
        } while ((retVal < 0) && (errno == EINTR));
        RemoveFile(file);
        return ret;
    } else {
        ret = 0;
    }

    do {
        retVal = close(fd);
    } while ((retVal < 0) && (errno == EINTR));

    fd = -1;

    ret = chmod(file, mode);
    if (ret != 0) {
        roce_err("file[%s] chmod failed, errno: %d.", file, ret);
        RemoveFile(file);
    }
    return ret;
}

void RemoveFile(const char *file)
{
    int ret, err;

    if (file == NULL) {
        roce_err("file is NULL, invalid");
        return;
    }

    ret = remove(file);
    if (ret < 0) {
        err = -errno;
        if (err != -ENOENT) {
            roce_err("remove end file failed %d", err);
            return;
        }
    }
    return;
}

int CheckFilePath(const char *path, mode_t mode)
{
    int ret;
    char realConfPath[PATH_MAX + 1] = {0}; // lint !e813

    if (path == NULL) {
        roce_err("path is NULL");
        return -EINVAL;
    }

    if (strlen(path) > PATH_MAX) {
        roce_err("path_len[%lu] > [%d](PATH_MAX)", strlen(path), PATH_MAX);
        return -EINVAL;
    }

    if (realpath(path, realConfPath) == NULL) {
        ret = -errno;
        if (ret == -ENOENT) {
            roce_warn("path[%s] does not exist, real_path[%s]", path, realConfPath);
            ret = mkdir(realConfPath, mode);
            if (ret) {
                roce_err("mkdir real_conf_path[%s] failed, ret[%d]", realConfPath, ret);
                return ret;
            }
            ret = chmod(realConfPath, mode);
            if (ret != 0) {
                roce_err("file[%s] chmod failed, errno: %d.", realConfPath, ret);
            }
            return ret;
        }

        roce_err("path[%s] is invalid, err[%d]", path, ret);
        return ret;
    }

    return 0;
}

static void CloseFdSecurity(int fd)
{
    int ret;
    int errNo = -1;

    do {
        ret = close(fd);
        if (ret < 0) {
            errNo = errno;
            if (errNo != EINTR) {
                roce_err("close fd[%d] failed, ret:%d, errNo[%d]", fd, ret, errNo);
                return;
            }
        }
    } while ((ret < 0) && (errNo == EINTR));

    return;
}

int GetFileLock(const char *path, int *lockFd)
{
    int ret, err;
    int fd = -1;

    if (lockFd == NULL || path == NULL) {
        roce_err("lock_fd is NULL or path is NULL.");
        return -EINVAL;
    }

    if (strlen(path) > PATH_MAX) {
        roce_err("path_len[%lu] > [%d](PATH_MAX)", strlen(path), PATH_MAX);
        return -EINVAL;
    }

    fd = open(path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        ret = -errno;
        roce_err("open real_path[%s] failed, ret[%d]", path, ret);
        return ret;
    }

    ret = flock(fd, LOCK_EX | LOCK_NB);
    if (ret) {
        err = -errno;
        CloseFdSecurity(fd);
        if (err == -EWOULDBLOCK) {
            return -EBUSY;
        }
        roce_err("lock fd[%d] failed, err[%d]", fd, err);
        return err;
    }

    *lockFd = fd;
    return 0;
}

int ReleaseFileLock(int lockFd, const char *lockFilePath)
{
    int ret, err;

    if (lockFd < 0) {
        roce_err("invalid lock_fd[%d]", lockFd);
        return -EINVAL;
    }

    ret = flock(lockFd, LOCK_UN);
    if (ret) {
        err = -errno;
        roce_err("release lock fd[%d] failed, err[%d]", lockFd, err);
        CloseFdSecurity(lockFd);
        return err;
    }

    CloseFdSecurity(lockFd);

    RemoveFile(lockFilePath);
    return 0;
}

STATIC int CfgInnerReadConfByfd(FILE *fp, const char *confName, char *confValue, unsigned int len)
{
    int ret = FILE_OPT_SYS_RD_FILE_NOT_FOUND;
    char *lineBuf = NULL;
    unsigned long lenBuf;
    unsigned int lenTmp;
    char *c = NULL;

    lineBuf = calloc(CONLINE_LEN, sizeof(char));
    if (lineBuf == NULL) {
        roce_err("calloc line_buf failed");
        return FILE_OPT_NO_MEM_ERR;
    }

    while (feof(fp) == 0 && fgets(lineBuf, CONLINE_LEN, fp) != NULL) {
        if ((lineBuf[0] == '#') || (strlen(lineBuf) < strlen("*=*"))) {
            continue;
        }
        c = (char *)strchr(lineBuf, '=');
        if (c == NULL) {
            continue;
        }

        lenBuf = strlen(lineBuf) - 1;
        if ((lenBuf < CONLINE_LEN) && (lineBuf[lenBuf] == '\n')) {
            lineBuf[lenBuf] = '\0';
        }

        lenTmp = (unsigned int)(c - lineBuf);
        if ((strncmp(lineBuf, confName, strlen(confName)) == 0) && (lenTmp == strlen(confName))) {
            ++c;
            ret = strcpy_s(confValue, len, c);
            if (ret) {
                roce_err("strcpy_s err[%d], len:%u", ret, len);
                ret = FILE_OPT_NO_MEM_ERR;
            }

            goto out;
        }
    }

out:
    free(lineBuf);
    lineBuf = NULL;
    return ret;
}

STATIC int CfgInnerReadConf(const char *confPath, const char *confName, char *confValue, unsigned int len)
{
    char realConfPath[PATH_MAX + 1] = {0}; // lint !e813
    int ret, retVal;
    FILE *fp = NULL;

    // file not exist, degrade log level
    if ((strlen(confPath) > PATH_MAX) || (realpath(confPath, realConfPath) == NULL)) {
        roce_warn("read path_len[%u] > PATH_MAX[%d] or conf_path is invalid, errno[%d]", strlen(confPath), PATH_MAX,
            errno);
        return FILE_OPT_INNER_PARAM_ERR;
    }

    fp = fopen(realConfPath, "r");
    if (fp == NULL) {
        roce_err("Open configure file failed errno[%d] real_conf_path[%s]", errno, realConfPath);
        return FILE_OPT_SYS_READ_FILE_ERR;
    }

    ret = flock(fileno(fp), LOCK_EX);
    if (ret) {
        roce_err("hccn.conf lock fd[%d] failed! ret[%d] errno[%d]", fileno(fp), ret, errno);
        ret = FILE_OPT_SYS_READ_FILE_ERR;
        goto out;
    }

    ret = fseek(fp, 0, SEEK_SET);
    if (ret) {
        roce_err("hccn.conf fseek fd[%d] failed! ret[%d] errno[%d]", fileno(fp), ret, errno);
        ret = FILE_OPT_SYS_READ_FILE_ERR;
        goto out;
    }

    ret = CfgInnerReadConfByfd(fp, confName, confValue, len);
out:
    retVal = fclose(fp);
    FILE_CHECK_RET_WITHOUT_RETURN(retVal, "fclose unsuccessful, retVal:%d, errno:%d", retVal, errno);
    fp = NULL;
    return ret;
}

int FileReadCfg(const char *filePath, int devId, const char *confName, char *confValue, unsigned int len)
{
    char conf[CONLINE_LEN] = {0};
    int ret;

    FILE_CHECK_PTR_VALID_RETURN_VAL(filePath, -EINVAL);
    FILE_CHECK_PTR_VALID_RETURN_VAL(confName, -EINVAL);
    FILE_CHECK_PTR_VALID_RETURN_VAL(confValue, -EINVAL);

    ret = sprintf_s(conf, CONLINE_LEN, "%s_%d", confName, devId);
    if (ret <= 0) {
        roce_err("conf str op failed! ret[%d] conf_name[%s] dev_id[%d]", ret, confName, devId);
        return -EINVAL;
    }

    ret = CfgInnerReadConf(filePath, conf, confValue, len);
    if (ret == FILE_OPT_SYS_RD_FILE_NOT_FOUND) {
        roce_info("the conf[%s] not found", conf);
        return ret;
    } else if (ret == FILE_OPT_INNER_PARAM_ERR) {
        roce_warn("cfg_inner_read_conf unsuccessful ret[%d]", ret);
        return ret;
    } else if (ret != 0) {
        roce_err("cfg_inner_read_conf failed! ret[%d] conf[%s] conf_value[%s]", ret, conf, confValue);
        return ret;
    }

    roce_info("read conf[%s] realconf[%s] val[%s]", conf, confName, confValue);

    return 0;
}
