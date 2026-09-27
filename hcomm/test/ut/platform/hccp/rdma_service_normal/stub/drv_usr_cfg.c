/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <securec.h>

#define TLS_MAGIC_WORDS_LEN 8
#define RS_SALT_MAX_LEN 48
#define RS_CERT_COUNT 15
#define TLS_SALT_LEN 48
#define IV_LEN 16
#define BLOCK_KY_LEN 32
#define PWD_MIN_LEN 8
#define PWD_MAX_LEN 15
#define FGETS_MAX_LEN 32
#define PWD_ENC_LEN 256
#define WORK_KEY_LEN 516
#define TAG_LEN 16
#define MAX_CERT_COUNT 15
#define RS_MAGIC_WORDS "1234567"

struct cert_infos {
    char cert_info[2048];
};

struct certs {
    struct cert_infos certs[RS_CERT_COUNT];
};

#define TLS_RES_LEN 1024
struct rs_cert_manage_info {
    char magic_words[TLS_MAGIC_WORDS_LEN]; /* 1234567 */
    unsigned int cert_count;               /* num of certs */
    int state;                             /* 0:not ok 1:ok */
    unsigned int ca_wcout;                 /* counts of ca writing flash */
    unsigned int cert_ky_wcout;            /* counts of eqpt and key writing flash */
    unsigned int crl_wcout;                /* counts of crl writing flash */
    unsigned int crl_len;                  /* len of crl */
    unsigned int ky_len;                   /* len of key */
    unsigned int ky_enc_len;               /* len of enc key */
    unsigned char salt[TLS_SALT_LEN];      /* salt */
    unsigned int salt_size;                /* len of salt */
    unsigned int cert_len[MAX_CERT_COUNT];
    unsigned int total_cert_len; /* not include head only len of certs */
    unsigned int tls_enable;
    unsigned int tls_alarm;
    unsigned int pwd_len;     /* len of pwd */
    unsigned int pwd_enc_len; /* len of enc pwd */
    unsigned char enc_pwd[PWD_ENC_LEN];
    unsigned int work_key_len; /* len of work_key */
    unsigned char work_key[WORK_KEY_LEN];
    unsigned char iv[IV_LEN]; /* initial vector */
    unsigned int iv_size;     /* len of initial vector */
    unsigned char tag[TAG_LEN];
    unsigned int tag_len;
    unsigned int save_mode;
    char res[TLS_RES_LEN];
};

int dev_read_flash(unsigned int dev_id, const char* name, unsigned char* buf, unsigned int* buf_size)
{
    int ret;
    if (strcmp(name, "hccp_certs_mng_cb") == 0) {
        struct rs_cert_manage_info* mng_infos = (struct rs_cert_manage_info*)buf;
        mng_infos->cert_count = 0;
        mng_infos->total_cert_len = 0;
        mng_infos->ky_len = 0;
        mng_infos->ky_enc_len = 0;
        mng_infos->tls_enable = 0;
        mng_infos->pwd_len = PWD_MAX_LEN;
        mng_infos->pwd_enc_len = PWD_ENC_LEN;
        mng_infos->work_key_len = WORK_KEY_LEN;
        mng_infos->salt_size = TLS_SALT_LEN;
        mng_infos->iv_size = IV_LEN;
        mng_infos->tag_len = TAG_LEN;
        mng_infos->save_mode = 0;
        ret = memcpy_s(mng_infos->magic_words, sizeof(mng_infos->magic_words), "1234567", sizeof("1234567"));
        mng_infos->salt_size = 7;
        return 0;
    } else if (strcmp(name, "hccp_certs_eqpt_cb") == 0) {
        return -1;
    } else if (strcmp(name, "hccp_pri_data_cb") == 0) {
        *buf_size = 5120;
        return -1;
    } else if (strcmp(name, "hccp_certs_revoc_cb") == 0) {
        *buf_size = 40960;
        return -1;
    } else {
        return -1;
    }
}

int tls_get_user_config(
    unsigned int save_mode, unsigned int chipId, const char* name, unsigned char* buf, unsigned int* buf_size)
{
    int ret;

    ret = dev_read_flash(chipId, name, buf, buf_size);

    return ret;
}

void tls_get_enable_info(unsigned int save_mode, unsigned int chipId, unsigned char* buf, unsigned int buf_size)
{
    return;
}

int get_saved_tls_config_file_path(char* path, unsigned int path_len, const char* name) { return 0; }

int ReadFileToBuf(const char* path, char* content, int* len) { return 0; }

int NetCommGetSelfHome(char* userNamePath, unsigned int pathLen)
{
    memcpy(userNamePath, "/tmp", strlen("/tmp"));
    return 0;
}

int get_tls_config_path(char* user_name_path, unsigned int path_len) { return 0; }

int NetGetGatewayAddress(
    unsigned int chipId, const char* inbuf, unsigned int size_in, char* outbuf, unsigned int* size_out)
{
    return 0;
}

int FileReadCfg(const char* filePath, int devId, const char* confName, char* confValue, unsigned int len)
{
    if (strncmp(confName, "udp_port_mode", strlen("udp_port_mode") + 1) == 0) {
        memcpy_s(confValue, len, "nslb_dp", strlen("nslb_dp"));
    } else {
        memcpy_s(confValue, len, "16666", strlen("16666"));
    }
    return 0;
}
