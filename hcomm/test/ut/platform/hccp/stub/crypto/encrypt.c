/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "encrypt.h"

int do_crypt(crypt_info* input, unsigned char* outbuf, unsigned int* outlen) { return 1; }

int crypto_gen_key_with_pbkdf2(hash_info* input, unsigned char* key, unsigned int keylen) { return 0; }

int crypto_encrypt_with_aes_gcm(crypt_info* info, unsigned char* outbuf, unsigned int* out_len) { return 0; }

int crypto_decrypt_with_aes_gcm(crypt_info* info, unsigned char* outbuf, unsigned int* out_len)
{
    *out_len = 512;
    return 0;
}
