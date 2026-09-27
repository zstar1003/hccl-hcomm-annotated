/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

int kmc_enc(char* secu_path, char* store_path, char* pln, unsigned int pln_len) { return 0; }

int kmc_dec(char* secu_path, char* store_path, void* pln, unsigned int pln_len) { return 0; }

int kmc_dec_data(struct kmc_enc_info* enc_info, unsigned char* outbuf, unsigned int* size_out) { return 0; }

int kmc_enc_data(unsigned char* inbuf, unsigned int size_in, struct kmc_enc_info* enc_info) { return 0; }
