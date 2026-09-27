/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NSLB_MD5_H
#define HCCL_NSLB_MD5_H

#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include "hccl_nslbdp_pub.h"

namespace hccl {
constexpr u32 NSLB_MD5_TOTAL = 64;
constexpr u32 NSLB_MD5_RESERVE = 32;
constexpr u32 NSLB_MD5_DIGEST = 16;
constexpr u32 NSLB_MD5ENCODE_COUNT = 8;
constexpr u32 NSLB_MD5_120 = 120;
constexpr u32 NSLB_MD5_56 = 56;

constexpr u32 NSLB_MD5_4 = 4;
constexpr u32 NSLB_MD5_5 = 5;
constexpr u32 NSLB_MD5_6 = 6;
constexpr u32 NSLB_MD5_7 = 7;
constexpr u32 NSLB_MD5_8 = 8;
constexpr u32 NSLB_MD5_9 = 9;
constexpr u32 NSLB_MD5_10 = 10;
constexpr u32 NSLB_MD5_11 = 11;
constexpr u32 NSLB_MD5_12 = 12;
constexpr u32 NSLB_MD5_13 = 13;
constexpr u32 NSLB_MD5_14 = 14;
constexpr u32 NSLB_MD5_15 = 15;
constexpr u32 NSLB_MD5_16 = 16;
constexpr u32 NSLB_MD5_17 = 17;
constexpr u32 NSLB_MD5_18 = 18;
constexpr u32 NSLB_MD5_19 = 19;
constexpr u32 NSLB_MD5_20 = 20;
constexpr u32 NSLB_MD5_21 = 21;
constexpr u32 NSLB_MD5_22 = 22;
constexpr u32 NSLB_MD5_23 = 23;
constexpr u32 NSLB_MD5_24 = 24;
constexpr u32 NSLB_MD5_25 = 25;

constexpr u32 NSLB_MD5_HASH1 = 1;
constexpr u32 NSLB_MD5_HASH2 = 2;
constexpr u32 NSLB_MD5_HASH3 = 3;
constexpr u32 NSLB_MD5_HASH4 = 4;
constexpr u32 NSLB_MD5_HASH5 = 5;
constexpr u32 NSLB_MD5_HASH6 = 6;
constexpr u32 NSLB_MD5_HASH7 = 7;
constexpr u32 NSLB_MD5_HASH8 = 8;
constexpr u32 NSLB_MD5_HASH9 = 9;
constexpr u32 NSLB_MD5_HASH10 = 10;
constexpr u32 NSLB_MD5_HASH11 = 11;
constexpr u32 NSLB_MD5_HASH12 = 12;
constexpr u32 NSLB_MD5_HASH13 = 13;
constexpr u32 NSLB_MD5_HASH14 = 14;
constexpr u32 NSLB_MD5_HASH15 = 15;

constexpr u32 NSLB_MD5_STATE1 = 1;
constexpr u32 NSLB_MD5_STATE2 = 2;
constexpr u32 NSLB_MD5_STATE3 = 3;
constexpr u32 NSLB_MD5_STATE4 = 4;

constexpr u32 NSLB_MD5_COUNTLEN3 = 3;
constexpr u32 NSLB_MD5_COUNTLEN29 = 29;

class NSLBMD5 {
public:
    NSLBMD5();
    explicit NSLBMD5(const std::string& text);
    void update(const unsigned char* input, size_t length);
    void update(const char* input, size_t length);
    NSLBMD5& finalize();
    std::string hexdigest() const;
    friend std::ostream& operator<<(std::ostream& out, NSLBMD5 md5);

    std::string md5(const std::string str);

    // 计算NslbDpRankInfo结构体向量的MD5
    static void calculateRankInfoMd5(const std::vector<NslbDpRankInfo>& rankInfo, uint8_t commMd5Sum[16]);

    // 计算TableFourRankInfo结构体向量的MD5
    static void calculateTableFourRankInfoMd5(const std::vector<TableFourRankInfo>& rankInfo, uint8_t commMd5Sum[16]);

    // 将MD5值转换为字符串
    static std::string md5ToString(const uint8_t md5[16]);

    void init();
    void transform(const unsigned char block[64]);
    static void decode(uint32_t output[], const unsigned char input[], size_t len);
    static void encode(unsigned char output[], const uint32_t input[], size_t len);

    bool finalized;
    unsigned char buffer[64]; // 缓冲区
    uint32_t count[2];        // 以位为单位的消息长度 (低, 高)
    uint32_t state[4];        // 状态 (ABCD)
    unsigned char digest[16]; // 存储计算结果

    // 填充字节
    static const unsigned char PADDING[64];
    static const char HEX[16];
};

} // namespace hccl

#endif /* HCCL_NSLB_MD5_H */
