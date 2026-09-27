/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "hccl_nslb_md5.h"

namespace hccl {

// 初始化常量
const unsigned char NSLBMD5::PADDING[64]
    = {0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const char NSLBMD5::HEX[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

// F, G, H, I 是4个基本MD5函数
inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }

inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }

inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }

inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }

// 左循环移位操作
inline uint32_t ROTATE_LEFT(uint32_t x, int n) { return (x << n) | (x >> (NSLB_MD5_RESERVE - n)); }

// FF, GG, HH, II 是四轮变换中的基本操作
inline void FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a += F(b, c, d) + x + ac;
    a = ROTATE_LEFT(a, s);
    a += b;
}

inline void GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a += G(b, c, d) + x + ac;
    a = ROTATE_LEFT(a, s);
    a += b;
}

inline void HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a += H(b, c, d) + x + ac;
    a = ROTATE_LEFT(a, s);
    a += b;
}

inline void II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a += I(b, c, d) + x + ac;
    a = ROTATE_LEFT(a, s);
    a += b;
}

void NSLBMD5::init()
{
    finalized = false;

    // 初始化状态为标准值
    count[0] = count[1] = 0;
    // 初始化MD5缓冲区
    state[0] = 0x67452301;
    state[NSLB_MD5_STATE1] = 0xEFCDAB89;
    state[NSLB_MD5_STATE2] = 0x98BADCFE;
    state[NSLB_MD5_STATE3] = 0x10325476;
}

NSLBMD5::NSLBMD5() { init(); }

NSLBMD5::NSLBMD5(const std::string& text)
{
    init();
    update(text.c_str(), text.length());
    finalize();
}

// 解码字节数组为32位整数数组 - 改进实现，避免使用强制类型转换
void NSLBMD5::decode(uint32_t output[], const uint8_t input[], size_t len)
{
    // 使用std::memcpy安全地复制内存，避免强制类型转换
    size_t num_ints = len / NSLB_MD5_4;
    for (size_t i = 0; i < num_ints; ++i) {
        (void)memcpy_s(&output[i], sizeof(uint32_t), &input[i * NSLB_MD5_4], sizeof(uint32_t));
    }
}

// 编码32位整数数组为字节数组
void NSLBMD5::encode(unsigned char output[], const uint32_t input[], size_t len)
{
    for (size_t i = 0, j = 0; j < len; i++, j += NSLB_MD5_4) {
        output[j] = input[i] & 0xff;
        output[j + NSLB_MD5_STATE1] = (input[i] >> NSLB_MD5_8) & 0xff;
        output[j + NSLB_MD5_STATE2] = (input[i] >> NSLB_MD5_16) & 0xff;
        output[j + NSLB_MD5_STATE3] = (input[i] >> NSLB_MD5_24) & 0xff;
    }
}

/**
 * 背景：标准的MD5算法，主要功能是64字节处理
 * 参考文献：RFC 1321: The MD5 Message-Digest Algorithm
 * 作者：Ron Rivest
 * 发布日期：1992年4月
 */
void NSLBMD5::update(const unsigned char input[], size_t length)
{
    // 计算还需要填充多少字节才能达到64字节的倍数
    unsigned int index = count[0] / NSLB_MD5ENCODE_COUNT % NSLB_MD5_TOTAL;

    // 更新消息长度
    if ((count[0] += (length << NSLB_MD5_COUNTLEN3)) < (length << NSLB_MD5_COUNTLEN3)) {
        count[1]++;
    }
    count[1] += (length >> NSLB_MD5_COUNTLEN29);

    // 计算需要处理多少个完整的64字节块
    unsigned int firstpart = NSLB_MD5_TOTAL - index;
    unsigned int i;

    // 处理消息的第一部分（如果有）
    if (length >= firstpart) {
        (void)memcpy_s(&buffer[index], firstpart, input, firstpart);
        transform(buffer);

        // 处理剩余的64字节块
        for (i = firstpart; i + (NSLB_MD5_TOTAL - 1) < length; i += NSLB_MD5_TOTAL)
            transform(&input[i]);

        index = 0;
    } else {
        i = 0;
    }

    // 存储剩余的字节
    (void)memcpy_s(&buffer[index], length - i, &input[i], length - i);
}

void NSLBMD5::update(const char input[], size_t length)
{
    // 转换为uint8_t指针，使用标准C++转换
    update(reinterpret_cast<const uint8_t*>(input), length);
}

/**
 * 背景：标准的MD5算法，主要功能是MD5算法的核心变换函数，处理一个64字节的块
 * 参考文献：RFC 1321: The MD5 Message-Digest Algorithm
 * 作者：Ron Rivest
 * 发布日期：1992年4月
 */
void NSLBMD5::transform(const unsigned char block[64])
{
    uint32_t a = state[0], b = state[NSLB_MD5_STATE1], c = state[NSLB_MD5_STATE2], d = state[NSLB_MD5_STATE3],
             x[NSLB_MD5_16];

    decode(x, block, NSLB_MD5_TOTAL);

    // 第一轮
    FF(a, b, c, d, x[0], NSLB_MD5_7, 0xd76aa478);
    FF(d, a, b, c, x[NSLB_MD5_HASH1], NSLB_MD5_12, 0xe8c7b756);
    FF(c, d, a, b, x[NSLB_MD5_HASH2], NSLB_MD5_17, 0x242070db);
    FF(b, c, d, a, x[NSLB_MD5_HASH3], NSLB_MD5_22, 0xc1bdceee);
    FF(a, b, c, d, x[NSLB_MD5_HASH4], NSLB_MD5_7, 0xf57c0faf);
    FF(d, a, b, c, x[NSLB_MD5_HASH5], NSLB_MD5_12, 0x4787c62a);
    FF(c, d, a, b, x[NSLB_MD5_HASH6], NSLB_MD5_17, 0xa8304613);
    FF(b, c, d, a, x[NSLB_MD5_HASH7], NSLB_MD5_22, 0xfd469501);
    FF(a, b, c, d, x[NSLB_MD5_HASH8], NSLB_MD5_7, 0x698098d8);
    FF(d, a, b, c, x[NSLB_MD5_HASH9], NSLB_MD5_12, 0x8b44f7af);
    FF(c, d, a, b, x[NSLB_MD5_HASH10], NSLB_MD5_17, 0xffff5bb1);
    FF(b, c, d, a, x[NSLB_MD5_HASH11], NSLB_MD5_22, 0x895cd7be);
    FF(a, b, c, d, x[NSLB_MD5_HASH12], NSLB_MD5_7, 0x6b901122);
    FF(d, a, b, c, x[NSLB_MD5_HASH13], NSLB_MD5_12, 0xfd987193);
    FF(c, d, a, b, x[NSLB_MD5_HASH14], NSLB_MD5_17, 0xa679438e);
    FF(b, c, d, a, x[NSLB_MD5_HASH15], NSLB_MD5_22, 0x49b40821);

    // 第二轮
    GG(a, b, c, d, x[NSLB_MD5_HASH1], NSLB_MD5_5, 0xf61e2562);
    GG(d, a, b, c, x[NSLB_MD5_HASH6], NSLB_MD5_9, 0xc040b340);
    GG(c, d, a, b, x[NSLB_MD5_HASH11], NSLB_MD5_14, 0x265e5a51);
    GG(b, c, d, a, x[0], NSLB_MD5_20, 0xe9b6c7aa);
    GG(a, b, c, d, x[NSLB_MD5_HASH5], NSLB_MD5_5, 0xd62f105d);
    GG(d, a, b, c, x[NSLB_MD5_HASH10], NSLB_MD5_9, 0x02441453);
    GG(c, d, a, b, x[NSLB_MD5_HASH15], NSLB_MD5_14, 0xd8a1e681);
    GG(b, c, d, a, x[NSLB_MD5_HASH4], NSLB_MD5_20, 0xe7d3fbc8);
    GG(a, b, c, d, x[NSLB_MD5_HASH9], NSLB_MD5_5, 0x21e1cde6);
    GG(d, a, b, c, x[NSLB_MD5_HASH14], NSLB_MD5_9, 0xc33707d6);
    GG(c, d, a, b, x[NSLB_MD5_HASH3], NSLB_MD5_14, 0xf4d50d87);
    GG(b, c, d, a, x[NSLB_MD5_HASH8], NSLB_MD5_20, 0x455a14ed);
    GG(a, b, c, d, x[NSLB_MD5_HASH13], NSLB_MD5_5, 0xa9e3e905);
    GG(d, a, b, c, x[NSLB_MD5_HASH2], NSLB_MD5_9, 0xfcefa3f8);
    GG(c, d, a, b, x[NSLB_MD5_HASH7], NSLB_MD5_14, 0x676f02d9);
    GG(b, c, d, a, x[NSLB_MD5_HASH12], NSLB_MD5_20, 0x8d2a4c8a);

    // 第三轮
    HH(a, b, c, d, x[NSLB_MD5_HASH5], NSLB_MD5_4, 0xfffa3942);
    HH(d, a, b, c, x[NSLB_MD5_HASH8], NSLB_MD5_11, 0x8771f681);
    HH(c, d, a, b, x[NSLB_MD5_HASH11], NSLB_MD5_16, 0x6d9d6122);
    HH(b, c, d, a, x[NSLB_MD5_HASH14], NSLB_MD5_23, 0xfde5380c);
    HH(a, b, c, d, x[NSLB_MD5_HASH1], NSLB_MD5_4, 0xa4beea44);
    HH(d, a, b, c, x[NSLB_MD5_HASH4], NSLB_MD5_11, 0x4bdecfa9);
    HH(c, d, a, b, x[NSLB_MD5_HASH7], NSLB_MD5_16, 0xf6bb4b60);
    HH(b, c, d, a, x[NSLB_MD5_HASH10], NSLB_MD5_23, 0xbebfbc70);
    HH(a, b, c, d, x[NSLB_MD5_HASH13], NSLB_MD5_4, 0x289b7ec6);
    HH(d, a, b, c, x[0], NSLB_MD5_11, 0xeaa127fa);
    HH(c, d, a, b, x[NSLB_MD5_HASH3], NSLB_MD5_16, 0xd4ef3085);
    HH(b, c, d, a, x[NSLB_MD5_HASH6], NSLB_MD5_23, 0x04881d05);
    HH(a, b, c, d, x[NSLB_MD5_HASH9], NSLB_MD5_4, 0xd9d4d039);
    HH(d, a, b, c, x[NSLB_MD5_HASH12], NSLB_MD5_11, 0xe6db99e5);
    HH(c, d, a, b, x[NSLB_MD5_HASH15], NSLB_MD5_16, 0x1fa27cf8);
    HH(b, c, d, a, x[NSLB_MD5_HASH2], NSLB_MD5_23, 0xc4ac5665);

    // 第四轮
    II(a, b, c, d, x[0], NSLB_MD5_6, 0xf4292244);
    II(d, a, b, c, x[NSLB_MD5_HASH7], NSLB_MD5_10, 0x432aff97);
    II(c, d, a, b, x[NSLB_MD5_HASH14], NSLB_MD5_15, 0xab9423a7);
    II(b, c, d, a, x[NSLB_MD5_HASH5], NSLB_MD5_21, 0xfc93a039);
    II(a, b, c, d, x[NSLB_MD5_HASH12], NSLB_MD5_6, 0x655b59c3);
    II(d, a, b, c, x[NSLB_MD5_HASH3], NSLB_MD5_10, 0x8f0ccc92);
    II(c, d, a, b, x[NSLB_MD5_HASH10], NSLB_MD5_15, 0xffeff47d);
    II(b, c, d, a, x[NSLB_MD5_HASH1], NSLB_MD5_21, 0x85845dd1);
    II(a, b, c, d, x[NSLB_MD5_HASH8], NSLB_MD5_6, 0x6fa87e4f);
    II(d, a, b, c, x[NSLB_MD5_HASH15], NSLB_MD5_10, 0xfe2ce6e0);
    II(c, d, a, b, x[NSLB_MD5_HASH6], NSLB_MD5_15, 0xa3014314);
    II(b, c, d, a, x[NSLB_MD5_HASH13], NSLB_MD5_21, 0x4e0811a1);
    II(a, b, c, d, x[NSLB_MD5_HASH4], NSLB_MD5_6, 0xf7537e82);
    II(d, a, b, c, x[NSLB_MD5_HASH11], NSLB_MD5_10, 0xbd3af235);
    II(c, d, a, b, x[NSLB_MD5_HASH2], NSLB_MD5_15, 0x2ad7d2bb);
    II(b, c, d, a, x[NSLB_MD5_HASH9], NSLB_MD5_21, 0xeb86d391);

    // 将变换结果添加到当前状态
    state[0] += a;
    state[NSLB_MD5_STATE1] += b;
    state[NSLB_MD5_STATE2] += c;
    state[NSLB_MD5_STATE3] += d;

    // 清除缓冲区
    (void)memset_s(x, sizeof(x), 0, sizeof(x));
}

// 完成MD5计算
NSLBMD5& NSLBMD5::finalize()
{
    static unsigned char bits[8];
    unsigned int index, padLen;

    // 存储消息长度
    encode(bits, count, NSLB_MD5ENCODE_COUNT);

    // 填充消息使其长度为56字节的倍数
    index = count[0] / NSLB_MD5ENCODE_COUNT % NSLB_MD5_TOTAL;
    padLen = (index < NSLB_MD5_56) ? (NSLB_MD5_56 - index) : (NSLB_MD5_120 - index);
    update(PADDING, padLen);

    // 附加长度
    update(bits, NSLB_MD5ENCODE_COUNT);

    // 存储状态到digest
    encode(digest, state, NSLB_MD5_DIGEST);

    // 清除敏感信息
    (void)memset_s(buffer, sizeof(buffer), 0, sizeof(buffer));
    (void)memset_s(count, sizeof(count), 0, sizeof(count));

    finalized = true;
    return *this;
}

// 转换为十六进制字符串
std::string NSLBMD5::hexdigest() const
{
    if (!finalized)
        return "";
    std::string result;
    result.reserve(NSLB_MD5_RESERVE);

    for (unsigned int i = 0; i < NSLB_MD5_DIGEST; i++) {
        // 手动转换为十六进制字符
        uint8_t byte = digest[i];
        result.push_back(HEX[(byte >> NSLB_MD5_STATE4) & 0xF]);
        result.push_back(HEX[byte & 0xF]);
    }

    return result;
}

std::ostream& operator<<(std::ostream& out, NSLBMD5 md5) { return out << md5.hexdigest(); }

std::string nslb_md5(const std::string str)
{
    NSLBMD5 md5 = NSLBMD5(str);
    return md5.hexdigest();
}

// 计算NslbDpRankInfo结构体向量的MD5
void NSLBMD5::calculateRankInfoMd5(const std::vector<NslbDpRankInfo>& rankInfo, uint8_t commMd5Sum[16])
{
    NSLBMD5 md5;

    // 遍历向量中的每个结构体元素
    for (const auto& info : rankInfo) {
        // 将结构体转换为字节数组并更新MD5
        md5.update(reinterpret_cast<const unsigned char*>(&info), sizeof(NslbDpRankInfo));
    }

    // 完成MD5计算
    md5.finalize();

    // 将结果复制到输出数组
    (void)memcpy_s(commMd5Sum, NSLB_MD5_DIGEST, md5.digest, NSLB_MD5_DIGEST);
}

// 新增：计算TableFourRankInfo结构体向量的MD5
void NSLBMD5::calculateTableFourRankInfoMd5(const std::vector<TableFourRankInfo>& rankInfo, uint8_t commMd5Sum[16])
{
    NSLBMD5 md5;

    // 遍历向量中的每个结构体元素
    for (const auto& info : rankInfo) {
        // 将结构体转换为字节数组并更新MD5
        md5.update(reinterpret_cast<const uint8_t*>(&info), sizeof(TableFourRankInfo));
    }

    // 完成MD5计算
    md5.finalize();
    // 将结果复制到输出数组
    (void)memcpy_s(commMd5Sum, NSLB_MD5_DIGEST, md5.digest, NSLB_MD5_DIGEST);
}

// 将MD5值转换为字符串
std::string NSLBMD5::md5ToString(const uint8_t md5[16])
{
    std::string result;
    result.reserve(NSLB_MD5_RESERVE);

    for (unsigned int i = 0; i < NSLB_MD5_DIGEST; i++) {
        // 手动转换为十六进制字符
        uint8_t byte = md5[i];
        result.push_back(HEX[(byte >> NSLB_MD5_STATE1) & 0xF]);
        result.push_back(HEX[byte & 0xF]);
    }

    return result;
}

} // namespace hccl
