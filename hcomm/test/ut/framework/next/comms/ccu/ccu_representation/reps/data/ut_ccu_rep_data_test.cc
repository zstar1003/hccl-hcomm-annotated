/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_read_v1.h"
#include "ccu_rep_write_v1.h"
#include "ccu_rep_bufread_v1.h"
#include "ccu_rep_bufwrite_v1.h"
#include "ccu_rep_buflocread_v1.h"
#include "ccu_rep_buflocwrite_v1.h"
#include "ccu_rep_bufreduce_v1.h"
#include "ccu_rep_loccpy_v1.h"
#include "ccu_rep_remMem_v1.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"

#include "hcomm_c_adpt.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace hcomm {
namespace CcuRep {
    namespace {

        class MockCcuUrmaChannel {
        public:
            MockCcuUrmaChannel(uint16_t channelId = 101, uint32_t dieId = 0) : channelId_(channelId), dieId_(dieId) {}
            uint16_t GetChannelId() { return channelId_; }
            uint32_t GetDieId() { return dieId_; }
            HcclResult GetRmtBuffer(uint64_t& addr, uint32_t& size, uint32_t& tokenId, uint32_t& tokenValue)
            {
                addr = 0x1000;
                size = 256;
                tokenId = 1;
                tokenValue = 100;
                return HCCL_SUCCESS;
            }

        private:
            uint16_t channelId_;
            uint32_t dieId_;
        };

        static MockCcuUrmaChannel* g_mockChannel = nullptr;

        HcclResult HcommChannelGet(ChannelHandle channel, void** channelPtr)
        {
            if (channelPtr == nullptr)
                return HCCL_E_PTR;
            *channelPtr = g_mockChannel;
            return HCCL_SUCCESS;
        }

        void SetMockCcuUrmaChannel(MockCcuUrmaChannel* channel) { g_mockChannel = channel; }

        void ResetStubs() { g_mockChannel = nullptr; }

        class CcuRepReadTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override
            {
                ResetStubs();
                mockChannel = new MockCcuUrmaChannel(100, 0);
                SetMockCcuUrmaChannel(mockChannel);
            }
            void TearDown() override
            {
                ResetStubs();
                delete mockChannel;
            }
            MockCcuUrmaChannel* mockChannel = nullptr;
        };

        class CcuRepWriteTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override
            {
                ResetStubs();
                mockChannel = new MockCcuUrmaChannel(100, 0);
                SetMockCcuUrmaChannel(mockChannel);
            }
            void TearDown() override
            {
                ResetStubs();
                delete mockChannel;
            }
            MockCcuUrmaChannel* mockChannel = nullptr;
        };

        class CcuRepBufReadTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override
            {
                ResetStubs();
                mockChannel = new MockCcuUrmaChannel(100, 0);
                SetMockCcuUrmaChannel(mockChannel);
            }
            void TearDown() override
            {
                ResetStubs();
                delete mockChannel;
            }
            MockCcuUrmaChannel* mockChannel = nullptr;
        };

        class CcuRepBufWriteTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override
            {
                ResetStubs();
                mockChannel = new MockCcuUrmaChannel(100, 0);
                SetMockCcuUrmaChannel(mockChannel);
            }
            void TearDown() override
            {
                ResetStubs();
                delete mockChannel;
            }
            MockCcuUrmaChannel* mockChannel = nullptr;
        };

        class CcuRepBufLocReadTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
            void TearDown() override {}
        };

        class CcuRepBufLocWriteTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
            void TearDown() override {}
        };

        class CcuRepBufReduceTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            void TearDown() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepLocCpyTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            void TearDown() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepRemMemTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override
            {
                ResetStubs();
                mockChannel = new MockCcuUrmaChannel(100, 0);
                SetMockCcuUrmaChannel(mockChannel);
            }
            void TearDown() override
            {
                ResetStubs();
                delete mockChannel;
            }
            MockCcuUrmaChannel* mockChannel = nullptr;
        };

        TEST_F(CcuRepReadTest, Constructor_Basic)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepRead rep(&insGen, 0, loc, rem, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::READ);
            EXPECT_EQ(rep.GetMask(), 0xF);
            EXPECT_EQ(rep.GetSemId(), sem.Id());
        }

        TEST_F(CcuRepReadTest, Constructor_WithDataType)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepRead rep(&insGen, 0, loc, rem, len, 1, 2, sem, 0xA);
            EXPECT_EQ(rep.Type(), CcuRepType::READ);
            EXPECT_EQ(rep.GetMask(), 0xA);
        }

        TEST_F(CcuRepReadTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepRead rep(&insGen, 0, loc, rem, len, sem, 0xF);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Read"), std::string::npos);
        }

        TEST_F(CcuRepReadTest, Translate)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepRead rep(&insGen, 0, loc, rem, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::READ);
        }

        TEST_F(CcuRepWriteTest, Constructor_Basic)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepWrite rep(&insGen, 0, rem, loc, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::WRITE);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepWriteTest, Constructor_WithDataType)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepWrite rep(&insGen, 0, rem, loc, len, 1, 2, sem, 0xA);
            EXPECT_EQ(rep.Type(), CcuRepType::WRITE);
        }

        TEST_F(CcuRepWriteTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepWrite rep(&insGen, 0, rem, loc, len, sem, 0xF);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Write"), std::string::npos);
        }

        TEST_F(CcuRepWriteTest, Translate)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr loc{addr, token};
            RemoteAddr rem{addr, token};

            CcuRepWrite rep(&insGen, 0, rem, loc, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::WRITE);
        }

        TEST_F(CcuRepBufReadTest, Constructor)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            RemoteAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufRead rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_READ);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepBufReadTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            RemoteAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufRead rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_READ);
        }

        TEST_F(CcuRepBufReadTest, Translate)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            RemoteAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufRead rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_READ);
        }

        TEST_F(CcuRepBufWriteTest, Constructor)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            RemoteAddr dst{addr, token};

            CcuRepBufWrite rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_WRITE);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepBufWriteTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            RemoteAddr dst{addr, token};

            CcuRepBufWrite rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_WRITE);
        }

        TEST_F(CcuRepBufWriteTest, Translate)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            RemoteAddr dst{addr, token};

            CcuRepBufWrite rep(&insGen, 0, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_WRITE);
        }

        TEST_F(CcuRepBufLocReadTest, Constructor)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufLocRead rep(&insGen, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_LOC_READ);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepBufLocReadTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufLocRead rep(&insGen, src, dst, len, sem, 0xF);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Read"), std::string::npos);
        }

        TEST_F(CcuRepBufLocReadTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr;
            Variable token;
            LocalAddr src{addr, token};
            CcuBuf dst{&context};

            CcuRepBufLocRead rep(&insGen, src, dst, len, sem, 0xF);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveChannalId[0] = 1;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
        }

        TEST_F(CcuRepBufLocWriteTest, Constructor)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            LocalAddr dst{addr, token};

            CcuRepBufLocWrite rep(&insGen, src, dst, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_LOC_WRITE);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepBufLocWriteTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            LocalAddr dst{addr, token};

            CcuRepBufLocWrite rep(&insGen, src, dst, len, sem, 0xF);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Write"), std::string::npos);
        }

        TEST_F(CcuRepBufLocWriteTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            CcuBuf src{&context};
            Address addr;
            Variable token;
            LocalAddr dst{addr, token};

            CcuRepBufLocWrite rep(&insGen, src, dst, len, sem, 0xF);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveChannalId[0] = 1;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
        }

        TEST_F(CcuRepBufReduceTest, Constructor)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 4, 1, 1, 0, sem, len, 1);
            EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
            EXPECT_EQ(rep.GetCount(), 4);
            EXPECT_EQ(rep.GetDataType(), 1);
            EXPECT_EQ(rep.GetMask(), 1);
        }

        TEST_F(CcuRepBufReduceTest, Describe)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}, CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 2, 1, 1, 0, sem, len, 1);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Reduce"), std::string::npos);
        }

        TEST_F(CcuRepBufReduceTest, Translate_Sum)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 4, 1, 1, 0, sem, len, 1);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
        }

        TEST_F(CcuRepBufReduceTest, Translate_Max)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 4, 1, 1, 1, sem, len, 1);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
        }

        TEST_F(CcuRepBufReduceTest, Translate_Min)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}, CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 4, 1, 1, 2, sem, len, 1);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
        }

        TEST_F(CcuRepBufReduceTest, Translate_CountLessThanMin)
        {
            CcuRepContext context;
            std::vector<CcuBuf> mem = {CcuBuf{&context}};
            Variable len{&context};
            CompletedEvent sem{&context};

            CcuRepBufReduce rep(&insGen, mem, 1, 1, 1, 0, sem, len, 1);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            EXPECT_THROW({ rep.Translate(nullptr, instrPtr, instrId, dep); }, Hccl::CcuApiException);
        }

        TEST_F(CcuRepLocCpyTest, Constructor_Basic)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr1;
            Variable token1;
            Address addr2;
            Variable token2;
            LocalAddr dst{addr1, token1};
            LocalAddr src{addr2, token2};

            CcuRepLocCpy rep(&insGen, dst, src, len, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::LOCAL_CPY);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepLocCpyTest, Constructor_WithDataType)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr1;
            Variable token1;
            Address addr2;
            Variable token2;
            LocalAddr dst{addr1, token1};
            LocalAddr src{addr2, token2};

            CcuRepLocCpy rep(&insGen, dst, src, len, 1, 2, sem, 0xA);
            EXPECT_EQ(rep.Type(), CcuRepType::LOCAL_REDUCE);
        }

        TEST_F(CcuRepLocCpyTest, Describe)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr1;
            Variable token1;
            Address addr2;
            Variable token2;
            LocalAddr dst{addr1, token1};
            LocalAddr src{addr2, token2};

            CcuRepLocCpy rep(&insGen, dst, src, len, sem, 0xF);
            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("LocalAddr"), std::string::npos);
        }

        TEST_F(CcuRepLocCpyTest, Translate_LocalCpy)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr1;
            Variable token1;
            Address addr2;
            Variable token2;
            LocalAddr dst{addr1, token1};
            LocalAddr src{addr2, token2};

            CcuRepLocCpy rep(&insGen, dst, src, len, sem, 0xF);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveChannalId[0] = 1;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
        }

        TEST_F(CcuRepLocCpyTest, Translate_LocalReduce)
        {
            CcuRepContext context;
            Variable len{&context};
            CompletedEvent sem{&context};
            Address addr1;
            Variable token1;
            Address addr2;
            Variable token2;
            LocalAddr dst{addr1, token1};
            LocalAddr src{addr2, token2};

            CcuRepLocCpy rep(&insGen, dst, src, len, 1, 2, sem, 0xF);
            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveChannalId[0] = 1;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
        }

        TEST_F(CcuRepRemMemTest, Constructor)
        {
            Address addr;
            Variable token;
            RemoteAddr rem{addr, token};
            CcuRepRemMem rep(&insGen, 0, rem);
            EXPECT_EQ(rep.Type(), CcuRepType::REM_MEM);
        }

        TEST_F(CcuRepRemMemTest, Translate)
        {
            Address addr;
            Variable token;
            RemoteAddr rem{addr, token};
            CcuRepRemMem rep(&insGen, 0, rem);
            EXPECT_EQ(rep.Type(), CcuRepType::REM_MEM);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
