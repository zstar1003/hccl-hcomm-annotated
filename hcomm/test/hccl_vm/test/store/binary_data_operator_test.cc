/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>

#include "store_binary_data_operator.h"
#include "sim_binary_data_type_pub.h"

namespace HcclSim {
class BinaryDataOperatorTest : public testing::Test {
protected:
    void SetUp() override
    {
        strcpy(testFilePath, "/tmp/binary_test_XXXXXX");
        int fd = mkstemp(testFilePath);
        ASSERT_NE(fd, -1);
        close(fd);
    }

    void TearDown() override
    {
        if (access(testFilePath, F_OK) == 0) {
            unlink(testFilePath);
        }
    }

    char testFilePath[256];
};

// ==================== FileHeader 测试 ====================

TEST_F(BinaryDataOperatorTest, FileHeaderWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    FileHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = 0x12345678;
    header.version = 1;
    header.header_size = sizeof(header);
    header.count = 10;

    HcclVmResult ret = FileHeaderWrite(fp, header);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, FileHeaderRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    FileHeader headerIn;
    memset(&headerIn, 0, sizeof(headerIn));
    headerIn.magic = 0x12345678;
    headerIn.version = 1;
    headerIn.header_size = sizeof(headerIn);
    headerIn.count = 10;

    HcclVmResult ret = FileHeaderWrite(fp, headerIn);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    FileHeader headerOut;
    ret = FileHeaderRead(fp, headerOut, 0x12345678);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(headerOut.magic, 0x12345678u);
    EXPECT_EQ(headerOut.version, 1u);
    EXPECT_EQ(headerOut.count, 10u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, FileHeaderRead_MagicMismatch_Fail)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    FileHeader headerIn;
    memset(&headerIn, 0, sizeof(headerIn));
    headerIn.magic = 0x12345678;
    headerIn.version = 1;

    FileHeaderWrite(fp, headerIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    FileHeader headerOut;
    HcclVmResult ret = FileHeaderRead(fp, headerOut, 0xAAAAAAAA);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_NOT_SUPPORT);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, FileHeaderRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    FileHeader header;
    HcclVmResult ret = FileHeaderRead(fp, header, 0x12345678);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

// ==================== ModelInfoComm 测试 ====================

TEST_F(BinaryDataOperatorTest, ModelInfoCommWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ModelInfoCommInner comm;
    memset(&comm, 0, sizeof(comm));
    comm.root = 0;
    comm.rank_size = 8;
    comm.op_type = 1;
    comm.data_type = 2;
    comm.data_count = 1024;

    HcclVmResult ret = ModelInfoCommWrite(fp, comm);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoCommRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ModelInfoCommInner commIn;
    memset(&commIn, 0, sizeof(commIn));
    commIn.root = 0;
    commIn.rank_size = 8;
    commIn.op_type = 1;
    commIn.data_type = 2;
    commIn.data_count = 1024;

    ModelInfoCommWrite(fp, commIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoCommInner commOut;
    HcclVmResult ret = ModelInfoCommRead(fp, commOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(commOut.root, 0u);
    EXPECT_EQ(commOut.rank_size, 8u);
    EXPECT_EQ(commOut.op_type, 1u);
    EXPECT_EQ(commOut.data_type, 2u);
    EXPECT_EQ(commOut.data_count, 1024u);

    fclose(fp);
}

// ==================== VDataDesTag 测试 ====================

TEST_F(BinaryDataOperatorTest, VDataDesTagWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDes;
    vDataDes.dataType = 1;
    vDataDes.count = 3;
    vDataDes.displs = {0, 100, 200};
    vDataDes.counts = {100, 100, 100};

    HcclVmResult ret = VDataDesTagWrite(fp, vDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, VDataDesTagRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDesIn;
    vDataDesIn.dataType = 1;
    vDataDesIn.count = 3;
    vDataDesIn.displs = {0, 100, 200};
    vDataDesIn.counts = {100, 100, 100};

    VDataDesTagWrite(fp, vDataDesIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDesOut;
    HcclVmResult ret = VDataDesTagRead(fp, vDataDesOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(vDataDesOut.dataType, 1u);
    EXPECT_EQ(vDataDesOut.count, 3u);
    EXPECT_EQ(vDataDesOut.displs.size(), 3u);
    EXPECT_EQ(vDataDesOut.counts.size(), 3u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, VDataDesTagWrite_Empty_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDes;
    vDataDes.dataType = 1;
    vDataDes.count = 0;

    HcclVmResult ret = VDataDesTagWrite(fp, vDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

// ==================== All2AllDataDesTag 测试 ====================

TEST_F(BinaryDataOperatorTest, All2AllDataDesTagWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDes;
    all2AllDataDes.sendType = 1;
    all2AllDataDes.recvType = 1;
    all2AllDataDes.sendCount = 100;
    all2AllDataDes.recvCount = 100;
    all2AllDataDes.count = 4;
    all2AllDataDes.sendCountMatrix = {10, 20, 30, 40};

    HcclVmResult ret = All2AllDataDesTagWrite(fp, all2AllDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, All2AllDataDesTagRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDesIn;
    all2AllDataDesIn.sendType = 1;
    all2AllDataDesIn.recvType = 1;
    all2AllDataDesIn.sendCount = 100;
    all2AllDataDesIn.recvCount = 100;
    all2AllDataDesIn.count = 4;
    all2AllDataDesIn.sendCountMatrix = {10, 20, 30, 40};

    All2AllDataDesTagWrite(fp, all2AllDataDesIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDesOut;
    HcclVmResult ret = All2AllDataDesTagRead(fp, all2AllDataDesOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(all2AllDataDesOut.sendType, 1u);
    EXPECT_EQ(all2AllDataDesOut.recvType, 1u);
    EXPECT_EQ(all2AllDataDesOut.sendCount, 100u);
    EXPECT_EQ(all2AllDataDesOut.recvCount, 100u);
    EXPECT_EQ(all2AllDataDesOut.count, 4u);
    EXPECT_EQ(all2AllDataDesOut.sendCountMatrix.size(), 4u);

    fclose(fp);
}

// ==================== ChannelInfo 测试 ====================

TEST_F(BinaryDataOperatorTest, ChannelInfoWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfo;
    chInfo.count = 2;
    ChannelData ch1 = {0, 0, 1, 0, 1};
    ChannelData ch2 = {1, 1, 0, 1, 0};
    chInfo.data = {ch1, ch2};

    HcclVmResult ret = ChannelInfoWrite(fp, chInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ChannelInfoRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfoIn;
    chInfoIn.count = 2;
    ChannelData ch1 = {0, 0, 1, 0, 1};
    ChannelData ch2 = {1, 1, 0, 1, 0};
    chInfoIn.data = {ch1, ch2};

    ChannelInfoWrite(fp, chInfoIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfoOut;
    HcclVmResult ret = ChannelInfoRead(fp, chInfoOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(chInfoOut.count, 2u);
    EXPECT_EQ(chInfoOut.data.size(), 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ChannelInfoWrite_Empty_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfo;
    chInfo.count = 0;

    HcclVmResult ret = ChannelInfoWrite(fp, chInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

// ==================== MemLayout 测试 ====================

TEST_F(BinaryDataOperatorTest, MemLayoutWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MemLayoutInfoInner memLayoutInfo;
    memLayoutInfo.count = 2;
    MemLayoutData mem1 = {0, 0, 0, 0x1000, 1024, 0};
    MemLayoutData mem2 = {1, 1, 0, 0x2000, 2048, 1024};
    memLayoutInfo.data = {mem1, mem2};

    HcclVmResult ret = MemLayoutWrite(fp, memLayoutInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MemLayoutRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MemLayoutInfoInner memLayoutInfoIn;
    memLayoutInfoIn.count = 2;
    MemLayoutData mem1 = {0, 0, 0, 0x1000, 1024, 0};
    MemLayoutData mem2 = {1, 1, 0, 0x2000, 2048, 1024};
    memLayoutInfoIn.data = {mem1, mem2};

    MemLayoutWrite(fp, memLayoutInfoIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MemLayoutInfoInner memLayoutInfoOut;
    HcclVmResult ret = MemLayoutRead(fp, memLayoutInfoOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(memLayoutInfoOut.count, 2u);
    EXPECT_EQ(memLayoutInfoOut.data.size(), 2u);

    fclose(fp);
}

// ==================== HcclVmSynData 测试 ====================

TEST_F(BinaryDataOperatorTest, HcclVmSynDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synData;
    memset(&synData, 0, sizeof(synData));
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.header.count = 1;

    synData.model_info.comm.root = 0;
    synData.model_info.comm.rank_size = 8;
    synData.model_info.comm.op_type = 1;

    synData.channel_info.count = 0;
    synData.memory_info.count = 0;

    HcclVmResult ret = HcclVmSynDataWrite(fp, synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataIn;
    memset(&synDataIn, 0, sizeof(synDataIn));
    synDataIn.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synDataIn.header.version = 1;
    synDataIn.header.header_size = sizeof(FileHeader);
    synDataIn.header.count = 1;

    synDataIn.model_info.comm.root = 0;
    synDataIn.model_info.comm.rank_size = 8;
    synDataIn.model_info.comm.op_type = 1;

    synDataIn.channel_info.count = 0;
    synDataIn.memory_info.count = 0;

    HcclVmSynDataWrite(fp, synDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataOut;
    HcclVmResult ret = HcclVmSynDataRead(fp, synDataOut, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synDataOut.header.magic, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(synDataOut.model_info.comm.root, 0u);
    EXPECT_EQ(synDataOut.model_info.comm.rank_size, 8u);

    fclose(fp);
}

// ==================== HcclVmFlagData 测试 ====================

TEST_F(BinaryDataOperatorTest, HcclVmFlagDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagData;
    memset(&flagData, 0, sizeof(flagData));
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.version = 1;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.runner_status = 1;

    HcclVmResult ret = HcclVmFlagDataWrite(fp, flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmFlagDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagDataIn;
    memset(&flagDataIn, 0, sizeof(flagDataIn));
    flagDataIn.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagDataIn.header.version = 1;
    flagDataIn.header.header_size = sizeof(FileHeader);
    flagDataIn.runner_status = 1;

    HcclVmFlagDataWrite(fp, flagDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagDataOut;
    HcclVmResult ret = HcclVmFlagDataRead(fp, flagDataOut, HCCLVM_FLAG_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(flagDataOut.header.magic, HCCLVM_FLAG_FILE_MAGIC);
    EXPECT_EQ(flagDataOut.runner_status, 1u);

    fclose(fp);
}

// ==================== 边界条件测试 ====================

TEST_F(BinaryDataOperatorTest, FileHeaderWrite_LargeCount_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    FileHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = 0x12345678;
    header.count = 1000000;

    HcclVmResult ret = FileHeaderWrite(fp, header);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ChannelInfoWrite_LargeData_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfo;
    chInfo.count = 100;
    for (uint32_t i = 0; i < 100; i++) {
        ChannelData ch = {static_cast<uint16_t>(i), 0, 1, i, i + 1};
        chInfo.data.push_back(ch);
    }

    HcclVmResult ret = ChannelInfoWrite(fp, chInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, VDataDesTagWrite_LargeCount_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDes;
    vDataDes.dataType = 1;
    vDataDes.count = 1000;
    for (uint32_t i = 0; i < 1000; i++) {
        vDataDes.displs.push_back(i * 100);
        vDataDes.counts.push_back(100);
    }

    HcclVmResult ret = VDataDesTagWrite(fp, vDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, All2AllDataDesTagWrite_LargeMatrix_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDes;
    all2AllDataDes.sendType = 1;
    all2AllDataDes.recvType = 1;
    all2AllDataDes.sendCount = 100;
    all2AllDataDes.recvCount = 100;
    all2AllDataDes.count = 100;
    for (uint32_t i = 0; i < 100; i++) {
        all2AllDataDes.sendCountMatrix.push_back(i);
    }

    HcclVmResult ret = All2AllDataDesTagWrite(fp, all2AllDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ModelInfoInner modelInfo;
    memset(&modelInfo, 0, sizeof(modelInfo));
    modelInfo.comm.root = 0;
    modelInfo.comm.rank_size = 8;
    modelInfo.vDataDes.dataType = 1;
    modelInfo.vDataDes.count = 0;
    modelInfo.all2AllDataDes.sendType = 1;
    modelInfo.all2AllDataDes.recvType = 1;
    modelInfo.all2AllDataDes.count = 0;

    HcclVmResult ret = ModelInfoWrite(fp, modelInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    ModelInfoInner modelInfoIn;
    memset(&modelInfoIn, 0, sizeof(modelInfoIn));
    modelInfoIn.comm.root = 0;
    modelInfoIn.comm.rank_size = 8;
    modelInfoIn.vDataDes.dataType = 1;
    modelInfoIn.vDataDes.count = 0;
    modelInfoIn.all2AllDataDes.sendType = 1;
    modelInfoIn.all2AllDataDes.recvType = 1;
    modelInfoIn.all2AllDataDes.count = 0;

    ModelInfoWrite(fp, modelInfoIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoInner modelInfoOut;
    HcclVmResult ret = ModelInfoRead(fp, modelInfoOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(modelInfoOut.comm.root, 0u);
    EXPECT_EQ(modelInfoOut.comm.rank_size, 8u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoCommRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoCommInner comm;
    HcclVmResult ret = ModelInfoCommRead(fp, comm);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, VDataDesTagRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDes;
    HcclVmResult ret = VDataDesTagRead(fp, vDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, All2AllDataDesTagRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDes;
    HcclVmResult ret = All2AllDataDesTagRead(fp, all2AllDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ChannelInfoRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfo;
    HcclVmResult ret = ChannelInfoRead(fp, chInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, JettyInfoRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfo;
    HcclVmResult ret = JettyInfoRead(fp, jettyInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MemLayoutRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MemLayoutInfoInner memLayoutInfo;
    HcclVmResult ret = MemLayoutRead(fp, memLayoutInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MicrocodeInstrRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstr;
    HcclVmResult ret = MicrocodeInstrRead(fp, mcInstr);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, TaskMetaRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskData;
    HcclVmResult ret = TaskMetaRead(fp, taskData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmTaskMetaDataRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = HcclVmTaskMetaDataRead(fp, taskMeta, HCCLVM_TASK_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmFlagDataRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagData;
    HcclVmResult ret = HcclVmFlagDataRead(fp, flagData, HCCLVM_FLAG_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synData;
    HcclVmResult ret = HcclVmSynDataRead(fp, synData, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmInstrDataRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrData;
    HcclVmResult ret = HcclVmInstrDataRead(fp, instrData, HCCLVM_INSTR_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoRead_InvalidFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoInner modelInfo;
    HcclVmResult ret = ModelInfoRead(fp, modelInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmFlagDataRead_MagicMismatch_Fail)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagDataIn;
    memset(&flagDataIn, 0, sizeof(flagDataIn));
    flagDataIn.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagDataIn.header.version = 1;
    flagDataIn.header.header_size = sizeof(FileHeader);
    flagDataIn.runner_status = 1;

    HcclVmFlagDataWrite(fp, flagDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagDataOut;
    HcclVmResult ret = HcclVmFlagDataRead(fp, flagDataOut, 0xAAAAAAAA);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmFlagDataWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmFlagData flagData;
    HcclVmResult ret = HcclVmFlagDataWrite(fp, flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataWrite_AivMode_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synData;
    memset(&synData, 0, sizeof(synData));
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.header.count = 1;

    synData.model_info.comm.root = 0;
    synData.model_info.comm.rank_size = 8;
    synData.model_info.comm.op_type = 1;
    synData.model_info.comm.op_expansion_mode = 2; // AIV mode

    synData.memory_info.count = 0;

    HcclVmResult ret = HcclVmSynDataWrite(fp, synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataRead_AivMode_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataIn;
    memset(&synDataIn, 0, sizeof(synDataIn));
    synDataIn.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synDataIn.header.version = 1;
    synDataIn.header.header_size = sizeof(FileHeader);
    synDataIn.header.count = 1;

    synDataIn.model_info.comm.root = 0;
    synDataIn.model_info.comm.rank_size = 8;
    synDataIn.model_info.comm.op_type = 1;
    synDataIn.model_info.comm.op_expansion_mode = 2; // AIV mode

    synDataIn.memory_info.count = 0;

    HcclVmSynDataWrite(fp, synDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataOut;
    HcclVmResult ret = HcclVmSynDataRead(fp, synDataOut, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synDataOut.model_info.comm.op_expansion_mode, 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, FileHeaderWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    FileHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = 0x12345678;

    HcclVmResult ret = FileHeaderWrite(fp, header);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoCommWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoCommInner comm;
    memset(&comm, 0, sizeof(comm));

    HcclVmResult ret = ModelInfoCommWrite(fp, comm);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, VDataDesTagWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    VDataDesTagInner vDataDes;
    vDataDes.dataType = 1;
    vDataDes.count = 0;

    HcclVmResult ret = VDataDesTagWrite(fp, vDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, All2AllDataDesTagWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    All2AllDataDesTagInner all2AllDataDes;
    all2AllDataDes.sendType = 1;
    all2AllDataDes.recvType = 1;
    all2AllDataDes.sendCount = 100;
    all2AllDataDes.recvCount = 100;
    all2AllDataDes.count = 0;

    HcclVmResult ret = All2AllDataDesTagWrite(fp, all2AllDataDes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ChannelInfoWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ChannelInfoInner chInfo;
    chInfo.count = 0;

    HcclVmResult ret = ChannelInfoWrite(fp, chInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, JettyInfoWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfo;
    jettyInfo.count = 0;

    HcclVmResult ret = JettyInfoWrite(fp, jettyInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MemLayoutWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MemLayoutInfoInner memLayoutInfo;
    memLayoutInfo.count = 0;

    HcclVmResult ret = MemLayoutWrite(fp, memLayoutInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MicrocodeInstrWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstr;
    memset(&mcInstr, 0, sizeof(mcInstr));
    mcInstr.desc.count = 0;

    HcclVmResult ret = MicrocodeInstrWrite(fp, mcInstr);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, TaskMetaWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskData;
    memset(&taskData, 0, sizeof(taskData));

    HcclVmResult ret = TaskMetaWrite(fp, taskData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmTaskMetaDataWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.header.count = 0;

    HcclVmResult ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synData;
    memset(&synData, 0, sizeof(synData));

    HcclVmResult ret = HcclVmSynDataWrite(fp, synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmInstrDataWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrData;
    memset(&instrData, 0, sizeof(instrData));
    instrData.header.count = 0;

    HcclVmResult ret = HcclVmInstrDataWrite(fp, instrData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, ModelInfoWrite_ReadOnlyFile_Fail)
{
    FILE* fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    ModelInfoInner modelInfo;
    memset(&modelInfo, 0, sizeof(modelInfo));

    HcclVmResult ret = ModelInfoWrite(fp, modelInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataWrite_JettyMode_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synData;
    memset(&synData, 0, sizeof(synData));
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.header.count = 1;

    synData.model_info.comm.root = 0;
    synData.model_info.comm.rank_size = 8;
    synData.model_info.comm.op_type = 1;
    synData.model_info.comm.op_expansion_mode = 0; // Jetty mode (neither CCU nor AIV)

    synData.memory_info.count = 0;

    HcclVmResult ret = HcclVmSynDataWrite(fp, synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmSynDataRead_JettyMode_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataIn;
    memset(&synDataIn, 0, sizeof(synDataIn));
    synDataIn.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synDataIn.header.version = 1;
    synDataIn.header.header_size = sizeof(FileHeader);
    synDataIn.header.count = 1;

    synDataIn.model_info.comm.root = 0;
    synDataIn.model_info.comm.rank_size = 8;
    synDataIn.model_info.comm.op_type = 1;
    synDataIn.model_info.comm.op_expansion_mode = 0; // Jetty mode

    synDataIn.memory_info.count = 0;

    HcclVmSynDataWrite(fp, synDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmSynData synDataOut;
    HcclVmResult ret = HcclVmSynDataRead(fp, synDataOut, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synDataOut.model_info.comm.op_expansion_mode, 0u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MicrocodeInstrWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstr;
    memset(&mcInstr, 0, sizeof(mcInstr));
    mcInstr.desc.count = 2;
    hcomm::CcuRep::CcuInstr instr1;
    hcomm::CcuRep::CcuInstr instr2;
    memset(&instr1, 0, sizeof(instr1));
    memset(&instr2, 0, sizeof(instr2));
    mcInstr.data = {instr1, instr2};

    HcclVmResult ret = MicrocodeInstrWrite(fp, mcInstr);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, MicrocodeInstrRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstrIn;
    memset(&mcInstrIn, 0, sizeof(mcInstrIn));
    mcInstrIn.desc.count = 2;
    hcomm::CcuRep::CcuInstr instr1;
    hcomm::CcuRep::CcuInstr instr2;
    memset(&instr1, 0, sizeof(instr1));
    memset(&instr2, 0, sizeof(instr2));
    mcInstrIn.data = {instr1, instr2};

    MicrocodeInstrWrite(fp, mcInstrIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstrOut;
    HcclVmResult ret = MicrocodeInstrRead(fp, mcInstrOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(mcInstrOut.desc.count, 2u);
    EXPECT_EQ(mcInstrOut.data.size(), 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, TaskMetaWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskData;
    memset(&taskData, 0, sizeof(taskData));
    taskData.rankId = 1;
    taskData.streamId = 2;

    HcclVmResult ret = TaskMetaWrite(fp, taskData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, TaskMetaRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskDataIn;
    memset(&taskDataIn, 0, sizeof(taskDataIn));
    taskDataIn.rankId = 1;
    taskDataIn.streamId = 2;

    TaskMetaWrite(fp, taskDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskDataOut;
    HcclVmResult ret = TaskMetaRead(fp, taskDataOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskDataOut.rankId, 1u);
    EXPECT_EQ(taskDataOut.streamId, 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmTaskMetaDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMeta.header.version = 1;
    taskMeta.header.header_size = sizeof(FileHeader);
    taskMeta.header.count = 2;

    HcclTaskMetaData task1;
    HcclTaskMetaData task2;
    memset(&task1, 0, sizeof(task1));
    memset(&task2, 0, sizeof(task2));
    task1.rankId = 1;
    task2.rankId = 2;
    taskMeta.task_meta = {task1, task2};

    HcclVmResult ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmTaskMetaDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMetaIn;
    memset(&taskMetaIn, 0, sizeof(taskMetaIn));
    taskMetaIn.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMetaIn.header.version = 1;
    taskMetaIn.header.header_size = sizeof(FileHeader);
    taskMetaIn.header.count = 2;

    HcclTaskMetaData task1;
    HcclTaskMetaData task2;
    memset(&task1, 0, sizeof(task1));
    memset(&task2, 0, sizeof(task2));
    task1.rankId = 1;
    task2.rankId = 2;
    taskMetaIn.task_meta = {task1, task2};

    HcclVmTaskMetaDataWrite(fp, taskMetaIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMetaOut;
    HcclVmResult ret = HcclVmTaskMetaDataRead(fp, taskMetaOut, HCCLVM_TASK_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMetaOut.header.count, 2u);
    EXPECT_EQ(taskMetaOut.task_meta.size(), 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmInstrDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrData;
    memset(&instrData, 0, sizeof(instrData));
    instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrData.header.version = 1;
    instrData.header.header_size = sizeof(FileHeader);
    instrData.header.count = 2;

    MicrocodeInstrInner mcInstr1;
    MicrocodeInstrInner mcInstr2;
    memset(&mcInstr1, 0, sizeof(mcInstr1));
    memset(&mcInstr2, 0, sizeof(mcInstr2));
    mcInstr1.desc.count = 0;
    mcInstr2.desc.count = 0;
    instrData.instr_data = {mcInstr1, mcInstr2};

    HcclVmResult ret = HcclVmInstrDataWrite(fp, instrData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, HcclVmInstrDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrDataIn;
    memset(&instrDataIn, 0, sizeof(instrDataIn));
    instrDataIn.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrDataIn.header.version = 1;
    instrDataIn.header.header_size = sizeof(FileHeader);
    instrDataIn.header.count = 2;

    MicrocodeInstrInner mcInstr1;
    MicrocodeInstrInner mcInstr2;
    memset(&mcInstr1, 0, sizeof(mcInstr1));
    memset(&mcInstr2, 0, sizeof(mcInstr2));
    mcInstr1.desc.count = 0;
    mcInstr2.desc.count = 0;
    instrDataIn.instr_data = {mcInstr1, mcInstr2};

    HcclVmInstrDataWrite(fp, instrDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrDataOut;
    HcclVmResult ret = HcclVmInstrDataRead(fp, instrDataOut, HCCLVM_INSTR_FILE_MAGIC);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrDataOut.header.count, 2u);
    EXPECT_EQ(instrDataOut.instr_data.size(), 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, JettyInfoWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfo;
    jettyInfo.count = 2;
    JettyData j1 = {0, 0, 1, 0};
    JettyData j2 = {1, 1, 0, 1};
    jettyInfo.data = {j1, j2};

    HcclVmResult ret = JettyInfoWrite(fp, jettyInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, JettyInfoRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfoIn;
    jettyInfoIn.count = 2;
    JettyData j1 = {0, 0, 1, 0};
    JettyData j2 = {1, 1, 0, 1};
    jettyInfoIn.data = {j1, j2};

    JettyInfoWrite(fp, jettyInfoIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfoOut;
    HcclVmResult ret = JettyInfoRead(fp, jettyInfoOut);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(jettyInfoOut.count, 2u);
    EXPECT_EQ(jettyInfoOut.data.size(), 2u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorTest, JettyInfoWrite_Empty_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    JettyInfoInner jettyInfo;
    jettyInfo.count = 0;

    HcclVmResult ret = JettyInfoWrite(fp, jettyInfo);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    fclose(fp);
}
} // namespace HcclSim
