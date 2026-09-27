/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TIKCFW_KERNEL_TILING_H_
#define __TIKCFW_KERNEL_TILING_H_

#if defined(ASCENDC_CPU_DEBUG)
#include <cstdint>
#include <cstring>
#endif

namespace AscendC {
namespace tiling {
#pragma pack(push, 8)
    struct LogSoftMaxTiling {
        uint32_t srcM = 0;
        uint32_t srcK = 0;
        uint32_t srcSize = 0;
        uint32_t outMaxM = 0;
        uint32_t outMaxK = 0;
        uint32_t outMaxSize = 0;
        uint32_t splitM = 0;
        uint32_t splitK = 0;
        uint32_t splitSize = 0;
        uint32_t reduceM = 0;
        uint32_t reduceK = 0;
        uint32_t reduceSize = 0;
        uint32_t rangeM = 0;
        uint32_t tailM = 0;
        uint32_t tailSplitSize = 0;
        uint32_t tailReduceSize = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct SoftMaxTiling {
        uint32_t srcM = 0;
        uint32_t srcK = 0;
        uint32_t srcSize = 0;
        uint32_t outMaxM = 0;
        uint32_t outMaxK = 0;
        uint32_t outMaxSize = 0;
        uint32_t splitM = 0;
        uint32_t splitK = 0;
        uint32_t splitSize = 0;
        uint32_t reduceM = 0;
        uint32_t reduceK = 0;
        uint32_t reduceSize = 0;
        uint32_t rangeM = 0;
        uint32_t tailM = 0;
        uint32_t tailSplitSize = 0;
        uint32_t tailReduceSize = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TConv3DApiTiling {
        uint64_t orgDo = 0;
        uint32_t orgCo = 0;
        uint64_t orgHo = 0;
        uint64_t orgWo = 0;
        uint64_t orgDi = 0;
        uint32_t orgCi = 0;
        uint64_t orgHi = 0;
        uint64_t orgWi = 0;
        uint32_t kernelD = 0;
        uint32_t kernelH = 0;
        uint32_t kernelW = 0;
        uint64_t singleCoreDo = 0;
        uint32_t singleCoreCo = 0;
        uint64_t singleCoreM = 0;
        uint32_t singleCoreGroupOpt = 0;
        uint32_t groups = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t strideD = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t dilationD = 0;
        uint32_t padHead = 0;
        uint32_t padTail = 0;
        uint32_t padUp = 0;
        uint32_t padDown = 0;
        uint32_t padLeft = 0;
        uint32_t padRight = 0;
        uint32_t mL0 = 0;
        uint32_t kL0 = 0;
        uint32_t nL0 = 0;
        uint32_t kAL1 = 0;
        uint32_t kAL1Tail = 0;
        uint32_t kBL1 = 0;
        uint32_t kBL1Tail = 0;
        uint32_t nBL1 = 0;
        uint32_t mAL1 = 0;
        uint32_t kBL1DivK0 = 0;
        uint32_t kBL1TailDivK0 = 0;
        uint32_t nBL1DivnL0 = 0;
        uint32_t mAL1DivmL0 = 0;
        uint32_t cin1InAL1 = 0;
        uint32_t cin1InAL1Tail = 0;
        uint32_t nL0xk0 = 0;
        uint64_t kL0xorgCoAlignN0 = 0;
        uint64_t kernelHxkernelW = 0;
        uint64_t cin1xOriHixOriWixk0 = 0;
        uint64_t oriHixOriWixk0 = 0;
        uint64_t oriWixk0 = 0;
        uint64_t orgHixWi = 0;
        uint64_t orgHoxWo = 0;
        uint32_t pBufferFlag = 0;
        uint32_t groupOpt = 0;
        uint32_t cinOpt = 0;
        uint32_t coutOpt = 0;
        int8_t offsetx = 0;
        uint8_t bl1FullLoad = 0;
        uint8_t al1FullLoad = 0;
        uint8_t bl1BypassFlag = 0;
        uint8_t iterateMNOrder = 0;
        uint8_t biasFullLoadFlag = 0;
        uint8_t fixpParamsFullLoadFlag = 0;
        uint8_t hf32Enable = 0;
        uint8_t hf32TransMode = 0;
        uint8_t resvered1 = 0;
        uint16_t resvered2 = 0;
        uint32_t resvered3 = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TConv3DBpFilterTiling {
        uint32_t batch = 0;
        uint32_t cin = 0;
        uint32_t cout = 0;
        uint32_t cin1G = 0;
        uint32_t cout1G = 0;
        uint32_t dout = 0;
        uint32_t ho = 0;
        uint32_t wo = 0;
        uint32_t di = 0;
        uint32_t hi = 0;
        uint32_t wi = 0;
        uint32_t dk = 0;
        uint32_t hk = 0;
        uint32_t wk = 0;
        uint32_t group = 0;
        uint32_t strideD = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t padFront = 0;
        uint32_t padBack = 0;
        uint32_t padUp = 0;
        uint32_t padDown = 0;
        uint32_t padLeft = 0;
        uint32_t padRight = 0;
        uint32_t dilationD = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t channelSize = 0;
        uint32_t al0Pbuffer = 0;
        uint32_t bl0Pbuffer = 0;
        uint32_t cl0Pbuffer = 0;
        uint32_t al1Pbuffer = 0;
        uint32_t bl1Pbuffer = 0;
        uint32_t baseM = 0;
        uint32_t baseK = 0;
        uint32_t baseN = 0;
        uint32_t m0 = 0;
        uint32_t k0 = 0;
        uint32_t n0 = 0;
        uint32_t stepM = 0;
        uint32_t stepN = 0;
        uint32_t stepKa = 0;
        uint32_t stepKb = 0;
        uint32_t iterateOrder = 0;
        uint32_t bl1Bound = 0;
        uint32_t hf32Flag = 0;
        uint32_t singleCoreDk = 0;
        uint32_t singleCoreGroup = 0;
        uint32_t singleCoreCout = 0;
        uint32_t singleCoreHo = 0;
        uint64_t singleCoreBatch = 0;
        uint64_t singleCoreCin = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Conv3DBpFilterParams {
        uint32_t totalL1Size = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TConv3DBpFilterBasicBlockTiling {
        uint32_t singleCoreM = 0;
        uint32_t singleCoreN = 0;
        uint32_t singleCoreK = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Conv3DBackpropFilterTilingData {
        Conv3DBpFilterParams params;
        TConv3DBpFilterTiling dwTiling;
        TConv3DBpFilterBasicBlockTiling basicBlockTiling;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TConv3DBackpropInputTiling {
        uint32_t batch = 0;
        uint32_t cin = 0;
        uint32_t cout = 0;
        uint32_t cout1 = 0;
        uint32_t cin1 = 0;
        uint32_t cout1G = 0;
        uint32_t cin1G = 0;
        uint32_t c0 = 0;
        uint32_t c0Bits = 0;
        uint32_t dout = 0;
        uint32_t ho = 0;
        uint32_t wo = 0;
        uint32_t di = 0;
        uint32_t hi = 0;
        uint32_t wi = 0;
        uint32_t dk = 0;
        uint32_t hk = 0;
        uint32_t wk = 0;
        uint32_t group = 0;
        uint32_t strideD = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t padFront = 0;
        uint32_t padBack = 0;
        uint32_t padUp = 0;
        uint32_t padDown = 0;
        uint32_t padLeft = 0;
        uint32_t padRight = 0;
        uint32_t backpropPadTail = 0;
        uint32_t backpropPadUp = 0;
        uint32_t backpropPadDown = 0;
        uint32_t backpropPadLeft = 0;
        uint32_t backpropPadRight = 0;
        uint32_t dilationD = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t al0Pbuffer = 0;
        uint32_t bl0Pbuffer = 0;
        uint32_t cl0Pbuffer = 0;
        uint32_t al1Pbuffer = 0;
        uint32_t bl1Pbuffer = 0;
        uint32_t singleCoreGroup = 0;
        uint32_t singleCoreCout = 0;
        uint32_t singleCoreCout1 = 0;
        uint32_t singleCoreCin1 = 0;
        uint32_t singleCoreDin = 0;
        uint32_t singleCoreHo = 0;
        uint32_t baseM = 0;
        uint32_t baseK = 0;
        uint32_t baseN = 0;
        uint32_t baseD = 0;
        uint32_t baseBatch = 0;
        uint32_t baseGroup = 0;
        uint32_t stepM = 0;
        uint32_t stepN = 0;
        uint32_t stepKa = 0;
        uint32_t stepKb = 0;
        uint32_t stepBatch = 0;
        uint32_t stepGroup = 0;
        uint32_t iterateOrder = 0;
        int32_t hf32Flag = 0;
        int32_t initOutputFlag = 0;
        int32_t reserved = 0;
        uint64_t singleCoreBatch = 0;
        uint64_t singleCoreM = 0;
        uint64_t singleCoreCin = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Conv3DBackpropInputTilingData {
        TConv3DBackpropInputTiling conv3DDxTiling;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Mc2ServerCfg {
        uint32_t version = 0;
        uint8_t debugMode = 0;
        uint8_t sendArgIndex = 0;
        uint8_t recvArgIndex = 0;
        uint8_t commOutArgIndex = 0;
        uint8_t reserved[8] = {};
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Mc2HcommCfg {
        uint8_t skipLocalRankCopy = 0;
        uint8_t skipBufferWindowCopy = 0;
        uint8_t stepSize = 0;
        char reserved[13] = {};
        char groupName[128] = {};
        char algConfig[128] = {};
        uint32_t opType = 0;
        uint32_t reduceType = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Mc2InitTiling {
        uint8_t reserved[64] = {};
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct Mc2CcTiling {
        uint8_t reserved[280] = {};
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TCubeTiling {
        int32_t usedCoreNum = 0;
        int32_t M = 0;
        int32_t N = 0;
        int32_t Ka = 0;
        int32_t Kb = 0;
        int32_t singleCoreM = 0;
        int32_t singleCoreN = 0;
        int32_t singleCoreK = 0;
        int32_t baseM = 0;
        int32_t baseN = 0;
        int32_t baseK = 0;
        int32_t depthA1 = 0;
        int32_t depthB1 = 0;
        int32_t stepM = 0;
        int32_t stepN = 0;
        int32_t isBias = 0;
        int32_t transLength = 0;
        int32_t iterateOrder = 0;
        int32_t shareMode = 0;
        int32_t shareL1Size = 0;
        int32_t shareL0CSize = 0;
        int32_t shareUbSize = 0;
        int32_t batchM = 0;
        int32_t batchN = 0;
        int32_t singleBatchM = 0;
        int32_t singleBatchN = 0;
        int32_t stepKa = 0;
        int32_t stepKb = 0;
        int32_t depthAL1CacheUB = 0;
        int32_t depthBL1CacheUB = 0;
        int32_t dbL0A = 0;
        int32_t dbL0B = 0;
        int32_t dbL0C = 0;
        int32_t ALayoutInfoB = 0;
        int32_t ALayoutInfoS = 0;
        int32_t ALayoutInfoN = 0;
        int32_t ALayoutInfoG = 0;
        int32_t ALayoutInfoD = 0;
        int32_t BLayoutInfoB = 0;
        int32_t BLayoutInfoS = 0;
        int32_t BLayoutInfoN = 0;
        int32_t BLayoutInfoG = 0;
        int32_t BLayoutInfoD = 0;
        int32_t CLayoutInfoB = 0;
        int32_t CLayoutInfoS1 = 0;
        int32_t CLayoutInfoN = 0;
        int32_t CLayoutInfoG = 0;
        int32_t CLayoutInfoS2 = 0;
        int32_t BatchNum = 0;
        int32_t mxTypePara = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct BatchNormTiling {
        uint32_t originalBLength = 0;
        uint32_t meanVarSize = 0;
        uint32_t meanTmpTensorPos = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t bshCurLength = 0;
        uint32_t shCurLength = 0;
        float firstDimValueBack = 0;
        uint32_t castHalfRepStride = 0;
        uint32_t shCurLengthBlockNum = 0;
        uint32_t castHalfOutRepStride = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct DeepNormTiling {
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        uint32_t inputXSize = 0;
        uint32_t meanVarSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t meanTmpTensorPos = 0;
        uint32_t meanTmpTensorSize = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t varianceTmpTensorSize = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputRoundSize = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarRoundSize = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t bshCurLength = 0;
        uint32_t bsCurLength = 0;
        float lastDimValueBack = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct GroupNormTiling {
        uint32_t n = 0;
        uint32_t c = 0;
        uint32_t hw = 0;
        uint32_t g = 0;
        uint32_t d = 0;
        uint32_t hwAlignSize = 0;
        uint32_t dhwAlignSize = 0;
        uint32_t inputXSize = 0;
        uint32_t meanVarSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t meanTmpTensorPos = 0;
        uint32_t meanTmpTensorSize = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t varianceTmpTensorSize = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputRoundSize = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarRoundSize = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t bshCurLength = 0;
        uint32_t bsCurLength = 0;
        float factor = 0;
        bool smallShape = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct LayerNormGradBetaTiling {
        uint32_t stackBufferSize = 0;
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        uint32_t bshLength = 0;
        uint32_t bsLength = 0;
        uint32_t oneCalSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t loopRound = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t bsTailSize = 0;
        uint32_t bshCurLength = 0;
        uint32_t bsCurLength = 0;
        uint32_t gammaTempTensorPos = 0;
        uint32_t betaTempTensorPos = 0;
        uint32_t inputDyTmpTensorPos = 0;
        uint32_t resForGammaTmpTensorPos = 0;
        uint32_t reserved = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct LayerNormGradTiling {
        uint32_t stackBufferSize = 0;
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        uint32_t oneCalSize = 0;
        uint32_t nohCalSize = 0;
        uint32_t loopNum = 0;
        uint32_t tailSize = 0;
        uint32_t nohTailSize = 0;
        uint32_t tmpTensorBSHPos = 0;
        uint32_t tmpTensorBSHSize = 0;
        uint32_t pdVarTensorPos = 0;
        uint32_t pdVarTensorSize = 0;
        uint32_t pdMeanTensorPos = 0;
        uint32_t pdMeanTensorSize = 0;
        uint32_t x1TensorPos = 0;
        uint32_t x1TensorSize = 0;
        uint32_t x2TensorPos = 0;
        uint32_t x2TensorSize = 0;
        uint32_t x3TensorPos = 0;
        uint32_t x3TensorSize = 0;
        uint32_t tmpTensorPos = 0;
        uint32_t tmpTensorSize = 0;
        uint32_t tmpTensor1Pos = 0;
        uint32_t tmpTensor1Size = 0;
        uint32_t tmpTensor2Pos = 0;
        uint32_t tmpTensor2Size = 0;
        uint32_t lastDimValueBack = 0;
        uint32_t lastDimValueBackMulTwo = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct LayerNormTiling {
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        uint32_t inputXSize = 0;
        uint32_t meanVarSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t meanTmpTensorPos = 0;
        uint32_t meanTmpTensorSize = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t varianceTmpTensorSize = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputRoundSize = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarRoundSize = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t bshCurLength = 0;
        uint32_t bsCurLength = 0;
        float lastDimValueBack = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct LayerNormSeparateTiling {
        uint32_t aLength = 0;
        uint32_t rLength = 0;
        uint32_t halfAddRepeatTimes = 0;
        uint32_t rHeadLength = 0;
        float k2Rec = 0;
        float k2RRec = 0;
        uint32_t inputXSize = 0;
        uint32_t meanVarSize = 0;
        uint32_t numberOfTmpBuf = 0;
        uint32_t varianceTmpTensorPos = 0;
        uint32_t varianceTmpTensorSize = 0;
        uint32_t tmpBufSize = 0;
        uint32_t oneTmpSize = 0;
        uint32_t firstTmpStartPos = 0;
        uint32_t secondTmpStartPos = 0;
        uint32_t thirdTmpStartPos = 0;
        uint32_t loopRound = 0;
        uint32_t inputRoundSize = 0;
        uint32_t inputTailSize = 0;
        uint32_t inputTailPos = 0;
        uint32_t meanVarRoundSize = 0;
        uint32_t meanVarTailSize = 0;
        uint32_t meanVarTailPos = 0;
        uint32_t arCurLength = 0;
        uint32_t aCurLength = 0;
        float rValueBack = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct RmsNormTiling {
        uint32_t bLength = 0;
        uint32_t sLength = 0;
        uint32_t hLength = 0;
        uint32_t originalHLength = 0;
        float reciprocalOfHLength = 0;
        uint32_t mainBshLength = 0;
        uint32_t mainBsLength = 0;
        uint32_t mainBsLengthAlign = 0;
        uint32_t loopRound = 0;
        uint32_t inputTailPos = 0;
        uint32_t tailBshLength = 0;
        uint32_t tailBsLength = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct UnPadTiling {
        uint32_t srcHeight = 0;
        uint32_t srcWidth = 0;
        uint32_t tmpBuffer1BlockNum = 0;
        uint32_t tmpBuffer1RowNum = 0;
        uint32_t tmpBuffer2Offset = 0;
        uint32_t widthTiling = 0;
        uint32_t widthFractal = 0;
        uint32_t widthFractalTail = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct PadTiling {
        uint32_t srcHeight = 0;
        uint32_t srcWidth = 0;
        uint32_t srcOriWidth = 0;
        uint32_t widthWithoutLastBlock = 0;
        uint32_t blocksPerRow = 0;
        uint32_t heightTiling = 0;
        uint32_t heightFractal = 0;
        uint32_t heightFractalTail = 0;
        uint32_t mainLoopOffset = 0;
        uint32_t tailBlockOffset = 0;
        uint32_t tmpBuffer1BlockNum = 0;
        uint32_t tmpBuffer1RowNum = 0;
        uint32_t tmpBuffer2Offset = 0;
        uint32_t widthTiling = 0;
        uint32_t widthFractal = 0;
        uint32_t widthFractalTail = 0;
        uint32_t widthFractalTailAlingned = 0;
        uint32_t brcbTiling = 0;
        uint32_t brcbFractal = 0;
        uint32_t brcbFractalTail = 0;
        uint32_t maxRepeatTimes = 0;
        uint32_t brcbTilingRepeatTimes = 0;
        uint32_t brcbTilingRepeatTimesTail = 0;
        uint32_t brcbFractalTailRepeatTimes = 0;
        uint32_t brcbFractalTailRepeatTimesTail = 0;
        uint32_t reserved = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct TopkTiling {
        int32_t tmpLocalSize = 0;
        int32_t allDataSize = 0;
        int32_t innerDataSize = 0;
        uint32_t sortRepeat = 0;
        int32_t mrgSortRepeat = 0;
        int32_t kAlignFourBytes = 0;
        int32_t kAlignTwoBytes = 0;
        int32_t maskOffset = 0;
        int32_t maskVreducev2FourBytes = 0;
        int32_t maskVreducev2TwoBytes = 0;
        int32_t mrgSortSrc1offset = 0;
        int32_t mrgSortSrc2offset = 0;
        int32_t mrgSortSrc3offset = 0;
        int32_t mrgSortTwoQueueSrc1Offset = 0;
        int32_t mrgFourQueueTailPara1 = 0;
        int32_t mrgFourQueueTailPara2 = 0;
        int32_t srcIndexOffset = 0;
        uint32_t copyUbToUbBlockCount = 0;
        int32_t topkMrgSrc1MaskSizeOffset = 0;
        int32_t topkNSmallSrcIndexOffset = 0;
        uint32_t vreduceValMask0 = 0;
        uint32_t vreduceValMask1 = 0;
        uint32_t vreduceIdxMask0 = 0;
        uint32_t vreduceIdxMask1 = 0;
        uint16_t vreducehalfValMask0 = 0;
        uint16_t vreducehalfValMask1 = 0;
        uint16_t vreducehalfValMask2 = 0;
        uint16_t vreducehalfValMask3 = 0;
        uint16_t vreducehalfValMask4 = 0;
        uint16_t vreducehalfValMask5 = 0;
        uint16_t vreducehalfValMask6 = 0;
        uint16_t vreducehalfValMask7 = 0;
    };
#pragma pack(pop)
#pragma pack(push, 8)
    struct ConfusionTransposeTiling {
        uint32_t param0 = 0;
        uint32_t param1 = 0;
        uint32_t param2 = 0;
        uint32_t param3 = 0;
        uint32_t param4 = 0;
        uint32_t param5 = 0;
        uint32_t param6 = 0;
        uint32_t param7 = 0;
        uint32_t param8 = 0;
        uint32_t param9 = 0;
        uint32_t param10 = 0;
        uint32_t param11 = 0;
        uint32_t param12 = 0;
        uint32_t param13 = 0;
        uint32_t param14 = 0;
        uint32_t param15 = 0;
        uint32_t param16 = 0;
        uint32_t param17 = 0;
    };
#pragma pack(pop)
} // namespace tiling
} // namespace AscendC

using AscendC::tiling::BatchNormTiling;
using AscendC::tiling::ConfusionTransposeTiling;
using AscendC::tiling::Conv3DBackpropFilterTilingData;
using AscendC::tiling::Conv3DBackpropInputTilingData;
using AscendC::tiling::Conv3DBpFilterParams;
using AscendC::tiling::DeepNormTiling;
using AscendC::tiling::GroupNormTiling;
using AscendC::tiling::LayerNormGradBetaTiling;
using AscendC::tiling::LayerNormGradTiling;
using AscendC::tiling::LayerNormSeparateTiling;
using AscendC::tiling::LayerNormTiling;
using AscendC::tiling::LogSoftMaxTiling;
using AscendC::tiling::Mc2CcTiling;
using AscendC::tiling::Mc2HcommCfg;
using AscendC::tiling::Mc2InitTiling;
using AscendC::tiling::Mc2ServerCfg;
using AscendC::tiling::PadTiling;
using AscendC::tiling::RmsNormTiling;
using AscendC::tiling::SoftMaxTiling;
using AscendC::tiling::TConv3DApiTiling;
using AscendC::tiling::TConv3DBackpropInputTiling;
using AscendC::tiling::TConv3DBpFilterBasicBlockTiling;
using AscendC::tiling::TConv3DBpFilterTiling;
using AscendC::tiling::TCubeTiling;
using AscendC::tiling::TopkTiling;
using AscendC::tiling::UnPadTiling;
#endif
