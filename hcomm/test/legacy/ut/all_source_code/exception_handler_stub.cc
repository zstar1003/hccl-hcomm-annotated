/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exception_handler.h"
#include <stdexcept>

namespace hccl {

HcclResult ExceptionHandler::HandleException(const char* functionName)
{
    (void)functionName;
    try {
        throw;
    } catch (const HcclException& e) {
        return e.code();
    } catch (const std::out_of_range&) {
        return HCCL_E_NOT_FOUND;
    } catch (const std::runtime_error&) {
        return HCCL_E_RUNTIME;
    } catch (const std::logic_error&) {
        return HCCL_E_INTERNAL;
    } catch (const std::exception&) {
        return HCCL_E_INTERNAL;
    } catch (...) {
        return HCCL_E_INTERNAL;
    }
}

void ExceptionHandler::ThrowIfErrorCode(
    HcclResult errorCode, const std::string& errString, const char* fileName, s32 lineNum, const char* functionName)
{
    (void)fileName;
    (void)lineNum;
    (void)functionName;
    if (errorCode == HCCL_SUCCESS) {
        return;
    }
    throw HcclException(errorCode, errString);
}

} // namespace hccl
