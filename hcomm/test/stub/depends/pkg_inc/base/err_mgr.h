/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PKG_INC_BASE_ERR_MGR_H_
#define PKG_INC_BASE_ERR_MGR_H_

#include <memory>
#include "base/err_msg.h"

namespace error_message {
/**
 * @brief init error manager with error message mode
 * @param [in] error_mode: error mode, see ErrorMessageMode definition
 * @return int32_t 0(success) -1(fail)
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY int32_t ErrMgrInit(ErrorMessageMode error_mode) WEAK_SYMBOL;

/**
 * @brief Get Error manager context
 * @return An error manager context
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ErrorManagerContext GetErrMgrContext() WEAK_SYMBOL;

/**
 * @brief Set Error manager context
 * @param [in] An error manager context
 * @return void
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY void SetErrMgrContext(ErrorManagerContext error_context) WEAK_SYMBOL;

/**
 * @brief Get error message from error manager
 * @return unique_const_char_array, error message
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY unique_const_char_array GetErrMgrErrorMessage() WEAK_SYMBOL;

/**
 * @brief Get warning message from error manager
 * @return unique_const_char_array, warning message
 */
GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY unique_const_char_array GetErrMgrWarningMessage() WEAK_SYMBOL;
} // namespace error_message

#endif // PKG_INC_BASE_ERR_MGR_H_
