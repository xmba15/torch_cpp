/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#if ENABLE_DEBUG
#include <iostream>
#endif

namespace _cv
{
#define INFO_LOG(...)                                                                                                  \
    {                                                                                                                  \
        char str[100];                                                                                                 \
        snprintf(str, sizeof(str), __VA_ARGS__);                                                                       \
        std::cout << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] >>> " << str << std::endl; \
    }

#if ENABLE_DEBUG
#define DEBUG_LOG(...)                                                                                                 \
    {                                                                                                                  \
        char str[100];                                                                                                 \
        snprintf(str, sizeof(str), __VA_ARGS__);                                                                       \
        std::cout << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] >>> " << str << std::endl; \
    }
#else
#define DEBUG_LOG(...)
#endif
}  // namespace _cv
