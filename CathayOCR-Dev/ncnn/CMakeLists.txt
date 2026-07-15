cmake_minimum_required(VERSION 3.20)
project(OCREngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT DEFINED LIB_ROOT)
    if(WIN32)
        set(LIB_ROOT "$ENV{USERPROFILE}/Documents/libs")
    else()
        set(LIB_ROOT "$ENV{HOME}/Documents/libs")
    endif()
endif()

option(ENABLE_VULKAN "Enable Vulkan GPU acceleration" OFF)

find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
find_package(OpenEXR QUIET)

find_package(OpenMP REQUIRED)

if(NOT DEFINED ncnn_DIR)
    set(ncnn_DIR "${LIB_ROOT}/ncnn/lib/cmake/ncnn")
endif()
find_package(ncnn REQUIRED)

set(LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/angle_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/crnn_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.offset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.rectclip.cpp
)

add_library(OCREngine_core STATIC ${LIB_SOURCES})
target_include_directories(OCREngine_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty
)

if(ENABLE_VULKAN)
    add_definitions(-DENABLE_VULKAN=1)
    set(NCNN_VULKAN_INCLUDE "${CMAKE_SOURCE_DIR}/libs/ncnn/x64/include")
    target_include_directories(OCREngine_core PUBLIC "${NCNN_VULKAN_INCLUDE}")
    message(STATUS "Vulkan enabled, adding include: ${NCNN_VULKAN_INCLUDE}")
endif()

target_link_libraries(OCREngine_core
    PUBLIC
        ${OpenCV_LIBS}
        ncnn
        OpenMP::OpenMP_CXX
)

if(ENABLE_VULKAN)
    set(EXECUTABLE_NAME "ppocr_ocr_vulkan")
else()
    set(EXECUTABLE_NAME "ppocr_ocr_cpu")
endif()

add_executable(${EXECUTABLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE OCREngine_core)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(PYBIND11_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/pybind11-3.0.4")
if(EXISTS "${PYBIND11_LOCAL_DIR}/CMakeLists.txt")
    add_subdirectory(${PYBIND11_LOCAL_DIR})
else()
    message(FATAL_ERROR "Local pybind11 not found: ${PYBIND11_LOCAL_DIR}")
endif()

pybind11_add_module(ocr_engine_python
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind_ocr.cpp
)
set_target_properties(ocr_engine_python PROPERTIES
    OUTPUT_NAME ocr_engine
    PREFIX ""
    SUFFIX ".pyd"
)
target_link_libraries(ocr_engine_python PRIVATE OCREngine_core)

add_custom_command(TARGET ocr_engine_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ocr_engine_python>
        ${CMAKE_CURRENT_SOURCE_DIR}/ocr_engine.pyd
    COMMENT "Copy ocr_engine.pyd to project root"
)
