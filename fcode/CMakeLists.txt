cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME online_example)

# Enable Fortran and C++ languages
project(${PROJECT_NAME} LANGUAGES Fortran CXX)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(CMAKE_PREFIX_PATH " /usr/local/libtorch")

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(online_ftorch main.f90)
target_link_libraries(ftorch_gpu PRIVATE FTorch::ftorch)

# Additional steps to ensure GPU support
find_package(Torch REQUIRED)

# Link the Torch libraries to your target
target_link_libraries(ftorch_gpu PRIVATE ${TORCH_LIBRARIES})

# Ensure that the C++14 standard is used (required by PyTorch)
set_property(TARGET ftorch_gpu PROPERTY CXX_STANDARD 14)