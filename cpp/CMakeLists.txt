cmake_minimum_required(VERSION 3.20)
project(mm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mm_orderbook src/orderbook.cpp)
target_include_directories(mm_orderbook PUBLIC include)
if(MSVC)
  target_compile_options(mm_orderbook PRIVATE /W4)
else()
  target_compile_options(mm_orderbook PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(mm_engine src/main.cpp)
target_link_libraries(mm_engine PRIVATE mm_orderbook)

option(MM_BUILD_TESTS "Build Catch2 unit tests (downloads Catch2 at configure time)" ON)
if(MM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
  FetchContent_MakeAvailable(Catch2)
  add_executable(mm_tests tests/test_orderbook.cpp)
  target_link_libraries(mm_tests PRIVATE mm_orderbook Catch2::Catch2WithMain)
  include(CTest)
  include(Catch)
  catch_discover_tests(mm_tests)
endif()
