cmake_minimum_required(VERSION 3.16..3.24)
project(CMAKe VERSION 0.0.1
                    DESCRIPTION "Our first project"
                    LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_library(cxx_setup INTERFACE)
target_compile_options(cxx_setup
  INTERFACE -Wall -Wpedantic -Wextra)
target_compile_features(cxx_setup INTERFACE cxx_std_17)
target_include_directories(cxx_setup
  INTERFACE ${PROJECT_SOURCE_DIR})

# 🚨 Make sure CMakeLists.txt file exists in the subdirectories!
add_subdirectory(${PROJECT_NAME})
add_subdirectory(executables)

# 💡 Some things (e.g. tests or installation) are missing here.
# 💡 Stay tuned for these in future lectures!
set(FOO 42)  # A local variable
message(STATUS "Value of FOO variable is: ${FOO}")
# The PROJECT_NAME variable is defined by project(...) command
message(STATUS "Project name is: ${PROJECT_NAME}")

if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE is not defined!")
endif()
if(NOT FOO)
  message(FATAL_ERROR "FOO is not defined!")
endif()
message("Always print this")


set(VAR "value" CACHE STRING "Description") # Cached variable only set if not found already in cache
message(STATUS "Value of VAR variable is: ${VAR}")

#set(VAR "value" CACHE INTERNAL "") #The values stays un cache || else local 
#set(VAR "value" CACHE STRING "Description")
#option(CUSTOM_OPTION "Description" OFF)