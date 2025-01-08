cmake_minimum_required(VERSION 3.10)
project(navground_onnx)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(navground_core REQUIRED)
find_package(onnxruntime REQUIRED)

include(GenerateExportHeader)

include_directories(include ${PROJECT_BINARY_DIR})

add_library(policy_behavior SHARED src/policy_behavior.cpp src/policy.cpp
                                   src/shared_policy.cpp src/tensor_utils.cpp)
target_link_libraries(policy_behavior navground_core::navground_core
                      onnxruntime::onnxruntime)
set_target_properties(policy_behavior PROPERTIES LINKER_LANGUAGE CXX)
generate_export_header(policy_behavior 
  BASE_NAME navground_onnx
  EXPORT_FILE_NAME navground_onnx/export.h)

register_navground_plugins(TARGETS policy_behavior DESTINATION
                           $<IF:$<BOOL:${WIN32}>,bin,lib>)

add_subdirectory(dev)

install(
  TARGETS policy_behavior
  EXPORT policy_behaviorTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

if(ament_cmake_FOUND)
  ament_package()
endif()
