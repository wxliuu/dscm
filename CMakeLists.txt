cmake_minimum_required(VERSION 3.13)

project(dscm)

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if( NOT CMAKE_BUILD_TYPE )
  message(STATUS "1 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") 
  set( CMAKE_BUILD_TYPE Release)
else()
  message(STATUS "2 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") 
endif()
message(STATUS "3 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

include_directories(
#   /root/SETUP/gperftools_install/include
${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link_directories(/root/SETUP/gperftools_install/lib)

# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g") # cmake default: "-O2 -g -DNDEBUG"
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# SET(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh_zvfh1p0") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zba_zbb_zbc_zbs -fno-tree-vectorize -fno-tree-loop-vectorize")

find_package(Eigen3 REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED)
# find_package(TBB REQUIRED)

add_executable(undistortImage src/undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})


add_executable(dscm1_distort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm1_distort.cpp
)

target_link_libraries(dscm1_distort ${OpenCV_LIBS})

add_executable(dscm2_distort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm2_distort.cpp
)

target_link_libraries(dscm2_distort ${OpenCV_LIBS})


add_executable(dscm3_distort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm3_distort.cpp
)

target_link_libraries(dscm3_distort ${OpenCV_LIBS})


add_executable(dscm4_projection
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm4_projection.cpp
)

target_link_libraries(dscm4_projection ${OpenCV_LIBS})

add_executable(dscm5_distort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm5_distort.cpp
)

target_link_libraries(dscm5_distort ${OpenCV_LIBS})


add_executable(dscm6_undistort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm6_undistort.cpp
)

target_link_libraries(dscm6_undistort ${OpenCV_LIBS})

add_executable(dscm7_undistort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dscm7_undistort.cpp
)

target_link_libraries(dscm7_undistort ${OpenCV_LIBS})

add_executable(fx src/calc_fx.cpp)

add_executable(stereorectify_undistort
  ${CMAKE_CURRENT_SOURCE_DIR}/src/stereorectify_undistort.cpp
)

target_link_libraries(stereorectify_undistort ${OpenCV_LIBS})


add_executable(stereorectify_undistort2
  ${CMAKE_CURRENT_SOURCE_DIR}/src/stereorectify_undistort2.cpp
)

target_link_libraries(stereorectify_undistort2 ${OpenCV_LIBS})

add_executable(compute_extrinsic
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compute_extrinsic.cpp
)

target_link_libraries(compute_extrinsic Eigen3::Eigen)

# add_executable(test1
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/test1.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/wx_lkpyramid.cpp
# )

# target_link_libraries(test1 Eigen3::Eigen)
# target_link_libraries(test1 ${OpenCV_LIBS} TBB::tbb)
# target_link_libraries(test2 ${OpenCV_LIBS})
# target_link_libraries(test1 -lpthread)