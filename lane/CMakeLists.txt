find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14 -O2")
include_directories(${OpenCV_INCLUDE_DIRS}
                    ./
                    ${CMAKE_CURRENT_SOURCE_DIR}/../
                    /usr/local/cuda/include
                    /usr/local/TensorRT/include
                    /usr/local/cudnn/include)
link_directories(/usr/lib/aarch64-linux-gnu
                 /usr/local/TensorRT/lib
                 /usr/local/cudnn/lib64
                 /usr/local/cuda/lib64/)

add_library(lane SHARED LaneDetect.cpp PluginFactoryLane.cpp)
add_library(lidarlane lidarlane.cpp)
target_link_libraries(lane
                      inference
                      ${OpenCV_LIBRARIES} ${CRMW_LIBRARIES} msg)

target_link_libraries(lidarlane 
                    ${CRMW_LIBRARIES} 
                    ${PCL_LIBRARIES} 
                    math 
                    pose_provider
                    proj
                    )