# Minimum CMake version required
cmake_minimum_required(VERSION 3.13)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Specify the library name
project(IMU_EKF)

# Add the library source files
add_library(${PROJECT_NAME} STATIC 

)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # Include the directory where its header is located
    ${CMAKE_SOURCE_DIR}/lib/Eigen    # Include Eigen headers
)

# Link against Pico SDK and Arduino libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    pico_stdlib 
    hardware_gpio 
    hardware_uart
)
