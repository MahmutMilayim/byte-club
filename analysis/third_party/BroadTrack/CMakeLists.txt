cmake_minimum_required(VERSION 3.6)
project(evs-main-camera-tracking)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Torch REQUIRED PATHS "/opt/libtorch")

add_executable(broadtrack main.cpp
    KeypointDetectionModel.cpp
    KeypointDetectionModel.h
    LineSegmentationModel.cpp
    LineSegmentationModel.h
    Camera.cpp
    Camera.h
    core.h
    core.cpp
    LineIoUScore.cpp
    LineIoUScore.h
    SoccerPitch3D.cpp
    SoccerPitch3D.h
    CameraTracker.cpp
    CameraTracker.h
    Residuals.h
    Chronometer.h
    JsonCalibDict.h
    JsonCalibDict.cpp
    PointExtractor.h
    PointExtractor.cpp
)



add_executable(reinit singleFrameCalibration.cpp
    Camera.cpp
    Camera.h
    core.h
    core.cpp
    KeypointDetectionModel.cpp
    KeypointDetectionModel.h
    LineSegmentationModel.cpp
    LineSegmentationModel.h
    LineIoUScore.cpp
    LineIoUScore.h
    SoccerPitch3D.cpp
    SoccerPitch3D.h
    CameraTracker.cpp
    CameraTracker.h
    Residuals.h
    Chronometer.h
    JsonCalibDict.h
    JsonCalibDict.cpp
    PointExtractor.h
    PointExtractor.cpp
)


target_link_libraries(broadtrack 
PUBLIC 
boost_filesystem
boost_system
boost_program_options
ceres
${OpenCV_LIBS}
${TORCH_LIBRARIES}

)

target_link_libraries(reinit 
PUBLIC 
boost_filesystem
boost_system
PUBLIC ceres
${OpenCV_LIBS}
${TORCH_LIBRARIES}
)

install(TARGETS broadtrack DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
