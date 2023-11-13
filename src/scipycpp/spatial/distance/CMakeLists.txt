add_library(scipycppSpatialDistance
            distance.cpp "${DIR_INCLUDE_SCIPYCPP}/spatial/distance/distance.h")
target_link_libraries(scipycppSpatialDistance PUBLIC Eigen3::Eigen)
target_include_directories(scipycppSpatialDistance
                           PRIVATE ${DIR_INCLUDE_SCIPYCPP})
