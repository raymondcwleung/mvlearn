add_library(sklearncppMetricsPairwise
            pairwise.cpp "${DIR_INCLUDE_SKLEARNCPP}/metrics/pairwise.h")
target_link_libraries(sklearncppMetricsPairwise PUBLIC Eigen3::Eigen)
target_include_directories(sklearncppMetricsPairwise
                           PRIVATE ${DIR_INCLUDE_SKLEARNCPP})
