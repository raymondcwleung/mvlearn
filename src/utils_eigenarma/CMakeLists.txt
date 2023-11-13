add_library(
  utilsEigenArma io.cpp conversions.cpp "${DIR_INCLUDE_UTILS_EIGENARMA}/io.h"
                 "${DIR_INCLUDE_UTILS_EIGENARMA}/conversions.h")
target_link_libraries(utilsEigenArma PUBLIC Eigen3::Eigen Armadillo)
target_include_directories(utilsEigenArma
                           PRIVATE ${DIR_INCLUDE_UTILS_EIGENARMA})
