cmake_minimum_required(VERSION 3.0)

PROJECT(OpenSubdiv-Model-Fitting)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---------------------------------------------------------------		
# Find dependencies
#---------------------------------------------------------------

# OpenSubDiv
IF(WIN32)
    #Warning!! If MSVC, include <iso646.h> within the code to define words like "and", "not"... as operators.
    SET(OSD_PATH "C:/dev/GitHub/OpenSubdiv/awf-install" CACHE PATH "Path to osd")
    SET(OSD_INCLUDE_DIRS "${OSD_PATH}/include")
    SET(OSD_LIBRARY_DIRS "${OSD_PATH}/lib")
    SET(OSD_LIB "${OSD_LIBRARY_DIRS}/$(Configuration)/osdCPU.lib")
ELSEIF(UNIX)
    # OSD Compiled using GLFW 3.1.2
    SET(OSD_PATH "/usr/local" CACHE PATH "Path to osd")
    SET(OSD_INCLUDE_DIRS "${OSD_PATH}/include/opensubdiv")
    SET(OSD_LIBRARY_DIRS "${OSD_PATH}/lib")
    SET(OSD_LIB "${OSD_LIBRARY_DIRS}/libosdCPU.so")
ELSE()
    MESSAGE(FATAL_ERROR "Unrecognized OS")
ENDIF()
INCLUDE_DIRECTORIES(${OSD_INCLUDE_DIRS})
LINK_DIRECTORIES(${OSD_LIBRARY_DIRS})

# Eigen
SET(CMAKE_PREFIX_PATH "C:/dev/eigen_pr/awf-install")
FIND_PACKAGE(Eigen3)
# Point to Eigen directly, if developing Eigen at the same time SET(EIGEN3_INCLUDE_DIR "C:/dev/eigen_pr")
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

IF(UNIX)
    ADD_DEFINITIONS("-std=c++11")
ENDIF()

#---------------------------------------------------------------
#Set the projects
#---------------------------------------------------------------		
ADD_EXECUTABLE(Fit-Subdiv-to-3D-Points
	fit-subdiv-to-3d-points.cpp
  MeshTopology.cpp
  log3d.cpp
	)
	
TARGET_LINK_LIBRARIES(Fit-Subdiv-to-3D-Points ${OSD_LIB})
