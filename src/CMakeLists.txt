cmake_minimum_required(VERSION 3.0)

PROJECT(OpenSubdiv-Model-Fitting)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---------------------------------------------------------------		
# Find dependencies
#---------------------------------------------------------------

# OpenSubDiv
IF(WIN32)
    #Warning!! If MSVC, include <iso646.h> within the code to define words like "and", "not"... as operators.
    SET(OSD_PATH "Z:/OpenSubdiv/install" CACHE PATH "Path to osd")
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
SET(CMAKE_PREFIX_PATH "Z:/Eigen_PR/build;Z:/Eigen_PR;Z:/OpenSubdiv/install/include;C:/Program Files/Eigen3/include/eigen3")
FIND_PACKAGE(Eigen3)
# Point to Eigen directly, if developing Eigen at the same time SET(EIGEN3_INCLUDE_DIR "C:/dev/eigen_pr")
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

IF(UNIX)
    ADD_DEFINITIONS("-std=c++11")
ENDIF()
if(WIN32)
	ADD_DEFINITIONS("-D_USE_MATH_DEFINES")
ENDIF()

#---------------------------------------------------------------
#Set the projects
#---------------------------------------------------------------		
# Main
ADD_EXECUTABLE(Fit-Subdiv-to-3D-Points
	fit-subdiv-to-3d-points.cpp
    MeshTopology.cpp
    log3d.cpp
	SubdivEvaluator.cpp
	RigidTransform.cpp
	PLYParser.cpp
	Logger.cpp
	FPJParser.cpp
	BezierPatch.cpp
	Optimization/BaseFunctor.cpp
	Optimization/PosOnlyFunctor.cpp
	Optimization/PosOnlyWithRegFunctor.cpp
	Optimization/PosAndNormalsFunctor.cpp
	Optimization/PosAndNormalsWithRegFunctor.cpp
    MeshTopology.h
    log3d.h
	SubdivEvaluator.h
	RigidTransform.h
	PLYParser.h
	Logger.h
	FPJParser.h
	BezierPatch.h
	Optimization/BaseFunctor.h
	Optimization/PosOnlyFunctor.h
	Optimization/PosOnlyWithRegFunctor.h
	Optimization/PosAndNormalsFunctor.h
	Optimization/PosAndNormalsWithRegFunctor.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(Fit-Subdiv-to-3D-Points ${OSD_LIB})

# Tests
ADD_EXECUTABLE(QR_Solver_Test
	qr_solver_test.cpp
    MeshTopology.cpp
    log3d.cpp
	SubdivEvaluator.cpp
	RigidTransform.cpp
	PLYParser.cpp
	Logger.cpp
	FPJParser.cpp
	BezierPatch.cpp
	Optimization/BaseFunctor.cpp
	Optimization/PosOnlyFunctor.cpp
	Optimization/PosOnlyWithRegFunctor.cpp
	Optimization/PosAndNormalsFunctor.cpp
	Optimization/PosAndNormalsWithRegFunctor.cpp
    MeshTopology.h
    log3d.h
	SubdivEvaluator.h
	RigidTransform.h
	PLYParser.h
	Logger.h
	FPJParser.h
	BezierPatch.h
	Optimization/BaseFunctor.h
	Optimization/PosOnlyFunctor.h
	Optimization/PosOnlyWithRegFunctor.h
	Optimization/PosAndNormalsFunctor.h
	Optimization/PosAndNormalsWithRegFunctor.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	)

ADD_EXECUTABLE(QR_Diagonalization_Test
	qr_diagonalization_test.cpp
    Logger.cpp
	Logger.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(QR_Solver_Test ${OSD_LIB})