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
    SET(SSP_PATH "Z:/suitesparse-metis-for-windows-master/suitesparse-metis-for-windows-master/build/install" CACHE PATH "Path to SuiteSparse")
    SET(SSP_INCLUDE_DIRS "${SSP_PATH}/include")
    SET(SSP_LIBRARY_DIRS "${SSP_PATH}/lib")
    SET(SSP_BLAS_LIBRARY_DIRS "${SSP_PATH}/lib64/lapack_blas_windows")
    SET(SSP_LIB "libamd.lib;libbtf.lib;libcamd.lib;libccolamd.lib;libcholmod.lib;libcolamd.lib;libcxsparse.lib;libklu.lib;libldl.lib;libspqr.lib;libumfpack.lib;metis.lib;suitesparseconfig.lib;libblas.lib;liblapack.lib")
ELSEIF(UNIX)
    # OSD Compiled using GLFW 3.1.2
    SET(OSD_PATH "/usr/local" CACHE PATH "Path to osd")
    SET(OSD_INCLUDE_DIRS "${OSD_PATH}/include/opensubdiv")
    SET(OSD_LIBRARY_DIRS "${OSD_PATH}/lib")
    SET(OSD_LIB "${OSD_LIBRARY_DIRS}/libosdCPU.so")
ELSE()
    MESSAGE(FATAL_ERROR "Unrecognized OS")
ENDIF()
INCLUDE_DIRECTORIES(${OSD_INCLUDE_DIRS} ${SSP_INCLUDE_DIRS})
LINK_DIRECTORIES(${OSD_LIBRARY_DIRS} ${SSP_LIBRARY_DIRS} ${SSP_BLAS_LIBRARY_DIRS})

# Eigen
SET(CMAKE_PREFIX_PATH "Z:/Eigen_PR/build;Z:/Eigen_PR;Z:/OpenSubdiv/install/include;C:/Program Files/Eigen3/include/eigen3")
FIND_PACKAGE(Eigen3)
# Point to Eigen directly, if developing Eigen at the same time SET(EIGEN3_INCLUDE_DIR "C:/dev/eigen_pr")
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# For debug builds
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

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
	fit_subdiv_to_3d_points.cpp
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
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	)
	
TARGET_LINK_LIBRARIES(Fit-Subdiv-to-3D-Points ${OSD_LIB} ${SSP_LIB})

ADD_EXECUTABLE(Fit-Subdiv-Jacobian-Test
	fit_subdiv_jacobian_test.cpp
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
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	)
	
TARGET_LINK_LIBRARIES(Fit-Subdiv-Jacobian-Test ${OSD_LIB} ${SSP_LIB})

ADD_EXECUTABLE(ICP-Align-3D-Points
	icp_align_3d_points.cpp
    MeshTopology.cpp
    log3d.cpp
	SubdivEvaluator.cpp
	RigidTransform.cpp
	PLYParser.cpp
	Logger.cpp
	FPJParser.cpp
	BezierPatch.cpp
	Optimization/BaseFunctor.cpp
	Optimization/ICPFunctor.cpp
    MeshTopology.h
    log3d.h
	SubdivEvaluator.h
	RigidTransform.h
	PLYParser.h
	Logger.h
	FPJParser.h
	BezierPatch.h
	Optimization/BaseFunctor.h
	Optimization/ICPFunctor.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	)
	
TARGET_LINK_LIBRARIES(ICP-Align-3D-Points ${OSD_LIB} ${SSP_LIB})


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
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	)

ADD_EXECUTABLE(QR_Diagonalization_Test
	qr_diagonalization_test.cpp
    Logger.cpp
	Logger.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	)
	
ADD_EXECUTABLE(QR_Banded_Test
	qr_banded_test.cpp
    Logger.cpp
	Logger.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	)
		
ADD_EXECUTABLE(QR_Banded_Blocked_Test
	qr_banded_blocked_test.cpp
    Logger.cpp
	Logger.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	Eigen_ext/SparseBandedBlockedQR_Ext.h
	Eigen_ext/SparseBandedBlockedQR_Ext2.h
	Eigen_ext/SparseBandedBlockedQR_Ext3.h
	)

ADD_EXECUTABLE(QR_Banded_General_Test
	qr_banded_general_test.cpp
    Logger.cpp
	Logger.h
	Eigen_ext/eigen_extras.h
	Eigen_ext/BlockSparseQR_Ext.h
	Eigen_ext/SparseSubblockQR_Ext.h
	Eigen_ext/BlockDiagonalSparseQR_Ext.h
	Eigen_ext/SparseQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext.h
	Eigen_ext/SparseBandedQR_Ext2.h
	Eigen_ext/SparseBandedBlockedQR_Ext.h
	Eigen_ext/SparseBandedBlockedQR_Ext2.h
	Eigen_ext/SparseBandedBlockedQR_Ext3.h
	Eigen_ext/SparseBandedBlockedQR_General.h
	)

TARGET_LINK_LIBRARIES(QR_Solver_Test ${OSD_LIB} ${SSP_LIB})
TARGET_LINK_LIBRARIES(QR_Diagonalization_Test ${OSD_LIB} ${SSP_LIB})
TARGET_LINK_LIBRARIES(QR_Banded_Test ${OSD_LIB} ${SSP_LIB})
TARGET_LINK_LIBRARIES(QR_Banded_Blocked_Test ${OSD_LIB} ${SSP_LIB})
TARGET_LINK_LIBRARIES(QR_Banded_General_Test ${OSD_LIB} ${SSP_LIB})