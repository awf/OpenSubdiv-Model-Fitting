#ifndef BASE_FUNCTOR_H
#define BASE_FUNCTOR_H

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include "../eigen_extras.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "../MeshTopology.h"
#include "../SubdivEvaluator.h"

#include "../RigidTransform.h"

using namespace Eigen;

struct BaseFunctor : Eigen::SparseFunctor<Scalar> {
	typedef Eigen::SparseFunctor<Scalar> Base;
	typedef typename Base::JacobianType JacobianType;

	// Variables for optimization live in InputType
	struct InputType {
		Matrix3X control_vertices;
		std::vector<SurfacePoint> us;
		RigidTransform rigidTransf;

		Index nVertices() const { return control_vertices.cols(); }
	};

	// And the optimization steps are computed using VectorType.
	// For subdivs (see xx), the correspondences are of type (int, Vec2) while the updates are of type (Vec2).
	// The interactions between InputType and VectorType are restricted to:
	//   The Jacobian computation takes an InputType, and its worows must easily convert to VectorType
	//   The increment_in_place operation takes InputType and StepType. 
	typedef VectorX VectorType;

	// Functor constructor
	BaseFunctor(Eigen::Index numParameters, Eigen::Index numResiduals, const Matrix3X& data_points, const MeshTopology& mesh);

	// Functor functions
	// 1. Evaluate the residuals at x
	virtual int operator()(const InputType& x, ValueType& fvec) = 0;

	// 2. Evaluate jacobian at x
	virtual int df(const InputType& x, JacobianType& fjac) = 0;

	// Update function
	virtual void increment_in_place(InputType* x, StepType const& p) = 0;

	// Input data
	Matrix3X data_points;

	// Topology (faces as vertex indices, fixed during shape optimization)
	MeshTopology mesh;

	// Subdivison surface evaluator
	SubdivEvaluator evaluator;

	// Workspace variables for evaluation
	Matrix3X S;
	Matrix3X dSdu;
	Matrix3X dSdv;
	Matrix3X dSduu, dSduv, dSdvv;
	SubdivEvaluator::triplets_t dSdX, dSudX, dSvdX;



	// Workspace initialization
	virtual void initWorkspace();

	// "Mesh walking" to update correspondences, as in Fig 3, Taylor et al, CVPR 2014, "Hand shape.."
	int increment_u_crossing_edges(Matrix3X const& X, int face, const Vector2& u, const Vector2& du, int* new_face_out, Vector2* new_u_out);

	Scalar estimateNorm(InputType const& x, StepType const& diag);

	// 5. Describe the QR solvers
	// For generic Jacobian, one might use this Dense QR solver.
	typedef SparseQR<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;

	// But for optimal performance, declare QRSolver that understands the sparsity structure.
	// Here it's block-diagonal LHS with dense RHS
	//
	// J1 = [J11   0   0 ... 0
	//         0 J12   0 ... 0
	//                   ...
	//         0   0   0 ... J1N];
	// And 
	// J = [J1 J2];

	// QR for J1 subblocks is 2x1
	typedef ColPivHouseholderQR<Matrix<Scalar, 3, 2> > DenseQRSolver3x2;

	// QR for J1 is block diagonal
	typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolver3x2> LeftSuperBlockSolver;

	// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;

	// QR for J is concatenation of the above.
	typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;

	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);
};

#endif

