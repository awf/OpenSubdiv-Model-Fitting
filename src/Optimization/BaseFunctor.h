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

template <int BlkRows, int BlkCols>
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
	//   The Jacobian computation takes an InputType, and its rows must easily convert to VectorType
	//   The increment_in_place operation takes InputType and StepType. 
	typedef VectorX VectorType;

	// Functor constructor
	BaseFunctor(Eigen::Index numParameters, Eigen::Index numResiduals, Eigen::Index numJacobianNonzeros, const Matrix3X& data_points, const MeshTopology& mesh);

	// Functor functions
	// 1. Evaluate the residuals at x
	int operator()(const InputType& x, ValueType& fvec);
	virtual void f_impl(const InputType& x, ValueType& fvec) = 0;

	// 2. Evaluate jacobian at x
	int df(const InputType& x, JacobianType& fjac);
	virtual void df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) = 0;

	// Update function
	void increment_in_place(InputType* x, StepType const& p);
	virtual void increment_in_place_impl(InputType* x, StepType const& p) = 0;

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

	const Index numParameters;
	const Index numResiduals;
	const Index numJacobianNonzeros;
	const Index rowStride;

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

	// QR for J1 for small subblocks
	typedef ColPivHouseholderQR<Matrix<Scalar, BlkRows, BlkCols> > DenseQRSolverSmallBlock;

	// QR for J1 is block diagonal
	typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolverSmallBlock> LeftSuperBlockSolver;

	// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;

	// QR for JPos is concatenation of the above.
	typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver_Pos;

	// QR for JNormal is concatenation of the above.
	typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver_Norm;

	// QR for JThinPlate is concatenation of the above.
	typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;

	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);


	/************ ENERGIES, GRADIENTS and UPDATES ************/
	void E_pos(const Matrix3X& S, const Matrix3X& data_points, const RigidTransform& rigidTransf, ValueType& fvec, const Eigen::Index rowOffset) {
		for (int i = 0; i < data_points.cols(); i++) {
			fvec.segment(i * this->rowStride + rowOffset, 3) = (S.col(i) - data_points.col(i));
		}
	}

	void E_normal(const Matrix3X& dSdu, const Matrix3X& dSdv, const Matrix3X& data_normals, const RigidTransform& rigidTransf, ValueType& fvec, const Eigen::Index rowOffset) {
		for (int i = 0; i < data_normals.cols(); i++) {
			// Compute normal from the first derivatives of the subdivision surface
			Vector3 normal = dSdu.col(i).cross(dSdv.col(i));
			normal.normalize();

			fvec.segment(i * this->rowStride + rowOffset, 3) = (normal - data_normals.col(i));
		}
	}
	
	void E_thinplate(const InputType& x, const RigidTransform& rigidTransf, ValueType &fvec, const Eigen::Index rowOffset) {
		/************************************************************************************************************/
		/* Evaluate subdivision surface at the control points */
		std::vector<SurfacePoint> us_cv;

		// Assign face index and UV coordinate to each control vertex
		int nFaces = int(mesh.quads.cols());

		// 1. Make a list of test points, e.g. corner of each face
		Matrix3X test_points(3, nFaces);
		std::vector<SurfacePoint> uvs{ size_t(nFaces),{ 0,{ 0.0, 0.0 } } };
		for (int i = 0; i < nFaces; ++i)
			uvs[i].face = i;
		evaluator.evaluateSubdivSurface(x.control_vertices, uvs, &test_points);

		for (int i = 0; i < x.control_vertices.cols(); i++) {
			// Closest test point
			Eigen::Index test_pt_index;
			(test_points.colwise() - x.control_vertices.col(i)).colwise().squaredNorm().minCoeff(&test_pt_index);
			us_cv[i] = uvs[test_pt_index];
		}

		// Now evaluate subdivision surface at USv corresponding to the control points
		Matrix3X S_cv;
		Matrix3X dSdu_cv, dSdv_cv;
		Matrix3X dSduu_cv, dSduv_cv, dSdvv_cv;
		evaluator.evaluateSubdivSurface(x.control_vertices, us_cv, &S_cv, 0, 0, 0, &dSdu_cv, &dSdv_cv, &dSduu_cv, &dSduv_cv, &dSdvv_cv);
		/************************************************************************************************************/

		// The thin plate energy can be evaluated at each control vertex separately
		// FixMe: Implement the correct integrated TP energy, this one doesn't correspond to the euqation from Tom's paper
		/*std::cout << dSduu_cv.rows() << std::endl;
		std::cout << dSduu_cv.cols() << std::endl;
		for (int i = 0; i < x.control_vertices.cols(); i++) {
			fvec.segment(rowOffset + i * 3, 3) = (dSduu_cv.col(i) + 2 * dSduv_cv.col(i) + dSdvv_cv.col(i));
		}*/
	}

	void dE_pos_d_X(const SubdivEvaluator::triplets_t &dSdX, const RigidTransform& rigidTransf,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		
		// Get the rotation as quaternion and then convert to matrix
		float lambda = rigidTransf.params().s1;

		std::cout << dSdX.size() << std::endl;
		for (int i = 0; i < dSdX.size(); ++i) {
			auto const& triplet = dSdX[i];
			assert(0 <= triplet.row() && triplet.row() < data_points.cols());
			assert(0 <= triplet.col() && triplet.col() < mesh.num_vertices);
			jvals.add(triplet.row() * this->rowStride + rowOffset + 0, colBase + triplet.col() * 3 + 0, lambda * triplet.value());
			jvals.add(triplet.row() * this->rowStride + rowOffset + 1, colBase + triplet.col() * 3 + 1, lambda * triplet.value());
			jvals.add(triplet.row() * this->rowStride + rowOffset + 2, colBase + triplet.col() * 3 + 2, lambda * triplet.value());
		}
	}

	void dE_pos_d_uv(const Matrix3X& dSdu, const Matrix3X& dSdv, const RigidTransform& rigidTransf,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		Eigen::Index nPoints = data_points.cols();

		// Get the rotation as quaternion and then convert to matrix
		float lambda = rigidTransf.params().s1;

		for (int i = 0; i < nPoints; i++) {
			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 0, lambda * dSdu(0, i));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 0, lambda * dSdu(1, i));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 0, lambda * dSdu(2, i));
			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 1, lambda * dSdv(0, i));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 1, lambda * dSdv(1, i));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 1, lambda * dSdv(2, i));
		}
	}

	void dE_normal_d_X(const SubdivEvaluator::triplets_t &dSudX, const SubdivEvaluator::triplets_t &dSvdX, 
		const Matrix3X& dSdu, const Matrix3X& dSdv, const RigidTransform& rigidTransf,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		for (int i = 0; i < dSudX.size(); ++i) {
			// Normals
			auto const& tripletSu = dSudX[i];
			auto const& tripletSv = dSvdX[i];
			Vector3 normal = dSdu.col(tripletSu.row()).cross(dSdv.col(tripletSu.row()));
			float nnorm = normal.norm();
			normal.normalize();

			Vector3 dndx, dndy, dndz;
			dndx << 0.0f, dSdu(2, tripletSu.row()) * tripletSv.value() - dSdv(2, tripletSu.row()) * tripletSu.value(),
				dSdv(1, tripletSu.row()) * tripletSu.value() - dSdu(1, tripletSu.row()) * tripletSv.value();
			dndy << dSdv(2, tripletSu.row()) * tripletSu.value() - dSdu(2, tripletSu.row()) * tripletSv.value(),
				0.0f,
				dSdu(0, tripletSu.row()) * tripletSv.value() - dSdv(0, tripletSu.row()) * tripletSu.value();
			dndz << dSdu(1, tripletSu.row()) * tripletSv.value() - dSdv(1, tripletSu.row()) * tripletSu.value(),
				dSdv(0, tripletSu.row()) * tripletSu.value() - dSdu(0, tripletSu.row()) * tripletSv.value(),
				0.0f;
			float ndndx = normal.transpose() * dndx;
			float ndndy = normal.transpose() * dndy;
			float ndndz = normal.transpose() * dndz;

			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 0, colBase + tripletSu.col() * 3 + 0, (1.0 / nnorm) * (dndx(0) - normal(0) * ndndx));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 1, colBase + tripletSu.col() * 3 + 0, (1.0 / nnorm) * (dndx(1) - normal(1) * ndndx));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 2, colBase + tripletSu.col() * 3 + 0, (1.0 / nnorm) * (dndx(2) - normal(2) * ndndx));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 0, colBase + tripletSu.col() * 3 + 1, (1.0 / nnorm) * (dndy(0) - normal(0) * ndndy));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 1, colBase + tripletSu.col() * 3 + 1, (1.0 / nnorm) * (dndy(1) - normal(1) * ndndy));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 2, colBase + tripletSu.col() * 3 + 1, (1.0 / nnorm) * (dndy(2) - normal(2) * ndndy));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 0, colBase + tripletSu.col() * 3 + 2, (1.0 / nnorm) * (dndz(0) - normal(0) * ndndz));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 1, colBase + tripletSu.col() * 3 + 2, (1.0 / nnorm) * (dndz(1) - normal(1) * ndndz));
			jvals.add(tripletSu.row() * this->rowStride + rowOffset + 2, colBase + tripletSu.col() * 3 + 2, (1.0 / nnorm) * (dndz(2) - normal(2) * ndndz));
		}
	}

	void dE_normal_d_uv(const Matrix3X& dSdu, const Matrix3X& dSdv, const Matrix3X& dSduu, const Matrix3X& dSduv, const Matrix3X& dSdvv, const RigidTransform& rigidTransf,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		Eigen::Index nPoints = data_points.cols();
		for (int i = 0; i < nPoints; i++) {
			// Normals
			Vector3 normal = dSdu.col(i).cross(dSdv.col(i));
			float nnorm = normal.norm();
			normal.normalize();

			Vector3 dndu = dSduu.col(i).cross(dSdv.col(i)) + dSdu.col(i).cross(dSduv.col(i));
			Vector3 dndv = dSduv.col(i).cross(dSdv.col(i)) + dSdu.col(i).cross(dSdvv.col(i));
			float ndndu = normal.transpose() * dndu;
			float ndndv = normal.transpose() * dndv;

			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 0, (1.0 / nnorm) * (dndu(0) - normal(0) * ndndu));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 0, (1.0 / nnorm) * (dndu(1) - normal(1) * ndndu));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 0, (1.0 / nnorm) * (dndu(2) - normal(2) * ndndu));
			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 1, (1.0 / nnorm) * (dndv(0) - normal(0) * ndndv));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 1, (1.0 / nnorm) * (dndv(1) - normal(1) * ndndv));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 1, (1.0 / nnorm) * (dndv(2) - normal(2) * ndndv));
		}
	}

	void inc_X(InputType* x, StepType const& p, const Eigen::Index colBase) {
		Index nPoints = data_points.cols();

		// Increment control vertices
		Index nVertices = x->nVertices();

		Map<VectorX>(x->control_vertices.data(), nVertices * 3) += p.segment(colBase, nVertices * 3);
	}

	void inc_uv(InputType* x, StepType const& p, const Eigen::Index colBase) {
		Index nPoints = data_points.cols();

		// Increment surface correspondences
		int loopers = 0;
		int totalhops = 0;
		for (int i = 0; i < nPoints; ++i) {
			Vector2 du = p.segment<2>(colBase + 2 * i);
			int nhops = increment_u_crossing_edges(x->control_vertices, x->us[i].face, x->us[i].u, du, &x->us[i].face, &x->us[i].u);
			if (nhops < 0)
				++loopers;
			totalhops += std::abs(nhops);
		}
		if (loopers > 0)
			std::cerr << "[" << totalhops / Scalar(nPoints) << " hops, " << loopers << " points looped]";
		else if (totalhops > 0)
			std::cerr << "[" << totalhops << "/" << Scalar(nPoints) << " hops]";
	}
};

// Hack to be able to split template classes into .h and .cpp files
#include "BaseFunctor.cpp"

#endif

