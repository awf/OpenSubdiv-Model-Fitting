#ifndef BASE_FUNCTOR_H
#define BASE_FUNCTOR_H

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include "../Eigen_ext/eigen_extras.h"
#include "../Eigen_ext/BlockSparseQR_Ext.h"

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

	struct DataConstraint {
		Eigen::Index cvIdx;	// Model control-vertex index
		Eigen::Index ptIdx;	// Data point index

		DataConstraint()
			: cvIdx(0), ptIdx(0) {
		}

		DataConstraint(const Eigen::Index cvi, const Eigen::Index pti)
			: cvIdx(cvi), ptIdx(pti) {
		}
	};
	// Type vector of point constraints
	typedef std::vector<DataConstraint> DataConstraints;

	// And the optimization steps are computed using VectorType.
	// For subdivs (see xx), the correspondences are of type (int, Vec2) while the updates are of type (Vec2).
	// The interactions between InputType and VectorType are restricted to:
	//   The Jacobian computation takes an InputType, and its rows must easily convert to VectorType
	//   The increment_in_place operation takes InputType and StepType. 
	typedef VectorX VectorType;

	// Functor constructor
	BaseFunctor(Eigen::Index numParameters, Eigen::Index numResiduals, Eigen::Index numJacobianNonzeros, const Matrix3X& data_points, const MeshTopology& mesh, const DataConstraints& constraints = DataConstraints());

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

	// Vector of data constraints
	DataConstraints data_constraints;
	
	// Input data
	Matrix3X data_points;

	// Topology (faces as vertex indices, fixed during shape optimization)
	MeshTopology mesh;

	// Subdivison surface evaluator
	SubdivEvaluator evaluator;

	Eigen::Index nDataPoints() const {
		return data_points.cols();
	}
	Eigen::Index nDataConstraints() const {
		return data_constraints.size();
	}

	// Workspace variables for evaluation
	struct SubdivSurface {
		Matrix3X S;
		Matrix3X dSdu;
		Matrix3X dSdv;
		Matrix3X dSduu, dSduv, dSdvv;
		SubdivEvaluator::triplets_t dSdX, dSudX, dSvdX;

		bool update;

		void init(Eigen::Index nPoints) {
			S.resize(3, nPoints);
			dSdu.resize(3, nPoints);
			dSdv.resize(3, nPoints);
			dSduu.resize(3, nPoints);
			dSduv.resize(3, nPoints);
			dSdvv.resize(3, nPoints);

			update = true;
		}
	};
	SubdivSurface ssurf;
	SubdivSurface ssurf_tsr;
	SubdivSurface ssurf_sr;
	SubdivSurface ssurf_r;

	const Index numParameters;
	const Index numResiduals;
	const Index numJacobianNonzeros;
	const Index rowStride;

	// Weighting parameters for the energy terms (where needed)
	struct EnergyWeights {
		double thinplate;	// Weight for the thin plate energy
		double constraints;	// Weight for the point constraints
	
		EnergyWeights() 
			: thinplate(1.0), constraints(1.0) {
		}
	};
	EnergyWeights eWeights;

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

	// QR solver is concatenation of the above.
	typedef BlockSparseQR_Ext<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;

	typedef SchurlikeQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(SchurlikeQRSolver &qr);

	/************ UTILITY FUNCTIONS FOR ENERGIES ************/
	void computUVAtControlPoints(const InputType& x, std::vector<SurfacePoint> &us_cv) {
		/* Evaluate subdivision surface at the control points */
		us_cv.resize(x.control_vertices.cols());

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
	}

	void computUVAtConstraintPoints(const InputType& x, const std::vector<DataConstraint>& c, std::vector<SurfacePoint> &us_c) {
		/* Evaluate subdivision surface at the control points */
		us_c.resize(c.size());

		// Assign face index and UV coordinate to each control vertex
		int nFaces = int(mesh.quads.cols());

		// 1. Make a list of test points, e.g. corner of each face
		Matrix3X test_points(3, nFaces);
		std::vector<SurfacePoint> uvs{ size_t(nFaces),{ 0,{ 0.0, 0.0 } } };
		for (int i = 0; i < nFaces; ++i)
			uvs[i].face = i;
		evaluator.evaluateSubdivSurface(x.control_vertices, uvs, &test_points);

		for (int i = 0; i < c.size(); i++) {
			// Closest test point
			Eigen::Index test_pt_index;
			(test_points.colwise() - x.control_vertices.col(c.at(i).cvIdx)).colwise().squaredNorm().minCoeff(&test_pt_index);
			us_c[i] = uvs[test_pt_index];
		}
	}

	void computeThinPlateMatrix(const InputType& x, MatrixXX &tpe) {
		/* Evaluate subdivision surface at the control points */
		std::vector<SurfacePoint> us_cv;
		this->computUVAtControlPoints(x, us_cv);

		// Retrieve bicubic patches around the control points of the subdivision surface
		evaluator.thinPlateEnergy(x.control_vertices, us_cv, tpe);
	}

	void evaluateSubdivisionSurfaceAtConstraintPoints(const InputType& x, const std::vector<DataConstraint>& c, SubdivSurface& ss, const Eigen::Matrix4f& transf = Eigen::Matrix4f::Identity()) {
		/* Evaluate subdivision surface at the control points */
		std::vector<SurfacePoint> us_c;
		this->computUVAtConstraintPoints(x, c, us_c);

		// Add rigid transformation parameters
		// Matrix implementation of (t + s * (R * pt))
		Matrix3X tCVs(3, x.nVertices());
		for (int i = 0; i < x.nVertices(); i++) {
			Eigen::Vector4f pt;
			pt << x.control_vertices(0, i), x.control_vertices(1, i), x.control_vertices(2, i), 1.0f;
			pt = transf * pt;
			tCVs(0, i) = pt(0);
			tCVs(1, i) = pt(1);
			tCVs(2, i) = pt(2);
		}
		evaluator.evaluateSubdivSurface(tCVs, us_c, &(ss.S), &(ss.dSdX), &(ss.dSudX), &(ss.dSvdX), &(ss.dSdu), &(ss.dSdv), &(ss.dSduu), &(ss.dSduv), &(ss.dSdvv));
	}

	void evaluateSubdivisionSurface(const InputType& x, SubdivSurface& ss) {
		if (ss.update) {
			evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &(ss.S), &(ss.dSdX), &(ss.dSudX), &(ss.dSvdX), &(ss.dSdu), &(ss.dSdv), &(ss.dSduu), &(ss.dSduv), &(ss.dSdvv));
			ss.update = false;
		}
	}

	void evaluateTransformedSubdivisionSurface(const InputType& x, const Eigen::Matrix4f& transf, SubdivSurface& ss) {
		if (ss.update) {
			// Add rigid transformation parameters
			// Matrix implementation of (t + s * (R * pt))
			Matrix3X tCVs(3, x.nVertices());
			for (int i = 0; i < x.nVertices(); i++) {
				Eigen::Vector4f pt;
				pt << x.control_vertices(0, i), x.control_vertices(1, i), x.control_vertices(2, i), 1.0f;
				pt = transf * pt;
				tCVs(0, i) = pt(0);
				tCVs(1, i) = pt(1);
				tCVs(2, i) = pt(2);
			}
			evaluator.evaluateSubdivSurface(tCVs, x.us, &(ss.S), &(ss.dSdX), &(ss.dSudX), &(ss.dSvdX), &(ss.dSdu), &(ss.dSdv), &(ss.dSduu), &(ss.dSduv), &(ss.dSdvv));
		
			ss.update = false;
		}
	}
	/************ ENERGIES, GRADIENTS and UPDATES ************/
	/************ ENERGIES ************/
	void E_pos(const InputType& x, const Matrix3X& data_points, ValueType& fvec, const Eigen::Index rowOffset) {
		// Compute subdivision surface
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.translation() * x.rigidTransf.scaling() * x.rigidTransf.rotation(), this->ssurf_tsr);

		for (int i = 0; i < this->nDataPoints(); i++) {
			fvec.segment(i * this->rowStride + rowOffset, 3) = (this->ssurf_tsr.S.col(i) - data_points.col(i));
		}
	}

	void E_normal(const InputType& x, const Matrix3X& data_normals, ValueType& fvec, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.rotation(), this->ssurf_r);
		//this->evaluateSubdivisionSurface(x, this->ssurf);

		for (int i = 0; i < data_normals.cols(); i++) {
			// Compute normal from the first derivatives of the subdivision surface
			Vector3 normal = this->ssurf_r.dSdu.col(i).cross(this->ssurf_r.dSdv.col(i));
			normal.normalize();

			fvec.segment(i * this->rowStride + rowOffset, 3) = (normal - data_normals.col(i));
		}
	}

	void E_constraints(const InputType& x, const Matrix3X& data_points, const std::vector<DataConstraint>& cs, ValueType& fvec, const Eigen::Index rowOffset) {
		// Evaluate subdivision surface at control points
		SubdivSurface ssurf_cs;
		ssurf_cs.init(this->nDataConstraints());
		this->evaluateSubdivisionSurfaceAtConstraintPoints(x, cs, ssurf_cs, x.rigidTransf.translation() * x.rigidTransf.scaling() * x.rigidTransf.rotation());

		// Replace the position residuals for the constrained points
		for (int i = 0; i < this->nDataConstraints(); i++) {
			fvec.segment(rowOffset + i * 3, 3) = this->eWeights.constraints * (ssurf_cs.S.col(i) - data_points.col(cs.at(i).ptIdx));
		}
	}
	
	void E_thinplate(const InputType& x, ValueType &fvec, const Eigen::Index rowOffset) {
		// Compute thin plate energy matrix
		MatrixXX tpe;
		this->computeThinPlateMatrix(x, tpe);
		
		// The thin plate energy evaluated at each control vertex
		tpe = tpe.sqrt() * x.control_vertices.transpose();
		for (int i = 0; i < tpe.rows(); i++) {
			fvec(rowOffset + i * 3 + 0) = this->eWeights.thinplate * tpe(i, 0);
			fvec(rowOffset + i * 3 + 1) = this->eWeights.thinplate * tpe(i, 1);
			fvec(rowOffset + i * 3 + 2) = this->eWeights.thinplate * tpe(i, 2);
		}
	}

	/************ GRADIENTS ************/
	void dE_pos_d_X(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.rotation(), this->ssurf_r);

		for (int i = 0; i < this->ssurf_r.dSdX.size(); ++i) {
			auto const& triplet = this->ssurf_r.dSdX[i];
			assert(0 <= triplet.row() && triplet.row() < this->nDataPoints());
			assert(0 <= triplet.col() && triplet.col() < mesh.num_vertices);
			jvals.add(triplet.row() * this->rowStride + rowOffset + 0, colBase + triplet.col() * 3 + 0, x.rigidTransf.params().s1 * triplet.value());
			jvals.add(triplet.row() * this->rowStride + rowOffset + 1, colBase + triplet.col() * 3 + 1, x.rigidTransf.params().s2 * triplet.value());
			jvals.add(triplet.row() * this->rowStride + rowOffset + 2, colBase + triplet.col() * 3 + 2, x.rigidTransf.params().s3 * triplet.value());
		}
	}

	void dE_pos_d_uv(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.rotation(), this->ssurf_r);

		for (int i = 0; i < this->nDataPoints(); i++) {
			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 0, x.rigidTransf.params().s1 * this->ssurf_r.dSdu(0, i));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 0, x.rigidTransf.params().s2 * this->ssurf_r.dSdu(1, i));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 0, x.rigidTransf.params().s3 * this->ssurf_r.dSdu(2, i));
			jvals.add(this->rowStride * i + rowOffset + 0, colBase + 2 * i + 1, x.rigidTransf.params().s1 * this->ssurf_r.dSdv(0, i));
			jvals.add(this->rowStride * i + rowOffset + 1, colBase + 2 * i + 1, x.rigidTransf.params().s2 * this->ssurf_r.dSdv(1, i));
			jvals.add(this->rowStride * i + rowOffset + 2, colBase + 2 * i + 1, x.rigidTransf.params().s3 * this->ssurf_r.dSdv(2, i));
		}
	}

	void dE_pos_d_rst(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateSubdivisionSurface(x, this->ssurf);
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.rotation(), this->ssurf_r);

		Eigen::Index t_base = colBase + 0;
		Eigen::Index s_base = colBase + 3;
		Eigen::Index r_base = colBase + 6;

		// Compute derivatives of the rotation wrt rotation parameters
		Eigen::Vector3f v;
		v << x.rigidTransf.params().r1, x.rigidTransf.params().r2, x.rigidTransf.params().r3;
		Eigen::Vector4f q;
		Eigen::Matrix4f R;
		Eigen::MatrixXf dRdv = Eigen::MatrixXf::Zero(9, 3);
		RigidTransform::rotationToQuaternion(v, q, &dRdv);

		for (int i = 0; i < this->nDataPoints(); i++) {
			jvals.add(this->rowStride * i + rowOffset + 0, t_base + 0, 1.0);
			jvals.add(this->rowStride * i + rowOffset + 1, t_base + 1, 1.0);
			jvals.add(this->rowStride * i + rowOffset + 2, t_base + 2, 1.0);

			jvals.add(this->rowStride * i + rowOffset + 0, s_base + 0, this->ssurf_r.S(0, i));
			jvals.add(this->rowStride * i + rowOffset + 1, s_base + 1, this->ssurf_r.S(1, i));
			jvals.add(this->rowStride * i + rowOffset + 2, s_base + 2, this->ssurf_r.S(2, i));

			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 0, (x.rigidTransf.params().s1 * dRdv(0, 0) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 0) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 * dRdv(6, 0) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 1, (x.rigidTransf.params().s1 * dRdv(0, 1) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 1) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 * dRdv(6, 1) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 2, (x.rigidTransf.params().s1 * dRdv(0, 2) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 2) * this->ssurf.S(1, i)
				+ x.rigidTransf.params().s3 *  dRdv(6, 2) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 0, (x.rigidTransf.params().s1 * dRdv(1, 0) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 0) * this->ssurf.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(7, 0) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 1, (x.rigidTransf.params().s1 * dRdv(1, 1) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 1) * this->ssurf.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(7, 1) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 2, (x.rigidTransf.params().s1 * dRdv(1, 2) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 2) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 * dRdv(7, 2) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 0, (x.rigidTransf.params().s1 * dRdv(2, 0) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 0) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 * dRdv(8, 0) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 1, (x.rigidTransf.params().s1 * dRdv(2, 1) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 1) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 *  dRdv(8, 1) * this->ssurf.S(2, i)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 2, (x.rigidTransf.params().s1 * dRdv(2, 2) * this->ssurf.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 2) * this->ssurf.S(1, i) 
				+ x.rigidTransf.params().s3 * dRdv(8, 2) * this->ssurf.S(2, i)));
		}
	}

	void dE_normal_d_X(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateSubdivisionSurface(x, this->ssurf);

		for (int i = 0; i < this->ssurf.dSudX.size(); ++i) {
			// Normals
			auto const& tripletSu = this->ssurf.dSudX[i];
			auto const& tripletSv = this->ssurf.dSvdX[i];
			Vector3 normal = this->ssurf.dSdu.col(tripletSu.row()).cross(this->ssurf.dSdv.col(tripletSu.row()));
			float nnorm = normal.norm();
			normal.normalize();

			Vector3 dndx, dndy, dndz;
			dndx << 0.0f, this->ssurf.dSdu(2, tripletSu.row()) * tripletSv.value() - this->ssurf.dSdv(2, tripletSu.row()) * tripletSu.value(),
				this->ssurf.dSdv(1, tripletSu.row()) * tripletSu.value() - this->ssurf.dSdu(1, tripletSu.row()) * tripletSv.value();
			dndy << this->ssurf.dSdv(2, tripletSu.row()) * tripletSu.value() - this->ssurf.dSdu(2, tripletSu.row()) * tripletSv.value(),
				0.0f,
				this->ssurf.dSdu(0, tripletSu.row()) * tripletSv.value() - this->ssurf.dSdv(0, tripletSu.row()) * tripletSu.value();
			dndz << this->ssurf.dSdu(1, tripletSu.row()) * tripletSv.value() - this->ssurf.dSdv(1, tripletSu.row()) * tripletSu.value(),
				this->ssurf.dSdv(0, tripletSu.row()) * tripletSu.value() - this->ssurf.dSdu(0, tripletSu.row()) * tripletSv.value(),
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

	void dE_normal_d_uv(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateSubdivisionSurface(x, this->ssurf);

		for (int i = 0; i < this->nDataPoints(); i++) {
			// Normals
			Vector3 normal = this->ssurf.dSdu.col(i).cross(this->ssurf.dSdv.col(i));
			float nnorm = normal.norm();
			normal.normalize();

			Vector3 dndu = this->ssurf.dSduu.col(i).cross(this->ssurf.dSdv.col(i)) + this->ssurf.dSdu.col(i).cross(this->ssurf.dSduv.col(i));
			Vector3 dndv = this->ssurf.dSduv.col(i).cross(this->ssurf.dSdv.col(i)) + this->ssurf.dSdu.col(i).cross(this->ssurf.dSdvv.col(i));
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

	void dE_normal_d_rst(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate surface at x
		this->evaluateSubdivisionSurface(x, this->ssurf);

		Eigen::Index t_base = colBase + 0;
		Eigen::Index s_base = colBase + 3;
		Eigen::Index r_base = colBase + 6;

		// Compute derivatives of the rotation wrt rotation parameters
		Eigen::Vector3f v;
		v << x.rigidTransf.params().r1, x.rigidTransf.params().r2, x.rigidTransf.params().r3;
		Eigen::Vector4f q;
		Eigen::Matrix4f R;
		Eigen::MatrixXf dRdv = Eigen::MatrixXf::Zero(9, 3);
		RigidTransform::rotationToQuaternion(v, q, &dRdv);

		for (int i = 0; i < this->nDataPoints(); i++) {
			// Compute normal from the first derivatives of the subdivision surface
			Vector3 normal = this->ssurf.dSdu.col(i).cross(this->ssurf.dSdv.col(i));
			normal.normalize();

			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 0, (x.rigidTransf.params().s1 * dRdv(0, 0) * normal(0) + x.rigidTransf.params().s2 * dRdv(3, 0) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(6, 0) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 1, (x.rigidTransf.params().s1 * dRdv(0, 1) * normal(0) + x.rigidTransf.params().s2 * dRdv(3, 1) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(6, 1) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 0, r_base + 2, (x.rigidTransf.params().s1 * dRdv(0, 2) * normal(0) + x.rigidTransf.params().s2 * dRdv(3, 2) * normal(1)
				+ x.rigidTransf.params().s3 *  dRdv(6, 2) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 0, (x.rigidTransf.params().s1 * dRdv(1, 0) * normal(0) + x.rigidTransf.params().s2 * dRdv(4, 0) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(7, 0) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 1, (x.rigidTransf.params().s1 * dRdv(1, 1) * normal(0) + x.rigidTransf.params().s2 * dRdv(4, 1) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(7, 1) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 1, r_base + 2, (x.rigidTransf.params().s1 * dRdv(1, 2) * normal(0) + x.rigidTransf.params().s2 * dRdv(4, 2) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(7, 2) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 0, (x.rigidTransf.params().s1 * dRdv(2, 0) * normal(0) + x.rigidTransf.params().s2 * dRdv(5, 0) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(8, 0) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 1, (x.rigidTransf.params().s1 * dRdv(2, 1) * normal(0) + x.rigidTransf.params().s2 * dRdv(5, 1) * normal(1)
				+ x.rigidTransf.params().s3 *  dRdv(8, 1) * normal(2)));
			jvals.add(this->rowStride * i + rowOffset + 2, r_base + 2, (x.rigidTransf.params().s1 * dRdv(2, 2) * normal(0) + x.rigidTransf.params().s2 * dRdv(5, 2) * normal(1)
				+ x.rigidTransf.params().s3 * dRdv(8, 2) * normal(2)));
		}
	}

	void dE_constraints_d_X(const InputType& x, std::vector<DataConstraint>& cs,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate subdivision surface at control points
		SubdivSurface ssurf_cs;
		ssurf_cs.init(this->nDataConstraints());
		this->evaluateSubdivisionSurfaceAtConstraintPoints(x, cs, ssurf_cs, x.rigidTransf.rotation());

		for (int i = 0; i < ssurf_cs.dSdX.size(); ++i) {
			auto const& triplet = ssurf_cs.dSdX[i];
			assert(0 <= triplet.row() && triplet.row() < this->nDataPoints());
			assert(0 <= triplet.col() && triplet.col() < mesh.num_vertices);
			jvals.add(rowOffset + triplet.row() * 3 + 0, colBase + triplet.col() * 3 + 0, this->eWeights.constraints * x.rigidTransf.params().s1 * triplet.value());
			jvals.add(rowOffset + triplet.row() * 3 + 1, colBase + triplet.col() * 3 + 1, this->eWeights.constraints * x.rigidTransf.params().s2 * triplet.value());
			jvals.add(rowOffset + triplet.row() * 3 + 2, colBase + triplet.col() * 3 + 2, this->eWeights.constraints * x.rigidTransf.params().s3 * triplet.value());
		}
	}

	void dE_constraints_d_rst(const InputType& x, std::vector<DataConstraint>& cs,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {
		// Evaluate subdivision surface at control points
		SubdivSurface ssurf_cs, ssurf_cs_r;
		ssurf_cs.init(this->nDataConstraints());
		ssurf_cs_r.init(this->nDataConstraints());
		this->evaluateSubdivisionSurfaceAtConstraintPoints(x, cs, ssurf_cs);
		this->evaluateSubdivisionSurfaceAtConstraintPoints(x, cs, ssurf_cs_r, x.rigidTransf.rotation());

		Eigen::Index t_base = colBase + 0;
		Eigen::Index s_base = colBase + 3;
		Eigen::Index r_base = colBase + 6;

		// Compute derivatives of the rotation wrt rotation parameters
		Eigen::Vector3f v;
		v << x.rigidTransf.params().r1, x.rigidTransf.params().r2, x.rigidTransf.params().r3;
		Eigen::Vector4f q;
		Eigen::Matrix4f R;
		Eigen::MatrixXf dRdv = Eigen::MatrixXf::Zero(9, 3);
		RigidTransform::rotationToQuaternion(v, q, &dRdv);
		
		for (int i = 0; i < this->nDataConstraints(); i++) {
			jvals.add(rowOffset + i * 3 + 0, t_base + 0, this->eWeights.constraints * 1.0);
			jvals.add(rowOffset + i * 3 + 1, t_base + 1, this->eWeights.constraints * 1.0);
			jvals.add(rowOffset + i * 3 + 2, t_base + 2, this->eWeights.constraints * 1.0);

			jvals.add(rowOffset + i * 3 + 0, s_base + 0, this->eWeights.constraints * ssurf_cs_r.S(0, i));
			jvals.add(rowOffset + i * 3 + 1, s_base + 1, this->eWeights.constraints * ssurf_cs_r.S(1, i));
			jvals.add(rowOffset + i * 3 + 2, s_base + 2, this->eWeights.constraints * ssurf_cs_r.S(2, i));

			jvals.add(rowOffset + i * 3 + 0, r_base + 0, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(0, 0) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 0) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(6, 0) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 0, r_base + 1, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(0, 1) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 1) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(6, 1) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 0, r_base + 2, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(0, 2) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(3, 2) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 *  dRdv(6, 2) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 1, r_base + 0, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(1, 0) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 0) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(7, 0) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 1, r_base + 1, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(1, 1) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 1) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(7, 1) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 1, r_base + 2, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(1, 2) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(4, 2) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(7, 2) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 2, r_base + 0, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(2, 0) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 0) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(8, 0) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 2, r_base + 1, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(2, 1) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 1) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(8, 1) * ssurf_cs.S(2, i)));
			jvals.add(rowOffset + i * 3 + 2, r_base + 2, this->eWeights.constraints * (x.rigidTransf.params().s1 * dRdv(2, 2) * ssurf_cs.S(0, i) + x.rigidTransf.params().s2 * dRdv(5, 2) * ssurf_cs.S(1, i)
				+ x.rigidTransf.params().s3 * dRdv(8, 2) * ssurf_cs.S(2, i)));
		}
	}

	void dE_thinplate_d_X(const InputType& x, 
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset) {

		// Compute thin plate energy matrix
		MatrixXX tpe;
		this->computeThinPlateMatrix(x, tpe);
		
		// FixMe: Ignore the off-diagonal elements (leave them 0)
		// Finite-difference derivatives would however compute some rather small off-diagonal values there
		tpe = tpe.sqrt();
		for (int i = 0; i < tpe.rows(); i++) {
			jvals.add(rowOffset + i * 3 + 0, colBase + i * 3 + 0, this->eWeights.thinplate * tpe(i, i));
			jvals.add(rowOffset + i * 3 + 1, colBase + i * 3 + 1, this->eWeights.thinplate * tpe(i, i));
			jvals.add(rowOffset + i * 3 + 2, colBase + i * 3 + 2, this->eWeights.thinplate * tpe(i, i));
		}
	}

	/************ UPDATES ************/
	void inc_X(InputType* x, StepType const& p, const Eigen::Index colBase) {
		// Increment control vertices
		Index nVertices = x->nVertices();

		Map<VectorX>(x->control_vertices.data(), nVertices * 3) += p.segment(colBase, nVertices * 3);
	}

	void inc_uv(InputType* x, StepType const& p, const Eigen::Index colBase) {
		// Increment surface correspondences
		int loopers = 0;
		int totalhops = 0;
		for (int i = 0; i < this->nDataPoints(); ++i) {
			Vector2 du = p.segment<2>(colBase + 2 * i);
			int nhops = increment_u_crossing_edges(x->control_vertices, x->us[i].face, x->us[i].u, du, &x->us[i].face, &x->us[i].u);
			if (nhops < 0)
				++loopers;
			totalhops += std::abs(nhops);
		}
		if (loopers > 0)
			std::cerr << "[" << totalhops / Scalar(this->nDataPoints()) << " hops, " << loopers << " points looped]";
		else if (totalhops > 0)
			std::cerr << "[" << totalhops << "/" << Scalar(this->nDataPoints()) << " hops]";
	}

	void inc_rst(InputType* x, StepType const& p, const Eigen::Index colBase) {
		// Increment transformation parameters
		Eigen::Index t_base = colBase + 0;
		Eigen::Index s_base = colBase + 3;
		Eigen::Index r_base = colBase + 6;

		// Increment translation parameteres
		Vector3 tNew = p.segment<3>(t_base);
		float t1 = x->rigidTransf.params().t1 + tNew(0);
		float t2 = x->rigidTransf.params().t2 + tNew(1);
		float t3 = x->rigidTransf.params().t3 + tNew(2);
		x->rigidTransf.setTranslation(t1, t2, t3);
		// Increment scaling parameters
		Vector3 sNew = p.segment<3>(s_base);
		float s1 = x->rigidTransf.params().s1 + sNew(0);
		float s2 = x->rigidTransf.params().s2 + sNew(1);
		float s3 = x->rigidTransf.params().s3 + sNew(2);
		x->rigidTransf.setScaling(s1, s2, s3);
		// Increment rotation parameters
		Vector3 rNew = p.segment<3>(r_base);
		float r1 = x->rigidTransf.params().r1 + rNew(0);
		float r2 = x->rigidTransf.params().r2 + rNew(1);
		float r3 = x->rigidTransf.params().r3 + rNew(2);
		x->rigidTransf.setRotation(r1, r2, r3);
	}
};

// Hack to be able to split template classes into .h and .cpp files
#include "BaseFunctor.cpp"

#endif

