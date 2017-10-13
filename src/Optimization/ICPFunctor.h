#ifndef ICP_FUNCTOR_H
#define ICP_FUNCTOR_H

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include "../Eigen_ext/eigen_extras.h"
#include "../Eigen_ext/SparseQR_Ext.h"

#include <Eigen/SparseCore>

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>

#include "../MeshTopology.h"
#include "../SubdivEvaluator.h"

#include "../RigidTransform.h"

using namespace Eigen;

typedef Index SparseDataType;
//typedef SuiteSparse_long SparseDataType;

struct ICPFunctor : Eigen::SparseFunctor<Scalar, SparseDataType> {
	typedef Eigen::SparseFunctor<Scalar, SparseDataType> Base;
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
	ICPFunctor(const Matrix3X& data_points, const MeshTopology& mesh);

	// Functor functions
	// 1. Evaluate the residuals at x
	int operator()(const InputType& x, ValueType& fvec);
	virtual void f_impl(const InputType& x, ValueType& fvec);

	// 2. Evaluate jacobian at x
	int df(const InputType& x, JacobianType& fjac);
	virtual void df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals);

	// Update function
	void increment_in_place(InputType* x, StepType const& p);
	virtual void increment_in_place_impl(InputType* x, StepType const& p);
	
	// Input data
	Matrix3X data_points;

	// Topology (faces as vertex indices, fixed during shape optimization)
	MeshTopology mesh;

	// Subdivison surface evaluator
	SubdivEvaluator evaluator;

	Eigen::Index nDataPoints() const {
		return data_points.cols();
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
	SubdivSurface ssurf_r;

	const Index numParameters;
	const Index numResiduals;
	const Index numJacobianNonzeros;
	const Index rowStride;

	// Workspace initialization
	virtual void initWorkspace();

	Scalar estimateNorm(InputType const& x, StepType const& diag);

	// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
	typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseQRSolver;
	typedef SparseQR<JacobianType, COLAMDOrdering<SparseDataType> > SparseQRSolver;

	typedef SparseQRSolver QRSolver;

	// And tell the algorithm how to set the QR parameters.
	virtual void initQRSolver(QRSolver &qr);

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
	void E_pos(const InputType& x, const Matrix3X& data_points, ValueType& fvec, const Eigen::Index rowOffset, const Eigen::Index blockOffset) {
		// Compute subdivision surface
		this->evaluateTransformedSubdivisionSurface(x, x.rigidTransf.translation() * x.rigidTransf.scaling() * x.rigidTransf.rotation(), this->ssurf_tsr);

		for (int i = 0; i < this->nDataPoints(); i++) {
			fvec.segment(rowOffset + i * this->rowStride + blockOffset, 3) = (this->ssurf_tsr.S.col(i) - data_points.col(i));
			//fvec(i * this->rowStride + rowOffset + 2) = 0.0;
		}
	}

	/************ GRADIENTS ************/
	void dE_pos_d_rst(const InputType& x,
		Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals, const Eigen::Index colBase, const Eigen::Index rowOffset, const Eigen::Index blockOffset) {
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
		Eigen::Matrix4f R = x.rigidTransf.rotation();
		Eigen::MatrixXf dRdv = Eigen::MatrixXf::Zero(9, 3);
		RigidTransform::rotationToQuaternion(v, q, &dRdv);

		for (int i = 0; i < this->nDataPoints(); i++) {
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 0, t_base + 0, 1.0);
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 1, t_base + 1, 1.0);
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 2, t_base + 2, 1.0);

			jvals.add(rowOffset + this->rowStride * i + blockOffset + 0, s_base + 0, this->ssurf_r.S(0, i));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 1, s_base + 1, this->ssurf_r.S(1, i));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 2, s_base + 2, this->ssurf_r.S(2, i));

			jvals.add(rowOffset + this->rowStride * i + blockOffset + 0, r_base + 0, x.rigidTransf.params().s1 * (dRdv(0, 0) * this->ssurf.S(0, i) + dRdv(3, 0) * this->ssurf.S(1, i) + dRdv(6, 0) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 0, r_base + 1, x.rigidTransf.params().s1 * (dRdv(0, 1) * this->ssurf.S(0, i) + dRdv(3, 1) * this->ssurf.S(1, i) + dRdv(6, 1) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 0, r_base + 2, x.rigidTransf.params().s1 * (dRdv(0, 2) * this->ssurf.S(0, i) + dRdv(3, 2) * this->ssurf.S(1, i) + dRdv(6, 2) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 1, r_base + 0, x.rigidTransf.params().s2 * (dRdv(1, 0) * this->ssurf.S(0, i) + dRdv(4, 0) * this->ssurf.S(1, i) + dRdv(7, 0) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 1, r_base + 1, x.rigidTransf.params().s2 * (dRdv(1, 1) * this->ssurf.S(0, i) + dRdv(4, 1) * this->ssurf.S(1, i) + dRdv(7, 1) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 1, r_base + 2, x.rigidTransf.params().s2 * (dRdv(1, 2) * this->ssurf.S(0, i) + dRdv(4, 2) * this->ssurf.S(1, i) + dRdv(7, 2) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 2, r_base + 0, x.rigidTransf.params().s3 * (dRdv(2, 0) * this->ssurf.S(0, i) + dRdv(5, 0) * this->ssurf.S(1, i) + dRdv(8, 0) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 2, r_base + 1, x.rigidTransf.params().s3 * (dRdv(2, 1) * this->ssurf.S(0, i) + dRdv(5, 1) * this->ssurf.S(1, i) + dRdv(8, 1) * this->ssurf.S(2, i)));
			jvals.add(rowOffset + this->rowStride * i + blockOffset + 2, r_base + 2, x.rigidTransf.params().s3 * (dRdv(2, 2) * this->ssurf.S(0, i) + dRdv(5, 2) * this->ssurf.S(1, i) + dRdv(8, 2) * this->ssurf.S(2, i)));
		}
	}

	/************ UPDATES ************/
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
		//x->rigidTransf.setRotation(r1, r2, r3);
	}
};

#endif

