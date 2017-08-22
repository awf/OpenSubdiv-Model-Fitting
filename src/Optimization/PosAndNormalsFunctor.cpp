#include "PosAndNormalsFunctor.h"

#include <iostream>

PosAndNormalsFunctor::PosAndNormalsFunctor(const Matrix3X& data_points, const Matrix3X &data_normals, const MeshTopology& mesh)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2,   /* number of parameters */
		data_points.cols() * 3 + data_normals.cols() * 3, /* number of residuals */
		data_points.cols() * 3 + data_points.cols() * 6 + data_points.cols() * 6 + data_points.cols() * 9,	/* number of Jacobian nonzeros */
		data_points,
		mesh),
		data_normals(data_normals) {
}

// Functor functions
// 1. Evaluate the residuals at x
void PosAndNormalsFunctor::f_impl(const InputType& x, ValueType& fvec) {
	// Compute subdivision surface
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	this->E_pos(S, data_points, x.rigidTransf, fvec, 0);

	this->E_normal(dSdu, dSdv, data_normals, x.rigidTransf, fvec, 3);
}

// 2. Evaluate jacobian at x
void PosAndNormalsFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	// Evaluate surface at x
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	/*****************************/
	Index X_base = data_points.cols() * 2;
	Index ubase = 0;

	// Fill Jacobian columns.  
	// 1. Derivatives wrt control vertices.
	this->dE_pos_d_X(dSdX, x.rigidTransf, jvals, X_base, 0);
	this->dE_normal_d_X(dSudX, dSvdX, dSdu, dSdv, x.rigidTransf, jvals, X_base, 3);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_uv(dSdu, dSdv, x.rigidTransf, jvals, ubase, 0);
	this->dE_normal_d_uv(dSdu, dSdv, dSduu, dSduv, dSdvv, x.rigidTransf, jvals, ubase, 3);
}

void PosAndNormalsFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index X_base = data_points.cols() * 2;
	Index ubase = 0;

	// Increment control vertices
	this->inc_X(x, p, X_base);

	// Increment surface correspondences
	this->inc_uv(x, p, ubase);
}