#include "PosOnlyFunctor.h"

#include <iostream>

PosOnlyFunctor::PosOnlyFunctor(const Matrix3X& data_points, const MeshTopology& mesh)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2,   /* number of parameters */
		data_points.cols() * 3, /* number of residuals */
		data_points.cols() * 3 + data_points.cols() * 6,	/* number of Jacobian nonzeros */
		data_points,
		mesh) {
}

// Functor functions
// 1. Evaluate the residuals at x
void PosOnlyFunctor::f_impl(const InputType& x, ValueType& fvec) {
	// Compute subdivision surface
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	this->E_pos(S, data_points, x.rigidTransf, fvec, 0);
}

// 2. Evaluate jacobian at x
void PosOnlyFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	// Evaluate surface at x
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	/*****************************/
	Index X_base = data_points.cols() * 2;
	Index ubase = 0;

	// Fill Jacobian columns.  
	// 1. Derivatives wrt control vertices.
	this->dE_pos_d_X(dSdX, x.rigidTransf, jvals, X_base, 0);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_uv(dSdu, dSdv, x.rigidTransf, jvals, ubase, 0);
}

void PosOnlyFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index X_base = data_points.cols() * 2;
	Index ubase = 0;

	// Increment control vertices
	this->inc_X(x, p, X_base);

	// Increment surface correspondences
	this->inc_uv(x, p, ubase);
}