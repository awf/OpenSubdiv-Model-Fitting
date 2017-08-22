#include "PosOnlyWithRegFunctor.h"

#include <iostream>

PosOnlyWithRegFunctor::PosOnlyWithRegFunctor(const Matrix3X& data_points, const MeshTopology& mesh)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2,   /* number of parameters */
		data_points.cols() * 3 + mesh.num_vertices * 3, /* number of residuals */
		data_points.cols() * 3 + data_points.cols() * 6 + mesh.num_vertices * 3,	/* number of Jacobian nonzeros */
		data_points,
		mesh) {

	// Weight the energy terms
	this->eWeights.thinplate = 0.5;
}

// Functor functions
// 1. Evaluate the residuals at x
void PosOnlyWithRegFunctor::f_impl(const InputType& x, ValueType& fvec) {
	// Compute subdivision surface
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	this->E_pos(S, data_points, x.rigidTransf, fvec, 0);

	this->E_thinplate(x, x.rigidTransf, fvec, data_points.cols() * 3);
}

// 2. Evaluate jacobian at x
void PosOnlyWithRegFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
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

	// 3. Derivatives of thinplate wrt control vertices
	this->dE_thinplate_d_X(x, x.rigidTransf, jvals, X_base, data_points.cols() * 3);
}

void PosOnlyWithRegFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index X_base = data_points.cols() * 2;
	Index ubase = 0;

	// Increment control vertices
	this->inc_X(x, p, X_base);

	// Increment surface correspondences
	this->inc_uv(x, p, ubase);
}