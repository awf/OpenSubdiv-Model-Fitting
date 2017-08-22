#include "PosOnlyWithRegFunctor.h"

#include <iostream>

PosOnlyWithRegFunctor::PosOnlyWithRegFunctor(const Matrix3X& data_points, const MeshTopology& mesh)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2 + 9,   /* number of parameters */
		data_points.cols() * 3 + mesh.num_vertices * 3, /* number of residuals */
		data_points.cols() * 3 + data_points.cols() * 6 + data_points.cols() * 15 + mesh.num_vertices * 3,	/* number of Jacobian nonzeros */
		data_points,
		mesh) {

	// Weight the energy terms
	this->eWeights.thinplate = 0.5;
}

// Functor functions
// 1. Evaluate the residuals at x
void PosOnlyWithRegFunctor::f_impl(const InputType& x, ValueType& fvec) {
	this->E_pos(x, data_points, fvec, 0);

	this->E_thinplate(x, fvec, data_points.cols() * 3);
}

// 2. Evaluate jacobian at x
void PosOnlyWithRegFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	Index ubase = 0;
	Index rst_base = data_points.cols() * 2;
	Index X_base = rst_base + 9;

	// Fill Jacobian columns.  
	// 1. Derivatives wrt control vertices.
	this->dE_pos_d_X(x, jvals, X_base, 0);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_uv(x, jvals, ubase, 0);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_rst(x, jvals, rst_base, 0);

	// 3. Derivatives of thinplate wrt control vertices
	this->dE_thinplate_d_X(x, jvals, X_base, data_points.cols() * 3);
}

void PosOnlyWithRegFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index ubase = 0;
	Index rst_base = data_points.cols() * 2;
	Index X_base = rst_base + 9;

	// Increment control vertices
	this->inc_X(x, p, X_base);

	// Increment surface correspondences
	this->inc_uv(x, p, ubase);

	// Increment rigid transformation parameters
	this->inc_rst(x, p, rst_base);
}