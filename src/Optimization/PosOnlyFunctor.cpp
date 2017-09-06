#include "PosOnlyFunctor.h"

#include <iostream>

PosOnlyFunctor::PosOnlyFunctor(const Matrix3X& data_points, const MeshTopology& mesh, const DataConstraints& constraints)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2 + 9,   /* number of parameters */
		data_points.cols() * 3 + constraints.size() * 3, /* number of residuals */
		data_points.cols() * 3 + data_points.cols() * 6 + data_points.cols() * 15 + constraints.size() * 3 + constraints.size() * 15,	/* number of Jacobian nonzeros */
		data_points,
		mesh,
		constraints) {

	// Weight the energy terms
	this->eWeights.thinplate = 0.5;
	this->eWeights.constraints = 5.0;
}

// Functor functions
// 1. Evaluate the residuals at x
void PosOnlyFunctor::f_impl(const InputType& x, ValueType& fvec) {
	this->E_pos(x, data_points, fvec, 0, 0);
	this->E_constraints(x, data_points, data_constraints, fvec, this->nDataPoints() * 3);
}

// 2. Evaluate jacobian at x
void PosOnlyFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	Index ubase = 0;
	Index rst_base = this->nDataPoints() * 2;
	Index X_base = rst_base + 9;

	// Fill Jacobian columns.  
	// 1. Derivatives wrt control vertices.
	this->dE_pos_d_X(x, jvals, X_base, 0, 0);
	this->dE_constraints_d_X(x, data_constraints, jvals, X_base, this->nDataPoints() * 3);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_uv(x, jvals, ubase, 0, 0);

	// 2. Derivatives wrt correspondences
	this->dE_pos_d_rst(x, jvals, rst_base, 0, 0);
	this->dE_constraints_d_rst(x, data_constraints, jvals, rst_base, this->nDataPoints() * 3);
}

void PosOnlyFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index ubase = 0;
	Index rst_base = this->nDataPoints() * 2;
	Index X_base = rst_base + 9;

	// Increment control vertices
	this->inc_X(x, p, X_base);

	// Increment surface correspondences
	this->inc_uv(x, p, ubase);

	// Increment rigid transformation parameters
	this->inc_rst(x, p, rst_base);
}