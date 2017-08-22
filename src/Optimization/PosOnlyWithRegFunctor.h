#ifndef POSONLYWITHREG_FUNCTOR_H
#define POSONLYWITHREG_FUNCTOR_H

#include "BaseFunctor.h"

struct PosOnlyWithRegFunctor : public BaseFunctor<3, 2> {
	PosOnlyWithRegFunctor(const Matrix3X& data_points, const MeshTopology& mesh);

	// Functor functions
	// 1. Evaluate the residuals at x
	virtual void f_impl(const InputType& x, ValueType& fvec);

	// 2. Evaluate jacobian at x
	virtual void df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals);

	// Update function
	virtual void increment_in_place_impl(InputType* x, StepType const& p);
};

#endif

