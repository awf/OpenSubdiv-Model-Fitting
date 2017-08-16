#ifndef POSONLY_FUNCTOR_H
#define POSONLY_FUNCTOR_H

#include "BaseFunctor.h"

struct PosOnlyFunctor : public BaseFunctor {
	PosOnlyFunctor(const Matrix3X& data_points, const MeshTopology& mesh);

	// Functor functions
	// 1. Evaluate the residuals at x
	virtual int operator()(const InputType& x, ValueType& fvec);

	// 2. Evaluate jacobian at x
	virtual int df(const InputType& x, JacobianType& fjac);

	// Update function
	virtual void increment_in_place(InputType* x, StepType const& p);
};

#endif

