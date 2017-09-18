#ifndef POSANDNORMALS_FUNCTOR_H
#define POSANDNORMALS_FUNCTOR_H

#include "BaseFunctor.h"

struct PosAndNormalsFunctor : public BaseFunctor<7, 2> {
	// Input normals
	Matrix3X data_normals;

	PosAndNormalsFunctor(const Matrix3X& data_points, const Matrix3X &data_normals, const MeshTopology& mesh, const DataConstraints& constraints = DataConstraints());

	// Functor functions
	// 1. Evaluate the residuals at x
	virtual void f_impl(const InputType& x, ValueType& fvec);

	// 2. Evaluate jacobian at x
	virtual void df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals);

	// Update function
	virtual void increment_in_place_impl(InputType* x, StepType const& p);
};

#endif

