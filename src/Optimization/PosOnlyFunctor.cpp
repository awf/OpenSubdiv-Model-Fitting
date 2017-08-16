#include "PosOnlyFunctor.h"

#include <iostream>

PosOnlyFunctor::PosOnlyFunctor(const Matrix3X& data_points, const MeshTopology& mesh)
	: BaseFunctor(mesh.num_vertices * 3 + data_points.cols() * 2 + 3 + 1,   /* number of parameters */
		data_points.cols() * 3, /* number of residuals */
		data_points,
		mesh) {
}

// Functor functions
// 1. Evaluate the residuals at x
int PosOnlyFunctor::operator()(const InputType& x, ValueType& fvec) {
	// E = sum over i (s_i - Phi(x_i))
	// Phi(x) = t + R * x 

	// Get the rotation as quaternion and then convert to matrix
	Eigen::Vector4f t;
	t << x.rigidTransf.params().t1, x.rigidTransf.params().t2, x.rigidTransf.params().t3, 0.0;
	float lambda = x.rigidTransf.params().s1;
	Matrix3X rCVs(3, x.nVertices());
	for (int i = 0; i < x.nVertices(); i++) {
		Eigen::Vector4f pt;
		pt << x.control_vertices(0, i), x.control_vertices(1, i), x.control_vertices(2, i), 0.0f;
		// The translation 't' zeroes out in the derivative, scale is added below
		pt = t + lambda * (pt);
		rCVs(0, i) = pt(0);
		rCVs(1, i) = pt(1);
		rCVs(2, i) = pt(2);
	}
	// Compute subdivision surface
	evaluator.evaluateSubdivSurface(rCVs, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	// Fill residuals
	for (int i = 0; i < data_points.cols(); i++) {
		fvec.segment(i * 3, 3) = (S.col(i) - data_points.col(i));
	}

	return 0;
}

// 2. Evaluate jacobian at x
int PosOnlyFunctor::df(const InputType& x, JacobianType& fjac) {
	// Evaluate surface at x
	evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv, &dSduu, &dSduv, &dSdvv);

	/*****************************/
	// Get the rotation as quaternion and then convert to matrix
	float lambda = x.rigidTransf.params().s1;
	/*****************************/
	Index nPoints = data_points.cols();
	Index n_base = nPoints * 3;
	Index X_base = nPoints * 2;
	Index ubase = 0;
	Index t_base = X_base + x.nVertices() * 3;
	Index s_base = t_base + 3;

	// Fill Jacobian columns.  
	//Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(nPoints * 3 * 3);	
	// vertices + uvs + R + t + lambda
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(nPoints * 3 + nPoints * 3 * 2 + nPoints * 3 * 3 + nPoints * 3);
	// 1. Derivatives wrt control vertices.
	for (int i = 0; i < dSdX.size(); ++i) {
		auto const& triplet = dSdX[i];
		assert(0 <= triplet.row() && triplet.row() < nPoints);
		assert(0 <= triplet.col() && triplet.col() < x.nVertices());
		jvals.add(triplet.row() * 3 + 0, X_base + triplet.col() * 3 + 0, lambda * triplet.value());
		jvals.add(triplet.row() * 3 + 1, X_base + triplet.col() * 3 + 1, lambda * triplet.value());
		jvals.add(triplet.row() * 3 + 2, X_base + triplet.col() * 3 + 2, lambda * triplet.value());
	}

	// 2. Derivatives wrt correspondences
	for (int i = 0; i < nPoints; i++) {
		jvals.add(3 * i + 0, ubase + 2 * i + 0, lambda * dSdu(0, i));
		jvals.add(3 * i + 1, ubase + 2 * i + 0, lambda * dSdu(1, i));
		jvals.add(3 * i + 2, ubase + 2 * i + 0, lambda * dSdu(2, i));
		jvals.add(3 * i + 0, ubase + 2 * i + 1, lambda * dSdv(0, i));
		jvals.add(3 * i + 1, ubase + 2 * i + 1, lambda * dSdv(1, i));
		jvals.add(3 * i + 2, ubase + 2 * i + 1, lambda * dSdv(2, i));
	}

	// 3. Derivatives wrt transofrmation parameters (rotation + translation + scale)
	for (int i = 0; i < nPoints; i++) {
		jvals.add(3 * i + 0, t_base + 0, 1.0f);
		jvals.add(3 * i + 1, t_base + 0, 0.0f);
		jvals.add(3 * i + 2, t_base + 0, 0.0f);
		jvals.add(3 * i + 0, t_base + 1, 0.0f);
		jvals.add(3 * i + 1, t_base + 1, 1.0f);
		jvals.add(3 * i + 2, t_base + 1, 0.0f);
		jvals.add(3 * i + 0, t_base + 2, 0.0f);
		jvals.add(3 * i + 1, t_base + 2, 0.0f);
		jvals.add(3 * i + 2, t_base + 2, 1.0f);

		jvals.add(3 * i + 0, s_base, S(0, i));
		jvals.add(3 * i + 1, s_base, S(1, i));
		jvals.add(3 * i + 2, s_base, S(2, i));
	}

	// (..., ... + 3 + 3 + 1) for the rotation, translation and scale parameters
	fjac.resize(3 * nPoints, 2 * nPoints + 3 * x.nVertices() + 3 + 1);
	fjac.setFromTriplets(jvals.begin(), jvals.end());
	fjac.makeCompressed();

	return 0;
}

void PosOnlyFunctor::increment_in_place(InputType* x, StepType const& p) {
	Index nPoints = data_points.cols();
	Index X_base = nPoints * 2;
	Index ubase = 0;
	Index t_base = X_base + x->nVertices() * 3;
	Index s_base = t_base + 3;

	// Increment control vertices
	Index nVertices = x->nVertices();

	assert(p.size() == nVertices * 3 + nPoints * 2 + 3 + 1);
	assert(x->us.size() == nPoints);

	//Map<VectorX>(x->control_vertices.data(), nVertices * 3) += p.tail(nVertices * 3);
	Map<VectorX>(x->control_vertices.data(), nVertices * 3) += p.segment(X_base, nVertices * 3);

	// Increment surface correspondences
	int loopers = 0;
	int totalhops = 0;
	for (int i = 0; i < nPoints; ++i) {
		Vector2 du = p.segment<2>(ubase + 2 * i);
		int nhops = increment_u_crossing_edges(x->control_vertices, x->us[i].face, x->us[i].u, du, &x->us[i].face, &x->us[i].u);
		if (nhops < 0)
			++loopers;
		totalhops += std::abs(nhops);
	}
	if (loopers > 0)
		std::cerr << "[" << totalhops / Scalar(nPoints) << " hops, " << loopers << " points looped]";
	else if (totalhops > 0)
		std::cerr << "[" << totalhops << "/" << Scalar(nPoints) << " hops]";

	// Increment translation parameteres
	Vector3 tNew = p.segment<3>(t_base);
	float t1 = x->rigidTransf.params().t1 + tNew(0);
	float t2 = x->rigidTransf.params().t2 + tNew(1);
	float t3 = x->rigidTransf.params().t3 + tNew(2);
	x->rigidTransf.setTranslation(t1, t2, t3);
	// Increment scaling parameters
	float sNew = p(s_base);
	float lambda = x->rigidTransf.params().s1 + sNew;
	//std::cout << std::endl << "Old: " << x->rigidTransf.params().s1 << std::endl;
	//std::cout << "Update: " << sNew << std::endl;
	//std::cout << "New: " << lambda << std::endl;
	x->rigidTransf.setScaling(lambda, lambda, lambda);
}