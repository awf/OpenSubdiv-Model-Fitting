// Beause the file is being included by BaseFunctor.h
#ifdef BASE_FUNCTOR_H
#ifndef BASE_FUNCTOR_CPP
#define BASE_FUNCTOR_CPP

#include "BaseFunctor.h"

#include <iostream>

template <int BlkRows, int BlkCols>
BaseFunctor<BlkRows, BlkCols>::BaseFunctor(Eigen::Index numParameters, Eigen::Index numResiduals, Eigen::Index numJacobianNonzeros, const Matrix3X& data_points, const MeshTopology& mesh, const DataConstraints& constraints) :
	Base(numParameters, numResiduals),                        
	data_points(data_points),
	mesh(mesh),
	data_constraints(constraints),
	evaluator(mesh),
	numParameters(numParameters),
	numResiduals(numResiduals),
	numJacobianNonzeros(numJacobianNonzeros),
	rowStride(BlkRows) {

	initWorkspace();
}

template <int BlkRows, int BlkCols>
void BaseFunctor<BlkRows, BlkCols>::initWorkspace() {
	Index nPoints = data_points.cols();
	this->ssurf.init(nPoints);
	this->ssurf_tsr.init(nPoints);
	this->ssurf_sr.init(nPoints);
	this->ssurf_r.init(nPoints);
}

// "Mesh walking" to update correspondences, as in Fig 3, Taylor et al, CVPR 2014, "Hand shape.."
template <int BlkRows, int BlkCols>
int BaseFunctor<BlkRows, BlkCols>::increment_u_crossing_edges(Matrix3X const& X, int face, const Vector2& u, const Vector2& du, int* new_face_out, Vector2* new_u_out) {
	const int MAX_HOPS = 7;

	Scalar u1_old = u[0];
	Scalar u2_old = u[1];
	Scalar du1 = du[0];
	Scalar du2 = du[1];
	Scalar u1_new = u1_old + du1;
	Scalar u2_new = u2_old + du2;

	for (int count = 0; ; ++count) {
		bool crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

		if (!crossing) {
			*new_face_out = face;
			*new_u_out << u1_new, u2_new;
			return count;
		}

		//Find the new face	and the coordinates of the crossing point within the old face and the new face
		int face_new;

		bool face_found = false;

		Scalar dif, aux, u1_cross, u2_cross;

		if (u1_new < 0.f)
		{
			dif = u1_old;
			const Scalar u2t = u2_old - du2*dif / du1;
			if ((u2t >= 0.f) && (u2t <= 1.f))
			{
				face_new = mesh.face_adj(3, face); aux = u2t; face_found = true;
				u1_cross = 0.f; u2_cross = u2t;
			}
		}
		if ((u1_new > 1.f) && (!face_found))
		{
			dif = 1.f - u1_old;
			const Scalar u2t = u2_old + du2*dif / du1;
			if ((u2t >= 0.f) && (u2t <= 1.f))
			{
				face_new = mesh.face_adj(1, face); aux = 1.f - u2t; face_found = true;
				u1_cross = 1.f; u2_cross = u2t;
			}
		}
		if ((u2_new < 0.f) && (!face_found))
		{
			dif = u2_old;
			const Scalar u1t = u1_old - du1*dif / du2;
			if ((u1t >= 0.f) && (u1t <= 1.f))
			{
				face_new = mesh.face_adj(0, face); aux = 1.f - u1t; face_found = true;
				u1_cross = u1t; u2_cross = 0.f;
			}
		}
		if ((u2_new > 1.f) && (!face_found))
		{
			dif = 1.f - u2_old;
			const Scalar u1t = u1_old + du1*dif / du2;
			if ((u1t >= 0.f) && (u1t <= 1.f))
			{
				face_new = mesh.face_adj(2, face); aux = u1t; face_found = true;
				u1_cross = u1t; u2_cross = 1.f;
			}
		}
		assert(face_found);

		// Find the coordinates of the crossing point as part of the new face, and update u_old (as that will be new u in next iter).
		unsigned int conf;
		for (unsigned int f = 0; f < 4; f++)
			if (mesh.face_adj(f, face_new) == face) { conf = f; }

		switch (conf) {
		case 0: u1_old = aux; u2_old = 0.f; break;
		case 1: u1_old = 1.f; u2_old = aux; break;
		case 2:	u1_old = 1.f - aux; u2_old = 1.f; break;
		case 3:	u1_old = 0.f; u2_old = 1.f - aux; break;
		}

		// Evaluate the subdivision surface at the edge (with respect to the original face)
		std::vector<SurfacePoint> pts;
		pts.push_back({ face,{ u1_cross, u2_cross } });
		pts.push_back({ face_new,{ u1_old, u2_old } });
		Matrix3X S(3, 2);
		Matrix3X Su(3, 2);
		Matrix3X Sv(3, 2);
		evaluator.evaluateSubdivSurface(X, pts, &S, 0, 0, 0, &Su, &Sv);

		Matrix<Scalar, 3, 2> J_Sa;
		J_Sa.col(0) = Su.col(0);
		J_Sa.col(1) = Sv.col(0);

		Matrix<Scalar, 3, 2> J_Sb;
		J_Sb.col(0) = Su.col(1);
		J_Sb.col(1) = Sv.col(1);

		//Compute the new u increments
		Vector2 du_remaining;
		du_remaining << u1_new - u1_cross, u2_new - u2_cross;
		Vector3 prod = J_Sa*du_remaining;
		Matrix22 AtA = J_Sb.transpose()*J_Sb;
		Vector2 AtB = J_Sb.transpose()*prod;

		//Vector2 du_new = AtA.ldlt().solve(AtB);
		Vector2  u_incr = AtA.inverse()*AtB;

		du1 = u_incr[0];
		du2 = u_incr[1];

		if (count == MAX_HOPS) {
			//std::cerr << "Problem!!! Many jumps between the mesh faces for the update of one correspondence. I remove the remaining u_increment!\n";
			auto dmax = std::max(du1, du2);
			Scalar scale = Scalar(0.5 / dmax);
			*new_face_out = face;
			//*new_u_out << u1_old + du1 * scale, u2_old + du2 * scale;
			*new_u_out << 0.5, 0.5;

			assert((*new_u_out)[0] >= 0 && (*new_u_out)[1] <= 1.0 && (*new_u_out)[1] >= 0 && (*new_u_out)[1] <= 1.0);
			return -count;
		}

		u1_new = u1_old + du1;
		u2_new = u2_old + du2;
		face = face_new;
	}
}

template <int BlkRows, int BlkCols>
Scalar BaseFunctor<BlkRows, BlkCols>::estimateNorm(InputType const& x, StepType const& diag)
{
	Index nVertices = x.nVertices();
	Map<VectorX> xtop{ (Scalar*)x.control_vertices.data(), nVertices * 3 };
	double total = xtop.cwiseProduct(diag.tail(nVertices * 3)).stableNorm();
	total = total*total;
	for (int i = 0; i < x.us.size(); ++i) {
		Vector2 const& u = x.us[i].u;
		Vector2 di = diag.segment<2>(2 * i);
		total += u.cwiseProduct(di).squaredNorm();
	}
	return Scalar(sqrt(total));
}

// And tell the algorithm how to set the QR parameters.
template <int BlkRows, int BlkCols>
void BaseFunctor<BlkRows, BlkCols>::initQRSolver(SchurlikeQRSolver &qr) {
	// set block size
	qr.setSparseBlockParams(data_points.cols() * 4, data_points.cols() * 2);
	//qr.getLeftSolver().setPruningEpsilon(1e-12);
	//qr.getLeftSolver().setBlockParams(4, 2);
	//qr.getLeftSolver().setDiagBlockParams(data_points.cols() * 3, data_points.cols() * 2);
	//qr.getLeftSolver().getDiagSolver().setSparseBlockParams(BlkRows, BlkCols);
}

// Functor functions
// 1. Evaluate the residuals at x
template <int BlkRows, int BlkCols>
int BaseFunctor<BlkRows, BlkCols>::operator()(const InputType& x, ValueType& fvec) {
	// Update subdivison surfaces
	this->ssurf_tsr.update = true;
	this->ssurf_sr.update = true;
	this->ssurf_r.update = true;
	this->ssurf.update = true;

	this->f_impl(x, fvec);

	return 0;
}

// 2. Evaluate jacobian at x
template <int BlkRows, int BlkCols>
int BaseFunctor<BlkRows, BlkCols>::df(const InputType& x, JacobianType& fjac) {
	// Fill Jacobian columns.  
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(this->numJacobianNonzeros);

	this->df_impl(x, jvals);

	fjac.resize(this->numResiduals, this->numParameters);
	// Do not redefine the functor treating duplicate entries!!! The implementation expects to sum them up as done by default.
	fjac.setFromTriplets(jvals.begin(), jvals.end());
	fjac.makeCompressed();

	return 0;
}

template <int BlkRows, int BlkCols>
void BaseFunctor<BlkRows, BlkCols>::increment_in_place(InputType* x, StepType const& p) {
	Index nPoints = data_points.cols();
	assert(p.size() == this->numParameters);
	assert(x->us.size() == nPoints);

	this->increment_in_place_impl(x, p);
}

#endif
#endif