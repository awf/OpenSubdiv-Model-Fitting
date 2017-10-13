#include "ICPFunctor.h"

#include <iostream>

ICPFunctor::ICPFunctor(const Matrix3X& data_points, const MeshTopology& mesh) :
	Base(numParameters, numResiduals),                        
	data_points(data_points),
	mesh(mesh),
	evaluator(mesh),
	numParameters(9),
	numResiduals(data_points.cols() * 3),
	numJacobianNonzeros(data_points.cols() * 3 * 9),
	rowStride(3) {

	initWorkspace();
}

void ICPFunctor::initWorkspace() {
	Index nPoints = data_points.cols();
	this->ssurf.init(nPoints);
	this->ssurf_tsr.init(nPoints);
	this->ssurf_r.init(nPoints);
}

Scalar ICPFunctor::estimateNorm(InputType const& x, StepType const& diag)
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
void ICPFunctor::initQRSolver(QRSolver &qr) {

}

// Functor functions
// 1. Evaluate the residuals at x
int ICPFunctor::operator()(const InputType& x, ValueType& fvec) {
	// Update subdivison surfaces
	this->ssurf_tsr.update = true;
	this->ssurf_r.update = true;
	this->ssurf.update = true;

	this->f_impl(x, fvec);

	return 0;
}

// 2. Evaluate jacobian at x
int ICPFunctor::df(const InputType& x, JacobianType& fjac) {
	// Fill Jacobian columns.  
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(this->numJacobianNonzeros);

	this->df_impl(x, jvals);

	fjac.resize(this->numResiduals, this->numParameters);
	// Do not redefine the functor treating duplicate entries!!! The implementation expects to sum them up as done by default.
	fjac.setFromTriplets(jvals.begin(), jvals.end());
	fjac.makeCompressed();

	return 0;
}

void ICPFunctor::increment_in_place(InputType* x, StepType const& p) {
	Index nPoints = data_points.cols();
	assert(p.size() == this->numParameters);
	assert(x->us.size() == nPoints);

	this->increment_in_place_impl(x, p);
}

// Functor functions
// 1. Evaluate the residuals at x
void ICPFunctor::f_impl(const InputType& x, ValueType& fvec) {
	this->E_pos(x, data_points, fvec, 0, 0);
}

// 2. Evaluate jacobian at x
void ICPFunctor::df_impl(const InputType& x, Eigen::TripletArray<Scalar, typename JacobianType::Index>& jvals) {
	Index rst_base = 0;

	// Fill Jacobian columns.  
	// 3. Derivatives wrt rigid transformation
	this->dE_pos_d_rst(x, jvals, rst_base, 0, 0);
}

void ICPFunctor::increment_in_place_impl(InputType* x, StepType const& p) {
	Index rst_base = 0;

	// Increment rigid transformation parameters
	this->inc_rst(x, p, rst_base);
}