#include "RigidTransform.h"

#include <iostream>

RigidTransform::RigidTransform(const float r1, const float r2, const float r3, const float t1, const float t2, const float t3, const float s1, const float s2, const float s3) {
	// Init translation mat
	this->setTranslation(t1, t2, t3);

	// Init rotation mat
	this->setRotation(r1, r2, r3);

	// Init scaling mat
	this->setScaling(s1, s2, s3);
}

RigidTransform::~RigidTransform() {

}

const float Eps = 1e-15;
float sinc(const float arg) {
	if (fabs(arg) < Eps) {
		return 1.0f;
	} else {
		return sin(arg) / arg;
	}
}

// Convert rotation to quaternion and compute derivatives as in F. Sebastian Grassia "Practical parametrization of rotations..."
void RigidTransform::rotationToQuaternion(const Eigen::Vector3f &v, Eigen::Vector4f &q, Eigen::MatrixXf *dRdv) {
	// Calculate quaternion out of the unit vector specifying the rotation axis in R^3
	float vnorm = v.norm();
	float sincHalf = sinc(vnorm / 2.0f) / 2.0f;	

	// Compute the quaternion representation
	q << sincHalf * v, cos(vnorm / 2.0f);

	// Compute dq/dv
	Eigen::MatrixXf dqdv = Eigen::MatrixXf::Zero(4, 3);
	dqdv.row(3) << -0.5f * v(0) * sincHalf, -0.5f * v(1) * sincHalf, -0.5f * v(2) * sincHalf;
	// For numerical stability, use Taylor series approx. for vnorm < Eps^(1/4)
	float mult = 0.0f;
	if (vnorm < sqrt(sqrt(Eps))) {
		mult = ((vnorm * vnorm) / 40.0f - 1.0f) / 24.0f;
	} else {
		mult = (cos(vnorm / 2.0f) / 2.0f - sincHalf) / (vnorm * vnorm);
	}
	dqdv.block<3, 3>(0, 0) << v(0) * v(0), v(0) * v(1), v(0) * v(2),
		v(1) * v(0), v(1) * v(1), v(1) * v(2),
		v(2) * v(0), v(2) * v(1), v(2) * v(2);
	dqdv.block<3, 3>(0, 0) *= mult;
	dqdv(0, 0) += sincHalf;
	dqdv(1, 1) += sincHalf;
	dqdv(2, 2) += sincHalf;

	// Compute dR/dq
	Eigen::MatrixXf dRdq = Eigen::MatrixXf::Zero(9, 4);
	dRdq.row(0) << 0.0f, -2.0f * q(1), -2.0f * q(2), 0.0f;
	dRdq.row(1) << q(1), q(0), q(3), q(2);
	dRdq.row(2) << q(2), -q(3), q(0), -q(1);
	dRdq.row(3) << q(1), q(1), -q(3), -q(2);
	dRdq.row(4) << -2.0f * q(0), 0.0f, -2.0 * q(2), 0.0f;
	dRdq.row(5) << q(3), q(2), q(1), q(0);
	dRdq.row(6) << q(2), q(3), q(0), q(1);
	dRdq.row(7) << -q(3), q(2), q(1), -q(0);
	dRdq.row(8) << -2.0f * q(0), -2.0 * q(1), 0.0f, 0.0f;
	dRdq *= 2.0f;

	// Use Chain rule to compute dRdv as dRdq * dqdv
	*dRdv = dRdq * dqdv;
}

void RigidTransform::quaternionToMatrix(const Eigen::Vector4f &q, Eigen::Matrix4f &R) {
	R = Eigen::Matrix4f::Zero();

	float xy, xz, yz, wx, wy, wz, xx, yy, zz;
	xy = 2.0f * q(0) * q(1); xz = 2.0f * q(0) * q(2); yz = 2.0f * q(1) * q(2);
	wx = 2.0f * q(3) * q(0); wy = 2.0f * q(3) * q(1); wz = 2.0f * q(3) * q(2);
	xx = 2.0f * q(0) * q(0); yy = 2.0f * q(1) * q(1); zz = 2.0f * q(2) * q(2);

	R(0, 0) = 1.0f - (yy + zz); R(0, 1) = (xy - wz); R(0, 2) = (xz + wy);
	R(1, 0) = (xy + wz); R(1, 1) = 1.0f - (xx + zz); R(1, 2) = (yz - wx);
	R(2, 0) = (xz - wy); R(2, 1) = (yz + wx); R(2, 2) = 1.0f - (xx + yy);

	R(3, 3) = 1.0f;
}

void RigidTransform::setTranslation(const float t1, const float t2, const float t3) {
	// Update params
	this->transfParams.t1 = t1;
	this->transfParams.t2 = t2;
	this->transfParams.t3 = t3;

	// Update translation matrix
	this->translationMat = Eigen::Matrix4f::Identity();
	this->translationMat(0, 3) = t1;
	this->translationMat(1, 3) = t2;
	this->translationMat(2, 3) = t3;
}

void RigidTransform::setRotation(const float r1, const float r2, const float r3) {
	// Update params
	this->transfParams.r1 = r1;
	this->transfParams.r2 = r2;
	this->transfParams.r3 = r3;

	// Update rotation matrix
	// Store in terms of matrix exponential as in T. Cashman et al. "What shape are dolphins"...
	Eigen::Matrix3f rm, mm;
	rm << 0, -r3, r2,
		r3, 0, -r1,
		-r2, r1, 0;
	this->rotationMat = Eigen::Matrix4f::Identity();
	this->rotationMat.block<3, 3>(0, 0) = rm.exp();
}

void RigidTransform::setScaling(const float s1, const float s2, const float s3) {
	// Update params
	this->transfParams.s1 = s1;
	this->transfParams.s2 = s2;
	this->transfParams.s3 = s3;

	// Update scaling matrix
	this->scalingMat = Eigen::Matrix4f::Identity();
	this->scalingMat(0, 0) *= s1;
	this->scalingMat(1, 1) *= s2;
	this->scalingMat(2, 2) *= s3;
}

Eigen::Matrix4f RigidTransform::translation() const {
	return this->translationMat;
}

Eigen::Matrix4f RigidTransform::rotation() const {
	return this->rotationMat;
}

Eigen::Matrix4f RigidTransform::scaling() const {
	return this->scalingMat;
}

Eigen::Vector3f RigidTransform::rotationVec() const {
	return Eigen::Vector3f(this->transfParams.r1, this->transfParams.r2, this->transfParams.r3);
}

Eigen::Vector4f RigidTransform::scalingVec() const {
	return Eigen::Vector4f(this->transfParams.s1, this->transfParams.s2, this->transfParams.s3, 1.0);
}

Eigen::Vector4f RigidTransform::translationVec() const {
	return Eigen::Vector4f(this->transfParams.t1, this->transfParams.t2, this->transfParams.t3, 0.0);
}

Eigen::Matrix4f RigidTransform::transformation() const {
	return this->translationMat * this->scalingMat * this->rotationMat;
}

RigidTransform::Params RigidTransform::params() const {
	return this->transfParams;
}