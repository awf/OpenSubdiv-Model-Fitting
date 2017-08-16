#include "BezierPatch.h"

BezierPatch::BezierPatch(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4) {
	controlPoints = Eigen::MatrixXf(BezierPatch::Degree + 1, 2);
	controlPoints.row(0) << pt1;
	controlPoints.row(1) << pt2;
	controlPoints.row(2) << pt3;
	controlPoints.row(3) << pt4;
}

BezierPatch::~BezierPatch() {
}

Eigen::Vector2f BezierPatch::evaluateAt(const float t) {
	Eigen::Vector2f ab = BezierPatch::linearInterpolate(this->controlPoints.row(0), this->controlPoints.row(1), t);
	Eigen::Vector2f bc = BezierPatch::linearInterpolate(this->controlPoints.row(1), this->controlPoints.row(2), t);
	Eigen::Vector2f cd = BezierPatch::linearInterpolate(this->controlPoints.row(2), this->controlPoints.row(3), t);

	return BezierPatch::linearInterpolate(BezierPatch::linearInterpolate(ab, bc, t), BezierPatch::linearInterpolate(bc, cd, t), t);
}

Eigen::Vector2f BezierPatch::evaluateAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t) {
	Eigen::Vector2f ab = BezierPatch::linearInterpolate(pt1, pt2, t);
	Eigen::Vector2f bc = BezierPatch::linearInterpolate(pt2, pt3, t);
	Eigen::Vector2f cd = BezierPatch::linearInterpolate(pt3, pt4, t);

	return BezierPatch::linearInterpolate(BezierPatch::linearInterpolate(ab, bc, t), BezierPatch::linearInterpolate(bc, cd, t), t);
}

Eigen::Vector2f BezierPatch::evaluateTangentAt(const float t) {
	Eigen::Vector2f ab = 3.0f * (this->controlPoints.row(1) - this->controlPoints.row(0));
	Eigen::Vector2f bc = 3.0f * (this->controlPoints.row(2) - this->controlPoints.row(1));
	Eigen::Vector2f cd = 3.0f * (this->controlPoints.row(3) - this->controlPoints.row(2));

	return ab * (1 - t) * (1 - t) + 2 * bc * (1 - t) + 3 * cd * t * t;
}

Eigen::Vector2f BezierPatch::evaluateTangentAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t) {
	Eigen::Vector2f ab = 3.0f * (pt2 - pt1);
	Eigen::Vector2f bc = 3.0f * (pt3 - pt2);
	Eigen::Vector2f cd = 3.0f * (pt4 - pt3);

	return ab * (1 - t) * (1 - t) + 2 * bc * (1 - t) + 3 * cd * t * t;
}

Eigen::Vector2f BezierPatch::evaluateNormalAt(const float t, bool normalsLeft) {
	Eigen::Vector2f tangent = this->evaluateTangentAt(t);
	if (normalsLeft) {
		return Eigen::Vector2f(-tangent(1), tangent(0));
	} else {
		return Eigen::Vector2f(tangent(2), -tangent(1));
	}
}

Eigen::Vector2f BezierPatch::evaluateNormalAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t, bool normalsLeft) {
	Eigen::Vector2f tangent = BezierPatch::evaluateTangentAt(pt1, pt2, pt3, pt4, t);
	if (normalsLeft) {
		return Eigen::Vector2f(-tangent(1), tangent(0));
	} else {
		return Eigen::Vector2f(tangent(2), -tangent(1));
	}
}


Eigen::Vector2f BezierPatch::linearInterpolate(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const float t) {
	return pt1 + (pt2 - pt1) * t;
}