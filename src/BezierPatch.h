#ifndef BEZIER_PATCH_H
#define BEZIER_PATCH_H

#include <Eigen/Eigen>

class BezierPatch {
public:
	static const unsigned int Degree = 3; // => 4 control points

	BezierPatch(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4);
	~BezierPatch();

	Eigen::Vector2f evaluateAt(const float t);
	static Eigen::Vector2f evaluateAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t);
	Eigen::Vector2f evaluateTangentAt(const float t);
	static Eigen::Vector2f evaluateTangentAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t);
	Eigen::Vector2f evaluateNormalAt(const float t, bool normalsLeft);
	static Eigen::Vector2f evaluateNormalAt(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const Eigen::Vector2f &pt3, const Eigen::Vector2f &pt4, const float t, bool normalsLeft);

private:
	Eigen::MatrixXf controlPoints;

	static Eigen::Vector2f linearInterpolate(const Eigen::Vector2f &pt1, const Eigen::Vector2f &pt2, const float t);
};

#endif

