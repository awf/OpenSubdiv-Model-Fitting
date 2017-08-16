#ifndef RIGID_TRANSFORM_H
#define RIGID_TRANSFORM_H

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

class RigidTransform {
public:
	struct Params {
		float t1, t2, t3;
		float r1, r2, r3;
		float s1, s2, s3;

		Params() 
		 : t1(0.0f), t2(0.0f), t3(0.0f), r1(0.0f), r2(0.0f), r3(0.0f), s1(1.0f), s2(1.0f), s3(1.0f) {
		}
		Params(const float r1, const float r2, const float r3, const float t1, const float t2, const float t3, const float s1, const float s2, const float s3)
		 : t1(t1), t2(t2), t3(t3), r1(r1), r2(r2), r3(r3), s1(s1), s2(s2), s3(s3) {
		}
	};

	static void rotationToQuaternion(const Eigen::Vector3f &v, Eigen::Vector4f &q, Eigen::MatrixXf *dRdv);
	static void quaternionToMatrix(const Eigen::Vector4f &q, Eigen::Matrix4f &R);

	RigidTransform() :
		transfParams(Params()){
	}
	RigidTransform(const float r1, const float r2, const float r3, const float t1, const float t2, const float t3, const float s1, const float s2, const float s3);
	~RigidTransform();

	void setTranslation(const float t1, const float t2, const float t3);
	void setRotation(const float r1, const float r2, const float r3);
	void setScaling(const float s1, const float s2, const float s3);

	Eigen::Matrix4f translation() const;
	Eigen::Matrix4f rotation() const;
	Eigen::Matrix4f scaling() const;
	Eigen::Matrix4f transformation() const;

	RigidTransform::Params params() const;

private:
	Eigen::Matrix4f rotationMat;
	Eigen::Matrix4f translationMat;
	Eigen::Matrix4f scalingMat;

	RigidTransform::Params transfParams;
};

#endif

