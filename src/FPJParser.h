#ifndef FPJ_PARSER_H
#define FPJ_PARSER_H

#include <iostream>
#include <string>

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>

#include "RigidTransform.h"

class FPJParser {
public:
	struct ImageFile {
		std::string fileName;
		Eigen::Vector2i imageSize;

		Eigen::MatrixXf silhouettePoints[4];
		bool normalsLeft;

		Eigen::VectorXi c3dIndices;
		Eigen::MatrixXf c2dPoints;
		Eigen::VectorXi cOnSilhouette;

		RigidTransform rigidTransf;
		/*Eigen::Matrix4f transl;
		Eigen::Matrix4f rot;
		Eigen::Matrix4f scale;
		Eigen::Matrix4f transform;*/
	};

	struct Project {
		std::string plyFileName;
		std::vector<ImageFile> images;
	};

	FPJParser(const std::string &_fileName);
	~FPJParser();

	bool parse();

	Project project() const;

private:
	std::string fileName;

	Project proj;
};

#endif