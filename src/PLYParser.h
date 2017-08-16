#ifndef PLY_PARSER_H
#define PLY_PARSER_H

#include <iostream>
#include <string>

#include <Eigen/Eigen>

class PLYParser {
public:
	struct Model {
		enum PrimitiveType {
			Triangles = 0,
			Quads = 1
		};

		Eigen::MatrixXf vertices;
		Eigen::MatrixXi faces;
		PrimitiveType primitiveType;

		size_t numVertices() const {
			return vertices.rows();
		}
		size_t numFaces() const {
			return faces.rows();
		}
		Eigen::Vector3f barycenter() const {
			Eigen::Vector3f b;
			b << vertices.col(0).sum() / vertices.rows(),
				vertices.col(1).sum() / vertices.rows(),
				vertices.col(2).sum() / vertices.rows();

			return b;
		}
	};

	PLYParser(const std::string &_fileName);
	~PLYParser();

	bool parse(Model::PrimitiveType type);

	Model model() const;

private:
	std::string fileName;

	Model plyModel;
};

#endif