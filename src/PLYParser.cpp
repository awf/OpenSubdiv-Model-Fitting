#include "PLYParser.h"

#include <fstream>
#include <sstream>

PLYParser::PLYParser(const std::string &_fileName)
	: fileName(_fileName) {

}

PLYParser::~PLYParser() {

}

PLYParser::Model PLYParser::model() const {
	return this->plyModel;
}

bool PLYParser::parse(Model::PrimitiveType type) {
	std::ifstream inFile(this->fileName);

	if (inFile.is_open()) {
		// Set primitive type
		this->plyModel.primitiveType = type;

		std::string line;
		// First two lines are header
		std::getline(inFile, line);
		if (line.compare("ply") != 0) {
			return false;
		}
		std::getline(inFile, line);
		if (line.compare("format ascii 1.0") != 0) {
			return false;
		}

		// 3rd line contains no. of vertices
		std::getline(inFile, line);
		unsigned int nVerts = 0;
		sscanf_s(line.c_str(), "element vertex %u", &nVerts);

		// Ignore next 3 lines
		inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		// 3rd line contains no. of vertices
		std::getline(inFile, line);
		unsigned int nFaces = 0;
		sscanf_s(line.c_str(), "element face %u", &nFaces);

		// Ignore next 2 lines
		inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		// Vertex data start here
		this->plyModel.vertices = Eigen::MatrixXf(nVerts, 3);
		float x, y, z;
		for (unsigned int i = 0; i < nVerts; i++) {
			std::getline(inFile, line);
			sscanf_s(line.c_str(), "%f %f %f", &x, &y, &z);
			this->plyModel.vertices.row(i) << x, y, z;
		}

		// Face data follow
		unsigned int f1, f2, f3, f4;
		switch (type) {
		case Model::PrimitiveType::Triangles:
			this->plyModel.faces = Eigen::MatrixXi(nFaces, 3);
			for (unsigned int i = 0; i < nFaces; i++) {
				std::getline(inFile, line);
				sscanf_s(line.c_str(), "3 %u %u %u", &f1, &f2, &f3);
				this->plyModel.faces.row(i) << f1, f2, f3;
			}
			break;
		case Model::PrimitiveType::Quads:
			this->plyModel.faces = Eigen::MatrixXi(nFaces, 4);
			for (unsigned int i = 0; i < nFaces; i++) {
				std::getline(inFile, line);
				sscanf_s(line.c_str(), "4 %u %u %u %u", &f1, &f2, &f3, &f4);
				this->plyModel.faces.row(i) << f1, f2, f3, f4;
			}
			break;
		}

	} else {
		return false;
	}

	return true;
}