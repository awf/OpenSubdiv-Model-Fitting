#include "FPJParser.h"

#include <fstream>
#include <sstream>

FPJParser::FPJParser(const std::string &_fileName) 
	: fileName(_fileName) {

}

FPJParser::~FPJParser() {

}

FPJParser::Project FPJParser::project() const {
	return this->proj;
}

bool FPJParser::parse() {
	std::ifstream inFile(this->fileName);

	if (inFile.is_open()) {
		std::string line;
		// First two lines are header
		std::getline(inFile, line);
		if (line.compare("fpj") != 0) {
			return false;
		}
		std::getline(inFile, line);
		if (line.compare("format ascii 1.0") != 0) {
			return false;
		}
		std::getline(inFile, this->proj.plyFileName);

	
		while (!inFile.eof()) {
			ImageFile imgFile;

			// 1st line is the file name
			std::getline(inFile, imgFile.fileName);

			// 2nd line is the image size
			std::getline(inFile, line);
			std::stringstream ss(line);
			ss >> imgFile.imageSize(0) >> imgFile.imageSize(1);

			// 3rd line is the points string
			std::getline(inFile, line);
			imgFile.normalsLeft = line.back() == 'L';

			// Remove last two characters of the string (ending zL/R)
			line.pop_back(); line.pop_back();

			// Get the first point encapsulated by "M" and "C"
			std::vector<float> xy2dPts, xy2dPts2, xy2dPts3;
			float s1, s2;
			sscanf_s(line.c_str(), "M%f,%fC", &s1, &s2);

			// Then Skip everything from the start until letter 'C'
			line = line.substr(line.find('C') + 1);
			std::string coords;
			std::string::size_type pos = 0, prev_pos = 0; 
			float x1, y1, x2, y2, x3, y3;
			while ((pos = line.find('C', pos)) != std::string::npos) {
				coords = line.substr(prev_pos, pos - prev_pos);
				prev_pos = ++pos;

				sscanf_s(coords.c_str(), "%f,%f,%f,%f,%f,%f", &x1, &y1, &x2, &y2, &x3, &y3);
				xy2dPts.push_back(x1); xy2dPts.push_back(y1);
				xy2dPts2.push_back(x2); xy2dPts2.push_back(y2);
				xy2dPts3.push_back(x3); xy2dPts3.push_back(y3);
			}
			coords = line.substr(prev_pos, pos - prev_pos);
			sscanf_s(coords.c_str(), "%f,%f,%f,%f,%f,%f", &x1, &y1, &x2, &y2, &x3, &y3);
			xy2dPts.push_back(x1); xy2dPts.push_back(y1);
			xy2dPts2.push_back(x2); xy2dPts2.push_back(y2);
			xy2dPts3.push_back(x3); xy2dPts3.push_back(y3);
			// Copy into eigen structures
			imgFile.silhouettePoints[1] = Eigen::Map<Eigen::MatrixXf>(xy2dPts.data(), 2, xy2dPts.size() / 2).transpose();
			imgFile.silhouettePoints[2] = Eigen::Map<Eigen::MatrixXf>(xy2dPts2.data(), 2, xy2dPts3.size() / 2).transpose();
			imgFile.silhouettePoints[3] = Eigen::Map<Eigen::MatrixXf>(xy2dPts3.data(), 2, xy2dPts2.size() / 2).transpose();
			imgFile.silhouettePoints[0] = Eigen::MatrixXf(imgFile.silhouettePoints[1].rows(), imgFile.silhouettePoints[1].cols());
			imgFile.silhouettePoints[0].block(1, 0, imgFile.silhouettePoints[3].rows() - 1, 2) = imgFile.silhouettePoints[3].block(0, 0, imgFile.silhouettePoints[3].rows() - 1, 2);
			imgFile.silhouettePoints[0](0, 0) = s1;
			imgFile.silhouettePoints[0](0, 1) = s2;
			// Apply scaling to the image silhouette points as in Cashman et al. "What shape are doplhins..."
			float scaleNorm = 2.0f / imgFile.imageSize(0);
			for (int i = 0; i < 4; i++) {
				imgFile.silhouettePoints[i].col(0) = (imgFile.silhouettePoints[i].col(0) * scaleNorm).array() - 1.0f;
				imgFile.silhouettePoints[i].col(1) = (scaleNorm * imgFile.imageSize(1) * 0.5f) - (imgFile.silhouettePoints[i].col(1) * scaleNorm).array();
		}

			// 4th line contains constraint related information
			std::getline(inFile, line);
			std::string constr;
			pos = 0; prev_pos = 0;
			std::vector<int> idxs3d, onSils;
			std::vector<float> xy2d;
			int i3d, onsil;
			float x2d, y2d;
			while ((pos = line.find(',', pos)) != std::string::npos) {
				constr = line.substr(prev_pos, pos - prev_pos);
				prev_pos = ++pos;
				
				sscanf_s(constr.c_str(), "%i:%f:%f:%i", &i3d, &x2d, &y2d, &onsil);
				idxs3d.push_back(i3d + 1);
				xy2d.push_back(x2d);
				xy2d.push_back(y2d);
				onSils.push_back(onsil);
			}
			constr = line.substr(prev_pos, pos - prev_pos);
			sscanf_s(constr.c_str(), "%i:%f:%f:%i", &i3d, &x2d, &y2d, &onsil);
			idxs3d.push_back(i3d + 1);
			xy2d.push_back(x2d);
			xy2d.push_back(y2d);
			onSils.push_back(onsil);
			// Copy into eigen structures
			imgFile.c3dIndices = Eigen::Map<Eigen::VectorXi>(idxs3d.data(), idxs3d.size());
			imgFile.c2dPoints = Eigen::Map<Eigen::MatrixXf>(xy2d.data(), 2, xy2d.size() / 2).transpose();
			imgFile.cOnSilhouette = Eigen::Map<Eigen::VectorXi>(onSils.data(), onSils.size());

			// 5th line contains the transformation coefficients
			std::getline(inFile, line);
			pos = 0; prev_pos = 0;
			std::vector<float> transf;
			float t;
			while ((pos = line.find(',', pos)) != std::string::npos) {
				coords = line.substr(prev_pos, pos - prev_pos);
				prev_pos = ++pos;

				sscanf_s(coords.c_str(), "%f", &t);
				transf.push_back(t);
			}
			coords = line.substr(prev_pos, pos - prev_pos);
			sscanf_s(coords.c_str(), "%f", &t);
			transf.push_back(t);
			// Copy into eigen structures
			// Translation
			imgFile.rigidTransf.setTranslation(transf[3], transf[4], 0.0);
			/*imgFile.transl = Eigen::Matrix4f::Identity();
			imgFile.transl(0, 3) = transf[3];
			imgFile.transl(1, 3) = transf[4];*/

			// Rotation
			imgFile.rigidTransf.setRotation(transf[0], transf[1], transf[2]);
			/*Eigen::matrix3f rm;
			rm << 0, -transf[2], transf[1],
				transf[2], 0, -transf[0], 
				-transf[1], transf[0], 0;
			imgfile.rot = eigen::matrix4f::identity();
			imgfile.rot.block<3, 3>(0, 0) = rm.exp();*/
			// Scale
			imgFile.rigidTransf.setScaling(transf[5], transf[5], transf[5]);
			/*imgFile.scale = Eigen::Matrix4f::Identity() * transf[5];	
			imgFile.scale(3, 3) = 1.0;*/
			// Final transformation
			//imgFile.transform = imgFile.transl * imgFile.rot * imgFile.scale;
			
			this->proj.images.push_back(imgFile);
		}
	} else {
		return false;
	}

	return true;
}