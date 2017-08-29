#ifndef MESH_TOPOLOGY_H
#define MESH_TOPOLOGY_H

#include "Eigen_ext/eigen_extras.h"
#include "PLYParser.h"

// Mesh topology
struct MeshTopology {
	Eigen::Array<int, 4, Eigen::Dynamic> quads;
	Eigen::Array<int, 4, Eigen::Dynamic> face_adj;

	static const unsigned int MaxNeighbors = 4;

	size_t  num_vertices;
	size_t  num_faces() const { 
		return quads.cols(); 
	}

	void update_adjacencies();

	static Eigen::Vector3f computeBarycenter(const Matrix3X &vertices);
	
	bool isAdjacentFace(const int f1, const int f2) const;

	MeshTopology();
	MeshTopology(const MeshTopology &mesh);
	
	MeshTopology& operator=(const MeshTopology& mesh);
};

void makeCube(MeshTopology* mesh, Matrix3X* verts);
void makeFromPLYModel(MeshTopology* mesh, Matrix3X* verts, const PLYParser::Model &model);

#endif