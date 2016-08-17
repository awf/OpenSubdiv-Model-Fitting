#pragma once

#include "eigen_extras.h"

// Mesh topology
struct MeshTopology {
  Eigen::Array<int, 4, Eigen::Dynamic> quads;
  Eigen::Array<int, 4, Eigen::Dynamic> face_adj;

  size_t  num_vertices;
  size_t  num_faces() const { return quads.cols(); }

  void update_adjacencies();
};

void makeCube(MeshTopology* mesh, Matrix3X* verts);
