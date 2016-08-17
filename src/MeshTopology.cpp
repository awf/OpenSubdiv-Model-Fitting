#include "MeshTopology.h"

void makeCube(MeshTopology* mesh, Matrix3X* verts)
{
  //Initial mesh - A cube/parallelepiped
  size_t num_verts = 8;
  size_t num_faces = 6;
  mesh->num_vertices = num_verts;

  // Init vertices
  verts->resize(3, num_verts);
  verts->col(0) << -1.0, -1.0, +1.0;
  verts->col(1) << +1.0, -1.0, +1.0;
  verts->col(2) << -1.0, +1.0, +1.0;
  verts->col(3) << +1.0, +1.0, +1.0;
  verts->col(4) << -1.0, +1.0, -1.0;
  verts->col(5) << +1.0, +1.0, -1.0;
  verts->col(6) << -1.0, -1.0, -1.0;
  verts->col(7) << +1.0, -1.0, -1.0;

  //Fill the vertices per face
  mesh->quads.resize(4, num_faces);
  mesh->quads.col(0) << 0, 1, 3, 2;
  mesh->quads.col(1) << 2, 3, 5, 4;
  mesh->quads.col(2) << 4, 5, 7, 6;
  mesh->quads.col(3) << 6, 7, 1, 0;
  mesh->quads.col(4) << 1, 7, 5, 3;
  mesh->quads.col(5) << 6, 0, 2, 4;

  mesh->update_adjacencies();
}

void MeshTopology::update_adjacencies()
{
  //Find the adjacent faces to every face
  face_adj.resize(4, num_faces());
  face_adj.fill(-1);
  for (size_t f = 0; f < num_faces(); f++)
    for (size_t k = 0; k < 4; k++)
    {
      // Find kth edge 
      unsigned int kinc = (int(k) + 1) % 4;
      int edge[2] = { quads(k,f), quads(kinc,f) };

      // And find the face that shares its reverse
      int found = 0;
      int other = -1;
      for (size_t fa = 0; fa < num_faces(); fa++)
      {
        if (f == fa) continue;
        for (size_t l = 0; l < 4; l++)
          if ((quads(l, fa) == edge[1]) && (quads((l + 1) % 4, fa) == edge[0])) {
            other = (int) fa;
            found++;
          }
      }
      assert(found == 1);

      face_adj(k, f) = other;
    }
}
