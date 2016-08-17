#pragma once

#include <Eigen/Eigen>

#include <iso646.h> //To define the words and, not, etc. as operators in Windows
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/far/stencilTable.h>

#include "eigen_extras.h"

#include "MeshTopology.h"

using namespace OpenSubdiv;

#define MAX_NUM_W  16		//If using ENDCAP_BSPLINE_BASIS

// Vertex container implementation for OSD
struct OSD_Vertex {
  OSD_Vertex() { }

  void Clear(void * = 0) {
    point.setZero();
  }

  void AddWithWeight(OSD_Vertex const & src, float weight) {
    point += weight * src.point;
  }

  void SetPosition(float x, float y, float z) {
    point << x, y, z;
  }

  Vector3 point;
};

// A (u,v) point on a particular face of a facewise parametric surface
struct SurfacePoint {
  int face;
  Vector2 u;

  //  SurfacePoint() {}
  //  SurfacePoint(int face, Vector2 const& u) :face(face), u(u) {}
};

struct SubdivEvaluator {
  typedef Eigen::TripletArray<Scalar> triplets_t;

  OpenSubdiv::Far::PatchTable *patchTable;

  size_t  nVertices;
  size_t  nRefinerVertices;
  size_t  nLocalPoints;

  mutable std::vector<OSD_Vertex> evaluation_verts_buffer;
  static const int maxlevel = 3;
  Far::TopologyRefiner * refiner2;
  void generate_refined_mesh(Matrix3X const& vert_coords, int levels, MeshTopology* mesh_out, Matrix3X* verts_out);

  SubdivEvaluator(MeshTopology const& mesh);
  void evaluateSubdivSurface(Matrix3X const& vert_coords,
    std::vector<SurfacePoint> const& uv,
    Matrix3X* out_S,
    triplets_t* out_dSdX = 0,
    triplets_t* out_dSudX = 0,
    triplets_t* out_dSvdX = 0,
    Matrix3X* out_Su = 0,
    Matrix3X* out_Sv = 0,
    Matrix3X* out_Suu = 0,
    Matrix3X* out_Suv = 0,
    Matrix3X* out_Svv = 0,
    Matrix3X* out_N = 0,
    Matrix3X* out_Nu = 0,
    Matrix3X* out_Nv = 0) const;

  SubdivEvaluator(SubdivEvaluator const& that) {
    *this = that;
  }

  SubdivEvaluator& operator=(SubdivEvaluator const& that) {
    this->nVertices = that.nVertices;
    this->nRefinerVertices = that.nRefinerVertices;
    this->nLocalPoints = that.nLocalPoints;
    this->patchTable = new OpenSubdiv::Far::PatchTable(*that.patchTable);
    this->evaluation_verts_buffer = that.evaluation_verts_buffer;
    return *this;
  }

  ~SubdivEvaluator() {
    delete patchTable;

    // xxawf delete refiner2
  }

};

SubdivEvaluator::SubdivEvaluator(MeshTopology const& mesh)
{
  nVertices = mesh.num_vertices;

  size_t  num_faces = mesh.num_faces();

  //Fill the topology of the mesh
  Far::TopologyDescriptor desc;
  desc.numVertices = (int) mesh.num_vertices;
  desc.numFaces = (int) num_faces;

  Eigen::VectorXi vertsperface((int) num_faces);
  vertsperface.setConstant((int) mesh.quads.rows());

  desc.numVertsPerFace = vertsperface.data();
  desc.vertIndicesPerFace = mesh.quads.data();

  //Instantiate a FarTopologyRefiner from the descriptor.
  Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
  // Adpative refinement is only supported for CATMARK
  // Scheme LOOP is only supported if the mesh is purely composed of triangles

  Sdc::Options options;
  options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);
  typedef Far::TopologyRefinerFactory<Far::TopologyDescriptor> Refinery;
  OpenSubdiv::Far::TopologyRefiner *refiner = Refinery::Create(desc, Refinery::Options(type, options));

  const int maxIsolation = 0; //Don't change it!
  refiner->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions(maxIsolation));

  // Generate a set of Far::PatchTable that we will use to evaluate the surface limit
  Far::PatchTableFactory::Options patchOptions;
  patchOptions.endCapType = Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;

  patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);

  // Compute the total number of points we need to evaluate patchtable.
  // we use local points around extraordinary features.
  nRefinerVertices = refiner->GetNumVerticesTotal();
  nLocalPoints = patchTable->GetNumLocalPoints();

  // Create a buffer to hold the position of the refined verts and
  // local points.
  evaluation_verts_buffer.resize(nRefinerVertices + nLocalPoints);

  // xxaqwf delete refiner here?

  // This refiner is to generate subdivided meshes
  // Instantiate a FarTopologyRefiner from the descriptor
  this->refiner2 = Refinery::Create(desc, Refinery::Options(type, options));

  // Uniformly refine the topolgy up to 'maxlevel'
  refiner2->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));
}

void SubdivEvaluator::generate_refined_mesh(Matrix3X const& vert_coords, int levels, MeshTopology* mesh_out, Matrix3X* verts_out)
{
  if (levels > maxlevel) {
    std::cerr << "SubdivEvaluator::generate_refined_mesh: level too high\n";
    levels = maxlevel;
  }

  // Allocate a buffer for vertex primvar data. The buffer length is set to
  // be the sum of all children vertices up to the highest level of refinement.
  std::vector<OSD_Vertex> vbuffer(refiner2->GetNumVerticesTotal());
  OSD_Vertex * verts = &vbuffer[0];


  // Initialize coarse mesh positions
  int nCoarseVerts = (int)vert_coords.cols();
  for (int i = 0; i<nCoarseVerts; ++i)
    verts[i].point = vert_coords.col(i);

  // Interpolate vertex primvar data
  Far::PrimvarRefiner primvarRefiner(*refiner2);

  OSD_Vertex * src = verts;
  for (int level = 1; level <= levels; ++level) {
    OSD_Vertex * dst = src + refiner2->GetLevel(level - 1).GetNumVertices();
    primvarRefiner.Interpolate(level, src, dst);
    src = dst;
  }

  Far::TopologyLevel const & refLastLevel = refiner2->GetLevel(levels);

  int nverts = refLastLevel.GetNumVertices();
  int nfaces = refLastLevel.GetNumFaces();

  // Print vertex positions
  int firstOfLastVerts = refiner2->GetNumVerticesTotal() - nverts;

  verts_out->resize(3, nverts);
  for (int vert = 0; vert < nverts; ++vert)
    verts_out->col(vert) = verts[firstOfLastVerts + vert].point;

  // Print faces
  mesh_out->num_vertices = nverts;
  mesh_out->quads.resize(4, nfaces);
  for (int face = 0; face < nfaces; ++face) {

    Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

    // all refined Catmark faces should be quads
    assert(fverts.size() == 4);

    for (int vert = 0; vert < fverts.size(); ++vert)
      mesh_out->quads(vert, face) = fverts[vert];
  }
  mesh_out->update_adjacencies();
}

void SubdivEvaluator::evaluateSubdivSurface(Matrix3X const& vert_coords,
  std::vector<SurfacePoint> const& uv,
  Matrix3X* out_S,
  triplets_t* out_dSdX,
  triplets_t* out_dSudX,
  triplets_t* out_dSvdX,
  Matrix3X* out_Su,
  Matrix3X* out_Sv,
  Matrix3X* out_Suu,
  Matrix3X* out_Suv,
  Matrix3X* out_Svv,
  Matrix3X* out_N,
  Matrix3X* out_Nu,
  Matrix3X* out_Nv) const
{
  // Check it's the same size vertex array
  assert(vert_coords.cols() == nVertices);
  // Check output size matches input
  assert(uv.size() == out_S->cols());
  assert(!out_Su || (uv.size() == out_Su->cols()));
  assert(!out_Sv || (uv.size() == out_Sv->cols()));
  assert(!out_Suu || (uv.size() == out_Suu->cols()));
  assert(!out_Suv || (uv.size() == out_Suv->cols()));
  assert(!out_Svv || (uv.size() == out_Svv->cols()));
  // Check that we can use raw pointers as iterators over the evaluation_verts
  //assert((uint8_t*) &evaluation_verts_buffer[1] - (uint8_t*) &evaluation_verts_buffer[0] == 12);

  if (0) {
    for (int i = 0; i < uv.size(); ++i) {
      out_S->col(i)[0] = uv[i].u[0];
      out_S->col(i)[1] = uv[i].u[1];
      out_S->col(i)[2] = 0.0;

      if (!out_Su) continue;

      out_Su->col(i)[0] = 1;
      out_Su->col(i)[1] = 0;
      out_Su->col(i)[2] = 0;

      out_Sv->col(i)[0] = 0;
      out_Sv->col(i)[1] = 1;
      out_Sv->col(i)[2] = 0;

    }
    return;
  }

  // then copy the coarse positions at the beginning from vert_coords
  for (size_t i = 0; i < nVertices; ++i)
    evaluation_verts_buffer[i].point = vert_coords.col(i);

  // Evaluate local points from interpolated vertex primvars.
  int numStencils = patchTable->GetLocalPointStencilTable()->GetNumStencils();

  patchTable->ComputeLocalPointValues(&evaluation_verts_buffer[0], &evaluation_verts_buffer[nRefinerVertices]);

  //Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
  //--------------------------------------------------------------------------------------------------
  Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
  size_t  nstencils = stenciltab->GetNumStencils();
  // printf("\n Num of stencils - %d", nstencils);
  std::vector<Far::Stencil> st(nstencils);

  for (size_t i = 0; i < nstencils; i++)
  {
    st[i] = stenciltab->GetStencil(Far::Index(i));
    if (0) {
      unsigned int size_st = st[i].GetSize();
      Far::Index const *ind = st[i].GetVertexIndices();
      float const *wei = st[i].GetWeights();
      printf("\n Stencil %zd: ", i);
      for (unsigned int i = 0; i < size_st; i++)
        printf("V=%d, W=%0.3f ,", ind[i], wei[i]);
    }
  }

  // Create a Far::PatchMap to help locating patches in the table
  Far::PatchMap patchmap(*patchTable);
  //Far::PtexIndices ptexIndices(*refiner);  // Far::PtexIndices helps to find indices of ptex faces.

  float
    pWeights[MAX_NUM_W],
    dsWeights[MAX_NUM_W],
    dtWeights[MAX_NUM_W],
    dssWeights[MAX_NUM_W],
    dttWeights[MAX_NUM_W],
    dstWeights[MAX_NUM_W];

  // Zero the output arrays
  if (out_S) out_S->setZero();
  if (out_Su) out_Su->setZero();
  if (out_Sv) out_Sv->setZero();
  if (out_Suu) out_Suu->setZero();
  if (out_Suv) out_Suv->setZero();
  if (out_Svv) out_Svv->setZero();
  if (out_N) out_N->setZero();
  if (out_Nu) out_Nu->setZero();
  if (out_Nv) out_Nv->setZero();

  // Preallocate triplet vectors to max feasibly needed
#define CLEAR(VAR)\
  if (VAR) {\
    VAR->data.reserve(MAX_NUM_W*uv.size());\
    VAR->data.resize(0);\
  }
  CLEAR(out_dSdX);
  CLEAR(out_dSudX);
  CLEAR(out_dSvdX);
#undef CLEAR

  //Evaluate the surface with parametric coordinates
  for (unsigned int i = 0; i < uv.size(); ++i) {
    int face = uv[i].face;
    Scalar u = uv[i].u[0];
    Scalar v = uv[i].u[1];

    // Locate the patch corresponding to the face ptex idx and (s,t)
    Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u, v);
    assert(handle);

    // Evaluate the patch weights, identify the CVs and compute the limit frame:
    patchTable->EvaluateBasis(*handle, u, v, pWeights, dsWeights, dtWeights,
      out_Suu ? dssWeights : 0, out_Svv ? dttWeights : 0, out_Suv ? dstWeights : 0);

    Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);
    for (int cv = 0; cv < cvs.size(); ++cv) {
#define UPDATE(var, weight)\
      if (var) var->col(i) += evaluation_verts_buffer[cvs[cv]].point * weight ## Weights[cv];
      UPDATE(out_S, p);
      UPDATE(out_Su, ds);
      UPDATE(out_Sv, dt);
      UPDATE(out_Suu, dss);
      UPDATE(out_Suv, dst);
      UPDATE(out_Svv, dtt);
#undef UPDATE

      if (out_N) {
        assert(out_Su && out_Sv);
        // Compute the normals xxfixme not normalized?
        Vector3 Su = out_Su->col(i);
        Vector3 Sv = out_Sv->col(i);
        out_N->col(i) = Su.cross(Sv);
      }

      assert(!out_Nu && !out_Nv); // unimplemented

                                  // Compute derivatives wrt control vertices
      int n_to_compute = (out_dSdX ? 1 : 0) + (out_dSudX ? 1 : 0) + (out_dSvdX ? 1 : 0);

      if (n_to_compute == 0)
        continue;

      //Compute the weights for the coordinates and the derivatives wrt the control vertices
      //Set all weights to zero
      MatrixXX accumulated_weights(n_to_compute, nVertices);
      accumulated_weights.setZero();
      std::vector<bool> has_nonzero_weight(nVertices, false);

      for (int cv = 0; cv < cvs.size(); ++cv)
      {
        if (cvs[cv] < nVertices)
        {
          int c = 0;
          if (out_dSdX)  accumulated_weights(c++, cvs[cv]) += pWeights[cv];
          if (out_dSudX) accumulated_weights(c++, cvs[cv]) += dsWeights[cv];
          if (out_dSvdX) accumulated_weights(c++, cvs[cv]) += dtWeights[cv];
          has_nonzero_weight[cv] = true;
        }
        else
        {
          size_t ind_offset = cvs[cv] - nVertices;
          //Look at the stencil associated to this local point and distribute its weight over the control vertices
          unsigned int size_st = st[ind_offset].GetSize();
          Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
          float const *st_weights = st[ind_offset].GetWeights();
          for (unsigned int s = 0; s < size_st; s++)
          {
            int c = 0;
            if (out_dSdX)  accumulated_weights(c++, st_ind[s]) += pWeights[cv] * st_weights[s];
            if (out_dSudX) accumulated_weights(c++, st_ind[s]) += dsWeights[cv] * st_weights[s];
            if (out_dSvdX) accumulated_weights(c++, st_ind[s]) += dtWeights[cv] * st_weights[s];
            has_nonzero_weight[st_ind[s]] = true;
          }
        }
      }

      //Store the weights
      float scale = 1.f / cvs.size(); // xxawf fixme -- what is the correct value?
      for (int cv = 0; cv < nVertices; cv++)
        if (has_nonzero_weight[cv])
        {
          int c = 0;
          if (out_dSdX)   out_dSdX->add(i, cv, accumulated_weights(c++, cv)*scale);
          if (out_dSudX) out_dSudX->add(i, cv, accumulated_weights(c++, cv)*scale);
          if (out_dSvdX) out_dSvdX->add(i, cv, accumulated_weights(c++, cv)*scale);
        }
    }
  }
}
