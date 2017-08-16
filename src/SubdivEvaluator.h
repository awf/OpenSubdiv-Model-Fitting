#ifndef SUBDIV_EVALUATOR_H
#define SUBDIV_EVALUATOR_H

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

#define MAX_NUM_W  16		// If using ENDCAP_BSPLINE_BASIS (no. of weights)

#define QUAD_SIZE 4			// No. of face vertices 

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
	
	SubdivEvaluator();
	SubdivEvaluator(MeshTopology const& mesh);
	SubdivEvaluator(SubdivEvaluator const& that);
	~SubdivEvaluator();

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
		triplets_t* out_dSuudX = 0,
		triplets_t* out_dSuvdX = 0,
		triplets_t* out_dSvvdX = 0,
		Matrix3X* out_N = 0,
		Matrix3X* out_Nu = 0,
		Matrix3X* out_Nv = 0) const;

	void generate_refined_mesh(Matrix3X const& vert_coords, int levels, MeshTopology* mesh_out, Matrix3X* verts_out);

	SubdivEvaluator& operator=(SubdivEvaluator const& that) {
		this->nVertices = that.nVertices;
		this->nRefinerVertices = that.nRefinerVertices;
		this->nLocalPoints = that.nLocalPoints;
		this->patchTable = new OpenSubdiv::Far::PatchTable(*that.patchTable);
		this->evaluation_verts_buffer = that.evaluation_verts_buffer;
		return *this;
	}
};

#endif