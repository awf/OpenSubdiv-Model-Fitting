#include "SubdivEvaluator.h"

SubdivEvaluator::SubdivEvaluator()
	: patchTable(NULL), refiner2(NULL), nVertices(0), nRefinerVertices(0), nLocalPoints(0) {

	this->initThinPlate();
}

SubdivEvaluator::SubdivEvaluator(SubdivEvaluator const& that) {
	*this = that;

	this->initThinPlate();
}

SubdivEvaluator::~SubdivEvaluator() {
	if (this->patchTable != NULL) {
		delete this->patchTable;
	}

// FixMe: Crashing if not commented!
//	if (this->refiner2 != NULL) {
//		delete this->refiner2;
//	}
}

SubdivEvaluator::SubdivEvaluator(MeshTopology const& mesh) {
	this->initThinPlate();

	this->nVertices = mesh.num_vertices;
	size_t num_faces = mesh.num_faces();

	//Fill the topology of the mesh
	Far::TopologyDescriptor desc;
	desc.numVertices = (int)mesh.num_vertices;
	desc.numFaces = (int)num_faces;

	Eigen::VectorXi vertsperface((int)num_faces);
	vertsperface.setConstant((int)mesh.quads.rows());

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
	this->patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);

	// Compute the total number of points we need to evaluate patchtable.
	// we use local points around extraordinary features.
	this->nRefinerVertices = refiner->GetNumVerticesTotal();
	this->nLocalPoints = this->patchTable->GetNumLocalPoints();

	// Create a buffer to hold the position of the refined verts and
	// local points.
	evaluation_verts_buffer.resize(nRefinerVertices + this->nLocalPoints);

	// Local refiner not needed anymore, delete it
	delete refiner;

	// This refiner is to generate subdivided meshes
	// Instantiate a FarTopologyRefiner from the descriptor
	this->refiner2 = Refinery::Create(desc, Refinery::Options(type, options));

	// Uniformly refine the topolgy up to 'maxlevel'
	this->refiner2->RefineUniform(Far::TopologyRefiner::UniformOptions(this->maxlevel));
}

void SubdivEvaluator::initThinPlate() {
	this->Q_thinplate = MatrixXX(16, 16);
	this->Q_thinplate.row(0) << 2312, 3744, -1824, -32, 3744, -5682, -5328, 966, -1824, -5328, 6048, 1104, -32, 966, 1104, 62;
	this->Q_thinplate.row(1) << 3744, 33528, 10752, -1824, -5682, -17904, -40386, -5328, -5328, -13536, 12816, 6048, 966, 10512, 10518, 1104;
	this->Q_thinplate.row(2) << -1824, 10752, 33528, 3744, -5328, -40386, -17904, -5682, 6048, 12816, -13536, -5328, 1104, 10518, 10512, 966;
	this->Q_thinplate.row(3) << -32, -1824, 3744, 2312, 966, -5328, -5682, 3744, 1104, 6048, -5328, -1824, 62, 1104, 966, -32;
	this->Q_thinplate.row(4) << 3744, -5682, -5328, 966, 33528, -17904, -13536, 10512, 10752, -40386, 12816, 10518, -1824, -5328, 6048, 1104;
	this->Q_thinplate.row(5) << -5682, -17904, -40386, -5328, -17904, 191112, -21072, -13536, -40386, -21072, -20658, 12816, -5328, -13536, 12816, 6048;
	this->Q_thinplate.row(6) << -5328, -40386, -17904, -5682, -13536, -21072, 191112, -17904, 12816, -20658, -21072, -40386, 6048, 12816, -13536, -5328;
	this->Q_thinplate.row(7) << 966, -5328, -5682, 3744, 10512, -13536, -17904, 33528, 10518, 12816, -40386, 10752, 1104, 6048, -5328, -1824;
	this->Q_thinplate.row(8) << -1824, -5328, 6048, 1104, 10752, -40386, 12816, 10518, 33528, -17904, -13536, 10512, 3744, -5682, -5328, 966;
	this->Q_thinplate.row(9) << -5328, -13536, 12816, 6048, -40386, -21072, -20658, 12816, -17904, 191112, -21072, -13536, -5682, -17904, -40386, -5328;
	this->Q_thinplate.row(10) << 6048, 12816, -13536, -5328, 12816, -20658, -21072, -40386, -13536, -21072, 191112, -17904, -5328, -40386, -17904, -5682;
	this->Q_thinplate.row(11) << 1104, 6048, -5328, -1824, 10518, 12816, -40386, 10752, 10512, -13536, -17904, 33528, 966, -5328, -5682, 3744;
	this->Q_thinplate.row(12) << -32, 966, 1104, 62, -1824, -5328, 6048, 1104, 3744, -5682, -5328, 966, 2312, 3744, -1824, -32;
	this->Q_thinplate.row(13) << 966, 10512, 10518, 1104, -5328, -13536, 12816, 6048, -5682, -17904, -40386, -5328, 3744, 33528, 10752, -1824;
	this->Q_thinplate.row(14) << 1104, 10518, 10512, 966, 6048, 12816, -13536, -5328, -5328, -40386, -17904, -5682, -1824, 10752, 33528, 3744;
	this->Q_thinplate.row(15) << 62, 1104, 966, -32, 1104, 6048, -5328, -1824, 966, -5328, -5682, 3744, -32, -1824, 3744, 2312;
	this->Q_thinplate = this->Q_thinplate / 302400.0f;
}

void SubdivEvaluator::generate_refined_mesh(Matrix3X const& vert_coords, int levels, MeshTopology* mesh_out, Matrix3X* verts_out)
{
	if (levels > this->maxlevel) {
		std::cerr << "SubdivEvaluator::generate_refined_mesh: level too high" << std::endl;
		levels = this->maxlevel;
	}

	// Allocate a buffer for vertex primvar data. The buffer length is set to
	// be the sum of all children vertices up to the highest level of refinement.
	std::vector<OSD_Vertex> vbuffer(this->refiner2->GetNumVerticesTotal());
	OSD_Vertex * verts = &vbuffer[0];

	// Initialize coarse mesh positions
	int nCoarseVerts = (int)vert_coords.cols();
	for (int i = 0; i < nCoarseVerts; ++i) {
		verts[i].point = vert_coords.col(i);
	}

	// Interpolate vertex primvar data
	Far::PrimvarRefiner primvarRefiner(*this->refiner2);

	OSD_Vertex * src = verts;
	for (int level = 1; level <= levels; ++level) {
		OSD_Vertex * dst = src + this->refiner2->GetLevel(level - 1).GetNumVertices();
		primvarRefiner.Interpolate(level, src, dst);
		src = dst;
	}

	Far::TopologyLevel const & refLastLevel = this->refiner2->GetLevel(levels);

	int nverts = refLastLevel.GetNumVertices();
	int nfaces = refLastLevel.GetNumFaces();

	// Output vertex positions
	verts_out->resize(3, nverts);
	for (int vert = 0; vert < nverts; ++vert) {
		verts_out->col(vert) = src[vert].point;
	}

	// Output faces
	mesh_out->num_vertices = nverts;
	mesh_out->quads.resize(QUAD_SIZE, nfaces);
	for (int face = 0; face < nfaces; ++face) {
		Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

		// All refined Catmark faces should be quads
		assert(fverts.size() == QUAD_SIZE);

		for (int vert = 0; vert < fverts.size(); ++vert)
			mesh_out->quads(vert, face) = fverts[vert];
	}
	mesh_out->update_adjacencies();
}

void SubdivEvaluator::thinPlateEnergy(Matrix3X const& vert_coords,
	std::vector<SurfacePoint> const& uv, 
	MatrixXX &thinPlateEnergy) const {

	// Check it's the same size vertex array
	assert(vert_coords.cols() == nVertices);

	// Then copy the coarse positions at the beginning from vert_coords
	for (size_t i = 0; i < nVertices; ++i) {
		evaluation_verts_buffer[i].point = vert_coords.col(i);
	}
	patchTable->ComputeLocalPointValues(&evaluation_verts_buffer[0], &evaluation_verts_buffer[nRefinerVertices]);

	// Get stencils for local points
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	size_t  nstencils = stenciltab->GetNumStencils();
	std::vector<Far::Stencil> st(nstencils);

	for (size_t i = 0; i < nstencils; i++) {
		st[i] = stenciltab->GetStencil(Far::Index(i));
	}

	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	MatrixXX energyCoeff = MatrixXX::Zero(nVertices, nVertices);
	// Evaluate the surface with parametric coordinates
	for (unsigned int i = 0; i < uv.size(); ++i) {
		int face = uv[i].face;
		Scalar u = uv[i].u[0];
		Scalar v = uv[i].u[1];

		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u, v);
		assert(handle);
		
		// ToDo:
		// Test - this could be used to retrieve indices of the 4x4 bicubic patch vertices
		// => evaluating subdivision surface at the control points we can easily arrive to 4x4 bicubic patch vertices for each control point
		Far::ConstIndexArray patchVertices = patchTable->GetPatchVertices(*handle);
		MatrixXX cv_Ws(16, nVertices);
		for (unsigned int j = 0; j < MAX_NUM_W; j++) {
			VectorX row = VectorX::Zero(nVertices);
			if (patchVertices[j] < nVertices) {
				row(patchVertices[j]) = 1.0;
			} else {
				unsigned int st_idx = patchVertices[j] - nVertices;
				Far::Index const *ind = st[st_idx].GetVertexIndices();
				float const *wei = st[st_idx].GetWeights();
				for (unsigned int k = 0; k < st[st_idx].GetSize(); k++) {
					row(ind[k]) = wei[k];
				}
			}
			cv_Ws.row(j) = row;
		}
		// Add to the energy
		energyCoeff += (cv_Ws.transpose() * this->Q_thinplate * cv_Ws);
	}

	// Preserve matrix symmetricity
	thinPlateEnergy = (energyCoeff + energyCoeff.transpose()) / 2.0;
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
	triplets_t* out_dSuudX,
	triplets_t* out_dSuvdX,
	triplets_t* out_dSvvdX,
	Matrix3X* out_N,
	Matrix3X* out_Nu,
	Matrix3X* out_Nv) const {
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

	// Then copy the coarse positions at the beginning from vert_coords
	for (size_t i = 0; i < nVertices; ++i) {
		evaluation_verts_buffer[i].point = vert_coords.col(i);
	}
	patchTable->ComputeLocalPointValues(&evaluation_verts_buffer[0], &evaluation_verts_buffer[nRefinerVertices]);

	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	size_t  nstencils = stenciltab->GetNumStencils();
	//printf("\n Num of stencils - %d\n", nstencils);
	std::vector<Far::Stencil> st(nstencils);

	for (size_t i = 0; i < nstencils; i++) {
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

	float pWeights[MAX_NUM_W],
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
    VAR->reserve(MAX_NUM_W*uv.size());\
    VAR->resize(0);\
  }
	CLEAR(out_dSdX);
	CLEAR(out_dSudX);
	CLEAR(out_dSvdX);
	CLEAR(out_dSuudX);
	CLEAR(out_dSuvdX);
	CLEAR(out_dSvvdX);
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
			out_Suu ? dssWeights : 0, out_Suv ? dstWeights : 0, out_Svv ? dttWeights : 0);
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
			int n_to_compute = (out_dSdX ? 1 : 0) + (out_dSudX ? 1 : 0) + (out_dSvdX ? 1 : 0) + (out_dSuudX ? 1 : 0) + +(out_dSuvdX ? 1 : 0) + +(out_dSvvdX ? 1 : 0);
			// If there are no derivatives requested, continue
			if (n_to_compute == 0)
				continue;

			// Compute the weights for the coordinates and the derivatives wrt the control vertices
			// Set all weights to zero
			MatrixXX accumulated_weights(n_to_compute, nVertices);
			accumulated_weights.setZero();
			std::vector<bool> has_nonzero_weight(nVertices, false);

			for (int cv = 0; cv < cvs.size(); ++cv) {
				if (cvs[cv] < nVertices) {
					int c = 0;
					if (out_dSdX)  accumulated_weights(c++, cvs[cv]) += pWeights[cv];
					if (out_dSudX) accumulated_weights(c++, cvs[cv]) += dsWeights[cv];
					if (out_dSvdX) accumulated_weights(c++, cvs[cv]) += dtWeights[cv];
					if (out_dSuudX) accumulated_weights(c++, cvs[cv]) += dssWeights[cv];
					if (out_dSuvdX) accumulated_weights(c++, cvs[cv]) += dstWeights[cv];
					if (out_dSvvdX) accumulated_weights(c++, cvs[cv]) += dttWeights[cv];
					has_nonzero_weight[cv] = true;
				}
				else {
					size_t ind_offset = cvs[cv] - nVertices;
					// Look at the stencil associated to this local point and distribute its weight over the control vertices
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++) {
						int c = 0;
						if (out_dSdX)  accumulated_weights(c++, st_ind[s]) += pWeights[cv] * st_weights[s];
						if (out_dSudX) accumulated_weights(c++, st_ind[s]) += dsWeights[cv] * st_weights[s];
						if (out_dSvdX) accumulated_weights(c++, st_ind[s]) += dtWeights[cv] * st_weights[s];
						if (out_dSuudX) accumulated_weights(c++, st_ind[s]) += dssWeights[cv] * st_weights[s];
						if (out_dSuvdX) accumulated_weights(c++, st_ind[s]) += dstWeights[cv] * st_weights[s];
						if (out_dSvvdX) accumulated_weights(c++, st_ind[s]) += dttWeights[cv] * st_weights[s];
						has_nonzero_weight[st_ind[s]] = true;
					}
				}
			}

			// Store the weights
			float scale = 1.f / cvs.size(); // xxawf fixme -- what is the correct value?
			for (int cv = 0; cv < nVertices; cv++)
				if (has_nonzero_weight[cv])
				{
					int c = 0;
					if (out_dSdX)   out_dSdX->add(i, cv, accumulated_weights(c++, cv)*scale);
					if (out_dSudX) out_dSudX->add(i, cv, accumulated_weights(c++, cv)*scale);
					if (out_dSvdX) out_dSvdX->add(i, cv, accumulated_weights(c++, cv)*scale);
					if (out_dSuudX) out_dSuudX->add(i, cv, accumulated_weights(c++, cv)*scale);
					if (out_dSuvdX) out_dSuvdX->add(i, cv, accumulated_weights(c++, cv)*scale);
					if (out_dSvvdX) out_dSvvdX->add(i, cv, accumulated_weights(c++, cv)*scale);
				}
		}
	}
}
