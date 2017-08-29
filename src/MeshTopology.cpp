#include "MeshTopology.h"

void makeCube(MeshTopology* mesh, Matrix3X* verts) {
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

void makeFromPLYModel(MeshTopology* mesh, Matrix3X* verts, const PLYParser::Model &model) {
	// Set mesh parameters
	mesh->num_vertices = model.numVertices();

	// Initialize vertices
	verts->resize(3, model.numVertices());
	for (int i = 0; i < model.numVertices(); i++) {
		verts->col(i) << model.vertices(i, 0), model.vertices(i, 1), model.vertices(i, 2);
	}
	
	// Initialize faces
	mesh->quads.resize(4, model.numFaces());
	for (int i = 0; i < model.numFaces(); i++) {
		// Order of those points matters to the optimization process!!!
		// See mesh walking implementation, it is indexing fixed  adjacency indices!!! (FixMe?)
		mesh->quads.col(i) << model.faces(i, 3), model.faces(i, 2), model.faces(i, 1), model.faces(i, 0);
	}
	mesh->update_adjacencies();
}

MeshTopology::MeshTopology() {

}

MeshTopology::MeshTopology(const MeshTopology &mesh) {
	this->num_vertices = mesh.num_vertices;
	this->quads.resize(4, mesh.num_faces());
	for (int i = 0; i < mesh.num_faces(); i++) {
		this->quads.col(i) << mesh.quads.col(i);
	}
	this->update_adjacencies();
}

MeshTopology& MeshTopology::operator=(const MeshTopology& mesh) {
	this->num_vertices = mesh.num_vertices;
	this->quads.resize(4, mesh.num_faces());
	for (int i = 0; i < mesh.num_faces(); i++) {
		this->quads.col(i) << mesh.quads.col(i);
	}
	this->update_adjacencies();

	return *this;
}

void MeshTopology::update_adjacencies() {
	//Find the adjacent faces to every face
	face_adj.resize(MeshTopology::MaxNeighbors, num_faces());
	face_adj.fill(-1);
	for (size_t f = 0; f < num_faces(); f++) {
		for (size_t k = 0; k < MeshTopology::MaxNeighbors; k++) {
			// Find kth edge 
			unsigned int kinc = (int(k) + 1) % MeshTopology::MaxNeighbors;
			int edge[2] = { quads(k,f), quads(kinc,f) };

			// And find the face that shares its reverse
			int found = 0;
			int other = -1;
			for (size_t fa = 0; fa < num_faces(); fa++)
			{
				if (f == fa) continue;
				for (size_t l = 0; l < MeshTopology::MaxNeighbors; l++)
					if ((quads(l, fa) == edge[1]) && (quads((l + 1) % MeshTopology::MaxNeighbors, fa) == edge[0])) {
						other = (int)fa;
						found++;
					}
			}
			assert(found == 1);

			face_adj(k, f) = other;
		}
	}
}

Eigen::Vector3f MeshTopology::computeBarycenter(const Matrix3X &vertices) {
	Eigen::Vector3f b;
	b << vertices.col(0).sum() / vertices.rows(),
		vertices.col(1).sum() / vertices.rows(),
		vertices.col(2).sum() / vertices.rows();

	return b;
}

bool MeshTopology::isAdjacentFace(const int f1, const int f2) const {
	for (size_t l = 0; l < MeshTopology::MaxNeighbors; l++) {
		if (quads(l, f1) == f2) {
			return true;
		}
	}

	return false;
}