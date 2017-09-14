#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>

#include <random>

#include <Eigen/Eigen>

#include "log3d.h"

#include "FPJParser.h"
#include "PLYParser.h"
#include "Logger.h"
#include "BezierPatch.h"
#include "RigidTransform.h"

#include "Optimization/PosOnlyFunctor.h"
#include "Optimization/PosOnlyWithRegFunctor.h"
#include "Optimization/PosAndNormalsFunctor.h"
#include "Optimization/PosAndNormalsWithRegFunctor.h"

//typedef PosOnlyFunctor OptimizationFunctor;
typedef PosOnlyWithRegFunctor OptimizationFunctor;
//typedef PosAndNormalsFunctor OptimizationFunctor;
//typedef PosAndNormalsWithRegFunctor OptimizationFunctor;

using namespace Eigen;


void logmesh(log3d& log, MeshTopology const& mesh, Matrix3X const& vertices) {
	Matrix3Xi tris(3, mesh.quads.cols() * 2);
	
	tris.block(0, 0, 1, mesh.quads.cols()) = mesh.quads.row(0);
	tris.block(1, 0, 1, mesh.quads.cols()) = mesh.quads.row(2);
	tris.block(2, 0, 1, mesh.quads.cols()) = mesh.quads.row(1);
	tris.block(0, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(0);
	tris.block(1, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(3);
	tris.block(2, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(2);
	/*
	tris.block(0, 0, 1, mesh.quads.cols()) = mesh.quads.row(3);
	tris.block(1, 0, 1, mesh.quads.cols()) = mesh.quads.row(1);
	tris.block(2, 0, 1, mesh.quads.cols()) = mesh.quads.row(2);
	tris.block(0, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(3);
	tris.block(1, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(0);
	tris.block(2, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(1);
	*/
	log.mesh(tris, vertices);
	
}

void logsubdivmesh(log3d& log, MeshTopology const& mesh, Matrix3X const& vertices) {
	log.wiremesh(mesh.quads, vertices);
	SubdivEvaluator evaluator(mesh);
	MeshTopology refined_mesh;
	Matrix3X refined_verts;
	evaluator.generate_refined_mesh(vertices, 3, &refined_mesh, &refined_verts);
	logmesh(log, refined_mesh, refined_verts);
}

// Initialize UVs to the middle of each face
void initializeUVs(MeshTopology &mesh, OptimizationFunctor::InputType &params, const Matrix3X &data) {
	int nFaces = int(mesh.quads.cols());
	int nDataPoints = int(data.cols());

	// 1. Make a list of test points, e.g. centre point of each face
	Matrix3X test_points(3, nFaces);
	std::vector<SurfacePoint> uvs{ size_t(nFaces),{ 0,{ 0.5, 0.5 } } };
	for (int i = 0; i < nFaces; ++i)
		uvs[i].face = i;

	SubdivEvaluator evaluator(mesh);
	evaluator.evaluateSubdivSurface(params.control_vertices, uvs, &test_points);
	
	for (int i = 0; i < nDataPoints; i++) {
		// Closest test point
		Eigen::Index test_pt_index;
		(test_points.colwise() - data.col(i)).colwise().squaredNorm().minCoeff(&test_pt_index);
		params.us[i] = uvs[test_pt_index];
	}
}

// Transformation of 3D model to the initial alignment
void transform3D(const Matrix3X &points3D, Matrix3X &points3DOut, const FPJParser::ImageFile &imageParams) {
	int nDataPts = int(points3D.cols());

	// Convert points into homogeneous coordinates
	MatrixXd pts3DTransf = MatrixXd::Ones(nDataPts, 4);
	pts3DTransf.block(0, 0, nDataPts, 3) << points3D.transpose();
	// Apply positioning transformation (translation, rotation, scale)
	pts3DTransf = pts3DTransf * imageParams.rigidTransf.transformation().cast<Scalar>();

	// Ortographic projection into 2D plane defined as Phi([x, y, z]) = [x, y]
	points3DOut << pts3DTransf.block(0, 0, nDataPts, 3).transpose();
}

// Projection of 3D model into 2D
void project3DTo2D(const Matrix3X &points3D, Matrix2X &points2D, const FPJParser::ImageFile &imageParams) {
	int nDataPts = int(points3D.cols());
	Matrix3X points3DTransf(3, nDataPts);
	transform3D(points3D, points3DTransf, imageParams);

	// Ortographic projection into 2D plane defined as Phi([x, y, z]) = [x, y]
	points2D << points3DTransf.block(0, 0, 2, nDataPts);
}

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "Computation STARTED!");

	std::cout << "Go\n";
	// CREATE DATA SAMPLES
	/*
	int nDataPoints = 200;
	Matrix3X data(3, nDataPoints);
	for (int i = 0; i < nDataPoints; i++) {
		if (0) {
			float t = float(i) / float(nDataPoints);
			data(0, i) = 0.1f + 1.3f*cos(80 * t);
			data(1, i) = -0.2f + 0.7f*sin(80 * t);
			data(2, i) = t;
		}
		else {
			Scalar t = rand() / Scalar(RAND_MAX);
			Scalar s = rand() / Scalar(RAND_MAX);

			auto u = Scalar(2 * EIGEN_PI * t);
			auto v = Scalar(EIGEN_PI * (s - 0.5));
			data(0, i) = 0.1f + 1.3f*cos(u)*cos(v);
			data(1, i) = -0.2f + 0.7f*sin(u)*cos(v);
			data(2, i) = sin(v);
		}
		if (1)
			log.position(log.CreateSphere(0, 0.02), data(0, i), data(1, i), data(2, i));
		else
			log.star(data.col(i));
	}
	*/
	// Load banana model
	//PLYParser plyParse("sphere_quad.ply");
	PLYParser plyParse("Z:/OpenSubdiv-Model-Fitting/build/Debug/banana_quad_coarse.ply");
	plyParse.parse(PLYParser::Model::Quads);
	// Get the user input image parameters
	FPJParser fpjParse("Z:/OpenSubdiv-Model-Fitting/build/Debug/projects/bananas.fpj");
	fpjParse.parse();
	// Prepare final 3d log
	log3d log("log3d.html", "fit-subdiv-to-3d-points");
	log.ArcRotateCamera();
	log.axes();
	log.color(1, 0, 0);
	/*int nDataPoints = int(plyParse.model().vertices.rows());
	Matrix3X data(3, nDataPoints);
	for (int i = 0; i < nDataPoints; i++) {
		data(0, i) = plyParse.model().vertices(i, 0);
		data(1, i) = plyParse.model().vertices(i, 1);
		data(2, i) = plyParse.model().vertices(i, 2);

		//logb.position(logb.CreateSphere(0, 0.02), data(0, i), data(1, i), data(2, i));
		//logb.position(logb.CreateSphere(0, 0.05), data(0, i), data(1, i), 0.0);
		log.position(log.CreateSphere(0, 0.05), data(0, i), data(1, i), data(2, i));
	}*/

	//const unsigned int nParamVals = 3;
	//float t[nParamVals] = { 0.0, 0.5, 1.0 };
	const unsigned int nParamVals = 10;
	float t[nParamVals] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	int nDataPoints = int(fpjParse.project().images[0].silhouettePoints[0].rows()) * nParamVals;
	std::stringstream ss;
	ss << "Number of data points: " << nDataPoints;
	Logger::instance()->log(Logger::Info, ss.str());
	Matrix3X data(3, nDataPoints);
	Matrix3X dataNormals(3, nDataPoints);
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < int(fpjParse.project().images[0].silhouettePoints[0].rows()); i++) {
		for (int j = 0; j < nParamVals; j++) {
			Eigen::Vector2f pt = BezierPatch::evaluateAt(fpjParse.project().images[0].silhouettePoints[0].row(i),
				fpjParse.project().images[0].silhouettePoints[1].row(i),
				fpjParse.project().images[0].silhouettePoints[2].row(i),
				fpjParse.project().images[0].silhouettePoints[3].row(i), t[j]);
			data(0, nParamVals * i + j) = pt(0);
			data(1, nParamVals * i + j) = pt(1);
			data(2, nParamVals * i + j) = 0.0f;// dist(gen) * 0.125 - 0.125 / 2.0;
			
			Eigen::Vector2f n = BezierPatch::evaluateNormalAt(fpjParse.project().images[0].silhouettePoints[0].row(i),
				fpjParse.project().images[0].silhouettePoints[1].row(i),
				fpjParse.project().images[0].silhouettePoints[2].row(i),
				fpjParse.project().images[0].silhouettePoints[3].row(i), t[j], false);
			n.normalize();
			dataNormals(0, nParamVals * i + j) = n(0);
			dataNormals(1, nParamVals * i + j) = n(1);
			dataNormals(2, nParamVals * i + j) = 0.0f;

			//logb.position(logb.CreateSphere(0, 0.02), data(0, i), data(1, i), data(2, i));
			//logb.position(logb.CreateSphere(0, 0.05), data(0, i), data(1, i), 0.0);
			log.position(log.CreateSphere(0, 0.05), data(0, nParamVals * i + j), data(1, nParamVals * i + j), data(2, nParamVals * i + j));
		}
	}

	std::cout << nDataPoints << std::endl;

	// Draw one of the silhouettes
	log3d logb("banana.html", "fit-subdiv-to-3d-points");
	logb.ArcRotateCamera();
	logb.axes();
	logb.color(1, 0, 0);
	//std::cout << int(fpjParse.project().images[0].silhouettePoints.rows()) << std::endl;
	//std::cout << fpjParse.project().images[0].silhouettePoints << std::endl;
	for (int i = 0; i < int(fpjParse.project().images[0].silhouettePoints[0].rows()); i++) {
		Eigen::Vector2f pt = BezierPatch::evaluateAt(fpjParse.project().images[0].silhouettePoints[0].row(i),
			fpjParse.project().images[0].silhouettePoints[1].row(i), 
			fpjParse.project().images[0].silhouettePoints[2].row(i), 
			fpjParse.project().images[0].silhouettePoints[3].row(i), 0.5);
		logb.position(logb.CreateSphere(0, 0.05), pt(0), pt(1), 0.0);
	}
	
	// Make "control" cube
	MeshTopology mesh;
	Matrix3X control_vertices_gt;
	makeFromPLYModel(&mesh, &control_vertices_gt, plyParse.model());
	//std::cout << mesh.quads << std::endl;
	//makeCube(&mesh, &control_vertices_gt);	

	// INITIAL PARAMS
	OptimizationFunctor::InputType params;
	params.control_vertices = control_vertices_gt;// +0.1 * MatrixXX::Random(3, control_vertices_gt.cols());
	params.us.resize(nDataPoints);
	//params.rigidTransf.setTranslation(fpjParse.project().images[0].rigidTransf.params().t1, 
	//	fpjParse.project().images[0].rigidTransf.params().t2, 
	//	fpjParse.project().images[0].rigidTransf.params().t3);
	//Eigen::Vector3f barycenter = plyParse.model().barycenter();
	Eigen::Vector3f barycenter = MeshTopology::computeBarycenter(data);
	log.color(0.0, 0.0, 1.0);
	log.position(log.CreateSphere(0, 0.05), barycenter(0), barycenter(1), barycenter(2));
	params.rigidTransf.setTranslation(barycenter(0), barycenter(1), barycenter(2));
	//params.rigidTransf.setRotation(fpjParse.project().images[0].rigidTransf.params().r1,
	//	fpjParse.project().images[0].rigidTransf.params().r2,
	//params.rigidTransf.setScaling(0.5, 0.5, 0.5);
	//params.rigidTransf.setScaling(1.0, 10.0, 5.0);
	params.rigidTransf.setScaling(fpjParse.project().images[0].rigidTransf.params().s1 * 2.0,
		fpjParse.project().images[0].rigidTransf.params().s2 * 2.0,
		fpjParse.project().images[0].rigidTransf.params().s3 * 2.0);
	
	// Log initialization
	{
		Matrix4f transf = params.rigidTransf.translation() * params.rigidTransf.scaling() * params.rigidTransf.rotation();
		Matrix3X tCVs(3, params.control_vertices.cols());
		for (int i = 0; i < params.control_vertices.cols(); i++) {
			Eigen::Vector4f pt;
			pt << params.control_vertices(0, i), params.control_vertices(1, i), params.control_vertices(2, i), 1.0f;
			pt = transf * pt;
			tCVs(0, i) = pt(0);
			tCVs(1, i) = pt(1);
			tCVs(2, i) = pt(2);
		}

		logsubdivmesh(log, mesh, tCVs);
	}

	// Initialize uvs.
	initializeUVs(mesh, params, data);
	//logsubdivmesh(log, mesh, params.control_vertices);
	
	/*MeshTopology mesh1;
	Matrix3X verts1;
	SubdivEvaluator evaluator(mesh);
	evaluator.generate_refined_mesh(params.control_vertices, 1, &mesh1, &verts1);
	params.control_vertices = verts1;
	initializeUVs(mesh1, params, data);
	mesh = mesh1;*/

	OptimizationFunctor::DataConstraints constraints;
	//constraints.push_back(OptimizationFunctor::DataConstraint(0, 1));
	OptimizationFunctor functor(data, mesh, constraints);
	//OptimizationFunctor functor(data, dataNormals, mesh, constraints);
	

	// Check Jacobian
	if (0) {
		std::cout << "Test Jacobian MODE" << std::endl;
		for (float eps = 1e-8f; eps < 1.1e-3f; eps *= 10.f) {
			NumericalDiff<OptimizationFunctor> fd{ functor, OptimizationFunctor::Scalar(eps) };
			OptimizationFunctor::JacobianType J;
			OptimizationFunctor::JacobianType J_fd;
			functor.df(params, J);
			fd.df(params, J_fd);
			double diff = (J - J_fd).norm();
			unsigned int num_uv = nDataPoints * 2;
			unsigned int num_tsr = 9;
			unsigned int num_xyz = mesh.num_vertices * 3;
			unsigned int num_res_p = nDataPoints * 3;
			unsigned int num_res_n = nDataPoints;// nDataPoints * 3;
			unsigned int num_res_c = constraints.size() * 3;
			unsigned int num_res_tp = num_xyz;
			if (diff > 0) {
				std::cout << "pn-xyz: " << (J.toDense().block(0, num_uv + num_tsr, num_res_p + num_res_n, num_xyz) - J_fd.toDense().block(0, num_uv + num_tsr, num_res_p + num_res_n, num_xyz)).norm() << std::endl;
				std::cout << "pn-uv: " << (J.toDense().block(0, 0, num_res_p + num_res_n, num_uv) - J_fd.toDense().block(0, 0, num_res_p + num_res_n, num_uv)).norm() << std::endl;
				std::cout << "pn-t: " << (J.toDense().block(0, num_uv, num_res_p + num_res_n, 3) - J_fd.toDense().block(0, num_uv, num_res_p + num_res_n, 3)).norm() << std::endl;
				std::cout << "pn-s: " << (J.toDense().block(0, num_uv + 3, num_res_p + num_res_n, 3) - J_fd.toDense().block(0, num_uv + 3, num_res_p + num_res_n, 3)).norm() << std::endl;
				std::cout << "pn-r: " << (J.toDense().block(0, num_uv + 6, num_res_p + num_res_n, 3) - J_fd.toDense().block(0, num_uv + 6, num_res_p + num_res_n, 3)).norm() << std::endl;
				std::cout << "pn-tsr: " << (J.toDense().block(0, num_uv, num_res_p + num_res_n, num_tsr) - J_fd.toDense().block(0, num_uv, num_res_p + num_res_n, num_tsr)).norm() << std::endl;
				std::cout << "c-xyz: " << (J.toDense().block(num_res_p + num_res_n, num_uv + num_tsr, num_res_c, num_xyz) - J_fd.toDense().block(num_res_p + num_res_n, num_uv + num_tsr, num_res_c, num_xyz)).norm() << std::endl;
				std::cout << "c-uv: " << (J.toDense().block(num_res_p + num_res_n, 0, num_res_c, num_uv) - J_fd.toDense().block(num_res_p + num_res_n, 0, num_res_c, num_uv)).norm() << std::endl;
				std::cout << "c-tsr: " << (J.toDense().block(num_res_p + num_res_n, num_uv, num_res_c, num_tsr) - J_fd.toDense().block(num_res_p + num_res_n, num_uv, num_res_c, num_tsr)).norm() << std::endl;
				std::cout << "tp-xyz: " << (J.toDense().block(num_res_p + num_res_n + num_res_c, num_uv + num_tsr, num_res_tp, num_xyz) - J_fd.toDense().block(num_res_p + num_res_n + num_res_c, num_uv + num_tsr, num_res_tp, num_xyz)).norm() << std::endl;
				std::cout << "tp-uv: " << (J.toDense().block(num_res_p + num_res_n + num_res_c, 0, num_res_tp, num_uv) - J_fd.toDense().block(num_res_p + num_res_n + num_res_c, 0, num_res_tp, num_uv)).norm() << std::endl;
				std::cout << "tp-tsr: " << (J.toDense().block(num_res_p + num_res_n + num_res_c, num_uv, num_res_tp, num_tsr) - J_fd.toDense().block(num_res_p + num_res_n + num_res_c, num_uv, num_res_tp, num_tsr)).norm() << std::endl;
				
				/*
				std::ofstream ofs("j_xyz_fd.csv");
				ofs << J_fd.toDense().block(0, num_uv + num_tsr, num_res_p + num_res_c + num_res_tp, num_xyz) << std::endl;
				ofs.close();

				std::ofstream ofs2("j_xyz_my.csv");
				ofs2 << J.toDense().block(0, num_uv + num_tsr, num_res_p + num_res_c + num_res_tp, num_xyz) << std::endl;
				ofs2.close();
				*/

				Logger::instance()->logMatrixCSV(J_fd.toDense().block(0, 0, num_res_p + num_res_n, num_uv), "j_pnuv_fd.csv");
				Logger::instance()->logMatrixCSV(J.toDense().block(0, 0, num_res_p + num_res_n, num_uv), "j_pnuv_my.csv");
				//Logger::instance()->logMatrixCSV(J_fd.toDense().block(num_res_p + num_res_c, num_uv, num_res_tp, num_tsr + num_xyz), "j_tp_fd.csv");
				//Logger::instance()->logMatrixCSV(J.toDense().block(num_res_p + num_res_c, num_uv, num_res_tp, num_tsr + num_xyz), "j_tp_my.csv");
				
				/*
				std::ofstream ofs("p_st_fd.csv");
				ofs << J_fd.toDense().block<560, 9>(0, 374) << std::endl;
				ofs.close();

				std::ofstream ofs2("p_st_my.csv");
				ofs2 << J.toDense().block<560, 9>(0, 374)<< std::endl;
				ofs2.close();
				*/
				/*std::ofstream ofs("tp-xyz_fd.csv");
				ofs << J_fd.toDense().block<24, 24>(153, 102) << std::endl;
				ofs.close();

				std::ofstream ofs2("tp-xyz_my.csv");
				ofs2 << J.toDense().block<24, 24>(153, 102) << std::endl;
				ofs2.close();*/

				std::stringstream ss;
				ss << "Jacobian diff(eps=" << eps << "), = " << diff;
				Logger::instance()->log(Logger::Debug, ss.str());
				Logger::instance()->logSparseMatrix(J, "J.txt");
				Logger::instance()->logSparseMatrix(J_fd, "J_fd.txt");
			}

			if (diff > 10.0) {
				std::cout << "Test Jacobian - ERROR TOO BIG, exitting..." << std::endl;
				return 0;
			}
		}
		std::cout << "Test Jacobian - DONE, exitting..." << std::endl;
		return 0;
	}

	// Set-up the optimization
	Eigen::LevenbergMarquardt< OptimizationFunctor > lm(functor);
	lm.setVerbose(true);
	lm.setMaxfev(40);
	
	Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
	//log.color(0, 1, 0);
	//logsubdivmesh(log, mesh, params.control_vertices);

	std::cerr << "Done: err = " << lm.fnorm() << "\n";
	// Now, on a refined mesh.
	const unsigned int numSteps = 3;
	MeshTopology currMesh = mesh;
	
	if (0) {
		for (int i = 0; i < numSteps; i++) {
			MeshTopology mesh1;
			Matrix3X verts1;
			SubdivEvaluator evaluator(currMesh);
			evaluator.generate_refined_mesh(params.control_vertices, (i < 2) ? 1 : 0, &mesh1, &verts1);

			{
				log3d log2("log2.html");
				log2.ArcRotateCamera();
				log2.axes();
				log2.wiremesh(mesh1.quads, verts1);
			}

			params.control_vertices = verts1;
			// Initialize uvs.
			initializeUVs(mesh1, params, data);
			//OptimizationFunctor functor1(data, dataNormals, mesh1, constraints);
			OptimizationFunctor functor1(data, mesh1, constraints);
			Eigen::LevenbergMarquardt< OptimizationFunctor > lm(functor1);
			lm.setVerbose(true);
			lm.setMaxfev(40);
			Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);

			std::cerr << "Done: err = " << lm.fnorm() << "\n";

			currMesh = mesh1;

			if (lm.fnorm() < 1e-12) {
				break;
			}
		}
	}
	
	if (1) {
		log.color(.5, .8, 0);

		Matrix4f transf = params.rigidTransf.translation() * params.rigidTransf.scaling() * params.rigidTransf.rotation();
		Matrix3X tCVs(3, params.control_vertices.cols());
		for (int i = 0; i < params.control_vertices.cols(); i++) {
			Eigen::Vector4f pt;
			pt << params.control_vertices(0, i), params.control_vertices(1, i), params.control_vertices(2, i), 1.0f;
			pt = transf * pt;
			tCVs(0, i) = pt(0);
			tCVs(1, i) = pt(1);
			tCVs(2, i) = pt(2);
		}

		logsubdivmesh(log, currMesh, tCVs);
		//logsubdivmesh(log, mesh1, params.control_vertices);
	}
	
	Logger::instance()->log(Logger::Info, "Computation DONE!");
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
