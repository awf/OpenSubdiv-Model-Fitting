
#define _wassert wassert_awf
#include <cassert>
#define _USE_MATH_DEFINES 
#include <cmath>

#include <iostream>
#include <iomanip>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include "eigen_extras.h"

#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "MeshTopology.h"
#include "SubdivEvaluator.h"
#include "log3d.h"

using namespace Eigen;

struct Subdiv3D_Functor : Eigen::SparseFunctor<Scalar>
{
  typedef Eigen::SparseFunctor<Scalar> Base;
  typedef typename Base::JacobianType JacobianType;

  // Input data
  Matrix3X data_points;

  // Topology (faces as vertex indices, fixed during shape optimization)
  MeshTopology mesh;

  SubdivEvaluator evaluator;

  // Functor constructor
  Subdiv3D_Functor(const Matrix3X& data_points, const MeshTopology& mesh) :
    Base(mesh.num_vertices*3 + data_points.cols()*2,   /* number of parameters */
         data_points.cols()*3),                        /* number of residuals */
    data_points(data_points), 
    mesh(mesh),
    evaluator(mesh)
  {
    initWorkspace();
  }

  // Variables for optimization live in InputType
  struct InputType {
    Matrix3X control_vertices;
    std::vector<SurfacePoint> us;

    Index nVertices() const { return control_vertices.cols();  }
  };

  // And the optimization steps are computed using VectorType.
  // For subdivs (see xx), the correspondences are of type (int, Vec2) while the updates are of type (Vec2).
  // The iteractions between InputType and VectorType are restricted to:
  //   The Jacobian computation takeas an InputType, and its worows must easily convert to VectorType
  //   The increment_in_place operation takes InputType and StepType. 
  typedef VectorX VectorType;

  // Workspace variables for evaluation
  Matrix3X S;
  Matrix3X dSdu;
  Matrix3X dSdv;
  SubdivEvaluator::triplets_t dSdX, dSudX, dSvdX;
  void initWorkspace()
  {
    Index nPoints = data_points.cols();
    S.resize(3, nPoints);
    dSdu.resize(3, nPoints);
    dSdv.resize(3, nPoints);
  }

  // Functor functions
  // 1. Evaluate the residuals at x
  int operator()(const InputType& x, ValueType& fvec) {
    evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S);

    // Fill residuals
    for (int i = 0; i < data_points.cols(); i++)
      fvec.segment(i * 3, 3) = S.col(i) - data_points.col(i);

    return 0;
  }

  // 2. Evaluate jacobian at x
  int df(const InputType& x, JacobianType& fjac) 
  {
    // Evaluate surface at x
    evaluator.evaluateSubdivSurface(x.control_vertices, x.us, &S, &dSdX, &dSudX, &dSvdX, &dSdu, &dSdv);

    Index nPoints = data_points.cols();
    Index X_base = nPoints * 2;
    Index ubase = 0;

    // Fill Jacobian columns.  
    // 1. Derivatives wrt control vertices.
    Eigen::TripletArray<Scalar> jvals(nPoints * 3 * 3);
    for (int i = 0; i < dSdX.data.size(); ++i) {
      auto const& triplet = dSdX.data[i];
      assert(0 <= triplet.row() && triplet.row() < nPoints);
      assert(0 <= triplet.col() && triplet.col() < x.nVertices());
      jvals.add(triplet.row() * 3 + 0, X_base + triplet.col() * 3 + 0, triplet.value());
      jvals.add(triplet.row() * 3 + 1, X_base + triplet.col() * 3 + 1, triplet.value());
      jvals.add(triplet.row() * 3 + 2, X_base + triplet.col() * 3 + 2, triplet.value());
    }

    // 2. Derivatives wrt correspondences
    for (int i = 0; i < nPoints; i++) {
      jvals.add(3 * i + 0, ubase + 2 * i + 0, dSdu(0, i));
      jvals.add(3 * i + 1, ubase + 2 * i + 0, dSdu(1, i));
      jvals.add(3 * i + 2, ubase + 2 * i + 0, dSdu(2, i));

      jvals.add(3 * i + 0, ubase + 2 * i + 1, dSdv(0, i));
      jvals.add(3 * i + 1, ubase + 2 * i + 1, dSdv(1, i));
      jvals.add(3 * i + 2, ubase + 2 * i + 1, dSdv(2, i));
    }

    fjac.resize(3 * nPoints, 2 * nPoints + 3 * x.nVertices());
    fjac.setFromTriplets(jvals.begin(), jvals.end());
    fjac.makeCompressed();

    return 0;
  }

  void increment_in_place(InputType* x, StepType const& p)
  {
    Index nPoints = data_points.cols();
    Index X_base = nPoints * 2;
    Index ubase = 0;

    // Increment control vertices
    Index nVertices = x->nVertices();

    assert(p.size() == nVertices * 3 + nPoints * 2);
    assert(x->us.size() == nPoints);

    Map<VectorX>(x->control_vertices.data(), nVertices * 3) += p.tail(nVertices * 3);
    
    // Increment surface correspondences
    int loopers = 0;
    int totalhops = 0;
    for (int i = 0; i < nPoints; ++i) {
      Vector2 du = p.segment<2>(ubase + 2 * i);
      int nhops = increment_u_crossing_edges(x->control_vertices, x->us[i].face, x->us[i].u, du, &x->us[i].face, &x->us[i].u);
      if (nhops < 0)
        ++loopers;
      totalhops += std::abs(nhops);
    }
    if (loopers > 0)
      std::cerr << "[" << totalhops / Scalar(nPoints) << " hops, " << loopers << " points looped]";
    else if (totalhops > 0)
      std::cerr << "[" << totalhops << "/"  << Scalar(nPoints) << " hops]";
  }

  // "Mesh walking" to update correspondences, as in Fig 3, Taylor et al, CVPR 2014, "Hand shape.."
  int increment_u_crossing_edges(Matrix3X const& X, int face, const Vector2& u, const Vector2& du, int* new_face_out, Vector2* new_u_out)
  {
    const int MAX_HOPS = 7;

    Scalar u1_old = u[0];
    Scalar u2_old = u[1];
    Scalar du1 = du[0];
    Scalar du2 = du[1];
    Scalar u1_new = u1_old + du1;
    Scalar u2_new = u2_old + du2;

    for (int count = 0; ; ++count) {
      bool crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

      if (!crossing) {
        *new_face_out = face;
        *new_u_out << u1_new, u2_new;
        return count;
      }

      //Find the new face	and the coordinates of the crossing point within the old face and the new face
      int face_new;

      bool face_found = false;

      Scalar dif, aux, u1_cross, u2_cross;

      if (u1_new < 0.f)
      {
        dif = u1_old;
        const Scalar u2t = u2_old - du2*dif / du1;
        if ((u2t >= 0.f) && (u2t <= 1.f))
        {
          face_new = mesh.face_adj(3, face); aux = u2t; face_found = true;
          u1_cross = 0.f; u2_cross = u2t;
        }
      }
      if ((u1_new > 1.f) && (!face_found))
      {
        dif = 1.f - u1_old;
        const Scalar u2t = u2_old + du2*dif / du1;
        if ((u2t >= 0.f) && (u2t <= 1.f))
        {
          face_new = mesh.face_adj(1, face); aux = 1.f - u2t; face_found = true;
          u1_cross = 1.f; u2_cross = u2t;
        }
      }
      if ((u2_new < 0.f) && (!face_found))
      {
        dif = u2_old;
        const Scalar u1t = u1_old - du1*dif / du2;
        if ((u1t >= 0.f) && (u1t <= 1.f))
        {
          face_new = mesh.face_adj(0, face); aux = 1.f - u1t; face_found = true;
          u1_cross = u1t; u2_cross = 0.f;
        }
      }
      if ((u2_new > 1.f) && (!face_found))
      {
        dif = 1.f - u2_old;
        const Scalar u1t = u1_old + du1*dif / du2;
        if ((u1t >= 0.f) && (u1t <= 1.f))
        {
          face_new = mesh.face_adj(2, face); aux = u1t; face_found = true;
          u1_cross = u1t; u2_cross = 1.f;
        }
      }
      assert(face_found);

      // Find the coordinates of the crossing point as part of the new face, and update u_old (as that will be new u in next iter).
      unsigned int conf;
      for (unsigned int f = 0; f < 4; f++)
        if (mesh.face_adj(f, face_new) == face) { conf = f; }

      switch (conf)
      {
      case 0: u1_old = aux; u2_old = 0.f; break;
      case 1: u1_old = 1.f; u2_old = aux; break;
      case 2:	u1_old = 1.f - aux; u2_old = 1.f; break;
      case 3:	u1_old = 0.f; u2_old = 1.f - aux; break;
      }

      // Evaluate the subdivision surface at the edge (with respect to the original face)
      std::vector<SurfacePoint> pts;
      pts.push_back({ face,{ u1_cross, u2_cross } });
      pts.push_back({ face_new, { u1_old, u2_old } });
      Matrix3X S(3, 2);
      Matrix3X Su(3, 2);
      Matrix3X Sv(3, 2);
      evaluator.evaluateSubdivSurface(X, pts, &S, 0, 0, 0, &Su, &Sv);

      Matrix<Scalar, 3, 2> J_Sa;
      J_Sa.col(0) = Su.col(0);
      J_Sa.col(1) = Sv.col(0);

      Matrix<Scalar, 3, 2> J_Sb;
      J_Sb.col(0) = Su.col(1);
      J_Sb.col(1) = Sv.col(1);

      //Compute the new u increments
      Vector2 du_remaining; 
      du_remaining << u1_new - u1_cross, u2_new - u2_cross;
      Vector3 prod = J_Sa*du_remaining;
      Matrix22 AtA = J_Sb.transpose()*J_Sb;
      Vector2 AtB = J_Sb.transpose()*prod;

      //Vector2 du_new = AtA.ldlt().solve(AtB);
      Vector2  u_incr = AtA.inverse()*AtB;

      du1 = u_incr[0];
      du2 = u_incr[1];

      if (count == MAX_HOPS) {
        //std::cerr << "Problem!!! Many jumps between the mesh faces for the update of one correspondence. I remove the remaining u_increment!\n";
        auto dmax = std::max(du1, du2);
        Scalar scale = Scalar(0.5 / dmax);
        *new_face_out = face;
        //*new_u_out << u1_old + du1 * scale, u2_old + du2 * scale;
        *new_u_out << 0.5, 0.5;

        assert((*new_u_out)[0] >= 0 && (*new_u_out)[1] <= 1.0 && (*new_u_out)[1] >= 0 && (*new_u_out)[1] <= 1.0);
        return -count;
      }

      u1_new = u1_old + du1;
      u2_new = u2_old + du2;
      face = face_new;
    }
  }

  Scalar estimateNorm(InputType const& x, StepType const& diag)
  {
    Index nVertices = x.nVertices();
    Map<VectorX> xtop{ (Scalar*)x.control_vertices.data(), nVertices * 3 };
    double total = xtop.cwiseProduct(diag.tail(nVertices*3)).stableNorm();
    total = total*total;
    for (int i = 0; i < x.us.size(); ++i) {
      Vector2 const& u = x.us[i].u;
      Vector2 di = diag.segment<2>(2 * i);
      total += u.cwiseProduct(di).squaredNorm();
    }
    return Scalar(sqrt(total));
  }

  // 5. Describe the QR solvers
  // For generic Jacobian, one might use this Dense QR solver.
  typedef SparseQR<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;

  // But for optimal performance, declare QRSolver that understands the sparsity structure.
  // Here it's block-diagonal LHS with dense RHS
  //
  // J1 = [J11   0   0 ... 0
  //         0 J12   0 ... 0
  //                   ...
  //         0   0   0 ... J1N];
  // And 
  // J = [J1 J2];

  // QR for J1 subblocks is 2x1
  typedef ColPivHouseholderQR<Matrix<Scalar, 3, 2> > DenseQRSolver3x2;

  // QR for J1 is block diagonal
  typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolver3x2> LeftSuperBlockSolver;

  // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;

  // QR for J is concatenation of the above.
  typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;

  typedef SchurlikeQRSolver QRSolver;

  // And tell the algorithm how to set the QR parameters.
  void initQRSolver(SchurlikeQRSolver &qr) {
    // set block size
    qr.setBlockParams(data_points.cols() * 2);
    qr.getLeftSolver().setSparseBlockParams(3, 2);
  }
};

void logmesh(log3d& log, MeshTopology const& mesh, Matrix3X const& vertices)
{
  Matrix3Xi tris(3, mesh.quads.cols() * 2);
  tris.block(0, 0, 1, mesh.quads.cols()) = mesh.quads.row(0);
  tris.block(1, 0, 1, mesh.quads.cols()) = mesh.quads.row(2);
  tris.block(2, 0, 1, mesh.quads.cols()) = mesh.quads.row(1);
  tris.block(0, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(0);
  tris.block(1, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(3);
  tris.block(2, mesh.quads.cols(), 1, mesh.quads.cols()) = mesh.quads.row(2);
  log.mesh(tris, vertices);
}

void logsubdivmesh(log3d& log, MeshTopology const& mesh, Matrix3X const& vertices)
{
  log.wiremesh(mesh.quads, vertices);
  SubdivEvaluator evaluator(mesh);
  MeshTopology refined_mesh;
  Matrix3X refined_verts;
  evaluator.generate_refined_mesh(vertices, 3, &refined_mesh, &refined_verts);
  logmesh(log, refined_mesh, refined_verts);
}

int main()
{
  std::cout << "Go\n";
  log3d log("log3d.html", "fit-subdiv-to-3d-points");
  log.ArcRotateCamera();
  log.axes();

  // CREATE DATA SAMPLES
  int nDataPoints = 200;
  Matrix3X data(3, nDataPoints);
  for (int i = 0; i < nDataPoints; i++) {
    if (0) {
      float t = float(i) / float(nDataPoints);
      data(0, i) = 0.1f + 1.3f*cos(80*t);
      data(1, i) = -0.2f + 0.7f*sin(80*t);
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
  

  MeshTopology mesh;
  Matrix3X control_vertices_gt;
  makeCube(&mesh, &control_vertices_gt);
  int nFaces = int(mesh.quads.cols());

  // INITIAL PARAMS
  typedef Subdiv3D_Functor Functor;
  
  Functor::InputType params;
  params.control_vertices = control_vertices_gt + 0.1 * MatrixXX::Random(3, control_vertices_gt.cols());
  params.us.resize(nDataPoints);

  // Initialize uvs.
  {
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

  logsubdivmesh(log, mesh, params.control_vertices);

  Functor functor(data, mesh);

  // Check Jacobian
  for (float eps = 1e-7; eps < 1.1e-3; eps*=10) {
    NumericalDiff<Functor> fd{ functor, Functor::Scalar(eps) };
    Functor::JacobianType J;
    Functor::JacobianType J_fd;
    functor.df(params, J);
    fd.df(params, J_fd);
    double diff = (J - J_fd).norm();
    if (diff > 0) {
      std::cerr << "Jacobian diff(eps=" << eps <<"), = " << diff << std::endl;
      write(J, "c:\\tmp\\J.txt");
      write(J_fd, "c:\\tmp\\J_fd.txt");
    }
  }

  Eigen::LevenbergMarquardt< Functor > lm(functor);
  lm.setVerbose(true);
  lm.setMaxfev(10);

  Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
  logsubdivmesh(log, mesh, params.control_vertices);

  std::cerr << "Done: err = "<< lm.fnorm() <<"\n";

  // Now, on a refined mesh.
  {
    MeshTopology mesh1;
    Matrix3X verts1;
    SubdivEvaluator evaluator(mesh);
    evaluator.generate_refined_mesh(params.control_vertices, 1, &mesh1, &verts1);
    
    {log3d log2("log2.html");
    log2.ArcRotateCamera();
    log2.axes();
    log2.wiremesh(mesh1.quads, verts1);
    log2.wiremesh(mesh1.quads, verts1 * 100);
    }

    for (int i = 0; i < nDataPoints; ++i)
      params.us[i].face *= 4;

    params.control_vertices = verts1;
    Functor functor1(data, mesh1);

    Eigen::LevenbergMarquardt< Functor > lm(functor1);
    lm.setVerbose(true);
    lm.setMaxfev(40);

    Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
    logsubdivmesh(log, mesh, params.control_vertices);

    std::cerr << "Done: err = " << lm.fnorm() << "\n";
  }
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message,_In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
  std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

  abort();
}
