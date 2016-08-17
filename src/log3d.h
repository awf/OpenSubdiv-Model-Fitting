#pragma once

#include <fstream>
#include "eigen_extras.h"

// Send 3D primitives to an HTML file, which can be viewed in a browser.
struct log3d {
  std::ofstream f;

  log3d(std::string filename, std::string tag = "");
  ~log3d();

  log3d& operator<<(std::string const& s);
  
  typedef std::string object_t;

  // Initialization stuff
  void ArcRotateCamera();
  void axes();

  object_t CreateSphere(int subdivisions, double radius);
  object_t CreatePlane(double v);

  void position(object_t obj, double x, double y, double z);
  void rotation(object_t obj, double x, double y, double z);

  void mesh(Eigen::Matrix3Xi const& triangle_indices, Matrix3X const& V);
  void wiremesh(Eigen::MatrixXi const& faces, Matrix3X const& V);
  void lines(Matrix3X const& V, bool closed = false);
  void star(Vector3 const & X);


  // Internals
private:
  // These are to be called in specific order, and there's no possibility of multiple canvases, so let's leave private.
  void head(std::string tag);
  void canvas();
  void done();
  void endcanvas();
  static int next_obj;
  object_t newobj(std::string prefix);
};
