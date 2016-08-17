#pragma once

#include <Eigen/Eigen>

typedef double Scalar;
// Like the eigen typedefs, but using the Scalar template parameter
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXX;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Matrix3X;
typedef Eigen::Matrix<Scalar, 2, Eigen::Dynamic> Matrix2X;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix22;
typedef Eigen::Matrix<Scalar, 3, 2> Matrix32;
typedef Eigen::Matrix<Scalar, 2, 3> Matrix23;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;


template <typename T, int _Options, typename _Index>
void write(Eigen::SparseMatrix<T, _Options, _Index> const& J, char const* filename)
{
  std::ofstream f{ filename };
  if (!f.good()) {
    std::cerr << "Failed to open [" << filename << "] for writing\n";
    return;
  }
  std::cout << "Writing " << J.rows() << "x" << J.cols() << " sparse to [" << filename << "]\n";
  for (int k = 0; k < J.outerSize(); ++k)
    for (Eigen::SparseMatrix<T, _Options, _Index>::InnerIterator it(J, k); it; ++it)
      f << it.row() << "\t" << it.col() << "\t" << it.value() << std::endl;
}

template <typename Derived>
void write(Eigen::MatrixBase<Derived> const& J, char const* filename)
{
  std::ofstream f{ filename };
  if (!f.good()) {
    std::cerr << "Failed to open [" << filename << "] for writing\n";
    return;
  }
  std::cout << "Writing " << J.rows() << "x" << J.cols() << " dense to [" << filename << "]\n";
  f << J;
}

/*
template <typename M1, typename M2>
auto hcat(M1 const& A, M2 const& B)
{
  M1 out(A.rows(), A.cols() + B.cols());
  out << A, B;
  return out;
}
*/

