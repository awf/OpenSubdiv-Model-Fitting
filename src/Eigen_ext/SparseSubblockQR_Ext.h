// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Sergio Garrido Jurado <>
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_SUBBLOCK_QR_H
#define EIGEN_SPARSE_SUBBLOCK_QR_H

#include <algorithm>
#include <ctime>
#include "../Logger.h"

namespace Eigen {

template<typename _MatrixType, typename _DiagSubblockSolver, typename _SuperblockSolver>
class SparseSubblockQR_Ext : public SparseSolverBase<SparseSubblockQR_Ext<_MatrixType,_DiagSubblockSolver,_SuperblockSolver> >
{
  protected:
    typedef SparseSubblockQR_Ext<_MatrixType, _DiagSubblockSolver, _SuperblockSolver> this_t;
    typedef SparseSolverBase<SparseSubblockQR_Ext<_MatrixType,_DiagSubblockSolver,_SuperblockSolver> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _MatrixType MatrixType;
    typedef _DiagSubblockSolver DiagBlockQRSolver;
    typedef _SuperblockSolver SuperblockQRSolver;
    typedef typename DiagBlockQRSolver::MatrixType DiagBlockMatrixType;
    typedef typename SuperblockQRSolver::MatrixType SuperblockMatrixType;
    typedef typename DiagBlockQRSolver::MatrixQType LeftBlockMatrixQType;
    //typedef typename SuperblockQRSolver::MatrixQType SuperblockMatrixQType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef typename MatrixType::Index Index;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

	typedef SparseMatrix<Scalar, RowMajor, StorageIndex> MatrixQType;
	typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
    //typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
	typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:
    SparseSubblockQR_Ext() : m_diagCols(1)
    { }

    /** Construct a QR factorization of the matrix \a mat.
      *
      * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
      *
      * \sa compute()
      */
    explicit SparseSubblockQR_Ext(const MatrixType& mat) : m_diagCols(1)
    {
      compute(mat);
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
      *
      * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
      *
      * \sa analyzePattern(), factorize()
      */
    void compute(const MatrixType& mat)
    {
      analyzePattern(mat);
      factorize(mat);
    }
    void analyzePattern(const MatrixType& mat);
    void factorize(const MatrixType& mat);

    /** \returns the number of rows of the represented matrix.
      */
    inline Index rows() const { return m_R.rows(); }

    /** \returns the number of columns of the represented matrix.
      */
    inline Index cols() const { return m_R.cols();}

    /** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
      * \warning The entries of the returned matrix are not sorted. This means that using it in algorithms
      *          expecting sorted entries will fail. This include random coefficient accesses (SpaseMatrix::coeff()),
      *          and coefficient-wise operations. Matrix products and triangular solves are fine though.
      *
      * To sort the entries, you can assign it to a row-major matrix, and if a column-major matrix
      * is required, you can copy it again:
      * \code
      * SparseMatrix<double>          R  = qr.matrixR();  // column-major, not sorted!
      * SparseMatrix<double,RowMajor> Rr = qr.matrixR();  // row-major, sorted
      * SparseMatrix<double>          Rc = Rr;            // column-major, sorted
      * \endcode
      */
    const MatrixRType& matrixR() const { return m_R; }

    /** \returns the number of non linearly dependent columns as determined by the pivoting threshold.
      *
      * \sa setPivotThreshold()
      */
    Index rank() const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      return m_nonzeropivots;
    }

    /** \returns the matrix Q 
    */
	MatrixQType matrixQ() const
	{
		return m_Q;
	}

    /** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
      * It is the combination of the fill-in reducing permutation and numerical column pivoting.
      */
    const PermutationType& colsPermutation() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_outputPerm_c;
    }

    /** \returns A string describing the type of error.
      * This method is provided to ease debugging, not to handle errors.
      */
    std::string lastErrorMessage() const { return m_lastError; }

    /** \internal */
    template<typename Rhs, typename Dest>
    bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");

      Index rank = this->rank();

      // Compute Q^T * b;
      typename Dest::PlainObject y, b;
      y = this->matrixQ().transpose() * B;
      b = y;

      // Solve with the triangular matrix R
      y.resize((std::max<Index>)(cols(),y.rows()),y.cols());
      y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
      y.bottomRows(y.rows()-rank).setZero();

      // Apply the column permutation
      if (colsPermutation().size() > 0)
        dest = colsPermutation() * y.topRows(cols());
      else
        dest = y.topRows(cols());

      m_info = Success;
      return true;
    }

    /** Sets the threshold that is used to determine linearly dependent columns during the factorization.
      *
      * In practice, if during the factorization the norm of the column that has to be eliminated is below
      * this threshold, then the entire column is treated as zero, and it is moved at the end.
      */
    void setPivotThreshold(const RealScalar& threshold)
    {
      m_useDefaultThreshold = false;
      m_threshold = threshold;
    }

    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const Solve<SparseSubblockQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<SparseSubblockQR_Ext, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<SparseSubblockQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
          eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
          eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
          return Solve<SparseSubblockQR_Ext, Rhs>(*this, B.derived());
    }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was successful,
      *          \c NumericalIssue if the QR factorization reports a numerical problem
      *          \c InvalidInput if the input matrix is invalid
      *
      * \sa iparm()
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }

    void setDiagBlockParams(Index blockRows, Index blockCols) {
		m_diagRows = blockRows;
        m_diagCols = blockCols;
    }

    DiagBlockQRSolver& getDiagSolver() { return m_diagSolver; }
    SuperblockQRSolver& getSuperSolver() { return m_superSolver; }


  protected:
    mutable ComputationInfo m_info;

	MatrixQType m_Q;				// Q for this solver (assuming it's sparse enough) 
    MatrixRType m_R;                // The triangular factor matrix
    ScalarVector m_hcoeffs;         // The Householder coefficients

    PermutationType m_outputPerm_c; // The final column permutation

    Index m_nonzeropivots;          // Number of non zero pivots found
    IndexVector m_etree;            // Column elimination tree
    IndexVector m_firstRowElt;      // First element in each row

    Index m_diagCols;                // Cols of first block
	Index m_diagRows;				  // Rows of the first block
									  // Every row below the first block is treated as a part of already upper triangular block)
    DiagBlockQRSolver m_diagSolver;
    SuperblockQRSolver m_superSolver;

    template <typename, typename > friend struct SparseQR_QProduct;

};

/** \brief Preprocessing step of a QR factorization
  *
  * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
  *
  * In this step, the fill-reducing permutation is computed and applied to the columns of A
  * and the column elimination tree is computed as well. Only the sparsity pattern of \a mat is exploited.
  *
  * \note In this step it is assumed that there is no empty row in the matrix \a mat.
  */
template <typename MatrixType, typename DiagBlockQRSolver, typename SuperblockQRSolver>
void SparseSubblockQR_Ext<MatrixType, DiagBlockQRSolver, SuperblockQRSolver>::analyzePattern(const MatrixType& mat)
{
  eigen_assert(mat.isCompressed() && "SparseQR requires a sparse matrix in compressed mode. Call .makeCompressed() before passing it to SparseQR");

  Index n = mat.cols();
  m_outputPerm_c.resize(n);
  m_outputPerm_c.indices().setLinSpaced(n, 0, StorageIndex(n - 1));
}

/** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
template <typename MatrixType, typename DiagBlockQRSolver, typename SuperblockQRSolver>
void SparseSubblockQR_Ext<MatrixType,DiagBlockQRSolver,SuperblockQRSolver>::factorize(const MatrixType& mat)
{
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
    typedef MatrixType::Index Index;
	Index m = mat.cols();
    Index n1 = m_diagRows;
	Index n2 = mat.rows() - m_diagRows;

	
	///	Decomposition of the form:
	///	Q * R = | Q1 R1 | = | Q1 R1 | = | Q1 0 | * | R1 |	R1 - upper triangular
	///			| Q2 R2 |	| I  R2 |   | 0  I |   | R2 |	R2 - not yet solved (non block-diagonal part of the matrix)
	///
	///	Final solution obtained as:
	///	| R1 | = Q3 * R3 => Q * R = | Q1 0 | * Q3 * R3
	///	| R2 |						| 0  I |
	///
	///	where:
	///	Q1 = (n1 x n1), R1 = (n1 x m)
	///	Q2 = (n2 x n2), R2 = (n2 x m)
	///	Q3 = (n1 + n2) x (n1 + n2), R3 = (n1 + n2) * m
	
	// Split the diagonal block from the separately treated bottom
	MatrixType diagBlock = mat.topRows(n1);
	MatrixType bottomBlock = mat.bottomRows(n2);

    // Compute QR for the block diagonal part
	m_diagSolver.compute(diagBlock);
	
	eigen_assert(m_diagSolver.info() == Success);

    typename DiagBlockQRSolver::MatrixRType R1 = m_diagSolver.matrixR();

	// Permute the bottom block according to the diagonal solver column permutations
	m_diagSolver.colsPermutation().applyThisOnTheRight(bottomBlock);
	
	/// | R1 |
	/// | R2 |
	// Create the superblock matrix
	Eigen::TripletArray<Scalar, Index> triplets(R1.nonZeros() + bottomBlock.nonZeros()); 
	for (Index k = 0; k < R1.outerSize(); ++k) {
		for (typename MatrixType::InnerIterator it(R1, k); it; ++it) {
			if (it.row() < n1) { // xxawf Hoist if R1.IsRowMajor?
				triplets.add(it.row(), it.col(), it.value());
			}
		}
	}
	for (Index k = 0; k < bottomBlock.outerSize(); ++k) {
		for (typename MatrixType::InnerIterator it(bottomBlock, k); it; ++it) {
			if (it.row() < n2) { // xxawf Hoist if bottomBlock.IsRowMajor?
				triplets.add(it.row() + n1, it.col(), it.value());
			}
		}
	}
	SuperblockMatrixType superblock;
	superblock.resize(n1 + n2, m);
	superblock.setFromTriplets(triplets.begin(), triplets.end());
	superblock.makeCompressed();
	superblock.prune(Scalar(0));

//	SuperblockMatrixType superblock(n1 + n2, m);
//	superblock.topRows(n1) = R1;
//	superblock.bottomRows(n2) = bottomBlock;

	/// | R1 | = Q3 * R3 
	/// | R2 |
	// Solve the superblock using general sparse QR
	m_superSolver.compute(superblock);
	eigen_assert(m_superSolver.info() == Success);

	// Output the final QR decomposition
	this->m_R = m_superSolver.matrixR();
	// Remember m_Q is RowMajor (easier to fill in this case)
	this->m_Q.resize(n1 + n2, n1 + n2);
	this->m_Q.setIdentity();
	DiagBlockQRSolver::MatrixQType diagQ = m_diagSolver.matrixQ();
	diagQ.conservativeResize(n1, n1 + n2);
	this->m_Q.topRows(n1) = diagQ;
	MatrixType superQ(n1 + n2, n1 + n2);
	superQ.setIdentity();
	superQ = m_superSolver.matrixQ() * superQ;
	this->m_Q = this->m_Q * superQ;

    // fill cols permutation
	//SuperblockQRSolver::PermutationType perm = m_superSolver.colsPermutation() * m_diagSolver.colsPermutation();
	PermutationType perm = m_superSolver.colsPermutation() * m_diagSolver.colsPermutation();
	for (Index j = 0; j < m; j++)
		m_outputPerm_c.indices()(j, 0) = perm.indices()(j, 0);//m_superSolver.colsPermutation().indices()(j,0);

	this->m_R.makeCompressed();
	this->m_Q.makeCompressed();
	
    //m_nonzeropivots = m_diagSolver.rank() + m_superSolver.rank();
	m_nonzeropivots = m_superSolver.rank();
	m_isInitialized = true;
    m_info = Success;
}


} // end namespace Eigen

#endif
