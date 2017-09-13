// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_BANDED_QR_EXT_H
#define EIGEN_SPARSE_BANDED_QR_EXT_H

#include <ctime>

namespace Eigen {

template<typename MatrixType, typename OrderingType> class SparseBandedQR_Ext;
template<typename SparseBandedQR_ExtType> struct SparseBandedQR_ExtMatrixQReturnType;
template<typename SparseBandedQR_ExtType> struct SparseBandedQR_ExtMatrixQTransposeReturnType;
template<typename SparseBandedQR_ExtType, typename Derived> struct SparseBandedQR_Ext_QProduct;
namespace internal {

  // traits<SparseBandedQR_ExtMatrixQ[Transpose]>
  template <typename SparseBandedQR_ExtType> struct traits<SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType> >
  {
    typedef typename SparseBandedQR_ExtType::MatrixType ReturnType;
    typedef typename ReturnType::StorageIndex StorageIndex;
    typedef typename ReturnType::StorageKind StorageKind;
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
  };

  template <typename SparseBandedQR_ExtType> struct traits<SparseBandedQR_ExtMatrixQTransposeReturnType<SparseBandedQR_ExtType> >
  {
    typedef typename SparseBandedQR_ExtType::MatrixType ReturnType;
  };

  template <typename SparseBandedQR_ExtType, typename Derived> struct traits<SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, Derived> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };

  // SparseBandedQR_Ext_traits
  template <typename T> struct SparseBandedQR_Ext_traits {  };
  template <class T, int Rows, int Cols, int Options> struct SparseBandedQR_Ext_traits<Matrix<T,Rows,Cols,Options>> { 
    typedef Matrix<T,Rows,1,Options> Vector;
  };
  template <class Scalar, int Options, typename Index> struct SparseBandedQR_Ext_traits<SparseMatrix<Scalar,Options,Index>> { 
    typedef SparseVector<Scalar,Options> Vector;
  };
} // End namespace internal

/**
  * \ingroup SparseBandedQR_Ext_Module
  * \class SparseBandedQR_Ext
  * \brief Sparse Householder QR Factorization for banded matrices
  * This implementation is not rank revealing and uses Eigen::HouseholderQR for solving the dense blocks.
  * 
  * Q is the orthogonal matrix represented as products of Householder reflectors. 
  * Use matrixQ() to get an expression and matrixQ().transpose() to get the transpose.
  * You can then apply it to a vector.
  * 
  * R is the sparse triangular or trapezoidal matrix. The later occurs when A is rank-deficient.
  * matrixR().topLeftCorner(rank(), rank()) always returns a triangular factor of full rank.
  * 
  * \tparam _MatrixType The type of the sparse matrix A, must be a column-major SparseMatrix<>
  * \tparam _OrderingType The fill-reducing ordering method. See the \link OrderingMethods_Module 
  *  OrderingMethods \endlink module for the list of built-in and external ordering methods.
  * 
  * \implsparsesolverconcept
  *
  * \warning The input sparse matrix A must be in compressed mode (see SparseMatrix::makeCompressed()).
  * 
  */
template<typename _MatrixType, typename _OrderingType>
class SparseBandedQR_Ext : public SparseSolverBase<SparseBandedQR_Ext<_MatrixType,_OrderingType> >
{
  protected:
    typedef SparseSolverBase<SparseBandedQR_Ext<_MatrixType,_OrderingType> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _MatrixType MatrixType;
    typedef _OrderingType OrderingType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

    typedef SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_Ext> MatrixQType;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
	typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    
  public:
    SparseBandedQR_Ext () :  m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true),m_isQSorted(false), m_sqrtEps(1e-8), m_blockRows(4), m_blockCols(2)
    { }
    
    /** Construct a QR factorization of the matrix \a mat.
      * 
      * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
      * 
      * \sa compute()
      */
    explicit SparseBandedQR_Ext(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true),m_isQSorted(false), m_sqrtEps(1e-8), m_blockRows(4), m_blockCols(2)
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
    inline Index rows() const { return m_pmat.rows(); }
    
    /** \returns the number of columns of the represented matrix. 
      */
    inline Index cols() const { return m_pmat.cols();}
    
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
    
    /** \returns an expression of the matrix Q as products of sparse Householder reflectors.
    * The common usage of this function is to apply it to a dense matrix or vector
    * \code
    * VectorXd B1, B2;
    * // Initialize B1
    * B2 = matrixQ() * B1;
    * \endcode
    *
    * To get a plain SparseMatrix representation of Q:
    * \code
    * SparseMatrix<double> Q;
    * Q = SparseBandedQR_Ext<SparseMatrix<double> >(A).matrixQ();
    * \endcode
    * Internally, this call simply performs a sparse product between the matrix Q
    * and a sparse identity matrix. However, due to the fact that the sparse
    * reflectors are stored unsorted, two transpositions are needed to sort
    * them before performing the product.
    */
    SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_Ext> matrixQ() const 
    { return SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_Ext>(*this); }

	void setPruningEpsilon(const RealScalar &_eps) {
		this->m_sqrtEps = std::sqrt(_eps);
	}

	void setBlockParams(const Index &_blockRows, const Index &_blockCols) {
		this->m_blockRows = _blockRows;
		this->m_blockCols = _blockCols;
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
      eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext::solve() : invalid number of rows in the right hand side matrix");

      Index rank = this->rank();
      
      // Compute Q^T * b;
      typename Dest::PlainObject y, b;
      y = this->matrixQ().transpose() * B; 
      b = y;
      
      // Solve with the triangular matrix R
      y.resize((std::max<Index>)(cols(),y.rows()),y.cols());
      y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
      y.bottomRows(y.rows()-rank).setZero();
      
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
    inline const Solve<SparseBandedQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const 
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext::solve() : invalid number of rows in the right hand side matrix");
      return Solve<SparseBandedQR_Ext, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<SparseBandedQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
          eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
          eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext::solve() : invalid number of rows in the right hand side matrix");
          return Solve<SparseBandedQR_Ext, Rhs>(*this, B.derived());
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


    /** \internal */
    inline void _sort_matrix_Q()
    {
      if(this->m_isQSorted) return;
      // The matrix Q is sorted during the transposition
      SparseMatrix<Scalar, RowMajor, Index> mQrm(this->m_Q);
      this->m_Q = mQrm;
      this->m_isQSorted = true;
    }

    
  protected:
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixQStorageType;

    bool m_analysisIsok;
    bool m_factorizationIsok;
    mutable ComputationInfo m_info;
    std::string m_lastError;
	MatrixQStorageType m_pmat;            // Temporary matrix
    MatrixRType m_R;               // The triangular factor matrix
    MatrixQStorageType m_Q;               // The orthogonal reflectors
    ScalarVector m_hcoeffs;         // The Householder coefficients
	PermutationType m_outputPerm_c; // The final column permutation (for compatibility here, set to identity)
    RealScalar m_threshold;         // Threshold to determine null Householder reflections
    bool m_useDefaultThreshold;     // Use default threshold
    Index m_nonzeropivots;          // Number of non zero pivots found
    bool m_isQSorted;               // whether Q is sorted or not
	RealScalar m_sqrtEps;

	Index m_blockRows;
	Index m_blockCols;

template <typename, typename > friend struct SparseBandedQR_Ext_QProduct;

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
template <typename MatrixType, typename OrderingType>
void SparseBandedQR_Ext<MatrixType, OrderingType>::analyzePattern(const MatrixType& mat)
{
	Index n = mat.cols();
	Index m = mat.rows();
	Index diagSize = (std::min)(m, n);

	m_Q.resize(mat.rows(), mat.cols());
	m_R.resize(mat.rows(), mat.cols());

	m_hcoeffs.resize(diagSize);

	m_analysisIsok = true;
}

/** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseBandedQR_Ext::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
template <typename MatrixType, typename OrderingType>
void SparseBandedQR_Ext<MatrixType, OrderingType>::factorize(const MatrixType& mat)
{
	// Not rank-revealing, column permutation is identity
	m_outputPerm_c.setIdentity(mat.cols());

	m_pmat = mat;

	typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

	Eigen::TripletArray<Scalar, typename MatrixType::Index> Qvals(2 * mat.nonZeros());
	Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

	Eigen::HouseholderQR<DenseMatrixType> houseqr;
	Index blockRows = m_blockRows;
	Index blockCols = m_blockCols;
	Index numBlocks = mat.cols() / blockCols;
	DenseMatrixType Ji = mat.block(0, 0, blockRows, blockCols);
	DenseMatrixType Ji2;
	DenseMatrixType tmp;

	for (Index i = 0; i < numBlocks; i++) {
		// Where does the current block start
		Index bs = i * blockCols;

		// Solve dense block using Householder QR
		houseqr.compute(Ji);

		// Update Q and Householder coefficients
		for (int bc = 0; bc < blockCols; bc++) {
			Qvals.add(bs + bc, bs + bc, 1.0);
			for (int r = 0; r < houseqr.householderQ().essentialVector(bc).rows(); r++) {
				Qvals.add(bs + r + 1 + bc, bs + bc, houseqr.householderQ().essentialVector(bc)(r));
			}
			m_hcoeffs(bs + bc) = houseqr.hCoeffs()(bc);
		}

		// Update R
		if (i == numBlocks - 1) {
			tmp = houseqr.householderQ().transpose() * Ji;// spJ.block(bs, bs, blockRows, blockCols).toDense();
			for (int br = 0; br < blockCols; br++) {
				for (int bc = 0; bc < blockCols; bc++) {
					Rvals.add(bs + br, bs + bc, tmp(br, bc));
				}
			}
		}
		else {
			Ji2 = DenseMatrixType::Zero(blockRows, blockCols * 2);
			Ji2 << Ji, mat.block(bs, bs + blockCols, blockRows, blockCols).toDense();
			//std::cout << Ji2 << "\n----\n";
			tmp = houseqr.householderQ().transpose() * Ji2;// spJ.block(bs, bs, blockRows, blockCols * 2).toDense();
			for (int br = 0; br < blockCols; br++) {
				for (int bc = 0; bc < blockCols * 2; bc++) {
					Rvals.add(bs + br, bs + bc, tmp(br, bc));
				}
			}

			// Update block rows
			blockRows += blockCols;

			// Update Ji
			Ji = mat.block(bs + blockCols, bs + blockCols, blockRows, blockCols).toDense();
			Ji.block(0, 0, blockRows - blockCols * 2, blockCols) = tmp.block(blockCols, blockCols, blockRows - blockCols * 2, blockCols);
		}

	}
  
  // Finalize the column pointers of the sparse matrices R and Q
  m_Q.setFromTriplets(Qvals.begin(), Qvals.end());
  m_Q.makeCompressed();
  m_Q.prune(Scalar(m_sqrtEps), m_sqrtEps);	// Prune the matrix to avoid numerical issues (impacts performance in sparse cases)
  m_R.setFromTriplets(Rvals.begin(), Rvals.end());// , [](const Scalar&, const Scalar &b) { return b; });
  m_R.makeCompressed();
  m_R.prune(Scalar(m_sqrtEps), m_sqrtEps); // Prune the matrix to avoid numerical issues (impacts performance in sparse cases)
  m_isQSorted = false;
  
  Logger::instance()->logMatrixCSV(m_Q.toDense(), "m_Q.csv");

  m_nonzeropivots = m_R.cols();	// Assuming all cols are nonzero

  m_isInitialized = true; 
  m_factorizationIsok = true;
  m_info = Success;
}

// xxawf boilerplate all this into BlockSparseBandedQR_Ext...
template <typename SparseBandedQR_ExtType, typename Derived>
struct SparseBandedQR_Ext_QProduct : ReturnByValue<SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, Derived> >
{
  typedef typename SparseBandedQR_ExtType::MatrixType MatrixType;
  typedef typename SparseBandedQR_ExtType::Scalar Scalar;

  typedef typename internal::SparseBandedQR_Ext_traits<MatrixType>::Vector SparseVector;

  // Get the references 
  SparseBandedQR_Ext_QProduct(const SparseBandedQR_ExtType& qr, const Derived& other, bool transpose) : 
  m_qr(qr),m_other(other),m_transpose(transpose) {}
  inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
  inline Index cols() const { return m_other.cols(); }
  
  // Assign to a vector
  template<typename DesType>
  void evalTo(DesType& res) const
  {
    Index m = m_qr.rows();
    Index n = m_qr.cols();
    Index diagSize = (std::min)(m,n);
	res = m_other;

	//clock_t begin = clock();

	// FixMe: Better estimation of nonzeros?
	Eigen::TripletArray<Scalar, typename MatrixType::Index> resVals(Index(res.rows() * res.cols() * 0.1));

	SparseVector resColJ;
	const Scalar Zero = Scalar(0);
	Scalar tau = Scalar(0);
    if (m_transpose)
    {
		eigen_assert(m_qr.m_Q.rows() == m_other.rows() && "Non conforming object sizes");
		//Compute res = Q' * other column by column
		for (Index j = 0; j < res.cols(); j++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJ = res.col(j).pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps);
			for (Index k = 0; k < diagSize; k++) {
				// Need to instantiate this to tmp to avoid various runtime fails (error in insertInnerOuter, mysterious collapses to zero)
				tau = m_qr.m_Q.col(k).dot(resColJ.pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps));
				if (tau == Zero)
					continue;
				tau = tau * m_qr.m_hcoeffs(k);
				resColJ -= tau *  m_qr.m_Q.col(k);
			}
			// Write the result back to j-th column of res
			//res.col(j) = resColJ.pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps);
			for (SparseVector::InnerIterator it(resColJ); it; ++it) {
				resVals.add(it.row(), j, it.value());
			}
		}
    }
    else
    {
		eigen_assert(m_qr.m_Q.rows() == m_other.rows() && "Non conforming object sizes");
		// Compute res = Q * other column by column
		for (Index j = 0; j < res.cols(); j++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJ = res.col(j).pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps);
			for (Index k = diagSize - 1; k >= 0; k--) {
				tau = m_qr.m_Q.col(k).dot(resColJ.pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps));
				if (tau == Zero) 
					continue;
				tau = tau * m_qr.m_hcoeffs(k);
				resColJ -= tau *  m_qr.m_Q.col(k);
			}
			// Write the result back to j-th column of res
			//res.col(j) = resColJ.pruned(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps);
			for (SparseVector::InnerIterator it(resColJ); it; ++it) {
				resVals.add(it.row(), j, it.value());
			}
		}
    }

	res.setFromTriplets(resVals.begin(), resVals.end());
	res.prune(Scalar(m_qr.m_sqrtEps), m_qr.m_sqrtEps);
	res.makeCompressed();

	//std::cout << "Elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
  }
  const SparseBandedQR_ExtType& m_qr;
  const Derived& m_other;
  bool m_transpose;
};

template<typename SparseBandedQR_ExtType>
struct SparseBandedQR_ExtMatrixQReturnType : public EigenBase<SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType> >
{  
  typedef typename SparseBandedQR_ExtType::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic
  };
  explicit SparseBandedQR_ExtMatrixQReturnType(const SparseBandedQR_ExtType& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType,Derived>(m_qr,other.derived(),false);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
  {
    return SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
  }
  SparseBandedQR_ExtMatrixQTransposeReturnType<SparseBandedQR_ExtType> adjoint() const
  {
    return SparseBandedQR_ExtMatrixQTransposeReturnType<SparseBandedQR_ExtType>(m_qr);
  }
  inline Index rows() const { return m_qr.rows(); }
  inline Index cols() const { return m_qr.rows(); }
  // To use for operations with the transpose of Q
  SparseBandedQR_ExtMatrixQTransposeReturnType<SparseBandedQR_ExtType> transpose() const
  {
    return SparseBandedQR_ExtMatrixQTransposeReturnType<SparseBandedQR_ExtType>(m_qr);
  }

  const SparseBandedQR_ExtType& m_qr;
};

template<typename SparseBandedQR_ExtType>
struct SparseBandedQR_ExtMatrixQTransposeReturnType
{
  explicit SparseBandedQR_ExtMatrixQTransposeReturnType(const SparseBandedQR_ExtType& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, Derived>(m_qr, other.derived(), true);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar,_Options,_Index>& other)
  {
    return SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
  }
  const SparseBandedQR_ExtType& m_qr;
};

namespace internal {
  
template<typename SparseBandedQR_ExtType>
struct evaluator_traits<SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType> >
{
  typedef typename SparseBandedQR_ExtType::MatrixType MatrixType;
  typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
  typedef SparseShape Shape;
};

template< typename DstXprType, typename SparseBandedQR_ExtType>
struct Assignment<DstXprType, SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Sparse>
{
  typedef SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType> SrcXprType;
  typedef typename DstXprType::Scalar Scalar;
  typedef typename DstXprType::StorageIndex StorageIndex;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar,Scalar> &/*func*/)
  {
    typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
    idMat.setIdentity();
    // Sort the sparse householder reflectors if needed
    const_cast<SparseBandedQR_ExtType *>(&src.m_qr)->_sort_matrix_Q();
    dst = SparseBandedQR_Ext_QProduct<SparseBandedQR_ExtType, DstXprType>(src.m_qr, idMat, false);
  }
};

template< typename DstXprType, typename SparseBandedQR_ExtType>
struct Assignment<DstXprType, SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Dense>
{
  typedef SparseBandedQR_ExtMatrixQReturnType<SparseBandedQR_ExtType> SrcXprType;
  typedef typename DstXprType::Scalar Scalar;
  typedef typename DstXprType::StorageIndex StorageIndex;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar,Scalar> &/*func*/)
  {
    dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
  }
};

} // end namespace internal

} // end namespace Eigen

#endif
