// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_BANDED_QR_EXT2_H
#define EIGEN_SPARSE_BANDED_QR_EXT2_H

#include <ctime>

namespace Eigen {

	template<typename MatrixType, typename OrderingType> class SparseBandedQR_Ext2;
	template<typename SparseBandedQR_Ext2Type> struct SparseBandedQR_Ext2MatrixQReturnType;
	template<typename SparseBandedQR_Ext2Type> struct SparseBandedQR_Ext2MatrixQTransposeReturnType;
	template<typename SparseBandedQR_Ext2Type, typename Derived> struct SparseBandedQR_Ext2_QProduct;
	namespace internal {

		// traits<SparseBandedQR_Ext2MatrixQ[Transpose]>
		template <typename SparseBandedQR_Ext2Type> struct traits<SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type> >
		{
			typedef typename SparseBandedQR_Ext2Type::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename SparseBandedQR_Ext2Type> struct traits<SparseBandedQR_Ext2MatrixQTransposeReturnType<SparseBandedQR_Ext2Type> >
		{
			typedef typename SparseBandedQR_Ext2Type::MatrixType ReturnType;
		};

		template <typename SparseBandedQR_Ext2Type, typename Derived> struct traits<SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// SparseBandedQR_Ext2_traits
		template <typename T> struct SparseBandedQR_Ext2_traits {  };
		template <class T, int Rows, int Cols, int Options> struct SparseBandedQR_Ext2_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct SparseBandedQR_Ext2_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	/**
	  * \ingroup SparseBandedQR_Ext2_Module
	  * \class SparseBandedQR_Ext2
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
	class SparseBandedQR_Ext2 : public SparseSolverBase<SparseBandedQR_Ext2<_MatrixType, _OrderingType> >
	{
	protected:
		typedef SparseSolverBase<SparseBandedQR_Ext2<_MatrixType, _OrderingType> > Base;
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

		typedef SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

		enum {
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

	public:
		SparseBandedQR_Ext2() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_isQSorted(false), m_eps(1e-16), m_blockRows(4), m_blockCols(2)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		  *
		  * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		  *
		  * \sa compute()
		  */
		explicit SparseBandedQR_Ext2(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_isQSorted(false), m_eps(1e-16), m_blockRows(4), m_blockCols(2)
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
		static void householder_product_transposed(const MatrixXd &H, const VectorXd &hCoeffs, MatrixXd &V);

		/** \returns the number of rows of the represented matrix.
		  */
		inline Index rows() const { return m_pmat.rows(); }

		/** \returns the number of columns of the represented matrix.
		  */
		inline Index cols() const { return m_pmat.cols(); }

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
		* Q = SparseBandedQR_Ext2<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2> matrixQ() const
		{
			return SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2>(*this);
		}

		// Return the matrix of householder vectors
		const MatrixRType& matrixH() const {
			return this->m_Q;
		}

		void setRoundoffEpsilon(const RealScalar &_eps) {
			this->m_eps = _eps;
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
			eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext2::solve() : invalid number of rows in the right hand side matrix");

			Index rank = this->rank();

			// Compute Q^T * b;
			typename Dest::PlainObject y, b;
			y = this->matrixQ().transpose() * B;
			b = y;

			// Solve with the triangular matrix R
			y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
			y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
			y.bottomRows(y.rows() - rank).setZero();

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
		inline const Solve<SparseBandedQR_Ext2, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext2::solve() : invalid number of rows in the right hand side matrix");
			return Solve<SparseBandedQR_Ext2, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<SparseBandedQR_Ext2, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseBandedQR_Ext2::solve() : invalid number of rows in the right hand side matrix");
			return Solve<SparseBandedQR_Ext2, Rhs>(*this, B.derived());
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
			if (this->m_isQSorted) return;
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
		RealScalar m_eps;

		Index m_blockRows;
		Index m_blockCols;

		template <typename, typename > friend struct SparseBandedQR_Ext2_QProduct;

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
	void SparseBandedQR_Ext2<MatrixType, OrderingType>::analyzePattern(const MatrixType& mat)
	{
		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		m_Q.resize(mat.rows(), mat.cols() * 2 - 2);
		m_R.resize(mat.rows(), mat.cols());

		m_hcoeffs.resize(mat.cols() * 2 - 2);

		m_analysisIsok = true;
	}

/*
 * Helper function performing product Qt * V, where Qt is represented as product of Householder vectors that stored as
 *	H - the Householder reflectors
 *  hCoeffs - the Householder coefficients
 * The operation is happening in-place => result is stored in V.
 * !!! This implementation is efficient only because the input matrices are assumed to be dense, thin and with reasonable amount of rows. !!!
 */
template <typename MatrixType, typename OrderingType>
void SparseBandedQR_Ext2<MatrixType, OrderingType>::householder_product_transposed(const MatrixXd &H, const VectorXd &hCoeffs, MatrixXd &V) {
	for (int j = 0; j < V.cols(); j++) {
		for (int k = 0; k < H.cols(); k++) {
			V.col(j) -= (H.col(k).dot(V.col(j)) * hCoeffs(k)) * H.col(k);
		}
	}
}

/** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseBandedQR_Ext2::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
template <typename MatrixType, typename OrderingType>
void SparseBandedQR_Ext2<MatrixType, OrderingType>::factorize(const MatrixType& mat)
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
	Index rowIncrement = m_blockRows - m_blockCols;
	Index numBlocks = mat.cols() / blockCols;
	DenseMatrixType Ji = mat.block(0, 0, blockRows, blockCols * 2);
	DenseMatrixType Ji2;
	DenseMatrixType tmp;

	// Number of maximum non-zero rows we want to keep, the implicit zeros will be filled in accordingly
	//#define NNZ_ROWS (m_blockRows * 12)
	//#define NNZ_ROWS (m_blockRows * 16)	
	//#define NNZ_ROWS (m_blockRows * 24)	
	#define NNZ_ROWS (m_blockRows * 2)	

	int numZeros = 0;				// Number of implicitly zero-ed rows
	int activeRows = blockRows;		// Number of non-zero rows for implicit zero-ing
	for (Index i = 0; i < numBlocks; i++) {
		// Where does the current block start
		Index bs = i * blockCols;
		Index bsh = i * (blockCols * 2);

		// Solve dense block using Householder QR
		houseqr.compute(Ji);

		// If it is the last block, it's just two columns
		int currBlockCols = (i == numBlocks - 1) ? blockCols : blockCols * 2;

		// Update Q and Householder coefficients
		MatrixXd H = MatrixXd::Zero(activeRows, currBlockCols);
		for(int bc = 0; bc < currBlockCols; bc++) {
			H(bc, bc) = 1.0;
			H.block(bc + 1, bc, houseqr.householderQ().essentialVector(bc).rows(), 1) = houseqr.householderQ().essentialVector(bc);

			m_hcoeffs(bsh + bc) = houseqr.hCoeffs()(bc);
		}
		for (int bc = 0; bc < currBlockCols; bc++) {
			Qvals.add_if_nonzero(bs + bc, bsh + bc, H(bc, bc));
			for (int r = bc + 1; r < activeRows; r++) {
				Qvals.add_if_nonzero(bs + r + numZeros, bsh + bc, H(r, bc));
			}
		}

		// Update R
		// Multiplication by the householder vectors
		MatrixXd V = Ji;
		householder_product_transposed(H, houseqr.hCoeffs(), V);

		tmp = V;

		for (int br = 0; br < blockCols; br++) {
			for (int bc = 0; bc < currBlockCols; bc++) {
				Rvals.add_if_nonzero(bs + br, bs + bc, tmp(br, bc));
			}
		}

		if (i < numBlocks - 1) {
			// Update block rows
			blockRows += rowIncrement;

			// How many rows to zero-out implicitly in this step
			if (blockRows > NNZ_ROWS) {
				numZeros = blockRows - NNZ_ROWS;
				activeRows = NNZ_ROWS;

				// Update Ji accordingly
				if (i < numBlocks - 2) {
					Ji = mat.block(bs + blockCols + numZeros, bs + blockCols, activeRows, blockCols * 2).toDense();
					Ji.block(0, 0, activeRows - rowIncrement - blockCols, blockCols) = tmp.block(blockCols, blockCols, activeRows - rowIncrement - blockCols, blockCols);
				} else {
					Ji = mat.block(bs + blockCols + numZeros, bs + blockCols, activeRows, blockCols).toDense();
					Ji.block(0, 0, activeRows - rowIncrement - blockCols, blockCols) = tmp.block(blockCols, blockCols, activeRows - rowIncrement - blockCols, blockCols);
				}
			} else {
				numZeros = 0;
				activeRows = blockRows;

				// Update Ji accordingly
				Ji = mat.block(bs + blockCols + numZeros, bs + blockCols, activeRows, blockCols * 2).toDense();
				Ji.block(0, 0, activeRows - rowIncrement - blockCols, blockCols) = tmp.block(blockCols, blockCols, activeRows - rowIncrement - blockCols, blockCols);
			}
		}
	}
  
  // Finalize the column pointers of the sparse matrices R and Q
  m_Q.setFromTriplets(Qvals.begin(), Qvals.end());
  m_Q.makeCompressed();
  m_R.setFromTriplets(Rvals.begin(), Rvals.end());
  m_R.makeCompressed();
  m_isQSorted = false;

  m_nonzeropivots = m_R.cols();	// Assuming all cols are nonzero

  m_isInitialized = true; 
  m_factorizationIsok = true;
  m_info = Success;
}

// xxawf boilerplate all this into BlockSparseBandedQR_Ext2...
template <typename SparseBandedQR_Ext2Type, typename Derived>
struct SparseBandedQR_Ext2_QProduct : ReturnByValue<SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, Derived> >
{
  typedef typename SparseBandedQR_Ext2Type::MatrixType MatrixType;
  typedef typename SparseBandedQR_Ext2Type::Scalar Scalar;

  typedef typename internal::SparseBandedQR_Ext2_traits<MatrixType>::Vector SparseVector;

  // Get the references 
  SparseBandedQR_Ext2_QProduct(const SparseBandedQR_Ext2Type& qr, const Derived& other, bool transpose) : 
  m_qr(qr),m_other(other),m_transpose(transpose) {}
  inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
  inline Index cols() const { return m_other.cols(); }
  
  // Assign to a vector
  template<typename DesType>
  void evalTo(DesType& res) const
  {
    Index m = m_qr.rows();
    Index n = m_qr.cols();
	Index diagSize = m_qr.m_Q.cols();//(std::min)(m,n);
	res = m_other;

	//clock_t begin = clock();

	// FixMe: Better estimation of nonzeros?
	Eigen::TripletArray<Scalar, typename MatrixType::Index> resVals(Index(res.rows() * res.cols() * 0.1));

#define IS_ZERO(x, eps) (std::abs(x) < eps)

	SparseVector resColJ;
	const Scalar Zero = Scalar(0);
	Scalar tau = Scalar(0);
    if (m_transpose)
    {
		eigen_assert(m_qr.m_Q.rows() == m_other.rows() && "Non conforming object sizes");
		//Compute res = Q' * other column by column
		for (Index j = 0; j < res.cols(); j++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJ = res.col(j);
			for (Index k = 0; k < diagSize; k++) {
				// Need to instantiate this to tmp to avoid various runtime fails (error in insertInnerOuter, mysterious collapses to zero)
				tau = m_qr.m_Q.col(k).dot(resColJ);
				//if(tau == Zero)
				if (IS_ZERO(tau, m_qr.m_eps))
					continue;
				tau = tau * m_qr.m_hcoeffs(k);
				resColJ -= tau *  m_qr.m_Q.col(k);
			}
			// Write the result back to j-th column of res
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
			resColJ = res.col(j);
			for (Index k = diagSize - 1; k >= 0; k--) {
				tau = m_qr.m_Q.col(k).dot(resColJ);
				//if (tau == Zero)
				if (IS_ZERO(tau, m_qr.m_eps))
					continue;
				tau = tau * m_qr.m_hcoeffs(k);
				resColJ -= tau *  m_qr.m_Q.col(k);
			}
			// Write the result back to j-th column of res
			for (SparseVector::InnerIterator it(resColJ); it; ++it) {
				resVals.add(it.row(), j, it.value());
			}
		}
    }

	res.setFromTriplets(resVals.begin(), resVals.end());
	res.makeCompressed();

	//std::cout << "Elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
  }
  const SparseBandedQR_Ext2Type& m_qr;
  const Derived& m_other;
  bool m_transpose;
};

template<typename SparseBandedQR_Ext2Type>
struct SparseBandedQR_Ext2MatrixQReturnType : public EigenBase<SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type> >
{  
  typedef typename SparseBandedQR_Ext2Type::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic
  };
  explicit SparseBandedQR_Ext2MatrixQReturnType(const SparseBandedQR_Ext2Type& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type,Derived>(m_qr,other.derived(),false);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
  {
    return SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
  }
  SparseBandedQR_Ext2MatrixQTransposeReturnType<SparseBandedQR_Ext2Type> adjoint() const
  {
    return SparseBandedQR_Ext2MatrixQTransposeReturnType<SparseBandedQR_Ext2Type>(m_qr);
  }
  inline Index rows() const { return m_qr.rows(); }
  inline Index cols() const { return m_qr.rows(); }
  // To use for operations with the transpose of Q
  SparseBandedQR_Ext2MatrixQTransposeReturnType<SparseBandedQR_Ext2Type> transpose() const
  {
    return SparseBandedQR_Ext2MatrixQTransposeReturnType<SparseBandedQR_Ext2Type>(m_qr);
  }

  const SparseBandedQR_Ext2Type& m_qr;
};

template<typename SparseBandedQR_Ext2Type>
struct SparseBandedQR_Ext2MatrixQTransposeReturnType
{
  explicit SparseBandedQR_Ext2MatrixQTransposeReturnType(const SparseBandedQR_Ext2Type& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, Derived>(m_qr, other.derived(), true);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar,_Options,_Index>& other)
  {
    return SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
  }
  const SparseBandedQR_Ext2Type& m_qr;
};

namespace internal {
  
template<typename SparseBandedQR_Ext2Type>
struct evaluator_traits<SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type> >
{
  typedef typename SparseBandedQR_Ext2Type::MatrixType MatrixType;
  typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
  typedef SparseShape Shape;
};

template< typename DstXprType, typename SparseBandedQR_Ext2Type>
struct Assignment<DstXprType, SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Sparse>
{
  typedef SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type> SrcXprType;
  typedef typename DstXprType::Scalar Scalar;
  typedef typename DstXprType::StorageIndex StorageIndex;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar,Scalar> &/*func*/)
  {
    typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
    idMat.setIdentity();
    // Sort the sparse householder reflectors if needed
    const_cast<SparseBandedQR_Ext2Type *>(&src.m_qr)->_sort_matrix_Q();
    dst = SparseBandedQR_Ext2_QProduct<SparseBandedQR_Ext2Type, DstXprType>(src.m_qr, idMat, false);
  }
};

template< typename DstXprType, typename SparseBandedQR_Ext2Type>
struct Assignment<DstXprType, SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Dense>
{
  typedef SparseBandedQR_Ext2MatrixQReturnType<SparseBandedQR_Ext2Type> SrcXprType;
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
