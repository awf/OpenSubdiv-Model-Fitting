// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_BANDED_BLOCKED_QR_EXT3_H
#define EIGEN_SPARSE_BANDED_BLOCKED_QR_EXT3_H

#include <ctime>

#define IS_ZERO(x, eps) (std::abs(x) < eps)

namespace Eigen {

	template<typename MatrixType, typename OrderingType> class SparseBandedBlockedQR_Ext3;
	template<typename SparseBandedBlockedQR_Ext3Type> struct SparseBandedBlockedQR_Ext3MatrixQReturnType;
	template<typename SparseBandedBlockedQR_Ext3Type> struct SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType;
	template<typename SparseBandedBlockedQR_Ext3Type, typename Derived> struct SparseBandedBlockedQR_Ext3_QProduct;
	namespace internal {

		// traits<SparseBandedBlockedQR_Ext3MatrixQ[Transpose]>
		template <typename SparseBandedBlockedQR_Ext3Type> struct traits<SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type> >
		{
			typedef typename SparseBandedBlockedQR_Ext3Type::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename SparseBandedBlockedQR_Ext3Type> struct traits<SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType<SparseBandedBlockedQR_Ext3Type> >
		{
			typedef typename SparseBandedBlockedQR_Ext3Type::MatrixType ReturnType;
		};

		template <typename SparseBandedBlockedQR_Ext3Type, typename Derived> struct traits<SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};

		// SparseBandedBlockedQR_Ext3_traits
		template <typename T> struct SparseBandedBlockedQR_Ext3_traits {  };
		template <class T, int Rows, int Cols, int Options> struct SparseBandedBlockedQR_Ext3_traits<Matrix<T, Rows, Cols, Options>> {
			typedef Matrix<T, Rows, 1, Options> Vector;
		};
		template <class Scalar, int Options, typename Index> struct SparseBandedBlockedQR_Ext3_traits<SparseMatrix<Scalar, Options, Index>> {
			typedef SparseVector<Scalar, Options> Vector;
		};
	} // End namespace internal

	/**
	  * \ingroup SparseBandedBlockedQR_Ext3_Module
	  * \class SparseBandedBlockedQR_Ext3
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
	class SparseBandedBlockedQR_Ext3 : public SparseSolverBase<SparseBandedBlockedQR_Ext3<_MatrixType, _OrderingType> >
	{
	protected:
		typedef SparseSolverBase<SparseBandedBlockedQR_Ext3<_MatrixType, _OrderingType> > Base;
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

		typedef SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3> MatrixQType;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

		enum {
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

	public:
		SparseBandedBlockedQR_Ext3() : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_isHSorted(false), m_eps(1e-16), m_blockRows(4), m_blockCols(2), m_blockOverlap(2)
		{ }
			
		SparseBandedBlockedQR_Ext3(const Index &_blockRows, const Index &_blockCols, const Index &_blockOverlap) 
			: m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_isHSorted(false), m_eps(1e-16), m_blockRows(_blockRows), m_blockCols(_blockCols), m_blockOverlap(_blockOverlap)
		{ }
		/** Construct a QR factorization of the matrix \a mat.
		  *
		  * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		  *
		  * \sa compute()
		  */
		explicit SparseBandedBlockedQR_Ext3(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_useDefaultThreshold(true), m_isHSorted(false), m_eps(1e-16), m_blockRows(4), m_blockCols(2), m_blockOverlap(2)
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
		static void yty_product_transposed(const MatrixXd &T, const MatrixXd &Y, MatrixXd &V);

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
		* Q = SparseBandedBlockedQR_Ext3<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
		*/
		SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3> matrixQ() const
		{
			return SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3>(*this);
		}

		// Return the matrices of the WY Householder representation
		const MatrixRType& matrixY() const {
			return this->m_Y;
		}
		const MatrixRType& matrixT() const {
			return this->m_T;
		}

		void setRoundoffEpsilon(const RealScalar &_eps) {
			this->m_eps = _eps;
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
			eigen_assert(this->rows() == B.rows() && "SparseBandedBlockedQR_Ext3::solve() : invalid number of rows in the right hand side matrix");

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
		inline const Solve<SparseBandedBlockedQR_Ext3, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseBandedBlockedQR_Ext3::solve() : invalid number of rows in the right hand side matrix");
			return Solve<SparseBandedBlockedQR_Ext3, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<SparseBandedBlockedQR_Ext3, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseBandedBlockedQR_Ext3::solve() : invalid number of rows in the right hand side matrix");
			return Solve<SparseBandedBlockedQR_Ext3, Rhs>(*this, B.derived());
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

	protected:
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixQStorageType;

		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;
		MatrixQStorageType m_pmat;            // Temporary matrix
		MatrixRType m_R;                // The triangular factor matrix
		std::vector<MatrixXd> m_denseT;	// Vector of blocks T of YTY' Householder blocks (compact WY' representation)
		std::vector<MatrixXd> m_denseY; // Vector of blocks Y of YTY' Householder blocks (compact WY' representation)
		std::vector<Vector4i> m_idxsY;  // Remembering block alignment necessary during the Householder product evaluation phase
		PermutationType m_outputPerm_c; // The final column permutation (for compatibility here, set to identity)
		RealScalar m_threshold;         // Threshold to determine null Householder reflections
		bool m_useDefaultThreshold;     // Use default threshold
		Index m_nonzeropivots;          // Number of non zero pivots found
		bool m_isHSorted;               // whether Q is sorted or not
		RealScalar m_eps;

		const Index m_blockRows;
		const Index m_blockCols;
		const Index m_blockOverlap;

		template <typename, typename > friend struct SparseBandedBlockedQR_Ext3_QProduct;

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
	void SparseBandedBlockedQR_Ext3<MatrixType, OrderingType>::analyzePattern(const MatrixType& mat)
	{
		Index n = mat.cols();
		Index m = mat.rows();
		Index diagSize = (std::min)(m, n);

		Index colIncrement = m_blockCols - m_blockOverlap;
		Index numBlocks = std::ceil(double(mat.cols()) / colIncrement);// mat.cols() / colIncrement;

		m_denseY.resize(numBlocks);
		m_denseT.resize(numBlocks);
		m_idxsY.resize(numBlocks);

		m_R.resize(mat.rows(), mat.cols());

		m_analysisIsok = true;
	}

/*
 * Helper function performing product Qt * V, where Qt is represented as product of Householder vectors that stored as
 *	T, Y - the block YTY' representation of the Householder reflectors
 * The operation is happening in-place => result is stored in V.
 * !!! This implementation is efficient only because the input matrices are assumed to be dense, thin and with reasonable amount of rows. !!!
 */
template <typename MatrixType, typename OrderingType>
void SparseBandedBlockedQR_Ext3<MatrixType, OrderingType>::yty_product_transposed(const MatrixXd &T, const MatrixXd &Y, MatrixXd &V) {
	for (int j = 0; j < V.cols(); j++) {
		V.col(j) += (Y * (T.transpose() * (Y.transpose() * V.col(j))));
		//V.col(j) = (MatrixXd::Identity(Y.rows(), Y.rows()) - Y * T * Y.transpose()).transpose() * V.col(j);

		//V.col(j) -= (Y * (W.transpose() * V.col(j)));
		//V.col(j) = (MatrixXd::Identity(W.rows(), W.rows()) - W * Y.transpose()).transpose() * V.col(j);
	}
}
/** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseBandedBlockedQR_Ext3::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
template <typename MatrixType, typename OrderingType>
void SparseBandedBlockedQR_Ext3<MatrixType, OrderingType>::factorize(const MatrixType& mat)
{
	// Not rank-revealing, column permutation is identity
	m_outputPerm_c.setIdentity(mat.cols());

	m_pmat = mat;

	typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

	Eigen::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

	Eigen::HouseholderQR<DenseMatrixType> houseqr;
	Index blockRows = m_blockRows;
	Index blockCols = m_blockCols;
	Index colIncrement = m_blockCols - m_blockOverlap;
	Index rowIncrement = m_blockRows - colIncrement;
	Index numBlocks = std::ceil(double(mat.cols()) / colIncrement);
	Index lastBlockCols = mat.cols() - (numBlocks - 1) * colIncrement;
	DenseMatrixType Ji = mat.block(0, 0, blockRows, blockCols);
	DenseMatrixType tmp;

	// Number of maximum non-zero rows we want to keep, the implicit zeros will be filled in accordingly
	#define NNZ_ROWS (m_blockRows * 2)	

	int numZeros = 0;				// Number of implicitly zero-ed rows
	int activeRows = blockRows;		// Number of non-zero rows for implicit zero-ing
	int currBlockCols = blockCols;
	for (Index i = 0; i < numBlocks; i++) {
		// Where does the current block start
		Index bs = i * colIncrement;
		Index bsh = i * blockCols;

		// Solve dense block using Householder QR
		houseqr.compute(Ji);

		// Update matrices W and Y
		MatrixXd T = MatrixXd::Zero(currBlockCols, currBlockCols);
		MatrixXd Y = MatrixXd::Zero(activeRows, currBlockCols);
		VectorXd v = VectorXd::Zero(activeRows);
		VectorXd z = VectorXd::Zero(activeRows);

		v(0) = 1.0;
		v.segment(1, houseqr.householderQ().essentialVector(0).rows()) = houseqr.householderQ().essentialVector(0);
		Y.col(0) = v;
		T(0, 0) = -houseqr.hCoeffs()(0);
		for (int bc = 1; bc < currBlockCols; bc++) {
			v.setZero();
			v(bc) = 1.0;
			v.segment(bc + 1, houseqr.householderQ().essentialVector(bc).rows()) = houseqr.householderQ().essentialVector(bc);
		
			z = -houseqr.hCoeffs()(bc) * (T * (Y.transpose() * v));
			
			Y.col(bc) = v;
			T.col(bc) = z;
			T(bc, bc) = -houseqr.hCoeffs()(bc);
		}
		m_denseT.at(i) = T;
		m_denseY.at(i) = Y;
		m_idxsY.at(i) = Vector4i(bs, numZeros, activeRows, currBlockCols);
	
		//	std::cout << "--- T ---\n" << T << std::endl;
		//	std::cout << "--- Y ---\n" << Y << std::endl;

		// Update R
		MatrixXd V = houseqr.matrixQR().template triangularView<Upper>();

		tmp = V;

		int solvedRows =  (i == numBlocks - 1) ? lastBlockCols : colIncrement;
		for (int br = 0; br < solvedRows; br++) {
			for (int bc = 0; bc < currBlockCols; bc++) {
				Rvals.add_if_nonzero(bs + br, bs + bc, tmp(br, bc));
			}
		}

		if (i < numBlocks - 1) {
			// Update block rows
			int newRows = ((i == numBlocks - 2) ? (mat.rows() - bs - blockRows - colIncrement) : rowIncrement);
			blockRows += newRows;

			// How many rows to zero-out implicitly in this step
			if (blockRows > NNZ_ROWS) {
				numZeros = blockRows - NNZ_ROWS;
				activeRows = NNZ_ROWS;

				// Update Ji accordingly
				if (i < numBlocks - 2) {
					Ji = mat.block(bs + colIncrement + numZeros, bs + colIncrement, activeRows, blockCols).toDense();
					Ji.block(0, 0, activeRows - newRows - colIncrement, m_blockOverlap) = tmp.block(colIncrement, colIncrement, activeRows - newRows - colIncrement, m_blockOverlap);
				} else {
					Ji = mat.block(mat.rows() - activeRows, mat.cols() - lastBlockCols, activeRows, lastBlockCols).toDense();
					Ji.block(0, 0, activeRows - newRows - colIncrement, m_blockOverlap) = tmp.block(colIncrement, colIncrement, activeRows - newRows - colIncrement, m_blockOverlap);
					currBlockCols = lastBlockCols;
				}
			} else {
				numZeros = 0;
				activeRows = blockRows;

				// Update Ji accordingly
				Ji = mat.block(bs + colIncrement + numZeros, bs + colIncrement, activeRows, blockCols).toDense();

				//std::cout << "--- Ji ---\n" << mat.block(bs + colIncrement + numZeros, bs + colIncrement, activeRows, blockCols).toDense() << std::endl;
				//std::cout << "--- tmp ---\n" << tmp << std::endl;
				//std::cout << "--- tmp block ---\n" << tmp.block(colIncrement, colIncrement, activeRows - rowIncrement - colIncrement, m_blockOverlap) << std::endl;

				Ji.block(0, 0, activeRows - rowIncrement - colIncrement, m_blockOverlap) = tmp.block(colIncrement, colIncrement, activeRows - rowIncrement - colIncrement, m_blockOverlap);
			}
		}
	}
  
  // Finalize the column pointers of the sparse matrices R, W and Y
  m_R.setFromTriplets(Rvals.begin(), Rvals.end());
  m_R.makeCompressed();
  m_isHSorted = false;

  m_nonzeropivots = m_R.cols();	// Assuming all cols are nonzero

  m_isInitialized = true; 
  m_factorizationIsok = true;
  m_info = Success;
}

//#define MULTITHREADED 1

// xxawf boilerplate all this into BlockSparseBandedBlockedQR_Ext3...
template <typename SparseBandedBlockedQR_Ext3Type, typename Derived>
struct SparseBandedBlockedQR_Ext3_QProduct : ReturnByValue<SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, Derived> >
{
  typedef typename SparseBandedBlockedQR_Ext3Type::MatrixType MatrixType;
  typedef typename SparseBandedBlockedQR_Ext3Type::Scalar Scalar;

  typedef typename internal::SparseBandedBlockedQR_Ext3_traits<MatrixType>::Vector SparseVector;

  // Get the references 
  SparseBandedBlockedQR_Ext3_QProduct(const SparseBandedBlockedQR_Ext3Type& qr, const Derived& other, bool transpose) : 
  m_qr(qr),m_other(other),m_transpose(transpose) {}
  inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
  inline Index cols() const { return m_other.cols(); }

  // Assign to a vector
  template<typename DesType>
  void evalTo(DesType& res) const
  {
    Index m = m_qr.rows();
    Index n = m_qr.cols();
	res = m_other;

	//clock_t begin = clock();

	// FixMe: Better estimation of nonzeros?
	Eigen::TripletArray<Scalar, typename MatrixType::Index> resVals(Index(res.rows() * res.cols() * 0.1));

	SparseVector resColJ;
	VectorXd resColJd;
	if (m_transpose)
    {
		eigen_assert(m_qr.m_Y.rows() == m_other.rows() && "Non conforming object sizes");

#ifdef MULTITHREADED
		// Compute res = Q' * other column by column using parallel for loop
		const size_t nloop = res.cols();
		const size_t nthreads = std::thread::hardware_concurrency();
		{
			std::vector<std::thread> threads(nthreads);
			std::mutex critical;
			for (int t = 0; t<nthreads; t++)
			{
				threads[t] = std::thread(std::bind(
					[&](const int bi, const int ei, const int t)
				{
					// loop over all items
					for (int j = bi; j<ei; j++)
					{
						// inner loop
						{
							VectorXd tmpResColJ;
							SparseVector resColJ;
							VectorXd resColJd;
							resColJd = res.col(j).toDense();
							for (Index k = 0; k < m_qr.m_denseY.size(); k++) {
								tmpResColJ = VectorXd(m_qr.m_idxsY.at(k)(2));
								tmpResColJ.segment(0, m_qr.m_idxsY.at(k)(3)) = resColJd.segment(m_qr.m_idxsY.at(k)(0), m_qr.m_idxsY.at(k)(3));
								int remaining = m_qr.m_idxsY.at(k)(2) - m_qr.m_idxsY.at(k)(3);
								if (remaining > 0) {
									tmpResColJ.segment(m_qr.m_idxsY.at(k)(3), remaining) = resColJd.segment(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + m_qr.m_idxsY.at(k)(3), remaining);
								}

								tmpResColJ += (m_qr.m_denseY.at(k) * (m_qr.m_denseT.at(k).transpose() * (m_qr.m_denseY.at(k).transpose() * tmpResColJ)));

								for (int i = 0; i < m_qr.m_idxsY.at(k)(3); i++) {
									resColJd(m_qr.m_idxsY.at(k)(0) + i) = tmpResColJ(i);
								}
								for (int i = m_qr.m_idxsY.at(k)(3); i < tmpResColJ.size(); i++) {
									resColJd(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + i) = tmpResColJ(i);
								}
							}

							std::lock_guard<std::mutex> lock(critical);
							// Write the result back to j-th column of res
							resColJ = resColJd.sparseView();
							for (SparseVector::InnerIterator it(resColJ); it; ++it) {
								resVals.add(it.row(), j, it.value());
							}

						}
					}
				}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
			}
			std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
		}
#else
		//Compute res = Q' * other column by column
		VectorXd tmpResColJ;
		for (Index j = 0; j < res.cols(); j++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJd = res.col(j).toDense();
			for (Index k = 0; k < m_qr.m_denseY.size(); k++) {
				tmpResColJ = VectorXd(m_qr.m_idxsY.at(k)(2));
				tmpResColJ.segment(0, m_qr.m_idxsY.at(k)(3)) = resColJd.segment(m_qr.m_idxsY.at(k)(0), m_qr.m_idxsY.at(k)(3));
				int remaining = m_qr.m_idxsY.at(k)(2) - m_qr.m_idxsY.at(k)(3);
				if (remaining > 0) {
					tmpResColJ.segment(m_qr.m_idxsY.at(k)(3), remaining) = resColJd.segment(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + m_qr.m_idxsY.at(k)(3), remaining);
				}

				tmpResColJ += (m_qr.m_denseY.at(k) * (m_qr.m_denseT.at(k).transpose() * (m_qr.m_denseY.at(k).transpose() * tmpResColJ)));

				for (int i = 0; i < m_qr.m_idxsY.at(k)(3); i++) {
					resColJd(m_qr.m_idxsY.at(k)(0) + i) = tmpResColJ(i);
				}
				for (int i = m_qr.m_idxsY.at(k)(3); i < tmpResColJ.size(); i++) {
					resColJd(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + i) = tmpResColJ(i);
				}
			}

			// Write the result back to j-th column of res
			resColJ = resColJd.sparseView();
			for (SparseVector::InnerIterator it(resColJ); it; ++it) {
				resVals.add(it.row(), j, it.value());
			}
		}
#endif
    }
    else
    {
		eigen_assert(m_qr.m_Y.rows() == m_other.rows() && "Non conforming object sizes");

		// Compute res = Q * other column by column using parallel for loop
#ifdef MULTITHREADED
		const size_t nloop = res.cols();
		const size_t nthreads = std::thread::hardware_concurrency();
		{
			std::vector<std::thread> threads(nthreads);
			std::mutex critical;
			for (int t = 0; t<nthreads; t++)
			{
				threads[t] = std::thread(std::bind(
					[&](const int bi, const int ei, const int t)
				{
					// loop over all items
					for (int j = bi; j<ei; j++)
					{
						// inner loop
						{
							VectorXd tmpResColJ;
							SparseVector resColJ;
							VectorXd resColJd;
							resColJd = res.col(j).toDense();
							for (Index k = m_qr.m_denseY.size() - 1; k >= 0; k--) {
								tmpResColJ = VectorXd(m_qr.m_idxsY.at(k)(2));
								tmpResColJ.segment(0, m_qr.m_idxsY.at(k)(3)) = resColJd.segment(m_qr.m_idxsY.at(k)(0), m_qr.m_idxsY.at(k)(3));
								int remaining = m_qr.m_idxsY.at(k)(2) - m_qr.m_idxsY.at(k)(3);
								if (remaining > 0) {
									tmpResColJ.segment(m_qr.m_idxsY.at(k)(3), remaining) = resColJd.segment(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + m_qr.m_idxsY.at(k)(3), remaining);
								}

								tmpResColJ += (m_qr.m_denseY.at(k) * (m_qr.m_denseT.at(k) * (m_qr.m_denseY.at(k).transpose() * tmpResColJ)));

								for (int i = 0; i < m_qr.m_idxsY.at(k)(3); i++) {
									resColJd(m_qr.m_idxsY.at(k)(0) + i) = tmpResColJ(i);
								}
								for (int i = m_qr.m_idxsY.at(k)(3); i < tmpResColJ.size(); i++) {
									resColJd(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + i) = tmpResColJ(i);
								}
							}

							std::lock_guard<std::mutex> lock(critical);
							// Write the result back to j-th column of res
							resColJ = resColJd.sparseView();
							for (SparseVector::InnerIterator it(resColJ); it; ++it) {
								resVals.add(it.row(), j, it.value());
							}

						}
					}
				}, t*nloop / nthreads, (t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads, t));
			}
			std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
		}
#else
		// Compute res = Q * other column by column
		VectorXd tmpResColJ;
		for (Index j = 0; j < res.cols(); j++) {
			// Use temporary vector resColJ inside of the for loop - faster access
			resColJd = res.col(j).toDense();
			for (Index k = m_qr.m_denseY.size() - 1; k >= 0; k--) {
				tmpResColJ = VectorXd(m_qr.m_idxsY.at(k)(2));
				tmpResColJ.segment(0, m_qr.m_idxsY.at(k)(3)) = resColJd.segment(m_qr.m_idxsY.at(k)(0), m_qr.m_idxsY.at(k)(3));
				int remaining = m_qr.m_idxsY.at(k)(2) - m_qr.m_idxsY.at(k)(3);
				if (remaining > 0) {
					tmpResColJ.segment(m_qr.m_idxsY.at(k)(3), remaining) = resColJd.segment(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + m_qr.m_idxsY.at(k)(3), remaining);
				}

				tmpResColJ += (m_qr.m_denseY.at(k) * (m_qr.m_denseT.at(k) * (m_qr.m_denseY.at(k).transpose() * tmpResColJ)));

				for (int i = 0; i < m_qr.m_idxsY.at(k)(3); i++) {
					resColJd(m_qr.m_idxsY.at(k)(0) + i) = tmpResColJ(i);
				}
				for (int i = m_qr.m_idxsY.at(k)(3); i < tmpResColJ.size(); i++) {
					resColJd(m_qr.m_idxsY.at(k)(0) + m_qr.m_idxsY.at(k)(1) + i) = tmpResColJ(i);
				}
			}

			// Write the result back to j-th column of res
			resColJ = resColJd.sparseView();
			for (SparseVector::InnerIterator it(resColJ); it; ++it) {
				resVals.add(it.row(), j, it.value());
			}
		}
#endif
    }

	res.setFromTriplets(resVals.begin(), resVals.end());
	res.makeCompressed();

	//std::cout << "Elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
  }
  const SparseBandedBlockedQR_Ext3Type& m_qr;
  const Derived& m_other;
  bool m_transpose;
};

template<typename SparseBandedBlockedQR_Ext3Type>
struct SparseBandedBlockedQR_Ext3MatrixQReturnType : public EigenBase<SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type> >
{  
  typedef typename SparseBandedBlockedQR_Ext3Type::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic
  };
  explicit SparseBandedBlockedQR_Ext3MatrixQReturnType(const SparseBandedBlockedQR_Ext3Type& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type,Derived>(m_qr,other.derived(),false);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
  {
    return SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
  }
  SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType<SparseBandedBlockedQR_Ext3Type> adjoint() const
  {
    return SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType<SparseBandedBlockedQR_Ext3Type>(m_qr);
  }
  inline Index rows() const { return m_qr.rows(); }
  inline Index cols() const { return m_qr.rows(); }
  // To use for operations with the transpose of Q
  SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType<SparseBandedBlockedQR_Ext3Type> transpose() const
  {
    return SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType<SparseBandedBlockedQR_Ext3Type>(m_qr);
  }

  const SparseBandedBlockedQR_Ext3Type& m_qr;
};

template<typename SparseBandedBlockedQR_Ext3Type>
struct SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType
{
  explicit SparseBandedBlockedQR_Ext3MatrixQTransposeReturnType(const SparseBandedBlockedQR_Ext3Type& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, Derived>(m_qr, other.derived(), true);
  }
  template<typename _Scalar, int _Options, typename _Index>
  SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar,_Options,_Index>& other)
  {
    return SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
  }
  const SparseBandedBlockedQR_Ext3Type& m_qr;
};

namespace internal {
  
template<typename SparseBandedBlockedQR_Ext3Type>
struct evaluator_traits<SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type> >
{
  typedef typename SparseBandedBlockedQR_Ext3Type::MatrixType MatrixType;
  typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
  typedef SparseShape Shape;
};

template< typename DstXprType, typename SparseBandedBlockedQR_Ext3Type>
struct Assignment<DstXprType, SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Sparse>
{
  typedef SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type> SrcXprType;
  typedef typename DstXprType::Scalar Scalar;
  typedef typename DstXprType::StorageIndex StorageIndex;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar,Scalar> &/*func*/)
  {
    typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
    idMat.setIdentity();
    // Sort the sparse householder reflectors if needed
    //const_cast<SparseBandedBlockedQR_Ext3Type *>(&src.m_qr)->_sort_matrix_Q();
    dst = SparseBandedBlockedQR_Ext3_QProduct<SparseBandedBlockedQR_Ext3Type, DstXprType>(src.m_qr, idMat, false);
  }
};

template< typename DstXprType, typename SparseBandedBlockedQR_Ext3Type>
struct Assignment<DstXprType, SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type>, internal::assign_op<typename DstXprType::Scalar,typename DstXprType::Scalar>, Sparse2Dense>
{
  typedef SparseBandedBlockedQR_Ext3MatrixQReturnType<SparseBandedBlockedQR_Ext3Type> SrcXprType;
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
