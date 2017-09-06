// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_DIAGONAL_SPARSE_QR_EXT_H
#define EIGEN_BLOCK_DIAGONAL_SPARSE_QR_EXT_H

namespace Eigen {


	/**
	* \ingroup SparseQR_Module
	* \class BlockDiagonalSparseQR_Ext
	* \brief QR factorization of block-diagonal matrix
	*
	* \implsparsesolverconcept
	*
	*/
	template<typename _MatrixType, typename _BlockQRSolver>
	class BlockDiagonalSparseQR_Ext : public SparseSolverBase<BlockDiagonalSparseQR_Ext<_MatrixType, _BlockQRSolver> >
	{
	protected:
		typedef SparseSolverBase<BlockDiagonalSparseQR_Ext<_MatrixType, _BlockQRSolver> > Base;
		using Base::m_isInitialized;
	public:
		using Base::_solve_impl;
		typedef _MatrixType MatrixType;
		typedef _BlockQRSolver BlockQRSolver;
		typedef typename BlockQRSolver::MatrixType BlockMatrixType;
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
		BlockDiagonalSparseQR_Ext() : m_blocksRows(1), m_blocksCols(1)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit BlockDiagonalSparseQR_Ext(const MatrixType& mat) : m_blocksRows(1), m_blocksCols(1)
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
		inline Index cols() const { return m_R.cols(); }

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
		* Q = SparseQR<SparseMatrix<double> >(A).matrixQ();
		* \endcode
		* Internally, this call simply performs a sparse product between the matrix Q
		* and a sparse identity matrix. However, due to the fact that the sparse
		* reflectors are stored unsorted, two transpositions are needed to sort
		* them before performing the product.
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
			y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
			y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
			y.bottomRows(y.rows() - rank).setZero();

			// Apply the column permutation
			if (colsPermutation().size())  dest = colsPermutation() * y.topRows(cols());
			else                  dest = y.topRows(cols());

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
		inline const Solve<BlockDiagonalSparseQR_Ext, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockDiagonalSparseQR_Ext, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<BlockDiagonalSparseQR_Ext, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockDiagonalSparseQR_Ext, Rhs>(*this, B.derived());
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


		void setSparseBlockParams(int blocksRows, int blocksCols) {
			m_blocksRows = blocksRows;
			m_blocksCols = blocksCols;
		}


	protected:


		mutable ComputationInfo m_info;

		MatrixRType m_R;               // The triangular factor matrix
		MatrixQType m_Q;               // The orthogonal reflectors
		ScalarVector m_hcoeffs;         // The Householder coefficients

		PermutationType m_outputPerm_c; // The final column permutation


		Index m_nonzeropivots;          // Number of non zero pivots found
		IndexVector m_etree;            // Column elimination tree
		IndexVector m_firstRowElt;      // First element in each row

		int m_blocksRows;               // Rows of each subblock in the diagonal
		int m_blocksCols;               // Cols of each subblock in the diagonal

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
	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::analyzePattern(const MatrixType& mat)
	{
		eigen_assert(mat.isCompressed() && "SparseQR requires a sparse matrix in compressed mode. Call .makeCompressed() before passing it to SparseQR");

		/// Check block structure is valid
		eigen_assert(mat.rows() % m_blocksRows == mat.cols() % m_blocksCols && mat.rows() / m_blocksRows == mat.cols() / m_blocksCols && mat.cols() % m_blocksCols == 0);

		Index n = mat.cols();

		m_outputPerm_c.resize(n);
		m_outputPerm_c.indices().setLinSpaced(n, 0, n - 1);

		assert(_CrtCheckMemory());
	}

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename BlockQRSolver>
	void BlockDiagonalSparseQR_Ext<MatrixType, BlockQRSolver>::factorize(const MatrixType& mat)
	{
		typedef MatrixType::Index Index;

		/// Check block structure is valid
		eigen_assert(mat.rows() % m_blocksRows == mat.cols() % m_blocksCols &&
			mat.rows() / m_blocksRows == mat.cols() / m_blocksCols &&
			mat.cols() % m_blocksCols == 0);
		int nBlocks = int(mat.rows() / m_blocksRows);

#ifndef EIGEN_NO_DEBUG
		/// Check mat is block diagonal with stated dimensions
		for (Index k = 0; k<mat.outerSize(); ++k)
			for (typename MatrixType::InnerIterator it(mat, k); it; ++it)
			{
				Index r = it.row();   // row index
				Index c = it.col();   // col index (here it is equal to k)
									  // For each row in block 
				Index block = r / m_blocksRows;
				Index colbegin = block*m_blocksCols;
				eigen_assert(c >= colbegin && c < colbegin + m_blocksCols);
			}
#endif

		// Q is rows x rows, R is rows x cols
		/// Extract QR of each block in mat, and assemble final Q,R
		//Eigen::TripletArray<Scalar> tripletsQ(nBlocks * m_blocksRows * m_blocksRows);
		Eigen::TripletArray<Scalar> tripletsR(nBlocks * m_blocksRows * m_blocksCols);
		m_Q.resize(mat.rows(), mat.rows());
		m_Q.reserve(nBlocks * m_blocksRows * m_blocksRows);

		int rank = 0;
		for (int i = 0, currRow = 0, currCol = 0; i<nBlocks; i++, currRow += m_blocksRows, currCol += m_blocksCols) {

			// Copy block into a temporary
			BlockMatrixType block_i(m_blocksRows, m_blocksCols);
			for (int j = 0; j<m_blocksRows; j++)
				for (int k = 0; k<m_blocksCols; k++)
					block_i.coeffRef(j, k) = mat.coeff(currRow + j, currCol + k);

			// Perform QR
			BlockQRSolver blockSolver;
			blockSolver.compute(block_i);
			rank += blockSolver.rank();

			// Matrix<MatrixType::Scalar, Dynamic, Dynamic> Qi = blockSolver.matrixQ();
			enum { QRows = BlockQRSolver::MatrixType::RowsAtCompileTime };
			Matrix<MatrixType::Scalar, QRows, QRows> Qi = blockSolver.matrixQ();
			typename BlockQRSolver::MatrixRType Ri = blockSolver.matrixR();

			// Assemble into final Q
			if (m_blocksRows > m_blocksCols) {
				// each rectangular Qi is partitioned into [U N] where U is rxc and N is rx(r-c)
				// All the Us are gathered in the leftmost nc columns of Q, all Ns to the right
				auto N_start = nBlocks * m_blocksCols;
				auto m1 = m_blocksRows - m_blocksCols;

				auto base_row = i * m_blocksRows;
				auto base_col = i * m_blocksCols;
				// Q
				for (int j = 0; j < m_blocksRows; j++) {
					m_Q.startVec(base_row + j);
					// Us
					for (int k = 0; k < m_blocksCols; k++)
						//tripletsQ.add(base_row + j, base_col + k, Qi.coeff(j, k));
						m_Q.insertBack(base_row + j, base_col + k) = Qi.coeff(j, k);
					// Ns
					for (int k = 0; k < m1; k++)
						m_Q.insertBack(base_row + j, N_start + i * m1 + k) = Qi.coeff(j, m_blocksCols + k);
				}

				// R
				// Only the top cxc of R is nonzero, so c rows at a time
				for (int j = 0; j < m_blocksCols; j++)
					for (int k = j; k < m_blocksCols; k++)
						tripletsR.add(base_col + j, base_col + k, Ri.coeff(j, k));
			}
			else {
				// Just concatenate everything -- it's upper triangular anyway (although not rank-revealing... xxfixme with colperm?)
				// xx and indeed for landscape, don't even need to compute QR after we've done the leftmost #rows columns

				assert(false);
				/*
				auto base_row = i * m_blocksRows;
				auto base_col = i * m_blocksCols;
				// Q
				for (int j = 0; j < m_blocksRows; j++)
				for (int k = 0; k < m_blocksRows; k++)
				tripletsQ.add(base_row + j, base_row + k, Qi.coeff(j, k));

				// R
				for (int j = 0; j < m_blocksRows; j++)
				for (int k = j; k < m_blocksCols; k++)
				tripletsR.add(base_row + j, base_col + k, Ri.coeff(j, k));
				*/
			}

			// fill cols permutation
			for (int j = 0; j<m_blocksCols; j++)
				m_outputPerm_c.indices()(i*m_blocksCols + j, 0) = i*m_blocksCols + blockSolver.colsPermutation().indices()(j, 0);
		}

		/// Now build Q and R from Qs and Rs of each block
		//m_Q.resize(mat.rows(), mat.rows());
		//m_Q.setZero();
		//m_Q.setFromTriplets(tripletsQ.begin(), tripletsQ.end());
		//m_Q.makeCompressed();
		m_Q.finalize();

		m_R.resize(mat.rows(), mat.cols());
		m_R.setZero();
		m_R.setFromTriplets(tripletsR.begin(), tripletsR.end());
		m_R.makeCompressed();

		m_nonzeropivots = rank;
		m_isInitialized = true;
		m_info = Success;
	}


} // end namespace Eigen

#endif
