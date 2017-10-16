// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
// Copyright (C) 2016 Sergio Garrido Jurado <>
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_ANGULAR_SPARSE_QR_H
#define EIGEN_BLOCK_ANGULAR_SPARSE_QR_H

#include <algorithm>
#include <ctime>
#include "../Logger.h"

namespace Eigen {

	template < typename MatrixType, typename LeftSolver, typename RightSolver > class BlockAngularSparseQR;
	template<typename SparseQRType> struct BlockAngularSparseQRMatrixQReturnType;
	template<typename SparseQRType> struct BlockAngularSparseQRMatrixQTransposeReturnType;
	template<typename SparseQRType, typename Derived> struct BlockAngularSparseQR_QProduct;

	namespace internal {

		// traits<SparseQRMatrixQ[Transpose]>
		template <typename SparseQRType> struct traits<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType ReturnType;
			typedef typename ReturnType::StorageIndex StorageIndex;
			typedef typename ReturnType::StorageKind StorageKind;
			enum {
				RowsAtCompileTime = Dynamic,
				ColsAtCompileTime = Dynamic
			};
		};

		template <typename SparseQRType> struct traits<BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType ReturnType;
		};

		template <typename SparseQRType, typename Derived> struct traits<BlockAngularSparseQR_QProduct<SparseQRType, Derived> >
		{
			typedef typename Derived::PlainObject ReturnType;
		};
	} // End namespace internal


	  /**
	  * \ingroup SparseQR_Module
	  * \class BlockAngularSparseQR
	  * \brief QR factorization of block matrix, specifying subblock solvers
	  *
	  * This implementation is restricted to 1x2 block structure, factorizing
	  * matrix A = [A1 A2].
	  *
	  * \tparam _BlockQRSolverLeft The type of the QR solver which will factorize A1
	  * \tparam _BlockQRSolverRight The type of the QR solver which will factorize Q1'*A2
	  *
	  * \implsparsesolverconcept
	  *
	  */

	template<typename _MatrixType, typename _BlockQRSolverLeft, typename _BlockQRSolverRight>
	class BlockAngularSparseQR : public SparseSolverBase<BlockAngularSparseQR<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> >
	{
	protected:
		typedef BlockAngularSparseQR<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> this_t;
		typedef SparseSolverBase<BlockAngularSparseQR<_MatrixType, _BlockQRSolverLeft, _BlockQRSolverRight> > Base;
		using Base::m_isInitialized;
	public:
		using Base::_solve_impl;
		typedef _MatrixType MatrixType;
		typedef _BlockQRSolverLeft BlockQRSolverLeft;
		typedef _BlockQRSolverRight BlockQRSolverRight;
		typedef typename BlockQRSolverLeft::MatrixType LeftBlockMatrixType;
		typedef typename BlockQRSolverRight::MatrixType RightBlockMatrixType;
		typedef typename BlockQRSolverLeft::MatrixQType LeftBlockMatrixQType;
		typedef typename BlockQRSolverRight::MatrixQType RightBlockMatrixQType;
		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef typename MatrixType::StorageIndex StorageIndex;
		typedef typename MatrixType::Index Index;
		typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
		typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

		typedef BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR> MatrixQType;
		typedef SparseMatrix<Scalar> MatrixRType;
		typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;

		enum {
			ColsAtCompileTime = MatrixType::ColsAtCompileTime,
			MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
		};

	public:
		BlockAngularSparseQR() : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_blockCols(1)
		{ }

		/** Construct a QR factorization of the matrix \a mat.
		*
		* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
		*
		* \sa compute()
		*/
		explicit BlockAngularSparseQR(const MatrixType& mat) : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_blockCols(1)
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

		/** \returns the matrix Q
		*/
		BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR> matrixQ() const
		{
			return BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR>(*this); // xxawf pass pointer not ref
		}

		/** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
		* It is the combination of the fill-in reducing permutation and numerical column pivoting.
		*/
		const PermutationType& colsPermutation() const
		{
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return m_outputPerm_c;
		}

		const PermutationType& rowsPermutation() const {
			eigen_assert(m_isInitialized && "Decomposition is not initialized.");
			return this->m_rowPerm;
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
			// No pivoting ...
		}

		/** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
		*
		* \sa compute()
		*/
		template<typename Rhs>
		inline const Solve<BlockAngularSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockAngularSparseQR, Rhs>(*this, B.derived());
		}
		template<typename Rhs>
		inline const Solve<BlockAngularSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
		{
			eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
			eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
			return Solve<BlockAngularSparseQR, Rhs>(*this, B.derived());
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

		void setSparseBlockParams(Index blockRows, Index blockCols) {
			m_blockRows = blockRows;
			m_blockCols = blockCols;
		}

		BlockQRSolverLeft& getLeftSolver() { return m_leftSolver; }
		BlockQRSolverRight& getRightSolver() { return m_rightSolver; }


	protected:
		bool m_analysisIsok;
		bool m_factorizationIsok;
		mutable ComputationInfo m_info;
		std::string m_lastError;

		MatrixRType m_R;                // The triangular factor matrix
		PermutationType m_outputPerm_c; // The final column permutation
		PermutationType m_rowPerm;		// The final row permutation
		Index m_nonzeropivots;          // Number of non zero pivots found
		IndexVector m_etree;            // Column elimination tree
		IndexVector m_firstRowElt;      // First element in each row
		bool m_isQSorted;               // whether Q is sorted or not

		Index m_blockCols;                // Cols of first block
		Index m_blockRows;				  // Rows of the first block
										  // Every row below the first block is treated as a part of already upper triangular block)
		BlockQRSolverLeft m_leftSolver;
		BlockQRSolverRight m_rightSolver;

		template <typename, typename > friend struct BlockAngularSparseQR_QProduct;

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
	template <typename MatrixType, typename BlockQRSolverLeft, typename BlockQRSolverRight>
	void BlockAngularSparseQR<MatrixType, BlockQRSolverLeft, BlockQRSolverRight>::analyzePattern(const MatrixType& mat)
	{
		eigen_assert(mat.isCompressed() && "SparseQR requires a sparse matrix in compressed mode. Call .makeCompressed() before passing it to SparseQR");

		StorageIndex n = mat.cols();
		m_outputPerm_c.resize(n);
		m_outputPerm_c.indices().setLinSpaced(n, 0, StorageIndex(n - 1));

		StorageIndex m = mat.rows();
		m_rowPerm.resize(m);
		m_rowPerm.indices().setLinSpaced(m, 0, StorageIndex(m - 1));
	}

	/** \brief Performs the numerical QR factorization of the input matrix
	*
	* The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
	* a matrix having the same sparsity pattern than \a mat.
	*
	* \param mat The sparse column-major matrix
	*/
	template <typename MatrixType, typename BlockQRSolverLeft, typename BlockQRSolverRight>
	void BlockAngularSparseQR<MatrixType, BlockQRSolverLeft, BlockQRSolverRight>::factorize(const MatrixType& mat)
	{
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		typedef MatrixType::Index Index;
		Index m1 = m_blockCols;
		Index m2 = mat.cols() - m_blockCols;
		Index n1 = m_blockRows;
		Index n2 = mat.rows() - m_blockRows;

		// Split the main block from the separately treated bottom
		MatrixType matBlock = mat.topRows(n1);
		MatrixType matBottom = mat.bottomRows(n2);

		/// mat = | J1 J2 |
		/// J1 has m1 cols 
		LeftBlockMatrixType J1 = matBlock.leftCols(m1);
		//MatrixType J2 = mat.rightCols(m2);

		/// Compute QR for simple (e.g. block diagonal) matrix J1
		m_leftSolver.compute(J1);
		eigen_assert(m_leftSolver.info() == Success);

		//m_Q1 = m_leftSolver.matrixQ();
		typename BlockQRSolverLeft::MatrixRType R1 = m_leftSolver.matrixR();

		/// A = Q^t * J2
		/// n x m2

		// '(Eigen::SparseQRMatrixQTransposeReturnType<SparseQRType>, const Eigen::Block<const Derived,-1,-1,true>)'
		MatrixType XX = matBlock.rightCols(m2); // awf fixme this needs to be inlined in the next line
		if(m_leftSolver.hasRowPermutation()) {
			XX = m_leftSolver.rowsPermutation() * XX;
		}
		MatrixType A = m_leftSolver.matrixQ().transpose() * XX;//
		
		/// A = | Atop |      m1 x m2
		///     | Abot |    n-m1 x m2
		MatrixType Atop = A.topRows(m1);
		RightBlockMatrixType Abot = A.bottomRows(n1 - m1);
		// Concatenate remaining Abot with already diagonal bottom rows and solve
		RightBlockMatrixType Rbot(n1 - m1 + n2, m2);
		Rbot.topRows(n1 - m1) = Abot;
		Rbot.bottomRows(n2) = matBottom.rightCols(m2);
		m_rightSolver.compute(Rbot);
		eigen_assert(m_rightSolver.info() == Success);

		/// Compute final Q and R

		/// Q Matrix
		/// Q = Q1 * | I 0  |     n x n * | m1xm1    0
		///          | 0 Q2 |             |     0   (n-m1)x(n-m1)
		//m_Q2 = m_rightSolver.matrixQ();

		/// R Matrix
		/// R = | head(R1,m1) Atop*P2  |      m1 rows
		///     | 0           R2       |
		{
			Index R2_rows = (n1 + n2) - m1;
			DenseMatrix R2 = m_rightSolver.matrixR().topRows(R2_rows).template triangularView<Upper>(); // xx fix
			m_R.resize(n1 + n2, m1 + m2);

			RightBlockMatrixType AtopP2 = Atop.eval();
			m_rightSolver.colsPermutation().applyThisOnTheRight(AtopP2);

			Index initial_size = R1.nonZeros() + R2.nonZeros() + AtopP2.nonZeros();
			Eigen::TripletArray<Scalar, Index> triplets(initial_size); // xxawf fixme better estimate of nonzeros

																	   // top left corner, head(R1, m1)
			for (Index k = 0; k<R1.outerSize(); ++k)
				for (typename MatrixType::InnerIterator it(R1, k); it; ++it)
					if (it.row() < m1) // xxawf Hoist if R1.IsRowMajor?
						triplets.add(it.row(), it.col(), it.value());

			// top right corner, Atop*P2
			for (Index i = 0; i < m1; i++)
				for (Index j = 0; j < m2; j++)
					triplets.add_if_nonzero(i, m1 + j, AtopP2.coeff(i, j));

			// bottom right corner, head(R2, m2)
			for (Index i = 0; i < (std::min)((n1 + n2) - m1, m2); i++)
				for (int j = 0; j < m2; j++)
					triplets.add_if_nonzero(m1 + i, m1 + j, R2(i, j));

			m_R.setFromTriplets(triplets.begin(), triplets.end());
		}

		// fill cols permutation
		for (Index j = 0; j < m1; j++)
			m_outputPerm_c.indices()(j, 0) = m_leftSolver.colsPermutation().indices()(j, 0);
		for (Index j = m1; j<matBlock.cols(); j++)
			m_outputPerm_c.indices()(j, 0) = Index(m1 + m_rightSolver.colsPermutation().indices()(j - m1, 0));

		// fill rows permutation
		// Top block will use row permutation from the left solver
		// Bottom block is not permuted - no change of indices needed
		if(m_leftSolver.hasRowPermutation()) {
			for (Index j = 0; j < n1; j++) {
				m_rowPerm.indices()(j, 0) = m_leftSolver.rowsPermutation().indices()(j, 0);
			}
		}

		m_nonzeropivots = m_leftSolver.rank() + m_rightSolver.rank();
		m_isInitialized = true;
		m_info = Success;

	}

	template <typename SparseQRType, typename Derived>
	struct BlockAngularSparseQR_QProduct : ReturnByValue<BlockAngularSparseQR_QProduct<SparseQRType, Derived> >
	{
		typedef typename SparseQRType::MatrixQType MatrixType;
		typedef typename SparseQRType::Scalar Scalar;

		// Get the references 
		BlockAngularSparseQR_QProduct(const SparseQRType& qr, const Derived& other, bool transpose) :
			m_qr(qr), m_other(other), m_transpose(transpose) {}

		inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }

		inline Index cols() const { return m_other.cols(); }

		// Assign to a vector
		template<typename DesType>
		void evalTo(DesType& res) const
		{
			Index n = m_qr.rows();
			Index m1 = m_qr.m_blockCols;
			Index n1 = m_qr.m_blockRows;

			eigen_assert(n == m_other.rows() && "Non conforming object sizes");

			if (m_transpose)
			{
				/// Q' Matrix
				/// Q = | I 0   | * Q1'    | m1xm1    0              | * n x n 
				///     | 0 Q2' |          |     0   (n-m1)x(n-m1)   |           

				/// Q v = | I 0   | * Q1' * v   = | I 0   | * [ Q1tv1 ]  = [ Q1tv1       ]
				///       | 0 Q2' |               | 0 Q2' |   [ Q1tv2 ]    [ Q2' * Q1tv2 ]    

				res = m_other;
				// jasvob FixMe: The multipliation has to be split on 3 lines like this in order for the Eigen type inference to work well. 
				DesType otherTopRows = m_other.topRows(n1);
				DesType resTopRows = m_qr.m_leftSolver.matrixQ().transpose() * otherTopRows;
				res.topRows(n1) = resTopRows;
				res.bottomRows(n - m1) = m_qr.m_rightSolver.matrixQ().transpose() * res.bottomRows(n - m1);
			}
			else
			{
				/// Q Matrix
				/// Q = Q1 * | I 0  |     n x n * | m1xm1    0            |
				///          | 0 Q2 |             |     0   (n-m1)x(n-m1) |

				/// Q v = Q1 * | I 0  | * | v1 | =  Q1 * | v1      | 
				///            | 0 Q2 |   | v2 |         | Q2 * v2 | 

				res = m_other;
				DesType Q2v2 = m_other.bottomRows(n - m1);
				res.bottomRows(n - m1) = m_qr.m_rightSolver.matrixQ() * Q2v2;
				res = (m_qr.m_leftSolver.matrixQ() * res).eval();
			}
		}

		const SparseQRType& m_qr;
		const Derived& m_other;
		bool m_transpose;
	};

	template<typename SparseQRType>
	struct BlockAngularSparseQRMatrixQReturnType : public EigenBase<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
	{
		typedef typename SparseQRType::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
		enum {
			RowsAtCompileTime = Dynamic,
			ColsAtCompileTime = Dynamic
		};
		explicit BlockAngularSparseQRMatrixQReturnType(const SparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		BlockAngularSparseQR_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BlockAngularSparseQR_QProduct<SparseQRType, Derived>(m_qr, other.derived(), false);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
		}
		BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> adjoint() const
		{
			return BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
		}
		inline Index rows() const { return m_qr.rows(); }
		inline Index cols() const { return m_qr.rows(); }
		// To use for operations with the transpose of Q
		BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> transpose() const
		{
			return BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
		}

		const SparseQRType& m_qr;
	};

	template<typename SparseQRType>
	struct BlockAngularSparseQRMatrixQTransposeReturnType
	{
		explicit BlockAngularSparseQRMatrixQTransposeReturnType(const SparseQRType& qr) : m_qr(qr) {}
		template<typename Derived>
		BlockAngularSparseQR_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
		{
			return BlockAngularSparseQR_QProduct<SparseQRType, Derived>(m_qr, other.derived(), true);
		}
		template<typename _Scalar, int _Options, typename _Index>
		BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
		{
			return BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
		}
		const SparseQRType& m_qr;
	};

	namespace internal {

		template<typename SparseQRType>
		struct evaluator_traits<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
		{
			typedef typename SparseQRType::MatrixType MatrixType;
			typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
			typedef SparseShape Shape;
		};

		template< typename DstXprType, typename SparseQRType>
		struct Assignment<DstXprType, BlockAngularSparseQRMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQRMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Sparse>
		{
			typedef BlockAngularSparseQRMatrixQReturnType<SparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
			{
				typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
				idMat.setIdentity();
				// Sort the sparse householder reflectors if needed
				const_cast<SparseQRType *>(&src.m_qr)->_sort_matrix_Q();
				dst = BlockAngularSparseQR_QProduct<SparseQRType, DstXprType>(src.m_qr, idMat, false);
			}
		};

		template< typename DstXprType, typename SparseQRType>
		struct Assignment<DstXprType, BlockAngularSparseQRMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQRMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Dense>
		{
			typedef BlockAngularSparseQRMatrixQReturnType<SparseQRType> SrcXprType;
			typedef typename DstXprType::Scalar Scalar;
			typedef typename DstXprType::StorageIndex StorageIndex;
			static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
			{
				dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
			}
		};

	} // end namespace internal



} // end namespace Eigen

#endif
