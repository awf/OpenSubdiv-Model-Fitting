#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <future>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include <Eigen/SparseCore>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <suitesparse/SuiteSparseQR.hpp>
#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include <Eigen/src/SPQRSupport/SuiteSparseQRSupport.h>

#include "Eigen_pull/SparseBandedBlockedQR.h"
#include "Eigen_pull/SparseBlockAngularQR.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>

#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "Logger.h"

using namespace Eigen;

//#define OUTPUT_MAT 1

typedef SparseMatrix<Scalar, ColMajor, SuiteSparse_long> JacobianType;
typedef SparseMatrix<Scalar, RowMajor, int> JacobianTypeRowMajor;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
typedef SparseBandedBlockedQR<JacobianType, NaturalOrdering<int>, 8, false> BandedBlockedQRSolver;

// QR for J1 is banded blocked QR
typedef BandedBlockedQRSolver LeftSuperBlockSolver;
// QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
typedef ColPivHouseholderQR<MatrixType> RightSuperBlockSolver;
// QR solver for sparse block angular matrix
typedef SparseBlockAngularQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> BlockAngularQRSolver;

typedef ColPivHouseholderQR<Matrix<Scalar, 7, 2> > DenseQRSolver7x2;
typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolver7x2> BlockDiagonalQRSolver;

typedef SPQR<JacobianType> SPQRSolver;

/*
 * Generate random sparse banded matrix.
 * The only assumption about the matrix is that it has some sort of banded block structure.
 */
void generate_random_banded_matrix(const Eigen::Index numParams, const Eigen::Index numResiduals, JacobianType &spJ, bool permuteRows = true) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	std::random_device rd;
	std::mt19937 genmt(rd());
	const int numIdxDists = 256;
	const int overlap = 2;
	int step = numParams / numIdxDists;
	std::uniform_int_distribution<int> idxDists[numIdxDists];
	for (int i = 0; i < numIdxDists; i++) {
		if (i < numIdxDists - 1) {
			idxDists[i] = std::uniform_int_distribution<int>(i * step, (i + 1) * step - 1 + overlap);
		}
		else {
			idxDists[i] = std::uniform_int_distribution<int>(i * step, (i + 1) * step - 1);
		}
	}
	std::uniform_int_distribution<int> drawIdxDist(0, numIdxDists - 1);

	/*
	* Set-up the problem to be solved
	*/
	int nnzPerRow = 12;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(nnzPerRow * numResiduals);
	for (int i = 0; i < numResiduals; i++) {
		std::uniform_int_distribution<int> currDist = idxDists[drawIdxDist(gen)];
		for (int j = 0; j < nnzPerRow; j++) {
			jvals.add(i, currDist(genmt), dist(gen));
		}
	}

	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	// Permute Jacobian rows (we want to see how our QR handles a general matrix)	
	if (permuteRows) {
		PermutationMatrix<Dynamic, Dynamic, SuiteSparse_long> perm(spJ.rows());
		perm.setIdentity();
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		spJ = perm * spJ;
	}
}

/*
* Generate block diagonal sparse matrix with overlapping diagonal blocks.
*/
void generate_overlapping_block_diagonal_matrix(const Eigen::Index numParams, const Eigen::Index numResiduals, JacobianType &spJ, bool permuteRows = true) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	int stride = 7;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(stride * numParams);
	for (int i = 0; i < numParams; i++) {
		for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
			jvals.add(i * stride, j, dist(gen));
			jvals.add(i * stride + 1, j, dist(gen));
			jvals.add(i * stride + 2, j, dist(gen));
			jvals.add(i * stride + 3, j, dist(gen));
			jvals.add(i * stride + 4, j, dist(gen));
			jvals.add(i * stride + 5, j, dist(gen));
			jvals.add(i * stride + 6, j, dist(gen));
			if (j < numParams - 2) {
				jvals.add(i * stride + 6, j + 2, dist(gen));
			}
		}
	}

	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	// Permute Jacobian rows (we want to see how our QR handles a general matrix)	
	if (permuteRows) {
		PermutationMatrix<Dynamic, Dynamic, SuiteSparse_long> perm(spJ.rows());
		perm.setIdentity();
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		spJ = perm * spJ;
	}
}

/*
* Generate block diagonal sparse matrix.
*/
void generate_block_diagonal_matrix(const Eigen::Index numParams, const Eigen::Index numResiduals, JacobianType &spJ, bool permuteRows = true) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	int stride = 7;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(stride * numParams);
	for (int i = 0; i < numParams; i++) {
		for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
			jvals.add(i * stride, j, dist(gen));
			jvals.add(i * stride + 1, j, dist(gen));
			jvals.add(i * stride + 2, j, dist(gen));
			jvals.add(i * stride + 3, j, dist(gen));
			jvals.add(i * stride + 4, j, dist(gen));
			jvals.add(i * stride + 5, j, dist(gen));
			jvals.add(i * stride + 6, j, dist(gen));
		}
	}

	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	// Permute Jacobian rows (we want to see how our QR handles a general matrix)	
	if(permuteRows) {
		PermutationMatrix<Dynamic, Dynamic, SuiteSparse_long> perm(spJ.rows());
		perm.setIdentity();
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		spJ = perm * spJ;
	}
}
/*
* Generate block angular sparse matrix with overlapping diagonal blocks.
*/
void generate_block_angular_matrix(const Eigen::Index numParams, const Eigen::Index numAngularParams, const Eigen::Index numResiduals, JacobianType &spJ) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	int stride = 7;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(stride * numParams + numResiduals * numAngularParams);
	for (int i = 0; i < numParams; i++) {
		for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
			jvals.add(i * stride, j, dist(gen));
			jvals.add(i * stride + 1, j, dist(gen));
			jvals.add(i * stride + 2, j, dist(gen));
			jvals.add(i * stride + 3, j, dist(gen));
			jvals.add(i * stride + 4, j, dist(gen));
			jvals.add(i * stride + 5, j, dist(gen));
			jvals.add(i * stride + 6, j, dist(gen));
			if (j < numParams - 2) {
				jvals.add(i * stride + 6, j + 2, dist(gen));
			}
		}
	}
	for (int i = 0; i < numResiduals; i++) {
		for (int j = 0; j < numAngularParams; j++) {
			jvals.add(i, numParams + j, dist(gen));
		}
	}

	spJ.resize(numResiduals, numParams + numAngularParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();
}


int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Banded Test STARTED!");

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	int nVecEvals = 1000;
	clock_t begin;

	/*
	 * Set-up the problem to be solved
	*/
	// Problem size
	Eigen::Index numVars = 1024;
	Eigen::Index numParams = numVars * 2;
	Eigen::Index numResiduals = numVars * 3 + numVars + numVars * 3;
	// Generate the sparse matrix
	JacobianType spJ;
	//generate_random_banded_matrix(numParams, numResiduals, spJ, true);
	//generate_overlapping_block_diagonal_matrix(numParams, numResiduals, spJ, true);
	generate_block_diagonal_matrix(numParams, numResiduals, spJ, false);

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
#endif

	// Auxiliary identity matrix (for later use)
	JacobianType I(spJ.rows(), spJ.rows());
	I.setIdentity();

	std::cout << "####################################################" << std::endl;
	std::cout << "Problem size (r x c): " << spJ.rows() << " x " << spJ.cols() << std::endl;
	std::cout << "####################################################" << std::endl;



	/*
	* Solve the problem using the block diagonal QR solver.
	*/
	std::cout << "Solver: Block Diagonal Sparse QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	BlockDiagonalQRSolver bdqr;
	bdqr.setSparseBlockParams(7, 2);

	// 1) Factorization
	begin = clock();
	bdqr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	// 2) Benchmark expressing full Q 
	JacobianType bdqrQ(spJ.rows(), spJ.rows());
	bdqrQ = bdqr.matrixQ();
	std::cout << "matrixQ(): " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	// 3) Benchmark LS solving
	std::cout << "Solve LS: " << std::endl;
	// Prepare the data
	Eigen::VectorXd bdqrXDense = Eigen::VectorXd::Random(spJ.cols());
	Eigen::VectorXd bdqrVecDense = spJ * bdqrXDense;
	// Solve LS
	begin = clock();
	Eigen::VectorXd bdqrResDense;
	for (int i = 0; i < nVecEvals; i++) {
		bdqrResDense = bdqr.matrixQ().transpose() * bdqrVecDense;//slvrVec;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";
	begin = clock();
	VectorXd bdqrSolved;
	for (int i = 0; i < nVecEvals; i++) {
		bdqrSolved = bdqr.matrixR().template triangularView<Upper>().solve(bdqrResDense);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	VectorXd bdqrSolvedBackperm = VectorXd::Zero(spJ.cols());
	for (int i = 0; i < spJ.cols(); i++) {
		bdqrSolvedBackperm(bdqr.colsPermutation().indices().coeff(i)) = bdqrSolved(i);
	}

	// 4) Apply computed column reordering
	JacobianType spJPerm = (spJ * bdqr.colsPermutation());

	// 5) Show statistics and residuals
	std::cout << "---------------------- Stats -----------------------" << std::endl;
	std::cout << "Q non-zeros: " << bdqrQ.nonZeros() << " (" << (bdqrQ.nonZeros() / double(bdqrQ.rows() * bdqrQ.cols())) * 100 << "%)" << std::endl;
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (bdqrQ * bdqr.matrixR() - spJPerm).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (bdqrQ.transpose() * spJPerm - bdqr.matrixR()).norm() << std::endl;
	std::cout << "||targetX  - X||_2 = " << (bdqrXDense - bdqrSolvedBackperm).norm() << std::endl;
	std::cout << "####################################################" << std::endl;



	/*
	* Solve the problem using the banded blocked QR solver.
	*/
	std::cout << "Solver: Sparse Banded Blocked QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	BandedBlockedQRSolver slvr;

	// 1) Factorization
	begin = clock();
	slvr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");
	
	// 2) Benchmark expressing full Q 
	// Q * I
	std::cout << "Express full Q: " << std::endl;
	begin = clock();
	JacobianType slvrQ(spJ.rows(), spJ.rows());
	slvrQ = slvr.matrixQ() * I;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");

	// Q.T * I
	begin = clock();
	JacobianType slvrQt(spJ.rows(), spJ.rows());
	slvrQt = slvr.matrixQ().transpose() * I;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	//Logger::instance()->logMatrixCSV(slvrQt.toDense(), "slvrQtranspose.csv");

	// 3) Benchmark LS solving
	std::cout << "Solve LS: " << std::endl;
	// Prepare the data
	Eigen::VectorXd slvrXDense = Eigen::VectorXd::Random(spJ.cols());
	Eigen::VectorXd slvrVecDense = spJ * slvrXDense;
	// Solve LS
	begin = clock();
	Eigen::VectorXd slvrResDense;
	for(int i = 0; i < nVecEvals; i++) {
		slvrResDense = slvr.matrixQ().transpose() * slvrVecDense;//slvrVec;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";
	begin = clock();
	VectorXd solved;
	for (int i = 0; i < nVecEvals; i++) {
		solved = slvr.matrixR().template triangularView<Upper>().solve(slvrResDense);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	VectorXd solvedBackperm = VectorXd::Zero(spJ.cols());
	for (int i = 0; i < spJ.cols(); i++) {
		solvedBackperm(slvr.colsPermutation().indices().coeff(i)) = solved(i);
	}

	// 4) Apply computed row reordering
	JacobianType spJRowPerm = (slvr.rowsPermutation() * spJ);

	// 5) Show statistics and residuals
	std::cout << "---------------------- Stats -----------------------" << std::endl;
	std::cout << "Q non-zeros: " << slvrQ.nonZeros() << " (" << (slvrQ.nonZeros() / double(slvrQ.rows() * slvrQ.cols())) * 100 << "%)" << std::endl;
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (slvrQ * slvr.matrixR() - spJRowPerm).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (slvrQ.transpose() * spJRowPerm - slvr.matrixR()).norm() << std::endl;
	std::cout << "||Qt.T * R - J||_2 = " << (slvrQt.transpose() * slvr.matrixR() - spJRowPerm).norm() << std::endl;
	std::cout << "||Qt   * J - R||_2 = " << (slvrQt * spJRowPerm - slvr.matrixR()).norm() << std::endl;
	std::cout << "||targetX  - X||_2 = " << (slvrXDense - solvedBackperm).norm() << std::endl;
	//std::cout << "||Q.T  * Q - I||_2 = " << (slvrQ.transpose() * slvrQ - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;




	/*
	* Solve the problem using SuiteSparse QR.
	*/
	std::cout << "Solver: SuiteSparse QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	SPQRSolver spqr;
	begin = clock();
	spqr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	std::cout << "Solve LS: " << std::endl;
	Eigen::VectorXd slvrXSP = Eigen::VectorXd::Random(spJ.cols());
	Eigen::VectorXd slvrVecSP = spJ * slvrXSP;
	begin = clock();
	Eigen::VectorXd resSP;
	for (int i = 0; i < nVecEvals; i++) {
		resSP = spqr.matrixQ().transpose() * slvrVecSP;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	begin = clock();
	VectorXd solvedSP;
	for (int i = 0; i < nVecEvals; i++) {
		solvedSP = spqr.matrixR().template triangularView<Upper>().solve(resSP);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";


	VectorXd solvedBackpermSP = VectorXd::Zero(spJ.cols());
	for (int i = 0; i < spJ.cols(); i++) {
		solvedBackpermSP(spqr.colsPermutation().indices().coeff(i)) = solvedSP(i);
	}

	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||targetX  - X||_2 = " << (slvrXSP - solvedBackpermSP).norm() << std::endl;
	std::cout << "####################################################" << std::endl;




	/*
	* Solve another problem using the block angular QR solver.
	*/
	// Generate new input 
	numVars = 1024;
	numParams = numVars * 2;
	numResiduals = numVars * 3 + numVars + numVars * 3;
	Eigen::Index numAngularParams = 384; // 128 control points
	generate_block_angular_matrix(numParams, numAngularParams, numResiduals, spJ);

	std::cout << "####################################################" << std::endl;
	std::cout << "Problem size (r x c): " << spJ.rows() << " x " << spJ.cols() << std::endl;
	std::cout << "Left block (sparse): " << numResiduals << " x " << numParams << std::endl;
	std::cout << "Right block (dense): " << numResiduals << " x " << numAngularParams << std::endl;
	std::cout << "####################################################" << std::endl;
	
	// 6) Solve sparse block angular matrix
	std::cout << "Solver: Sparse Block Angular QR" << std::endl;
	std::cout << " Left sub-solver: Sparse Banded Blocked QR" << std::endl;
	std::cout << " Right sub-solver: Dense Column Piv House QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	// Factorize
	BlockAngularQRSolver baqr;
	begin = clock();
	baqr.setSparseBlockParams(numResiduals, numParams);
	baqr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	// Benchmark LS solving
	std::cout << "Solve LS: " << std::endl;
	// Prepare the data
	Eigen::VectorXd baqrXDense = Eigen::VectorXd::Random(spJ.cols());
	Eigen::VectorXd baqrVecDense = spJ * baqrXDense;
	// Apply row permutation before solving
	baqrVecDense = baqr.rowsPermutation() * baqrVecDense;
	// Solve LS
	begin = clock();
	Eigen::VectorXd baqrResDense;
	for (int i = 0; i < nVecEvals; i++) {
		baqrResDense = baqr.matrixQ().transpose() * baqrVecDense;//slvrVec;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";
	begin = clock();
	VectorXd baqrSolved;
	for (int i = 0; i < nVecEvals; i++) {
		baqrSolved = baqr.matrixR().template triangularView<Upper>().solve(baqrResDense);
	}
	VectorXd baqrSolvedBackperm = VectorXd::Zero(spJ.cols());
	for (int i = 0; i < spJ.cols(); i++) {
		baqrSolvedBackperm(baqr.colsPermutation().indices().coeff(i)) = baqrSolved(i);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||targetX - X||_2 = " << (baqrXDense - baqrSolvedBackperm).norm() << std::endl;
	std::cout << "####################################################" << std::endl;


#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");
	//Logger::instance()->logMatrixCSV(slvr.matrixY().toDense(), "slvrY.csv");
	//Logger::instance()->logMatrixCSV(slvr.matrixT().toDense(), "slvrT.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");
#endif

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
