#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include "Eigen_ext/eigen_extras.h"
#include "Eigen_ext/BlockSparseQR_Ext.h"
#include "Eigen_ext/BlockDiagonalSparseQR_Ext.h"
#include "Eigen_ext/SparseSubblockQR_Ext.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "Logger.h"

using namespace Eigen;

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Diagonalization Test STARTED!");

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	
	Eigen::Index numVars = 187;
	Eigen::Index numParams = numVars * 2;
	Eigen::Index numResiduals = numVars * 3 + numVars;
	Eigen::Index numRightParams = 105;

	typedef SparseMatrix<Scalar, ColMajor, int> JacobianType;
	typedef SparseQR<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;
	//typedef SparseQR<JacobianType, NaturalOrdering<Eigen::Index> > GeneralQRSolver;
	typedef ColPivHouseholderQR<Matrix<Scalar, 3, 2> > DenseQRSolverSmallBlock;
	//typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolverSmallBlock> GeneralQRSolver;
	typedef GeneralQRSolver SparseSuperblockSolver;
	typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseQRSolverSmallBlock> DiagonalSubblockSolver;
	typedef SparseSubblockQR_Ext<JacobianType, DiagonalSubblockSolver, SparseSuperblockSolver> SpecializedSparseSolver;


	// QR for J1 is block diagonal
	//typedef BlockDiagonalSparseQR_Ext2<JacobianType, DenseQRSolverSmallBlock> LeftSuperBlockSolver;
	// FixMe: The expected diagonal strucutre of Jacobian was broken when uv continuity constraint was added - general SparseQR has to be used	
	/*
	JacobianType tst;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> tvals(12);
	tvals.add(0, 0, 1);
	tvals.add(0, 1, 3);
	tvals.add(1, 0, 2);	
	tvals.add(1, 1, 5);
	tvals.add(2, 0, 8);
	tvals.add(2, 1, 9);
	tvals.add(3, 2, 5);
	tvals.add(3, 3, 7);
	tvals.add(4, 2, 6);
	tvals.add(4, 3, 3);
	tvals.add(5, 2, 4);
	tvals.add(5, 3, 2);
	
	tst.resize(6, 4);
	tst.setFromTriplets(tvals.begin(), tvals.end());
	tst.makeCompressed();

	Logger::instance()->logMatrixCSV(tst.toDense(), "tst.csv");

	BDQRSolver slvr;
	slvr.setSparseBlockParams(3, 2);
	slvr.compute(tst);
	JacobianType Qtst;
	Qtst = slvr.matrixQ();
	Logger::instance()->logMatrixCSV(Qtst.toDense(), "Qtst.csv");

	return 0;
	*/
	JacobianType rightBlock;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> rbvals(numRightParams * numResiduals);
	for (int i = 0; i < numResiduals; i++) {
		for (int j = 0; j < numRightParams; j++) {
			rbvals.add(i, j, dist(gen));
		}
	}
	rightBlock.resize(numResiduals, numRightParams);
	rightBlock.setFromTriplets(rbvals.begin(), rbvals.end());
	rightBlock.makeCompressed();

	JacobianType spJ;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(64);
/*
	for (int i = 0; i < numResiduals; i++) {


		for (int j = i * 2; j < i * 2 + 4 && j < numParams; j++) {
			jvals.add(i, j, dist(gen));
		}
	}*/
	//for (int i = 0; i < numResiduals; i++) {
	for (int i = 0; i < numVars; i++) {
		
			for (int j = numParams - 1; j >= i; j--) {
				//jvals.add(i, j, dist(gen));
				//jvals.add(i * 2, j, dist(gen));
				//jvals.add(i * 2 + 1, j, dist(gen));
				//jvals.add(i + numParams * 2, j, dist(gen));
			}

			for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
				jvals.add(i * 3, j, dist(gen));
				jvals.add(i * 3 + 1, j, dist(gen));
				jvals.add(i * 3 + 2, j, dist(gen));
				/*
				jvals.add(i + numVars * 3, j, dist(gen));
				if (j < numParams - 2) {
				jvals.add(i + numVars * 3, j + 2, dist(gen));
				}
				*/


				jvals.add(i + numVars * 3, j, dist(gen));
				if (j < numParams - 2) {
					jvals.add(i + numVars * 3, j + 2, dist(gen));
				}
			}

		}
	
	/*for (int i = numResiduals - numVars; i < numResiduals; i++) {
		

		for (int j = (i - (numResiduals - numVars)) * 2; j < (i - (numResiduals - numVars)) * 2 + 4 && j < numParams; j++) {
			jvals.add(i, j, dist(gen));
		}
	}*/

	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	Logger::instance()->logMatrixCSV(spJ.toDense(), "spJ.csv");

	//GeneralQRSolver solver;
	SpecializedSparseSolver solver;
	solver.setDiagBlockParams(numVars * 3, numVars * 2);
	solver.getDiagSolver().setSparseBlockParams(3, 2);
	clock_t begin = clock();
	solver.compute(spJ);

	std::cout << "Compute elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	JacobianType R;
	begin = clock();

	JacobianType Q;
	//Q2 = solver.matrixQ2();// .transpose();
	//JacobianType Q(spJ.rows(), spJ.rows());
	//Q.setIdentity();
	Q = solver.matrixQ();
	//R = solver.matrixR();
	//Q = (solver.matrixQ().transpose() * spJ);
	
	std::cout << "MatrixQ elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	std::cout << "MatrixQ nnz: " << Q.nonZeros() << std::endl;

	// Compute errors in reconstruction of Jacobian and R and output matrices
	R = solver.matrixR();
	solver.colsPermutation().applyThisOnTheRight(R);
	std::cout << "J err norm: " << ((Q * R).toDense() - spJ.toDense()).norm() << std::endl;

	solver.colsPermutation().applyThisOnTheRight(spJ);
	std::cout << "R err norm: " << ((Q.transpose() * spJ).toDense() - solver.matrixR().toDense()).norm() << std::endl;

	Logger::instance()->logMatrixCSV((Q.transpose() * spJ).toDense(), "QtJ.csv");
	Logger::instance()->logMatrixCSV((Q * R).toDense(), "QR.csv");
	Logger::instance()->logMatrixCSV(Q.toDense(), "Q.csv");
	Logger::instance()->logMatrixCSV(Q.cwiseAbs().toDense(), "Qabs.csv");
	Logger::instance()->logMatrixCSV(solver.matrixR().toDense(), "R.csv");
	return 0;
	
	/*
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Eigen::Index> perm(numResiduals);
	perm.setIdentity();
	//std::cout << perm.indices() << std::endl;
	for (Eigen::Index i = 0; i < numParams; i++) {
		perm.indices().row(i * 2) << i;
		perm.indices().row(i * 2 + 1) << i + numParams * 3;
		perm.indices().row(i + numParams * 2) << i + numParams;
		perm.indices().row(i + numParams * 3) << i + numParams * 2;
	}
	//perm.applyThisOnTheLeft(spJ);
	spJ = perm * spJ;
	*/
	JacobianType spJ2;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals2(64);

	for (int i = 0; i < numResiduals; i++) {
		for (int j = numParams - 1; j >= i; j--) {
			jvals2.add(i * 2, j, dist(gen));
		}
	}
	for (int i = 0; i < numResiduals; i++) {
		for (int j = i * 2; j < i * 2 + 4 && j < numParams; j++) {
			jvals2.add(i * 2 + 1, j, dist(gen));
		}
		/*for (int j = i; j < 4 && j < numParams; j++) {
			jvals2.add(i * 2 + 1, j, dist(gen));
		}*/
		/*for (int j = numParams - 1; j >= i; j--) {
			jvals2.add(i * 2 + 1, j, dist(gen));
		}*/
	}
	spJ2.resize(numResiduals, numParams);
	spJ2.setFromTriplets(jvals2.begin(), jvals2.end());
	spJ2.makeCompressed();


	Logger::instance()->logMatrixCSV(spJ2.toDense(), "spJ_r.csv");

	GeneralQRSolver solver2;
	
	begin = clock();
	solver2.compute(spJ2);

	JacobianType Q2;
	//Q2 = solver.matrixQ2();// .transpose();
	Q2 = solver2.matrixQ();

	//JacobianType res = solver.matrixQ().transpose() * rightBlock;
	std::cout << "Reordered elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	return 0;

	/*
	// Create testing matrix
	MatrixXX J = MatrixXX::Zero(6, 4);
	J.row(0) << 3, 1, 0, 0;
	J.row(1) << 0, 1, 0, 0;
	J.row(2) << 4, 3, 2, 2;
	J.row(3) << 0, 0, 5, 7;
	J.row(4) << 0, 0, 8, 6;
	J.row(5) << 0, 0, 9, 1;
	*/
	// Use "inverse" Givens rotations to zero out the last two elements of row(2) -> it is corrupting the diagonal
	/*
	 * | c  -s | * | x_i | = |           0           |
	 * | s   c |   | x_j |   | (x_i^2 + x_j^2)^(1/2) |
	 *
	 * Solution can be found as:
	 * c = x_j / (x_i^2 + x_j^2)^(1/2)
	 * s = x_i / (x_i^2 + x_j^2)^(1/2)
	 *
	 */
	/*
	MatrixXX R = J;

	std::cout << "R = \n" << R << std::endl;

	Scalar c = R(1, 0) / sqrt(R(1, 0) * R(1, 0) + R(2, 0) * R(2, 0));
	Scalar s = R(2, 0) / sqrt(R(1, 0) * R(1, 0) + R(2, 0) * R(2, 0));
	MatrixXX G22 = MatrixXX::Identity(6, 6);
	G22(1, 1) = c; G22(1, 2) = -s;
	G22(2, 1) = s; G22(2, 2) = c;
	R = G22 * R;

	std::cout << "G22 = \n" << G22 << std::endl;
	std::cout << "R = \n" << R << std::endl;

	Scalar c2 = R(3, 3) / sqrt(R(2, 3) * R(2, 3) + R(3, 3) * R(3, 3));
	Scalar s2 = R(2, 3) / sqrt(R(2, 3) * R(2, 3) + R(3, 3) * R(3, 3));
	MatrixXX G23 = MatrixXX::Identity(6, 6);
	G23(2, 2) = c2; G23(2, 3) = -s2;
	G23(3, 2) = s2; G23(3, 3) = c2;
	R = G23 * R;

	std::cout << "G23 = \n" << G23 << std::endl;
	std::cout << "R = \n" << R << std::endl;

	MatrixXX Q = (G22.transpose() * G23.transpose());
	 
	std::cout << "Q = \n" << Q << std::endl;
	std::cout << "R = \n" << R << std::endl;
	*/
	Logger::instance()->log(Logger::Info, "QR Diagonalization Test DONE!");

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
