#ifndef SPARSE_BLOCK_COO_H
#define SPARSE_BLOCK_COO_H

#include <Eigen/Eigen>
#include <set>

template <typename IndexType>
class BlockYTY {
public:
	BlockYTY() {
	}

	BlockYTY(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &T, const IndexType numZeros, const IndexType rowStart)  
		: matY(Y), matT(T), nzrs(numZeros) {

		this->nnz_matY = this->matY.cast<int>().unaryExpr([](double x) {return int(std::abs(x) > 0); });
		this->nnz_matT = this->matT.cast<int>().unaryExpr([](double x) {return int(std::abs(x) > 0); });
		// Precomptue this for nnz counting (doesn't cost almost anything)
		this->matYTY = (this->matY * (this->matT * (this->matY.transpose())));//.cast<int>().unaryExpr([](double x) {return int(std::abs(x) > 0); });
		// Store sets of nonzeros indices in YTY product matrix
		for (int r = 0; r < this->matYTY.rows(); r++) {
			std::set<int> rowNnzs;
			for (int c = 0; c < this->matYTY.cols(); c++) {
				if (std::abs(this->matYTY(r, c)) > 0) {
					if (r < this->matY.cols()) {
						//std::cout << rowStart +  r << ", ";
						rowNnzs.insert(rowStart + r);
					} else {
						//std::cout << rowStart + this->matYTY.cols() + numZeros + r << ", ";
						rowNnzs.insert(rowStart + this->matYTY.cols() + numZeros + r);
					}

				}
			}
			//std::cout << "---" << std::endl;
			nnzSets.push_back(rowNnzs);

		}
		// Store indices of rows affected by this YTY
		for (int r = 0; r < this->matYTY.rows(); r++) {
			if (r < this->matY.cols()) {
				rowIdxs.push_back(rowStart + r);
			}
			else {
				rowIdxs.push_back(rowStart + this->matYTY.cols() + numZeros + r);
			}
		}
	}

	Eigen::MatrixXd& Y() {
		return this->matY;
	}

	Eigen::MatrixXd& T() {
		return this->matT;
	}

	Eigen::MatrixXi& YTY() {
		return this->matYTY;
	}

	IndexType rows() const {
		return this->matY.rows();
	}
	IndexType cols() const {
		return this->matY.cols();
	}

	IndexType numZeros() const {
		return this->nzrs;
	}

	BlockYTY transpose() const {
		return BlockYTY(matY, matT.transpose(), nzrs);
	}

	// FixMe: Is there a better way? (This is faster than calling .transpose() *
	Eigen::VectorXd multTransposed(const Eigen::VectorXd &other) const {
		return (this->matY * (this->matT.transpose() * (this->matY.transpose() * other)));
	}
	/*
	* 1) Unfinished nonzero cunting try using dense vector ...
	*/
	Eigen::VectorXi multNnz(const Eigen::VectorXi &other) const {
		Eigen::VectorXi nnz(other.size());
		for (int i = 0; i < other.size(); i++) {
			nnz(i) = this->matYTY.row(i).binaryExpr(other, [](int x, int y) { return x & y; }).sum();
		}
		return nnz;
		//return (this->nnz_matY * (this->nnz_matT * (this->nnz_matY.transpose() * other)));
	}
	/*
	* 2) Nonzero counting try using sparse vector and iterators
	*/
	template <typename SparseVector>
	Eigen::VectorXi multNnzSp(SparseVector &other, const int rowIdx) const {
		Eigen::VectorXi nnz(other.size());
		int firstBlockStart = rowIdx; int firstBlockEnd = rowIdx + this->matY.cols();
		int secondBlockStart = firstBlockEnd + this->nzrs; int secondBlockEnd = secondBlockStart + this->matY.rows() - this->matY.cols();
		for (SparseVector::InnerIterator it(other); it; ++it) {
			if (it.row() < firstBlockStart) {
				continue;
			}
			else if (it.row() > firstBlockStart && it.row() < firstBlockEnd) {
				//nnz |= (this->matYTY.col(it.row() - firstBlockStart) & 1);
				nnz = nnz.binaryExpr(this->matYTY.col(it.row() - firstBlockStart).unaryExpr([](int x) { return x & 1; }), [](int a, int b) { return a | b; });
			}
			else if (it.row() > secondBlockStart && it.row() < secondBlockEnd) {
				//nnz |= (this->matYTY.col(it.row() - secondBlockStart) & 1);
				nnz = nnz.binaryExpr(this->matYTY.col(it.row() - secondBlockStart).unaryExpr([](int x) { return x & 1; }), [](int a, int b) { return a | b; });
			}
			else if (it.row() >= secondBlockEnd) {
				break;
			}
		}
		return Eigen::VectorXi();
	}
	/*
	* 3) Nonzero counting try using std::set
	*/
	void multNnzSp2(std::set<int> &vecNnzs) const {
		std::vector<int> res;
		std::vector<int> isect(rowIdxs.size());
		for (int i = 0; i < rowIdxs.size(); i++) {
			/*
			for (auto it = nnzSets[i].begin(); it != nnzSets[i].end(); ++it) {
				std::cout << *it << ", ";
			}
			std::cout << std::endl;
			for (auto it = vecNnzs.begin(); it != vecNnzs.end(); ++it) {
				std::cout << *it << "; ";
			}
			std::cout << std::endl;
			*/
			if (std::set_intersection(nnzSets[i].begin(), nnzSets[i].end(), vecNnzs.begin(), vecNnzs.end(), isect.begin()) != isect.begin()) {
				// We know that the dot product will be nonzero
				for (int i = 0; i < rowIdxs.size(); i++) {
					vecNnzs.insert(rowIdxs[i]);
				}
				return;
				//res.push_back(rowIdxs[i]);
				//break;
			}
		}
		/*
		for (int i = 0; i < res.size(); i++) {
			vecNnzs.insert(res[i]);
		}*/
	}

	Eigen::VectorXd operator*(const Eigen::VectorXd &other) const {
		return (this->matY * (this->matT * (this->matY.transpose() * other)));
	}

private:
	Eigen::MatrixXd matY;
	Eigen::MatrixXd matT;
	Eigen::MatrixXd matYTY;
	Eigen::MatrixXi nnz_matY;
	Eigen::MatrixXi nnz_matT;
	//std::vector<std::vector<int> > ytyNNzIdxs;
	std::vector<std::set<int> > nnzSets;
	std::vector<int> rowIdxs;

	IndexType nzrs;
};

template <typename ValueType, typename IndexType>
class SparseBlockCOO {
public:
	struct Element {
		IndexType row;
		IndexType col;

		ValueType value;

		Element()
			: row(0), col(0) {
		}

		Element(const IndexType row, const IndexType col, const ValueType &val)
			: row(row), col(col), value(val) {
		}
	};
	typedef std::vector<Element> ElementsVec;

	SparseBlockCOO() 
		: nRows(0), nCols(0) {
	}

	SparseBlockCOO(const IndexType &rows, const IndexType &cols)
		: nRows(rows), nCols(cols) {
	}

	void insert(const Element &elem) {
		this->elems.push_back(elem);
	}

	IndexType size() const {
		return this->elems.size();
	}

	Element& operator[](IndexType i) {
		return this->elems[i];
	}
	const Element& operator[](IndexType i) const {
		return this->elems[i];
	}

private:
	ElementsVec elems;

	IndexType nRows;
	IndexType nCols;
};


#endif

