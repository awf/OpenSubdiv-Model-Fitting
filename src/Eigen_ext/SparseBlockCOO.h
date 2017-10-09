#ifndef SPARSE_BLOCK_COO_H
#define SPARSE_BLOCK_COO_H

#include <Eigen/Eigen>

template <typename IndexType>
class BlockYTY {
public:
	BlockYTY() {
	}

	BlockYTY(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &T, const IndexType numZeros)  
		: matY(Y), matT(T), nzrs(numZeros) {
	}

	Eigen::MatrixXd& Y() {
		return this->matY;
	}

	Eigen::MatrixXd& T() {
		return this->matT;
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
	
	Eigen::VectorXd operator*(const Eigen::VectorXd &other) const {
		return (this->matY * (this->matT * (this->matY.transpose() * other)));
	}

private:
	Eigen::MatrixXd matY;
	Eigen::MatrixXd matT;

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

