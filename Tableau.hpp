#pragma once

#include <Eigen/Dense>

namespace AQSystemSolver {
    template<typename MatrixType=Eigen::MatrixXd, typename VectorType=Eigen::VectorXd, typename RowVectorType=Eigen::RowVectorXd>
    class Tableau;

    template<typename T>
    using TableauType=T;
    //concept TableauType = SM_utils::is_base_of_template<T, Tableau>::value;   

    template<typename MatrixType, typename VectorType, typename RowVectorType>
    class Tableau{
    private:
        MatrixType coefficients;
        VectorType constants;
    public:
        void assignRowFromTableau(Eigen::Index lhsRowIndex, const auto& rhsTableau, Eigen::Index rhsRowIndex) {
            coefficients.row(lhsRowIndex)=rhsTableau.getCoefficients().row(rhsRowIndex);
            constants.coeffRef(lhsRowIndex)=rhsTableau.getConstants().coeff(rhsRowIndex);
        }
        void assignRow(Eigen::Index lhsRowIndex, const auto& row, typename std::decay_t<decltype(row)>::Scalar constant) {
            coefficients.row(lhsRowIndex)=row;
            constants.coeffRef(lhsRowIndex)=constant;
        }         
        [[nodiscard]] const auto& getCoefficients() const {
            return coefficients;
        }
        [[nodiscard]] const auto& getCoefficient(Eigen::Index row, Eigen::Index column) const {
            return coefficients.coeff(row, column);
        }
        [[nodiscard]] const auto& getConstants() const {
            return constants;
        }
        [[nodiscard]] const auto& getConstant(Eigen::Index i) const {
            return constants.coeff(i);
        }
        [[nodiscard]] auto rows() const {
            return constants.rows();
        }
        [[nodiscard]] auto cols() const {
            return coefficients.cols();
        }
        template<int implicitPower>
        void groupTerm(Eigen::Index row, Eigen::Index column){
            auto& power=coefficients.coeffRef(row, column);
            if(power==implicitPower-1){
                return; //it won't do anything, so we skip it.
            }
            if(power==implicitPower){
                throw std::runtime_error("Power is the same as the implicit power, and we are trying to eliminate a replacement. This is probably a Gibbs Rule violation.");
            }
            const auto factor=1/(implicitPower-power);
            coefficients.row(row)*=factor;
            constants.coeffRef(row)=pow(constants.coeff(row), factor);
            power=0;
        }

        [[nodiscard]] VectorType evalTerms(const RowVectorType& x) const {
            MatrixType terms(coefficients.rows(),coefficients.cols());
            for(Eigen::Index i=0; i<coefficients.rows(); ++i){
                terms.row(i)=pow(x.array(), coefficients.row(i).array());
            }
            return terms.rowwise().prod().array()*constants.array();
        }
        [[nodiscard]] MatrixType evalAddends(VectorType speciesConcentrations) const {
            return coefficients.array().colwise()*speciesConcentrations.array();
        }
        [[nodiscard]] RowVectorType eval(const RowVectorType& x) const {
            return evalAddends(evalTerms(x)).colwise().sum();
        }
        [[nodiscard]] RowVectorType eval(const VectorType& speciesConcentrations) const {
            return evalAddends(speciesConcentrations).colwise().sum();
        }
        [[nodiscard]] RowVectorType eval(const MatrixType& addends) const {
            return addends.colwise().sum();
        }
        void resize(Eigen::Index rows, Eigen::Index cols) {
            coefficients.resize(rows, cols);
            constants.resize(rows);
        }
        void conservativeResize(Eigen::Index rows, Eigen::Index cols) {
            coefficients.conservativeResize(rows, cols);
            constants.conservativeResize(rows);
        }

        template<bool eliminate_column=false>
        void substituteRowAndCol(const /*TableauType*/ auto& replacementTableau, Eigen::Index row, const /*TableauType*/ auto& originalTableau, Eigen::Index col){
            coefficients+=originalTableau.getCoefficients().col(col)*replacementTableau.getCoefficients().row(row);
            if constexpr(eliminate_column){
                coefficients.col(col)-=originalTableau.getCoefficients().col(col);
            }
            constants.array()*=pow(replacementTableau.getConstants().coeff(row), originalTableau.getCoefficients().col(col).array());
        }
        [[nodiscard]] auto reducedCopy(const auto& v1, const auto& v2) const {
            return Tableau{coefficients(v1, v2), constants(v1)};
        }
        [[nodiscard]] auto reducedCopy(const decltype(Eigen::all)& v1, const auto& v2) const {
            return Tableau{coefficients(v1, v2), constants};
        }
        [[nodiscard]] bool operator==(const auto& rhs) const {
            return coefficients==rhs.getCoefficients() && constants==rhs.getConstants();
        }
        Tableau(MatrixType coefficients_, VectorType constants_) : coefficients{std::move(coefficients_)}, constants{std::move(constants_)} {}
        Tableau() = default;
    };
    template<typename MatrixType=Eigen::MatrixXd, typename VectorType=Eigen::VectorXd, typename RowVectorType=Eigen::RowVectorXd>
    class TableauWithTotals : private Tableau<MatrixType, VectorType, RowVectorType>{
    private:
        using parent=Tableau<MatrixType, VectorType, RowVectorType>;
        RowVectorType totals;
    public:
        [[nodiscard]] const auto& getTotals(){
            return totals;
        }
        [[nodiscard]] const auto& getTotal(Eigen::Index col){
            return totals.coeff(col);
        }
        using parent::assignRowFromTableau;
        using parent::assignRow;
        using parent::getCoefficients;
        using parent::getCoefficient;
        using parent::getConstants;
        using parent::getConstant;
        using parent::rows;
        using parent::cols;
        using parent::evalTerms;
        using parent::evalAddends;

        [[nodiscard]] auto reducedCopy(const auto& v1, const auto& v2) const {
            return TableauWithTotals{parent::reducedCopy(v1, v2), totals(v2)};
        }
        [[nodiscard]] auto reducedCopy(const auto& v1, const decltype(Eigen::all)& v2) const {
            return TableauWithTotals{parent::reducedCopy(v1, v2), totals};
        }
        [[nodiscard]] auto reducedCopy(const decltype(Eigen::all)& v1, const auto& v2) const {
            return TableauWithTotals{parent::reducedCopy(v1, v2), totals(v2)};
        }
        template<bool eliminate_column>
        void substituteRowAndCol(const /*TableauType*/ auto& replacementTableau, Eigen::Index row, const TableauWithTotals& originalTableau, Eigen::Index col) {
            static_assert(!eliminate_column);
            parent::template substituteRowAndCol<false>(replacementTableau, row, originalTableau, col);
            totals+=originalTableau.totals.coeff(col)*replacementTableau.getCoefficients().row(row);
        }
        [[nodiscard]] RowVectorType eval(const auto& x) const {
            return parent::eval(x)-totals;
        }
        [[nodiscard]] RowVectorType evalWithoutTotal(const auto& x) const {
            return parent::eval(x);
        }
        void resize(Eigen::Index rows, Eigen::Index cols){
            parent::resize(rows, cols);
            totals.resize(cols); 
        }
        void conservativeResize(Eigen::Index rows, Eigen::Index cols){
            parent::conservativeResize(rows, cols);
            totals.conservativeResize(cols); 
        }
        [[nodiscard]] bool operator==(const TableauWithTotals& rhs) const {
            return parent::operator==(rhs) && totals==rhs.totals;
        }
        TableauWithTotals(parent tableau_, RowVectorType totals_) : parent{std::move(tableau_)}, totals{std::move(totals_)} {}
        TableauWithTotals() = default;
    };
} // namespace AQSystemSolver