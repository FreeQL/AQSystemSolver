
#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <queue>
#include <optional>
#include <memory>

#define EIGEN_NO_AUTOMATIC_RESIZING 

#include <Eigen/Dense>

#include "SM_utils.hpp"

namespace AQSystemSolver {
    template<typename MatrixType=Eigen::MatrixXd, typename VectorType=Eigen::VectorXd, typename RowVectorType=Eigen::RowVectorXd>
    class Tableau;

    template<typename T>
    using TableauType=T;
    //concept TableauType = SM_utils::is_base_of_template<T, Tableau>::value;

    
    template<typename Comp>
    using SolidVectorHeapIndexCompare=std::priority_queue<Eigen::Index, std::vector<Eigen::Index>, SM_utils::IndexCompare<Eigen::VectorXd, Comp, Eigen::Index>>;


    template<typename MatrixType, typename VectorType, typename RowVectorType>
    class Tableau{
    public:
        MatrixType coefficients;
        VectorType constants;

        Eigen::Index rows() const{
            return constants.rows();
        }
        Eigen::Index cols() const{
            return coefficients.cols();
        }
        VectorType evalTerms(const RowVectorType& x) const{
            MatrixType terms(coefficients.rows(),coefficients.cols());
            for(Eigen::Index i=0; i<coefficients.rows(); ++i){
                terms.row(i)=pow(x.array(), coefficients.row(i).array());
            }
            return terms.rowwise().prod().array()*constants.array();
        }
        MatrixType evalAddends(VectorType speciesConcentrations) const{
            return coefficients.array().colwise()*speciesConcentrations.array();
        }
        RowVectorType eval(const RowVectorType& x) const{
            return evalAddends(evalTerms(x)).colwise().sum();
        }
        RowVectorType eval(const VectorType& speciesConcentrations) const{
            return evalAddends(speciesConcentrations).colwise().sum();
        }
        RowVectorType eval(const MatrixType& addends) const{
            return addends.colwise().sum();
        }
        void resize(Eigen::Index rows, Eigen::Index cols){
            coefficients.resize(rows, cols);
            constants.resize(rows);
        }
        void conservativeResize(Eigen::Index rows, Eigen::Index cols){
            coefficients.conservativeResize(rows, cols);
            constants.conservativeResize(rows);
        }

        void substituteRowAndCol(const /*TableauType*/ auto& replacementTableau, Eigen::Index row, const /*TableauType*/ auto& originalTableau, Eigen::Index col){
            coefficients+=originalTableau.coefficients.col(col)*replacementTableau.coefficients.row(row); 
            constants.array()*=pow(replacementTableau.constants.coeff(row), originalTableau.coefficients.col(col).array());
        }
        Tableau reducedCopy(const auto& v1, const auto& v2) const {
            return {coefficients(v1, v2), constants(v1)};
        }
        Tableau reducedCopy(const decltype(Eigen::all)& v1, const auto& v2) const {
            return {coefficients(v1, v2), constants};
        }
        bool operator==(const Tableau& rhs) const {
            return coefficients==rhs.coefficients && constants==rhs.constants;
        }
    };
    template<typename MatrixType=Eigen::MatrixXd, typename VectorType=Eigen::VectorXd, typename RowVectorType=Eigen::RowVectorXd>
    class TableauWithTotals : public Tableau<MatrixType, VectorType, RowVectorType>{
    private:
        using parent=Tableau<MatrixType, VectorType, RowVectorType>;
    public:
        RowVectorType total;

        TableauWithTotals reducedCopy(const auto& v1, const auto& v2) const {
            return {parent::reducedCopy(v1, v2), total(v2)};
        }
        TableauWithTotals reducedCopy(const auto& v1, const decltype(Eigen::all)& v2) const {
            return {parent::reducedCopy(v1, v2), total};
        }
        TableauWithTotals reducedCopy(const decltype(Eigen::all)& v1, const auto& v2) const {
            return {parent::reducedCopy(v1, v2), total(v2)};
        }
        void substituteRowAndCol(const /*TableauType*/ auto& replacementTableau, Eigen::Index row, const TableauWithTotals& originalTableau, Eigen::Index col) {
            parent::substituteRowAndCol(replacementTableau, row, originalTableau, col);
            total+=originalTableau.total.coeff(col)*replacementTableau.coefficients.row(row);
        }
        RowVectorType eval(const auto& x) const {
            return parent::eval(x)-total;
        }
        void resize(Eigen::Index rows, Eigen::Index cols){
            parent::resize(rows, cols);
            total.resize(cols); 
        }
        void conservativeResize(Eigen::Index rows, Eigen::Index cols){
            parent::conservativeResize(rows, cols);
            total.conservativeResize(cols); 
        }
        bool operator==(const TableauWithTotals& rhs) const {
            return parent::operator==(rhs) && total==rhs.total;
        }
    };

    struct Solid{
        const Eigen::Index solidIndex;
        const Eigen::RowVectorXd row;
        const double constant;
        Eigen::Index column;
        template<typename RowType>
        Solid(Eigen::Index solidIndex_, RowType&& row_, double constant_, Eigen::Index column_) : 
        solidIndex{solidIndex_},
        row{std::forward<RowType>(row_)},
        constant{constant_},
        column{column_}
        {}
    };

    struct SolidCompare {
        using is_transparent=void;
        template<typename T, typename U>
        typename std::enable_if_t<SM_utils::is_pointer_fancy_v<T> && SM_utils::is_pointer_fancy_v<U>, bool> 
        operator()(const T& l, const U& r) const{
            return l->solidIndex<r->solidIndex;
        }
        template<typename T>
        typename std::enable_if_t<SM_utils::is_pointer_fancy_v<T>, bool> 
        operator()(const T& l, Eigen::Index r) const {
            return l->solidIndex<r;
        }
        template<typename U>
        typename std::enable_if_t<SM_utils::is_pointer_fancy<U>::value, bool> 
        operator()(Eigen::Index l, const U& r) const {
            return l<r->solidIndex;
        }
        bool operator()(Eigen::Index l, Eigen::Index r) const{
            return l<r;
        }
    };
    template<typename T>
    class SolidIndexIndexer{
    private:
        const T& set;
    public:
        Eigen::Index operator[](std::size_t i) const {
            return set[i]->solidIndex;
        }
        std::size_t size() const {
            return set.size();
        }
        SolidIndexIndexer(const T& set_): set{set_} {}
    };
    template<typename T>
    class ColumnIndexer{
    private:
        const T& set;
    public:
        Eigen::Index operator[](std::size_t i) const {
            return set[i]->column;
        }
        std::size_t size() const {
            return set.size();
        }
        ColumnIndexer(const T& set_): set{set_} {}
    };
    class SolidOwningSet {
    private:
        using set_type=SM_utils::flat_set<std::unique_ptr<Solid>, SolidCompare>;
        using const_iterator=SM_utils::UnowningIterator<set_type::const_iterator>;
        set_type flat_set;
    public:
        std::ptrdiff_t presenceIndexOf(const Solid* solid) const {
            return std::distance(flat_set.begin(), flat_set.find(solid));
        }
        std::ptrdiff_t lower_bound_index(const Solid* solid) const {
            return std::distance(flat_set.begin(), std::lower_bound(flat_set.begin(), flat_set.end(), solid, SolidCompare()));
        }
        template<typename U>
        const_iterator find(const U& t) const {
            return flat_set.find(t);
        }
        template<typename U>
        set_type::iterator find_extract(const U& t){
            return flat_set.find(t);
        }
        void insert(std::unique_ptr<Solid>&& in){
            flat_set.insert(std::move(in));
        }
        template<typename... Args>
        void emplace(Args&&... args){
            flat_set.emplace(std::forward<Args>(args)...);
        }
        void erase(set_type::iterator it){
            flat_set.erase(it);
        }
        SolidIndexIndexer<set_type> indexBySolidIndexes() const {
            return {flat_set};
        }
        ColumnIndexer<set_type> indexByColumn() const {
            return {flat_set};
        }
        Solid * get(Eigen::Index presenceIndex) {
            return flat_set[presenceIndex].get();
        }
        const Solid * get(Eigen::Index presenceIndex) const {
            return flat_set[presenceIndex].get();
        }
        std::size_t size() const {
            return flat_set.size();
        }
        void reserve(std::size_t s){
            flat_set.reserve(s);
        }
        const_iterator begin() const {
            return {flat_set.begin()};
        }
        const_iterator end() const {
            return {flat_set.end()};
        }
    };

    class SolidSystem{
    public:
        Eigen::Index numPresent;
        const Eigen::Index size;
        const Eigen::Index cols;
        const Tableau<> tableau;
        SolidOwningSet solidsPresent;
        SolidOwningSet solidsNotPresent;
    private:
        std::unordered_set<std::size_t> combinationsHash;
        std::size_t currentCombinationHash;

        template<bool adding>
        void addOrRemove(Solid* solidBeingChanged){
            SolidOwningSet& addingToContainer=adding ? solidsPresent : solidsNotPresent;
            SolidOwningSet& removingFromContainer=!adding ? solidsPresent : solidsNotPresent;

            auto extractedSolidIterator=removingFromContainer.find_extract(solidBeingChanged);
            addingToContainer.insert(std::move(*extractedSolidIterator));
            removingFromContainer.erase(extractedSolidIterator);
            if(adding){
                ++numPresent;
            } else {
                --numPresent;
            }
        }

        void add(Solid* solidBeingChanged){
            addOrRemove<true>(solidBeingChanged);
        }
        void remove(Solid* solidBeingChanged){
            addOrRemove<false>(solidBeingChanged);
        }

        bool conditionallyAddHash(const Solid* solid){
            const std::size_t newHash=currentCombinationHash^(std::size_t(1)<<solid->solidIndex);
            if(combinationsHash.insert(newHash).second){
                currentCombinationHash=newHash;
                return true;
            } else {
                return false;
            }
        }

        bool conditionallyAdd(Solid* solid){
            if(conditionallyAddHash(solid)){
                add(solid);
                return true;
            } else {
                return false;
            }
        }
        bool conditionallyRemove(Solid* solid){
            if(conditionallyAddHash(solid)){
                remove(solid);
                return true;
            } else {
                return false;
            }
        }
    public:
        Eigen::VectorXd calculateSolidAmts(const Eigen::RowVectorXd& leftOvers) const {
            Eigen::MatrixXd solidAmtEqns(solidsPresent.size(), numPresent);
            for(const Solid* solid : solidsPresent){
                solidAmtEqns.col(solidsPresent.presenceIndexOf(solid))=solid->row(solidsPresent.indexByColumn()).transpose();
            }
            const Eigen::VectorXd solidAmtLeftOver=leftOvers.transpose()(solidsPresent.indexByColumn());
            return solidAmtEqns.partialPivLu().solve(solidAmtLeftOver);
        }

        struct SolidChangeAttempt {
            bool success;
            Solid* solid;
        };
        
        SolidChangeAttempt possiblyRemoveSolid(const Eigen::VectorXd& solidAmts){
            Solid* solidNeedsToDisolve=nullptr;
            //we use a heap cause we usually wont need the full sort
            for(
                auto indexHeap=SolidVectorHeapIndexCompare<std::greater<void>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solidAmts.rows()), {solidAmts}); 
                Eigen::Index iThSolidPresent : SM_utils::ConsumingRange(indexHeap)
            ) {
                if(solidAmts.coeff(iThSolidPresent)<0.0) [[unlikely]] {
                    //not const because if we remove the solid then we're changing it
                    Solid* solid=solidsPresent.get(iThSolidPresent);
                    if(conditionallyRemove(solid)) [[likely]] {
                        return {true, solid};
                    } else if(!solidNeedsToDisolve){
                        //std::cout<<"WARNING: EITHER NEARLY LOOPED OR GIBBS RULE FAILED (remove)"<<std::endl;
                        solidNeedsToDisolve=solid;
                    }
                } else {
                    return {false, solidNeedsToDisolve};
                }
            }
            return {false, solidNeedsToDisolve};
        }
        SolidChangeAttempt possiblyAddSolid(const Eigen::VectorXd& solubilityProducts){
            Solid* solidNeedsToForm=nullptr;
            for(
                #ifndef NDEBUG
                    //sort the solids in the wrong order so that we are much more likely to trigger a removal
                    auto indexHeap=SolidVectorHeapIndexCompare<std::greater<Eigen::Index>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solubilityProducts.rows()), {solubilityProducts});
                #else
                    auto indexHeap=SolidVectorHeapIndexCompare<std::less<Eigen::Index>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solubilityProducts.rows()), {solubilityProducts});
                #endif
                Eigen::Index iThSolidNotPresent : SM_utils::ConsumingRange(indexHeap)
            ) {

                if(solubilityProducts.coeff(iThSolidNotPresent)>1.0) [[likely]] { //the long running cases will have lots of solids being added
                    Solid* solid=solidsNotPresent.get(iThSolidNotPresent);
                    if(conditionallyAdd(solid)) [[likely]] {
                        return {true, solid};
                    } else if(!solidNeedsToForm){
                        //std::cout<<"WARNING: EITHER NEARLY LOOPED OR GIBBS RULE FAILED (add)"<<std::endl;
                        solidNeedsToForm=solid;
                    }
                } else {
                    #ifdef NDEBUG
                        //short circuit if we're going in the right order 
                        return {false, solidNeedsToForm};
                    #endif
                } 
            }
            return {false, solidNeedsToForm};
        }

        template<typename T=Tableau<>, typename SetType=std::unordered_set<Eigen::Index>>
        SolidSystem(T&& tableau_, const SetType& starting_solids):
            numPresent{0},
            size{tableau_.rows()},
            cols{tableau_.cols()},
            tableau{std::forward<T>(tableau_)}
        {
            solidsPresent.reserve(size);
            solidsNotPresent.reserve(size);
            for(Eigen::Index i=0; i<tableau.rows(); ++i){
                const bool starting=starting_solids.find(i)!=starting_solids.end();
                const std::size_t hash=std::size_t(1)<<i;
                if(starting){
                    ++numPresent;
                    currentCombinationHash^=hash;
                    solidsPresent.emplace(std::make_unique<Solid>(i, tableau.coefficients.row(i), tableau.constants.coeff(i), -1));
                } else {
                    solidsNotPresent.emplace(std::make_unique<Solid>(i, tableau.coefficients.row(i), tableau.constants.coeff(i), -1));
                }
            }
            combinationsHash.insert(currentCombinationHash);
        }
    private:
        template<typename T=Tableau<>>
        SolidSystem(T&& tableau_, std::size_t currentCombinationHash_, SolidOwningSet&& solidsPresent_, SolidOwningSet&& solidsNotPresent_) : 
            numPresent{static_cast<Eigen::Index>(solidsPresent_.size())},
            size{tableau_.rows()},
            cols{tableau_.cols()},
            tableau{std::forward<T>(tableau_)},
            solidsPresent{std::move(solidsPresent_)},
            solidsNotPresent{std::move(solidsNotPresent_)},
            combinationsHash{currentCombinationHash_},
            currentCombinationHash{currentCombinationHash_}
        { }
    public:
        SolidSystem createNewWithInitialConditions() const {
            SolidOwningSet solidsPresent_;
            for(Solid * solidPresent : solidsPresent){
                solidsPresent_.emplace(std::make_unique<Solid>(solidPresent->solidIndex, solidPresent->row, solidPresent->constant, -1));
            }
            SolidOwningSet solidsNotPresent_;
            for(Solid * solidNotPresent : solidsNotPresent){
                solidsNotPresent_.emplace(std::make_unique<Solid>(solidNotPresent->solidIndex, solidNotPresent->row, solidNotPresent->constant, -1));
            }
            return SolidSystem(tableau, currentCombinationHash, std::move(solidsPresent_), std::move(solidsNotPresent_));
        }
    };


    class ReplacementDict{
    public:
        Eigen::Index cols;
    private:
        //which terms have been replaced out of existence
        SM_utils::flat_set<Eigen::Index> columns;
        SM_utils::flat_set<Eigen::Index> columnsNotReplaced;
        SM_utils::IncreasingPQ<Eigen::Index> nextRowToFill;
        std::vector<Eigen::Index> columnToRow;//because columns is not in order, and we always append to our replacement, this tells which column goes to which row
        Tableau<> unsimplifiedReplacement;
        Tableau<> replacement;
        Tableau<> replacementWithoutColumns;

        void groupTerms(){
            for(auto column : columns){
                auto& power=replacement.coefficients.coeffRef(columnToRow[column], column);
                if(power==0){
                    continue; //it won't do anything, so we skip it.
                } else if(power==1){
                    throw std::runtime_error("Power is 1, we are trying to eliminate a replacement. This is probably a Gibbs Rule violation.");
                }
                double powerInv=1/(1-power);
                replacement.coefficients.row(columnToRow[column])*=powerInv;
                replacement.constants.coeffRef(columnToRow[column])=pow(replacement.constants.coeff(columnToRow[column]), powerInv);
                power=0;
            }
        }

        bool substituteTerms(){
            bool replacedAnything=false;
            const Tableau<> copyReplacement=replacement;
            for(auto column : columns){
                if(std::any_of(SM_utils::NestingIterator(columnToRow, columns.begin()), SM_utils::NestingIterator(columnToRow, columns.end()), [&](Eigen::Index row) { return replacement.coefficients.coeff(row, column)!=0; })) {
                    //std::cout<<"replaced"<<std::endl<<std::endl;
                    replacedAnything=true;
                    replacement.substituteRowAndCol(copyReplacement, columnToRow[column], copyReplacement, column);
                    replacement.coefficients.col(column)-=copyReplacement.coefficients.col(column);
                }
            }
            return replacedAnything;
        }

        void simplify(){
            do{
                groupTerms();
            } while(substituteTerms());
            //the last substitute terms doesn't change anything so we dont need to regroup terms
            replacementWithoutColumns=replacement.reducedCopy(Eigen::all, columnsNotReplaced);
        }

        void simplifyFromUnsimplified(){
            replacement=unsimplifiedReplacement;
            simplify();
        }

        Eigen::Index addColumn(const Eigen::Index column){
            columnToRow[column]=nextRowToFill.top();
            nextRowToFill.pop();
            columns.insert(column);
            columnsNotReplaced.erase(column);
            return columnToRow[column];
        }

        void removeColumn(const Eigen::Index column){
            nextRowToFill.push(columnToRow[column]);
            columns.erase(column);
            columnsNotReplaced.insert(column);
        }

        void solveForTermAndAddToRow(const auto& rowVect, const double constant, const Eigen::Index term, const Eigen::Index row){
            const double termPowerInverse=-1.0/rowVect.coeff(term);

            unsimplifiedReplacement.coefficients.row(row)=termPowerInverse*rowVect.array();
            unsimplifiedReplacement.coefficients.coeffRef(row, term)=0.0;
            unsimplifiedReplacement.constants.coeffRef(row)=pow(constant, termPowerInverse);
            replacement.coefficients.row(row)=unsimplifiedReplacement.coefficients.row(row);
            replacement.constants.coeffRef(row)=unsimplifiedReplacement.constants.coeff(row);
        }

        void addSolidInternal(Solid* solid){
            auto columnIt=std::find_if(columnsNotReplaced.begin(), columnsNotReplaced.end(), [&](Eigen::Index term){ return solid->row.coeff(term)!=0.0; });
            if(columnIt==columnsNotReplaced.end()){
                throw std::runtime_error("Couldn't find a clean column. This is probably a Gibbs Rule violation.");
            }
            const Eigen::Index column=*columnIt;
            Eigen::Index row=addColumn(column);
            solid->column=column;
            solveForTermAndAddToRow(solid->row, solid->constant, column, row);
        }

    public:
        Eigen::Index size() const {
            assert(static_cast<Eigen::Index>(columns.size())==replacement.rows());
            return columns.size();
        }


        void addSolid(Solid* solid){
            addSolidInternal(solid);
            //mostly already simplifed, just the last row, but we still have to iterate over all the columns
            simplify();
        }


        void removeSolid(Solid* solid){
            removeColumn(solid->column);

            solid->column=-1;

            //TODO We should probably be able to plug it back into all our equations and then resimplify, but sounds hard and I know this works
            simplifyFromUnsimplified();
        }

        void addSolidSystem(const SolidSystem& solidSystem){
            assert(cols==solidSystem.cols);
            replacement.conservativeResize(replacement.rows()+solidSystem.size, cols);
            unsimplifiedReplacement.conservativeResize(replacement.rows()+solidSystem.size, cols);
            for(Solid * solid : solidSystem.solidsPresent) {
                addSolidInternal(solid); //avoid simplifying on every loop
            }
            simplify();
        }

        auto createReplacedTableau(const /*TableauType*/ auto& orig) const {
            auto replaced=orig.reducedCopy(Eigen::all, columnsNotReplaced);
            if(orig.rows()) [[likely]] {
                for(auto column : columns){
                    replaced.substituteRowAndCol(replacementWithoutColumns, columnToRow[column], orig, column);
                }
            }
            return replaced;
        }
        
        ReplacementDict(const /*std::ranges::range*/ auto& columns_, const Tableau<>& replacement_):
        cols{replacement_.cols()},
        columnsNotReplaced{SM_utils::CountingIterator(0), SM_utils::CountingIterator(replacement_.cols())},
        nextRowToFill{0},
        columnToRow(cols),
        unsimplifiedReplacement{replacement_},
        replacement{replacement_}
        {
            Eigen::Index i{0};
            for(const Eigen::Index column: columns_) {
                const Eigen::Index row=addColumn(column); //this will start at 0 and increment up as there are no removals
                assert(replacement_.coefficients.coeff(i, column)!=0);
                solveForTermAndAddToRow(replacement_.coefficients.row(i), replacement_.constants(i), column, row);
                ++i;
            }
            simplifyFromUnsimplified();
        }
    };

    inline SolidSystem::SolidChangeAttempt possiblyAddSolid(const Eigen::VectorXd& solubilityProducts, ReplacementDict& replacementDict, SolidSystem& solidSystem){
        const SolidSystem::SolidChangeAttempt addAttempt=solidSystem.possiblyAddSolid(solubilityProducts);
        if(addAttempt.success) [[likely]] {
            replacementDict.addSolid(addAttempt.solid);
        }
        return addAttempt;
    }
    inline SolidSystem::SolidChangeAttempt possiblyRemoveSolid(const Eigen::VectorXd& solidAmts, ReplacementDict& replacementDict, SolidSystem& solidSystem){
        const SolidSystem::SolidChangeAttempt removeAttempt=solidSystem.possiblyRemoveSolid(solidAmts);
        if(removeAttempt.success) [[unlikely]] {
            replacementDict.removeSolid(removeAttempt.solid);
        }
        return removeAttempt;
    }

    inline std::pair<Eigen::RowVectorXd, Eigen::VectorXd> solveWithReplacement(const TableauWithTotals<>& replacedTableau) {
        
        Eigen::RowVectorXd currentSolution=Eigen::RowVectorXd::Constant(replacedTableau.cols(), 1e-5);
        for(std::size_t iter=0; iter<30; ++iter){
            const Eigen::VectorXd speciesConcentrations=replacedTableau.evalTerms(currentSolution);

            const Eigen::MatrixXd addends=replacedTableau.evalAddends(speciesConcentrations);

            const Eigen::RowVectorXd maxAddend=abs(addends.array()).colwise().maxCoeff();

            //x^c*c/x=(x^c)'
            const Eigen::MatrixXd jacobian=addends.transpose()*(replacedTableau.coefficients.array().rowwise()/currentSolution.array()).matrix();

            const Eigen::RowVectorXd yResult=replacedTableau.eval(addends);

            if(abs(yResult.array()/maxAddend.array()).maxCoeff()<1e-5) [[unlikely]] {
                return {std::move(currentSolution), std::move(speciesConcentrations)};
            }

            Eigen::RowVectorXd delta=jacobian.partialPivLu().solve(yResult.transpose()).transpose();

            //don't go into the negatives, instead divide by 10
            delta=(delta.array()<currentSolution.array()).select(delta, currentSolution*0.9);
            

            currentSolution-=delta;
        }
        throw std::runtime_error("Failed to converge");
    }

    struct SolidPresent{
        Eigen::Index solidIndex;
        double concentration;

        friend bool operator<(const SolidPresent& solid1, const SolidPresent& solid2){
            return solid1.solidIndex<solid2.solidIndex;
        }
        operator Eigen::Index() const {
            return solidIndex;
        }

        SolidPresent(Eigen::Index solidIndex_, double concentration_) : 
        solidIndex{solidIndex_},
        concentration{concentration_}
        {}
    };
    struct SolidNotPresent{
        Eigen::Index solidIndex;
        double solubilityProduct;

        friend bool operator<(const SolidNotPresent& solid1, const SolidNotPresent& solid2){
            return solid1.solidIndex<solid2.solidIndex;
        }
        operator Eigen::Index() const {
            return solidIndex;
        }
        SolidNotPresent(Eigen::Index solidIndex_, double solubilityProduct_) : 
        solidIndex{solidIndex_},
        solubilityProduct{solubilityProduct_}
        {}
    };
        
    class Equilibrium {
    private:
        TableauWithTotals<> tableau;
        Tableau<> solidsTableau;
        ReplacementDict finalReplacements;
        Eigen::RowVectorXd finalSolution;
    public:

        Eigen::VectorXd tableauConcentrations;

        SM_utils::flat_set<SolidPresent> solidsPresent;
        SM_utils::flat_set<SolidNotPresent> solidsNotPresent;

        //this is useful a) to know error and b) to get the totals if you have a replacement
        Eigen::RowVectorXd getTotalConcentrations() const {
            Eigen::RowVectorXd totalConcentrations=static_cast<Tableau<>>(tableau).eval(tableauConcentrations);
            Eigen::VectorXd solidConcentrations(solidsPresent.size());
            {
                auto concIt=solidConcentrations.begin();
                for(const auto& solidPresent: solidsPresent){
                    *concIt=solidPresent.concentration;
                    ++concIt;
                }
            }
            totalConcentrations+=solidsTableau.reducedCopy(solidsPresent, Eigen::all).eval(solidConcentrations);
            return totalConcentrations;
        }
        Eigen::VectorXd getExtraSolubilityProducts(const Tableau<>& extraSolids) const {
            return finalReplacements.createReplacedTableau(extraSolids).evalTerms(finalSolution);
        }

        template<typename Tabl, typename RepDict, typename Sol, typename Conc, typename SolidSys>
        Equilibrium(
            Tabl&& tableau_,
            RepDict&&  replacementDict_, Sol&& finalSolution_, Conc&& speciesConcentrations_,
            SolidSys&& solidSystem_, const Eigen::VectorXd& solidAmts_, const Eigen::VectorXd& solubilityProducts_
        ) : 
        tableau{std::forward<Tabl>(tableau_)},
        solidsTableau{std::forward<decltype(std::forward<SolidSys>(solidSystem_).tableau)>(std::forward<SolidSys>(solidSystem_).tableau)},
        finalReplacements{std::forward<RepDict>(replacementDict_)},
        finalSolution{std::forward<Sol>(finalSolution_)},
        tableauConcentrations{std::forward<Conc>(speciesConcentrations_)}
        {
            for(const Solid * solidPresent: solidSystem_.solidsPresent){
                solidsPresent.emplace(solidPresent->solidIndex, solidAmts_.coeff(solidSystem_.solidsPresent.presenceIndexOf(solidPresent)));
            }
            for(const Solid * solidNotPresent: solidSystem_.solidsNotPresent){
                solidsNotPresent.emplace(solidNotPresent->solidIndex, solubilityProducts_.coeff(solidSystem_.solidsNotPresent.presenceIndexOf(solidNotPresent)));
            }
        }
    };

    inline Equilibrium solveForEquilibrium(const TableauWithTotals<>& tableau, const SolidSystem& initialSolidSystem, const ReplacementDict& origReplacementDict){
        Eigen::RowVectorXd currentSolution;
        Eigen::VectorXd speciesConcentrations;
        Eigen::VectorXd solubilityProducts;
        Eigen::VectorXd solidAmts;

        ReplacementDict replacementDict{origReplacementDict};
        SolidSystem solidSystem=initialSolidSystem.createNewWithInitialConditions();
        replacementDict.addSolidSystem(solidSystem);


        for(;;){
            const auto currentReplacedTableau=replacementDict.createReplacedTableau(tableau);
            if(currentReplacedTableau.cols()){
                std::tie(currentSolution, speciesConcentrations)=solveWithReplacement(currentReplacedTableau);
            } else {
                currentSolution=Eigen::VectorXd(0);
                speciesConcentrations=currentReplacedTableau.constants;
            }
            const auto solidsNotHereTableau=replacementDict.createReplacedTableau(solidSystem.tableau.reducedCopy(solidSystem.solidsNotPresent.indexBySolidIndexes(), Eigen::all));
            assert(solidsNotHereTableau.cols()==currentReplacedTableau.cols());
            if(currentReplacedTableau.cols()){
                solubilityProducts=solidsNotHereTableau.evalTerms(currentSolution);
            } else {
                solubilityProducts=solidsNotHereTableau.constants;
            }

            solidAmts=solidSystem.calculateSolidAmts(-tableau.eval(speciesConcentrations)); 

            SolidSystem::SolidChangeAttempt removalAttempt=possiblyRemoveSolid(solidAmts, replacementDict, solidSystem);
            if(removalAttempt.success) {
                continue;
            }
            SolidSystem::SolidChangeAttempt addAttempt=possiblyAddSolid(solubilityProducts, replacementDict, solidSystem);
            if(addAttempt.success) {
                continue;
            }

            
            //there used to be a huge piece of code to try and switch two solids (remove one and add another within a single iteration). It was a mess, huge, and didn't really help recovery        
            if(removalAttempt.solid){
                //we've looped, and the adding didn't help
                throw std::runtime_error("failed to recover from loop. You may have to provide an initial guess for which solids are present.");
            }
            if(addAttempt.solid){
                //we've looped while trying to add.
                throw std::runtime_error("failed to recover from loop. You may have to provide an initial guess for which solids are present.");
            }
            break;
        }
        return Equilibrium{
            tableau,
            std::move(replacementDict), std::move(currentSolution), std::move(speciesConcentrations),
            std::move(solidSystem), std::move(solidAmts), std::move(solubilityProducts)
        };
    }
}