#pragma once

#include "Tableau.hpp"
#include "Solids.hpp"
#include "SM_utils.hpp"

#include <Eigen/Dense>

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace AQSystemSolver {
    class ReplacementDict{
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
                replacement.groupTerm<1>(column, columnToRow[column]);
            }
        }

        [[nodiscard]] auto substituteTerms(){
            bool replacedAnything=false;
            const Tableau<> copyReplacement=replacement;
            for(auto column : columns){
                if(std::any_of(SM_utils::NestingIterator(columnToRow, columns.begin()), SM_utils::NestingIterator(columnToRow, columns.end()), [&](Eigen::Index row) { return replacement.getCoefficient(row, column)!=0; })) {
                    //std::cout<<"replaced"<<std::endl<<std::endl;
                    replacedAnything=true;
                    replacement.substituteRowAndCol<true>(copyReplacement, columnToRow[column], copyReplacement, column);
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

        [[nodiscard]] auto addColumn(const Eigen::Index column){
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

        void solveForTermAndAddToRow(const Eigen::RowVectorXd& rowVect, double constant, Eigen::Index term, Eigen::Index row){
            unsimplifiedReplacement.assignRow(row, rowVect, constant);
            unsimplifiedReplacement.groupTerm<0>(row, term);
            replacement.assignRowFromTableau(row, unsimplifiedReplacement, row);
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
        [[nodiscard]] auto size() const {
            assert(static_cast<Eigen::Index>(columns.size())==replacement.rows());
            return columns.size();
        }
        [[nodiscard]] auto cols() const {
            return replacement.cols();
        }



        void addSolid(Solid* solid){
            addSolidInternal(solid);
            //mostly already simplifed, just the last row, but we still have to iterate over all the columns
            simplify();
        }


        void removeSolid(Solid* solid){
            removeColumn(solid->column);

            solid->column=-1;

            //TODO(SoAsEr) Plug row back in and then simplify
            simplifyFromUnsimplified();
        }

        void addSolidSystem(const SolidSystem& solidSystem){
            assert(cols()==solidSystem.cols());
            replacement.conservativeResize(replacement.rows()+solidSystem.size(), cols());
            unsimplifiedReplacement.conservativeResize(replacement.rows()+solidSystem.size(), cols());
            for(Solid * solid : solidSystem.getSolidsPresent()) {
                addSolidInternal(solid); //avoid simplifying on every loop
            }
            simplify();
        }

        [[nodiscard]] auto createReplacedTableau(const /*TableauType*/ auto& orig) const {
            auto replaced=orig.reducedCopy(Eigen::all, columnsNotReplaced);
            if(orig.rows()) [[likely]] {
                for(auto column : columns){
                    replaced.template substituteRowAndCol<false>(replacementWithoutColumns, columnToRow[column], orig, column);
                }
            }
            return replaced;
        }
        
        ReplacementDict(const /*std::ranges::range*/ auto& columns_, const Tableau<>& replacement_):
        columnsNotReplaced{SM_utils::CountingIterator(0), SM_utils::CountingIterator(replacement_.cols())},
        nextRowToFill{0},
        columnToRow(replacement_.cols()),
        unsimplifiedReplacement{replacement_},
        replacement{replacement_}
        {
            Eigen::Index i{0};
            for(const Eigen::Index column: columns_) {
                const Eigen::Index row=addColumn(column);
                assert(replacement_.getCoefficient(i, column)!=0);
                solveForTermAndAddToRow(replacement_.getCoefficients().row(i), replacement_.getConstant(i), column, row);
                ++i;
            }
            simplifyFromUnsimplified();
        }
    };
} //namespace AQSystemSolver