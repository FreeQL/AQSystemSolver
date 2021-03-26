#pragma once

#include "Tableau.hpp"
#include "Solids.hpp"
#include "ReplacementDictionary.hpp"

#include <Eigen/Dense>

#include <utility>

namespace AQSystemSolver{
    struct SolidPresent{
        Eigen::Index solidIndex;
        double concentration;

        [[nodiscard]] bool friend operator<(const SolidPresent& solid1, const SolidPresent& solid2){
            return solid1.solidIndex<solid2.solidIndex;
        }
    };
    
    struct SolidNotPresent{
        Eigen::Index solidIndex;
        double solubilityProduct;

        [[nodiscard]] bool friend operator<(const SolidNotPresent& solid1, const SolidNotPresent& solid2){
            return solid1.solidIndex<solid2.solidIndex;
        }
    };
    
        
    class Equilibrium {
    private:
        TableauWithTotals<> tableau;
        Tableau<> solidsTableau;
        ReplacementDict finalReplacements;
        Eigen::RowVectorXd finalSolution;

        Eigen::VectorXd tableauConcentrations;

        SM_utils::flat_set<SolidPresent> solidsPresent;
        SM_utils::flat_set<SolidNotPresent> solidsNotPresent;
    public:

        [[nodiscard]] const auto& getTableauConcentrations() const {
            return tableauConcentrations;
        }
        [[nodiscard]] const auto& getSolidsPresent() const {
            return solidsPresent;
        }
        [[nodiscard]] const auto& getSolidsNotPresent() const {
            return solidsNotPresent;
        }
        [[nodiscard]] const auto& cols() const {
            return tableau.cols();
        }

        //this is useful a) to know error and b) to get the totals if you have a replacement
        [[nodiscard]] Eigen::RowVectorXd getTotalConcentrations() const {
            Eigen::RowVectorXd totalConcentrations=tableau.evalWithoutTotal(tableauConcentrations);
            Eigen::VectorXd solidConcentrations(solidsPresent.size());
            {
                auto concIt=solidConcentrations.begin();
                for(const auto& solidPresent: solidsPresent){
                    *concIt=solidPresent.concentration;
                    ++concIt;
                }
            }
            totalConcentrations+=solidsTableau.reducedCopy(SolidIndexIndexer{solidsPresent}, Eigen::all).eval(solidConcentrations);
            return totalConcentrations;
        }
        [[nodiscard]] Eigen::VectorXd getExtraSolubilityProducts(const Tableau<>& extraSolids) const {
            return finalReplacements.createReplacedTableau(extraSolids).evalTerms(finalSolution);
        }

        Equilibrium(
            TableauWithTotals<> tableau_,
            ReplacementDict replacementDict_, Eigen::RowVectorXd finalSolution_, Eigen::VectorXd speciesConcentrations_,
            SolidSystem solidSystem_, const Eigen::VectorXd& solidAmts_, const Eigen::VectorXd& solubilityProducts_
        ) : 
        tableau{std::move(tableau_)},
        solidsTableau{std::move(solidSystem_).getTableau()},
        finalReplacements{std::move(replacementDict_)},
        finalSolution{std::move(finalSolution_)},
        tableauConcentrations{std::move(speciesConcentrations_)}
        {
            for(const Solid * solidPresent: solidSystem_.getSolidsPresent()){
                solidsPresent.insert(SolidPresent{solidPresent->solidIndex, solidAmts_.coeff(solidSystem_.getSolidsPresent().presenceIndexOf(solidPresent))});
            }
            for(const Solid * solidNotPresent: solidSystem_.getSolidsNotPresent()){
                solidsNotPresent.insert(SolidNotPresent{solidNotPresent->solidIndex, solubilityProducts_.coeff(solidSystem_.getSolidsNotPresent().presenceIndexOf(solidNotPresent))});
            }
        }
    };
} //namespace AQSystemSolver