
#pragma once

#define EIGEN_NO_AUTOMATIC_RESIZING 

#include "Tableau.hpp"
#include "SM_utils.hpp"
#include "Solids.hpp"
#include "ReplacementDictionary.hpp"
#include "Equilibrium.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace AQSystemSolver {
    [[nodiscard]] inline auto possiblyAddSolid(const Eigen::VectorXd& solubilityProducts, ReplacementDict& replacementDict, SolidSystem& solidSystem){
        const auto addAttempt=solidSystem.getSolidToAdd(solubilityProducts);
        if(addAttempt.success) [[likely]] {
            replacementDict.addSolid(addAttempt.solid);
        }
        return addAttempt;
    }
    [[nodiscard]] inline auto possiblyRemoveSolid(const Eigen::VectorXd& solidAmts, ReplacementDict& replacementDict, SolidSystem& solidSystem){
        const auto removeAttempt=solidSystem.getSolidToRemove(solidAmts);
        if(removeAttempt.success) [[unlikely]] {
            replacementDict.removeSolid(removeAttempt.solid);
        }
        return removeAttempt;
    }

    inline constexpr double GUESS_TOTALS_FACTOR=1e-5;

    inline constexpr double MAX_PERCENT_ERROR_ADDEND=1e-5;

    inline constexpr std::size_t MAX_ITER=30;

    inline constexpr double STEP_WHEN_NEGATIVE=0.9;

    [[nodiscard]] std::pair<Eigen::RowVectorXd, Eigen::VectorXd> inline solveWithReplacement(const TableauWithTotals<>& replacedTableau) {
        
        Eigen::RowVectorXd currentSolution=Eigen::RowVectorXd::Constant(replacedTableau.cols(), GUESS_TOTALS_FACTOR);
        for(std::size_t iter=0; iter<MAX_ITER; ++iter){
            Eigen::VectorXd speciesConcentrations=replacedTableau.evalTerms(currentSolution);

            const Eigen::MatrixXd addends=replacedTableau.evalAddends(speciesConcentrations);

            const Eigen::RowVectorXd maxAddend=abs(addends.array()).colwise().maxCoeff();

            //x^c*c/x=(x^c)'
            const Eigen::MatrixXd jacobian=addends.transpose()*(replacedTableau.getCoefficients().array().rowwise()/currentSolution.array()).matrix();

            const Eigen::RowVectorXd yResult=replacedTableau.eval(addends);

            if(abs(yResult.array()/maxAddend.array()).maxCoeff()<MAX_PERCENT_ERROR_ADDEND) [[unlikely]] {
                return std::make_pair(std::move(currentSolution), std::move(speciesConcentrations));
            }

            Eigen::RowVectorXd delta=jacobian.partialPivLu().solve(yResult.transpose()).transpose();

            //don't go into the negatives, instead divide by 10
            delta=(delta.array()<currentSolution.array()).select(delta, currentSolution*STEP_WHEN_NEGATIVE);
            

            currentSolution-=delta;
        }
        throw std::runtime_error("Failed to converge");
    }

    inline auto solveForEquilibrium(const TableauWithTotals<>& tableau, const SolidSystem& initialSolidSystem, const ReplacementDict& origReplacementDict){
        Eigen::RowVectorXd currentSolution;
        Eigen::VectorXd speciesConcentrations;
        Eigen::VectorXd solubilityProducts;
        Eigen::VectorXd solidAmts;

        ReplacementDict replacementDict{origReplacementDict};
        SolidSystem solidSystem=initialSolidSystem.createNewWithInitialConditions();
        replacementDict.addSolidSystem(solidSystem);


        for(;;){
            const auto currentReplacedTableau=replacementDict.createReplacedTableau(tableau);
            if(currentReplacedTableau.cols()!=0){
                std::tie(currentSolution, speciesConcentrations)=solveWithReplacement(currentReplacedTableau);
            } else {
                currentSolution=Eigen::RowVectorXd(0);
                speciesConcentrations=currentReplacedTableau.getConstants();
            }
            solubilityProducts=solidSystem.calculateSolubilityProducts(currentSolution, replacementDict);

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
            if(removalAttempt.solid!=nullptr){
                //we've looped, and the adding didn't help
                throw std::runtime_error("failed to recover from loop. You may have to provide an initial guess for which solids are present.");
            }
            if(addAttempt.solid!=nullptr){
                //we've looped while trying to add.
                throw std::runtime_error("failed to recover from loop. You may have to provide an initial guess for which solids are present.");
            }
            break;
        }
        return Equilibrium{
            tableau,
            std::move(replacementDict), std::move(currentSolution), std::move(speciesConcentrations),
            std::move(solidSystem), solidAmts, solubilityProducts
        };
    }
} // namespace AQSystemSolver