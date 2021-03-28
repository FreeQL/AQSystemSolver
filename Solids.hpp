#pragma once

#include "SM_utils.hpp"
#include "Tableau.hpp"

#include <Eigen/Dense>

#include <queue>
#include <vector>
#include <utility>
#include <type_traits>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <functional>

namespace AQSystemSolver {

    template<typename Comp>
    using SolidVectorHeapIndexCompare=std::priority_queue<Eigen::Index, std::vector<Eigen::Index>, SM_utils::IndexCompare<Eigen::VectorXd, Comp, Eigen::Index>>;

    struct Solid{
        const Eigen::Index solidIndex;
        const Eigen::RowVectorXd row;
        const double constant;
        Eigen::Index column;

        Solid(Eigen::Index solidIndex_, Eigen::RowVectorXd row_, double constant_, Eigen::Index column_) : 
        solidIndex{solidIndex_},
        row{std::move(row_)},
        constant{constant_},
        column{column_}
        {}
    };

    struct SolidCompare {
        using is_transparent=void;
        [[nodiscard]] auto operator()(const auto& l, const auto& r) const 
        -> typename std::enable_if_t<SM_utils::is_pointer_fancy_v<decltype(l)> && SM_utils::is_pointer_fancy_v<decltype(r)>, bool> 
        {
            return l->solidIndex<r->solidIndex;
        } 
        [[nodiscard]] auto operator()(const auto& l, Eigen::Index r) const
        -> typename std::enable_if_t<SM_utils::is_pointer_fancy_v<decltype(l)>, bool>
        {
            return l->solidIndex<r;
        }
        [[nodiscard]] auto operator()(Eigen::Index l, const auto& r) const
        -> typename std::enable_if_t<SM_utils::is_pointer_fancy_v<decltype(r)>, bool>
        {
            return l<r->solidIndex;
        }
        [[nodiscard]] auto operator()(Eigen::Index l, Eigen::Index r) const {
            return l<r;
        }
    };
    template<typename T, typename... Args>
    class SolidIndexIndexer{
    private:
        const SM_utils::flat_set<T, Args...>& set;
    public:
        [[nodiscard]] auto operator[](std::size_t i) const {
            if constexpr(SM_utils::is_pointer_fancy_v<T>){
                return set[i]->solidIndex;
            } else {
                return set[i].solidIndex;
            }
        }
        [[nodiscard]] auto size() const {
            return set.size();
        }
        explicit SolidIndexIndexer(const SM_utils::flat_set<T, Args...>& set_): set{set_} {}
    };
    template<typename T, typename... Args>
    class ColumnIndexer{
    private:
        const SM_utils::flat_set<T, Args...>& set;
    public:
        [[nodiscard]] auto operator[](std::size_t i) const {
            if constexpr(SM_utils::is_pointer_fancy_v<T>){
                return set[i]->column;
            } else {
                return set[i].column;
            }
        }
        [[nodiscard]] auto size() const {
            return set.size();
        }
        explicit ColumnIndexer(const SM_utils::flat_set<T, Args...>& set_): set{set_} {}
    };
    class SolidOwningSet {
    private:
        using set_type=SM_utils::flat_set<std::unique_ptr<Solid>, SolidCompare>;
        using const_iterator=SM_utils::UnowningIterator<set_type::const_iterator>;
        set_type flat_set;
    public:
        [[nodiscard]] auto presenceIndexOf(const Solid* solid) const {
            return std::distance(flat_set.begin(), flat_set.find(solid));
        }
        [[nodiscard]] auto lower_bound_index(const Solid* solid) const {
            return std::distance(flat_set.begin(), std::lower_bound(flat_set.begin(), flat_set.end(), solid, SolidCompare()));
        }
        [[nodiscard]] const_iterator find(const auto& t) const {
            return flat_set.find(t);
        }
        [[nodiscard]] typename set_type::iterator find_extract(const auto& t) {
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
        [[nodiscard]] auto indexBySolidIndexes() const {
            return SolidIndexIndexer{flat_set};
        }
        [[nodiscard]] auto indexByColumn() const {
            return ColumnIndexer{flat_set};
        }
        [[nodiscard]] Solid* get(Eigen::Index presenceIndex) {
            return flat_set[presenceIndex].get();
        }
        [[nodiscard]] const Solid* get(Eigen::Index presenceIndex) const {
            return flat_set[presenceIndex].get();
        }
        [[nodiscard]] auto size() const {
            return flat_set.size();
        }
        void reserve(std::size_t s){
            flat_set.reserve(s);
        }
        [[nodiscard]] auto begin() const {
            return const_iterator{flat_set.begin()};
        }
        [[nodiscard]] auto end() const {
            return const_iterator{flat_set.end()};
        }
    };

    class SolidSystem{
    private:
        Eigen::Index numPresent;
        Tableau<> tableau;
        SolidOwningSet solidsPresent;
        SolidOwningSet solidsNotPresent;

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

        [[nodiscard]] auto conditionallyAddHash(const Solid* solid){
            const std::size_t newHash=currentCombinationHash^(std::size_t(1)<<std::size_t(solid->solidIndex));
            if(combinationsHash.insert(newHash).second){
                currentCombinationHash=newHash;
                return true;
            }
            return false;
        }

        [[nodiscard]] auto conditionallyAdd(Solid* solid){
            if(conditionallyAddHash(solid)){
                add(solid);
                return true;
            }
            return false;
        }
        [[nodiscard]] auto conditionallyRemove(Solid* solid){
            if(conditionallyAddHash(solid)){
                remove(solid);
                return true;
            }
            return false;
        }
    public:
        [[nodiscard]] auto size() const {
            return tableau.rows();
        }
        [[nodiscard]] auto cols() const {
            return tableau.cols();
        }
        [[nodiscard]] const auto& getSolidsPresent() const {
            return solidsPresent;
        }
        [[nodiscard]] const auto& getSolidsNotPresent() const {
            return solidsNotPresent;
        }
        [[nodiscard]] const auto& getTableau() const& {
            return tableau;
        }
        [[nodiscard]] auto getTableau() && {
            return std::move(tableau);
        }

        [[nodiscard]] Eigen::VectorXd calculateSolidAmts(const Eigen::RowVectorXd& leftOvers) const {
            Eigen::MatrixXd solidAmtEqns(solidsPresent.size(), numPresent);
            for(const Solid* solid : solidsPresent){
                solidAmtEqns.col(solidsPresent.presenceIndexOf(solid))=solid->row(solidsPresent.indexByColumn()).transpose();
            }
            const Eigen::VectorXd solidAmtLeftOver=leftOvers.transpose()(solidsPresent.indexByColumn());
            return solidAmtEqns.partialPivLu().solve(solidAmtLeftOver);
        }
        //auto in order to avoid circular dependency
        [[nodiscard]] Eigen::VectorXd calculateSolubilityProducts(const Eigen::RowVectorXd& currentSolution, const auto& replacementDict){
            auto replacedTableauNotPresent=replacementDict.createReplacedTableau(tableau.reducedCopy(solidsNotPresent.indexBySolidIndexes(), Eigen::all));
            if(replacedTableauNotPresent.cols()){
                return replacedTableauNotPresent.evalTerms(currentSolution);
            }
            return replacedTableauNotPresent.getConstants();
        }

        struct SolidChangeAttempt {
            bool success;
            Solid* solid;
        };
        
        [[nodiscard]] auto getSolidToRemove(const Eigen::VectorXd& solidAmts){
            Solid* solidNeedsToDisolve=nullptr;
            //we use a heap cause we usually wont need the full sort
            for(
                auto indexHeap=SolidVectorHeapIndexCompare<std::greater<>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solidAmts.rows()), SM_utils::IndexCompare<Eigen::VectorXd, std::greater<>, Eigen::Index>{solidAmts}); 
                Eigen::Index iThSolidPresent : SM_utils::ConsumingRange(indexHeap)
            ) {
                if(solidAmts.coeff(iThSolidPresent)<0.0) [[unlikely]] {
                    //not const because if we remove the solid then we're changing it
                    Solid* solid=solidsPresent.get(iThSolidPresent);
                    if(conditionallyRemove(solid)) [[likely]] {
                        return SolidChangeAttempt{true, solid};
                    }
                    if(solidNeedsToDisolve!=nullptr){
                        //std::cout<<"WARNING: EITHER NEARLY LOOPED OR GIBBS RULE FAILED (remove)"<<std::endl;
                        solidNeedsToDisolve=solid;
                    }
                } else {
                    return SolidChangeAttempt{false, solidNeedsToDisolve};
                }
            }
            return SolidChangeAttempt{false, solidNeedsToDisolve};
        }
        [[nodiscard]] auto getSolidToAdd(const Eigen::VectorXd& solubilityProducts){
            Solid* solidNeedsToForm=nullptr;

            //sort the solids in the wrong order so that we are much more likely to trigger a removal
            auto indexHeap=[&]() {
                if constexpr (SM_utils::debug) {
                    return SolidVectorHeapIndexCompare<std::greater<>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solubilityProducts.rows()), SM_utils::IndexCompare<Eigen::VectorXd, std::greater<>, Eigen::Index>{solubilityProducts});
                } else {
                    return SolidVectorHeapIndexCompare<std::less<>>(SM_utils::CountingIterator(0), SM_utils::CountingIterator(solubilityProducts.rows()), SM_utils::IndexCompare<Eigen::VectorXd, std::less<>, Eigen::Index>{solubilityProducts});
                }
            }();
            for(Eigen::Index iThSolidNotPresent : SM_utils::ConsumingRange(indexHeap)) {
                if(solubilityProducts.coeff(iThSolidNotPresent)>1.0) [[likely]] { //the long running cases will have lots of solids being added
                    Solid* solid=solidsNotPresent.get(iThSolidNotPresent);
                    if(conditionallyAdd(solid)) [[likely]] {
                        return SolidChangeAttempt{true, solid};
                    }
                    if(solidNeedsToForm!=nullptr){
                        //std::cout<<"WARNING: EITHER NEARLY LOOPED OR GIBBS RULE FAILED (add)"<<std::endl;
                        solidNeedsToForm=solid;
                    }
                } else {
                    if constexpr(!SM_utils::debug) {
                        //short circuit if we're going in the right order 
                        return SolidChangeAttempt{false, solidNeedsToForm};
                    }
                } 
            }
            return SolidChangeAttempt{false, solidNeedsToForm};
        }

        template<typename T=Tableau<>, typename SetType=std::unordered_set<Eigen::Index>>
        SolidSystem(T&& tableau_, const SetType& starting_solids):
            numPresent{0},
            tableau{std::forward<T>(tableau_)}
        {
            solidsPresent.reserve(size());
            solidsNotPresent.reserve(size());
            for(Eigen::Index i=0; i<tableau.rows(); ++i){
                const bool starting=starting_solids.find(i)!=starting_solids.end();
                const std::size_t hash=std::size_t(1)<<i;
                if(starting){
                    ++numPresent;
                    currentCombinationHash^=hash;
                    solidsPresent.emplace(std::unique_ptr<Solid>(new Solid{i, tableau.getCoefficients().row(i), tableau.getConstant(i), -1}));
                } else {
                    solidsNotPresent.emplace(std::unique_ptr<Solid>(new Solid{i, tableau.getCoefficients().row(i), tableau.getConstant(i), -1}));
                }
            }
            combinationsHash.insert(currentCombinationHash);
        }
    private:
        template<typename T=Tableau<>>
        SolidSystem(T&& tableau_, std::size_t currentCombinationHash_, SolidOwningSet&& solidsPresent_, SolidOwningSet&& solidsNotPresent_) : 
            numPresent{static_cast<Eigen::Index>(solidsPresent_.size())},
            tableau{std::forward<T>(tableau_)},
            solidsPresent{std::move(solidsPresent_)},
            solidsNotPresent{std::move(solidsNotPresent_)},
            combinationsHash{currentCombinationHash_},
            currentCombinationHash{currentCombinationHash_}
        { }
    public:
        [[nodiscard]] auto createNewWithInitialConditions() const {
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
} // namespace AQSystemSolver