#pragma once

#include <random>
#include <algorithm>
#include <utility>
#include <set>
#include <vector>
#include <queue>
#include <iostream>
#include <cassert>
//#include <ranges>


//random utilities, 
namespace SM_utils{

    template<typename T>
    using numeric=T;
    //concept numeric=std::integral<T> || std::floating_point<T>;

    template<typename T, typename Compare=std::less<T>>
    class flat_set {
        private:
            std::vector<T> vect;
            const Compare compare={};
        public:
            using iterator=typename std::vector<T>::iterator;
            using const_iterator=typename std::vector<T>::const_iterator;
            iterator begin(){
                return vect.begin();
            }
            const_iterator begin() const {
                return vect.begin();
            }
            auto cbegin() const {
                return vect.cbegin();
            }
            iterator end(){
                return vect.end();
            }
            const_iterator end() const {
                return vect.end();
            }
            auto cend() const {
                return vect.cend();
            }
            T* data(){
                return vect.data();
            }
            const T* data() const {
                return vect.data();
            }
            T& operator[](std::size_t i){
                return vect[i];
            }
            const T& operator[](std::size_t i) const {
                return vect[i];
            }
            std::size_t size() const {
                return vect.size();
            }
            void reserve(std::size_t s){
                vect.reserve(s);
            }
            template<typename... Args>
            void emplace(Args&&... args){
                T t(std::forward<Args>(args)...);
                vect.insert(std::lower_bound(vect.begin(), vect.end(), t, compare), std::move(t));
            }
            void insert(T&& t) {
                vect.insert(std::lower_bound(vect.begin(), vect.end(), t, compare), std::move(t));
            }
            void insert(const T& t) {
                vect.insert(std::lower_bound(vect.begin(), vect.end(), t, compare), t);
            }
            void erase(const T& t) {
                vect.erase(std::lower_bound(vect.begin(), vect.end(), t, compare));
            }
            void erase(iterator t) {
                vect.erase(t);
            }
            void erase(const_iterator t) {
                vect.erase(t);
            }
            template<typename U>
            iterator find(const U& t) {
                const auto ret = std::lower_bound(vect.begin(), vect.end(), t, compare);
                if(ret!=vect.end() && !compare(*ret, t) && !compare(t, *ret)){
                    return ret;
                } else {
                    return vect.end();
                }
            }
            template<typename U>
            const_iterator find(const U& t) const {
                const auto ret = std::lower_bound(vect.begin(), vect.end(), t, compare);
                if(ret!=vect.end() && !compare(*ret, t) && !compare(t, *ret)){
                    return ret;
                } else {
                    return vect.end();
                }
            }
            bool contains(const T& t) const{
                return std::binary_search(vect.begin(), vect.end(), t, compare);
            }
            flat_set()=default;
            flat_set(auto begin, auto end): vect(begin, end) {}
    };

    template<typename T, typename Enable = std::void_t<>>
    struct is_pointer_fancy : std::false_type {};
    
    template<class T>
    struct is_pointer_fancy<T, std::void_t<typename T::element_type>> : std::true_type {};

    template<class T>
    struct is_pointer_fancy<T*> : std::true_type {}; 

    template<typename T>
    constexpr inline bool is_pointer_fancy_v=is_pointer_fancy<T>::value;

    template<typename OriginalIterator>
    class UnowningIterator {
        using pointer = typename OriginalIterator::value_type::element_type**;
        using value_type = typename OriginalIterator::value_type::element_type*;
        using reference = typename OriginalIterator::value_type::element_type*;
        using difference_type = typename OriginalIterator::difference_type;
        using iterator_category = typename OriginalIterator::iterator_category;
        private:
            OriginalIterator inner_it;
        public:
            value_type operator*() {
                return inner_it->get();
            }
            const value_type& operator*() const {
                return inner_it->get();
            }
            UnowningIterator operator++(){
                return ++inner_it;
            }
            UnowningIterator operator++(int){
                return inner_it++;
            }
            UnowningIterator& operator+(std::size_t rhs){
                return inner_it+=rhs;
            }
            UnowningIterator& operator-(std::size_t rhs){
                return inner_it-=rhs;
            }
            std::ptrdiff_t operator-(const UnowningIterator& rhs){
                return inner_it-rhs.inner_it;
            }
            bool operator!=(const UnowningIterator& rhs) const{
                return inner_it!=rhs.inner_it;
            }
            UnowningIterator(OriginalIterator inner_it_): inner_it{inner_it_} {}
    };

    template</*std::ranges::random_access_range*/typename ContainerType, /*std::forward_iterator*/typename iterator>
    class NestingIterator: public iterator{
        public:
            using pointer = typename ContainerType::value_type*;
            using value_type = typename ContainerType::value_type;
            using reference = typename ContainerType::value_type&;
            using difference_type = typename iterator::difference_type;
            using iterator_category = typename iterator::iterator_category;
        private:
            ContainerType& outerArray;
        public:
            typename ContainerType::value_type operator*() const {
                return outerArray[iterator::operator*()];
            }
        NestingIterator(ContainerType& outerArray_, iterator currentLocation) : 
            iterator(currentLocation),
            outerArray{outerArray_}
            {}
    };
    //template<typename T>
    //concept priority_queue=requires(T c) { c.top(); c.pop(); c.size();};
    
    template</*priority_queue*/ typename ContainerType>
    class ConsumingIterator{
        private:
            ContainerType& container;
        public:
            ConsumingIterator(ContainerType& container_) : container{container_} {}
            const typename ContainerType::value_type& operator*() const {
                return container.top();
            }
            void operator++(){
                return container.pop();
            }
            bool operator!=([[maybe_unused]] ConsumingIterator<ContainerType>& end) const {
                return container.size();
            }
    };


    template</*priority_queue*/ typename ContainerType>
    class ConsumingRange{
        private:
            ContainerType& container;
        public:
            ConsumingRange(ContainerType& container_) : container{container_} { }
            
            ConsumingIterator<ContainerType> begin() const {
                return {container};
            }

            ConsumingIterator<ContainerType> end() const {
                return {container};
            }
    };

    //intended usage is to top and pop and then occassionally reinsert numbers that have been already popped (so all numbers inserted are less than max)
    template</*std::integral*/typename T>
    class IncreasingPQ{
        private:
            T max;
            std::priority_queue<T, std::vector<T>, std::greater<T>> reinserted;
        public:
            const T& top() const {
                if(reinserted.empty()){
                    return max;
                } else {
                    return reinserted.top();
                }
            }
            void pop(){
                if(reinserted.empty()){
                    ++max;
                } else {
                    reinserted.pop();
                }
            }
            void push(const T& value){
                assert(value<max);
                reinserted.push(value);
            }
            IncreasingPQ(T starting) : max{starting} {}
    };


    class CountingIterator{
        private:
            std::size_t value;
        public:
            using difference_type = std::ptrdiff_t;
            using value_type = std::size_t;
            using pointer = std::size_t*;
            using reference = std::size_t&;
            using iterator_category = std::random_access_iterator_tag;

            explicit CountingIterator(std::size_t initialValue) : value(initialValue){}
            void operator++(){
                ++value;
            };
            void operator--(){
                --value;
            };
            CountingIterator operator+(const std::size_t val){
                return CountingIterator(value+val);
            };
            void operator+=(const std::size_t val){
                value+=val;
            };
            difference_type operator-(const CountingIterator& rhs){
                return value-rhs.value;
            }
            bool operator==(CountingIterator const& it) const {
                return value==it.value;
            }
            bool operator!=(CountingIterator const& it) const {
                //std::cout<<value<<" "<<it.value<<std::endl;
                return value!=it.value;
            }
            std::size_t operator*() const {
                return value;
            }
    };

    template</*std::ranges::random_access_range*/typename ContainerType, typename Compare, /*std::integral*/typename IndexType>
    struct IndexCompare{
        const ContainerType& container;
        const Compare compare={};
        IndexCompare(const ContainerType& container_) : container{container_} { }
        bool operator()(IndexType a, IndexType b) const{
            return compare(container[a], container[b]);
        }
    };


    //template magic? template magic. Of course it's stack overflow. NOTE THAT IT's REVERSED
    //It goes is_base_template<ExpectedDerived, ExpectedBase>
    template <template <typename...> class C, typename...Ts>
    std::true_type is_base_of_template_impl(const C<Ts...>*);

    template <template <typename...> class C>
    std::false_type is_base_of_template_impl(...);

    template <typename T, template <typename...> class C>
    using is_base_of_template = decltype(is_base_of_template_impl<C>(std::declval<T*>()));

    void print_vect(const auto& v){
        for(const auto& elem : v){
            std::cout<<elem<<" ";
        }
        std::cout<<std::endl;
    }
    
}