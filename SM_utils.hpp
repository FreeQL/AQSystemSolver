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
        [[nodiscard]] auto begin(){
            return vect.begin();
        }
        [[nodiscard]] auto begin() const {
            return vect.begin();
        }
        [[nodiscard]] auto cbegin() const {
            return vect.cbegin();
        }
        [[nodiscard]] auto end(){
            return vect.end();
        }
        [[nodiscard]] auto end() const {
            return vect.end();
        }
        [[nodiscard]] auto cend() const {
            return vect.cend();
        }
        [[nodiscard]] auto data(){
            return vect.data();
        }
        [[nodiscard]] auto data() const {
            return vect.data();
        }
        [[nodiscard]] auto& operator[](std::size_t i) {
            return vect[i];
        }
        [[nodiscard]] const auto& operator[](std::size_t i) const {
            return vect[i];
        }
        [[nodiscard]] auto size() const {
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
        [[nodiscard]] auto find(const U& t) {
            const auto ret = std::lower_bound(vect.begin(), vect.end(), t, compare);
            if(ret!=vect.end() && !compare(*ret, t) && !compare(t, *ret)){
                return ret;
            }
            return vect.end();
        }
        template<typename U>
        [[nodiscard]] auto find(const U& t) const {
            const auto ret = std::lower_bound(vect.begin(), vect.end(), t, compare);
            if(ret!=vect.end() && !compare(*ret, t) && !compare(t, *ret)){
                return ret;
            }
            return vect.end();
        }
        [[nodiscard]] auto contains(const T& t) const{
            return std::binary_search(vect.begin(), vect.end(), t, compare);
        }
        flat_set()=default;
        flat_set(auto begin, auto end): vect(begin, end) {}
    };

    template<typename T, typename Enable = std::void_t<>>
    struct is_pointer_fancy_impl : std::false_type {};
    
    template<typename T>
    struct is_pointer_fancy_impl<T, std::void_t<typename T::element_type>> : std::true_type {};

    template<typename T>
    struct is_pointer_fancy_impl<T*> : std::true_type {}; 

    template<typename T>
    struct is_pointer_fancy : is_pointer_fancy_impl<std::decay_t<T>> {};

    template<typename T>
    constexpr inline bool is_pointer_fancy_v=is_pointer_fancy<T>::value;

    template<typename OriginalIterator>
    class UnowningIterator {
    private:
        using pointer = typename OriginalIterator::value_type::element_type**;
        using value_type = typename OriginalIterator::value_type::element_type*;
        using reference = typename OriginalIterator::value_type::element_type*;
        using difference_type = typename OriginalIterator::difference_type;
        using iterator_category = typename OriginalIterator::iterator_category;
        OriginalIterator inner_it;
    public:
        [[nodiscard]] value_type operator*() const  {
            return inner_it->get();
        }
        auto operator++() {
            return UnowningIterator{++inner_it};
        }
        auto operator++(int) & {
            return UnowningIterator{inner_it++};
        }
        auto operator+(std::size_t rhs) {
            return UnowningIterator{inner_it+rhs};
        }
        auto operator-(std::size_t rhs) {
            return UnowningIterator{inner_it-rhs};
        }
        [[nodiscard]] difference_type operator-(const UnowningIterator& rhs) const {
            return inner_it-rhs.inner_it;
        }
        [[nodiscard]] bool operator!=(const UnowningIterator& rhs) const {
            return inner_it!=rhs.inner_it;
        }
        explicit UnowningIterator(OriginalIterator inner_it_): inner_it{inner_it_} {}
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
        [[nodiscard]] auto& operator*() {
            return outerArray[iterator::operator*()];
        }
        [[nodiscard]] const auto& operator*() const {
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
        explicit ConsumingIterator(ContainerType& container_) : container{container_} {}
        [[nodiscard]] auto operator*() const {
            return container.top();
        }
        void operator++(){
            return container.pop();
        }
        [[nodiscard]] bool operator!=([[maybe_unused]] ConsumingIterator<ContainerType>& end) const {
            return container.size();
        }
    };


    template</*priority_queue*/ typename ContainerType>
    class ConsumingRange{
    private:
        ContainerType& container;
    public:
        explicit ConsumingRange(ContainerType& container_) : container{container_} { }
        
        [[nodiscard]] auto begin() const {
            return ConsumingIterator<ContainerType>{container};
        }

        [[nodiscard]] auto end() const {
            return ConsumingIterator<ContainerType>{container};
        }
    };

    //intended usage is to top and pop and then occassionally reinsert numbers that have been already popped (so all numbers inserted are less than max)
    template</*std::integral*/typename T>
    class IncreasingPQ{
    private:
        T max;
        std::priority_queue<T, std::vector<T>, std::greater<T>> reinserted;
    public:
        [[nodiscard]] const auto& top() const {
            if(reinserted.empty()){
                return max;
            }
            return reinserted.top();
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
        explicit IncreasingPQ(T starting) : max{starting} {}
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
        [[nodiscard]] auto operator+(const std::size_t val) const {
            return CountingIterator{value+val};
        };
        void operator+=(const std::size_t val) {
            value+=val;
        };
        [[nodiscard]] difference_type operator-(const CountingIterator& rhs) const{
            return value-rhs.value;
        }
        [[nodiscard]] bool operator==(CountingIterator const& it) const {
            return value==it.value;
        }
        [[nodiscard]] bool operator!=(CountingIterator const& it) const {
            //std::cout<<value<<" "<<it.value<<std::endl;
            return value!=it.value;
        }
        [[nodiscard]] value_type operator*() const {
            return value;
        }
    };

    template</*std::ranges::random_access_range*/typename ContainerType, typename Compare=std::less<>, /*std::integral*/typename IndexType=std::size_t>
    class IndexCompare{
    private:
        const ContainerType& container;
        const Compare compare={};
    public:
        explicit IndexCompare(const ContainerType& container_) : container{container_} { }
        [[nodiscard]] auto operator()(IndexType a, IndexType b) const{
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

    #ifdef NDEBUG
        inline constexpr bool debug=false;
    #else 
        inline constexpr bool debug=true;
    #endif
} // namespace SM_utils