/*
 * Copyright 2017 Jacob Lifshay
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#ifndef UTIL_ARRAY_STACK_H_
#define UTIL_ARRAY_STACK_H_

#include <cassert>
#include <utility>

namespace lossless_neural_sound
{
namespace util
{
template <typename T, std::size_t N>
class Array_stack
{
private:
    T array[N];
    std::size_t stack_top;

public:
    constexpr Array_stack() noexcept(noexcept(T{})) : array{}, stack_top(N)
    {
    }
    constexpr T &top() noexcept
    {
        assert(stack_top < N);
        return array[stack_top];
    }
    constexpr const T &top() const noexcept
    {
        assert(stack_top < N);
        return array[stack_top];
    }
    constexpr bool empty() const noexcept
    {
        return stack_top == N;
    }
    constexpr std::size_t size() const noexcept
    {
        return N - stack_top;
    }
    constexpr std::size_t max_size() const noexcept
    {
        return N;
    }
    constexpr void pop() noexcept
    {
        assert(stack_top < N);
        stack_top++;
    }
    constexpr void push(T &&value) noexcept(noexcept(std::declval<T &>() = std::declval<T &&>()))
    {
        assert(stack_top > 0);
        --stack_top;
        top() = std::move(value);
    }
    constexpr void push(const T &value) noexcept(noexcept(std::declval<T &>() = std::declval<const T &>()))
    {
        assert(stack_top > 0);
        --stack_top;
        top() = std::move(value);
    }
};
}
}

#endif // UTIL_ARRAY_STACK_H_
