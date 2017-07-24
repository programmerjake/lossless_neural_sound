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
#ifndef MATH_MATRIX_H_
#define MATH_MATRIX_H_

#include "util/constexpr_array.h"
#include <ostream>
#include <type_traits>
#include <utility>

namespace lossless_neural_sound
{
namespace math
{
template <std::size_t Width, std::size_t Height, typename T>
class Matrix
{
private:
    util::Constexpr_array<util::Constexpr_array<T, Width>, Height> values;

public:
    static constexpr std::size_t width = Width;
    static constexpr std::size_t height = Height;
    constexpr T &operator()(std::size_t y, std::size_t x) noexcept
    {
        return values[y][x];
    }
    constexpr const T &operator()(std::size_t y, std::size_t x) const noexcept
    {
        return values[y][x];
    }
    template <typename T2 = void>
    constexpr
        typename std::enable_if<std::is_void<T2>::value && (Width == 1 || Height == 1), T>::type &
        operator[](std::size_t index) noexcept
    {
        if(Height == 1)
            return (*this)(0, index);
        return (*this)(index, 0);
    }
    template <typename T2 = void>
    constexpr const typename std::enable_if<std::is_void<T2>::value && (Width == 1 || Height == 1),
                                            T>::type &
        operator[](std::size_t index) const noexcept
    {
        if(Height == 1)
            return (*this)(0, index);
        return (*this)(index, 0);
    }
    constexpr Matrix() noexcept : values{}
    {
    }
    constexpr Matrix(T v) noexcept : values{}
    {
        for(std::size_t y = 0; y < Height; y++)
            for(std::size_t x = 0; x < Width; x++)
                (*this)(y, x) = v;
    }
    template <typename T2 = void>
    operator typename std::enable_if<std::is_void<T2>::value && Width == 1 && Height == 1,
                                     T>::type() const noexcept
    {
        return values[0][0];
    }
    static constexpr Matrix zero() noexcept
    {
        return Matrix(T(0));
    }
    static constexpr Matrix identity() noexcept
    {
        Matrix retval = zero();
        for(std::size_t i = 0; i < Width && i < Height; i++)
            retval(i, i) = T(1);
        return retval;
    }
    template <std::size_t Result_width,
              typename Result_type = decltype(std::declval<T>() * std::declval<T>()
                                              + std::declval<T>() * std::declval<T>())>
    constexpr Matrix<Result_width, Height, Result_type> operator*(
        const Matrix<Result_width, Width, T> &rt) const noexcept
    {
        Matrix<Result_width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
        {
            for(std::size_t j = 0; j < Result_width; j++)
            {
                Result_type value = (*this)(i, 0) * rt(0, j);
                for(std::size_t k = 1; k < Width; k++)
                {
                    value += (*this)(i, k) * rt(k, j);
                }
                retval(i, j) = value;
            }
        }
        return retval;
    }
    template <typename Result_type = decltype(std::declval<T>() * std::declval<T>())>
    constexpr Matrix<Width, Height, Result_type> operator*(T rt) const noexcept
    {
        Matrix<Width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(i, j) = (*this)(i, j) * rt;
        return retval;
    }
    template <typename Result_type = decltype(std::declval<T>() * std::declval<T>())>
    friend constexpr Matrix<Width, Height, Result_type> operator*(T l, const Matrix &r) noexcept
    {
        Matrix<Width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(i, j) = l * r(i, j);
        return retval;
    }
    template <typename T2, typename Result_type = decltype(std::declval<T>() * std::declval<T2>())>
    constexpr Matrix<Width, Height, Result_type> elementwise_product(
        const Matrix<Width, Height, T2> &rt) const noexcept
    {
        Matrix<Width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(i, j) = (*this)(i, j) * rt(i, j);
        return retval;
    }
    template <typename T2, typename Result_type = decltype(std::declval<T>() + std::declval<T2>())>
    constexpr Matrix<Width, Height, Result_type> operator+(
        const Matrix<Width, Height, T2> &rt) const noexcept
    {
        Matrix<Width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(i, j) = (*this)(i, j) + rt(i, j);
        return retval;
    }
    template <typename T2, typename Result_type = decltype(std::declval<T>() - std::declval<T2>())>
    constexpr Matrix<Width, Height, Result_type> operator-(
        const Matrix<Width, Height, T2> &rt) const noexcept
    {
        Matrix<Width, Height, Result_type> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(i, j) = (*this)(i, j) - rt(i, j);
        return retval;
    }
    template <typename Retval = Matrix>
    constexpr typename std::enable_if<std::is_same<Retval, Matrix>::value
                                          && std::is_same<decltype(std::declval<Retval>()
                                                                   * std::declval<Retval>()),
                                                          Matrix>::value,
                                      Matrix>::type &
        operator*=(const Matrix &rt) noexcept
    {
        *this = *this * rt;
        return *this;
    }
    template <typename Retval = Matrix>
    constexpr typename std::enable_if<std::is_same<Retval, Matrix>::value
                                          && std::is_same<decltype(std::declval<Retval>()
                                                                   * std::declval<T>()),
                                                          Matrix>::value,
                                      Matrix>::type &
        operator*=(T rt) noexcept
    {
        *this = *this * rt;
        return *this;
    }
    template <typename Retval = Matrix>
    constexpr typename std::enable_if<std::is_same<Retval, Matrix>::value
                                          && std::is_same<decltype(std::declval<Retval>()
                                                                   - std::declval<Retval>()),
                                                          Matrix>::value,
                                      Matrix>::type &
        operator-=(const Matrix &rt) noexcept
    {
        *this = *this - rt;
        return *this;
    }
    template <typename Retval = Matrix>
    constexpr typename std::enable_if<std::is_same<Retval, Matrix>::value
                                          && std::is_same<decltype(std::declval<Retval>()
                                                                   + std::declval<Retval>()),
                                                          Matrix>::value,
                                      Matrix>::type &
        operator+=(const Matrix &rt) noexcept
    {
        *this = *this + rt;
        return *this;
    }
    constexpr Matrix<Height, Width, T> get_transpose() const noexcept
    {
        Matrix<Height, Width, T> retval;
        for(std::size_t i = 0; i < Height; i++)
            for(std::size_t j = 0; j < Width; j++)
                retval(j, i) = (*this)(i, j);
        return retval;
    }
    template <typename Char_type, typename Traits>
    friend std::basic_ostream<Char_type, Traits> &operator<<(
        std::basic_ostream<Char_type, Traits> &os, const Matrix &v)
    {
        for(std::size_t i = 0; i < Height; i++)
        {
            os << "| ";
            for(std::size_t j = 0; j < Width; j++)
            {
                os << v(i, j);
                os << ' ';
            }
            os << "|\n";
        }
        return os;
    }
};

template <std::size_t Width, std::size_t Height, typename T>
constexpr std::size_t Matrix<Width, Height, T>::width;

template <std::size_t Width, std::size_t Height, typename T>
constexpr std::size_t Matrix<Width, Height, T>::height;

template <std::size_t N, typename T>
using Row_vector = Matrix<N, 1, T>;

template <std::size_t N, typename T>
using Column_vector = Matrix<1, N, T>;
}
}

#endif // MATH_MATRIX_H_
