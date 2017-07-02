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
#ifndef NEURAL_NEURAL_H_
#define NEURAL_NEURAL_H_

#include <cstddef>
#include "util/constexpr_array.h"
#include <type_traits>
#include <cmath>
#include <cstdint>
#include <ratio>
#include <random>
#include <ostream>

namespace lossless_neural_sound
{
namespace neural
{
typedef float Number_type;
constexpr std::size_t number_type_write_precision = 7;

constexpr Number_type operator""_n(long double v) noexcept
{
    return v;
}

constexpr Number_type operator""_n(unsigned long long v) noexcept
{
    return v;
}

constexpr Number_type factorial(unsigned n) noexcept
{
    if(n <= 1)
        return 1_n;
    Number_type retval = 1_n;
    for(; n > 1; n--)
        retval *= static_cast<Number_type>(n);
    return retval;
}

template <unsigned Derivative_count = 0>
constexpr Number_type transfer_function(Number_type v) noexcept
{
    Number_type abs_v = v;
    Number_type sign = 1_n;
    if(v < 0)
    {
        abs_v = -v;
        sign = -1_n;
    }
    Number_type inv_abs_v_plus_1 = 1_n / (abs_v + 1_n);
    switch(Derivative_count)
    {
    case 0:
        return v / (1_n + abs_v);
    case 1:
        return inv_abs_v_plus_1 - abs_v * inv_abs_v_plus_1 * inv_abs_v_plus_1;
    case 2:
        return 2_n * v * (inv_abs_v_plus_1 * inv_abs_v_plus_1 * inv_abs_v_plus_1)
               - 2_n * sign * (inv_abs_v_plus_1 * inv_abs_v_plus_1);
    }
    static_assert(Derivative_count <= 2, "not implemented for higher derivatives");
}

template <std::size_t X, std::size_t Y>
class Matrix
{
private:
    util::Constexpr_array<util::Constexpr_array<Number_type, X>, Y> values;

public:
    static constexpr std::size_t width = X;
    static constexpr std::size_t height = Y;
    constexpr Number_type &operator()(std::size_t x, std::size_t y) noexcept
    {
        return values[y][x];
    }
    constexpr const Number_type &operator()(std::size_t x, std::size_t y) const noexcept
    {
        return values[y][x];
    }
    template <typename T = void>
    constexpr
        typename std::enable_if<std::is_void<T>::value && (X == 1) != (Y == 1), Number_type>::type &
        operator[](std::size_t index) noexcept
    {
        if(X == 1)
            return (*this)(0, index);
        return (*this)(index, 0);
    }
    template <typename T = void>
    constexpr const typename std::enable_if<std::is_void<T>::value && (X == 1) != (Y == 1),
                                            Number_type>::type &
        operator[](std::size_t index) const noexcept
    {
        if(X == 1)
            return (*this)(0, index);
        return (*this)(index, 0);
    }
    constexpr Matrix() noexcept : values{}
    {
    }
    template <typename T = void>
    constexpr Matrix(typename std::enable_if<std::is_void<T>::value && X == 1 && Y == 1,
                                             Number_type>::type v) noexcept : values{}
    {
        values[0][0] = v;
    }
    template <typename T = void>
    operator typename std::enable_if<std::is_void<T>::value && X == 1 && Y == 1,
                                     Number_type>::type() const noexcept
    {
        return values[0][0];
    }
    static constexpr Matrix zero() noexcept
    {
        Matrix retval;
        for(std::size_t y = 0; y < Y; y++)
            for(std::size_t x = 0; x < X; x++)
                retval(x, y) = 0_n;
        return retval;
    }
    static constexpr Matrix identity() noexcept
    {
        Matrix retval = zero();
        for(std::size_t i = 0; i < X && i < Y; i++)
            retval(i, i) = 0_n;
        return retval;
    }
    template <std::size_t Y2>
    constexpr Matrix<X, Y2> operator*(const Matrix<Y, Y2> &rt) const noexcept
    {
        Matrix<X, Y2> retval;
        for(std::size_t j = 0; j < Y2; j++)
        {
            for(std::size_t i = 0; i < X; i++)
            {
                Number_type value = (*this)(i, 0) * rt(0, j);
                for(std::size_t k = 1; k < Y; k++)
                {
                    value += (*this)(i, k) * rt(k, j);
                }
                retval(i, j) = value;
            }
        }
        return retval;
    }
    template <typename Retval = Matrix>
    constexpr typename std::enable_if<std::is_same<Retval, Matrix>::value
                                          && Retval::width == Retval::height,
                                      Matrix>::type &
        operator*=(const Matrix &rt) noexcept
    {
        *this = *this * rt;
        return *this;
    }
    constexpr Matrix &operator-=(const Matrix &rt) noexcept
    {
        for(std::size_t y = 0; y < Y; y++)
            for(std::size_t x = 0; x < X; x++)
                (*this)(x, y) -= rt(x, y);
        return *this;
    }
    constexpr Matrix<Y, X> get_transpose() const noexcept
    {
        Matrix<Y, X> retval;
        for(std::size_t y = 0; y < Y; y++)
            for(std::size_t x = 0; x < X; x++)
                retval(y, x) = (*this)(x, y);
        return retval;
    }
};

template <std::size_t X, std::size_t Y>
constexpr std::size_t Matrix<X, Y>::width;

template <std::size_t X, std::size_t Y>
constexpr std::size_t Matrix<X, Y>::height;

template <std::size_t X, std::size_t Y>
std::ostream &operator<<(std::ostream &os, const Matrix<X, Y> &v)
{
    for(std::size_t y = 0; y < Y; y++)
    {
        os << "| ";
        for(std::size_t x = 0; x < X; x++)
        {
            auto old_precision = os.precision(number_type_write_precision);
            auto old_width = os.width(number_type_write_precision + 5);
            try
            {
                os << v(x, y);
            }
            catch(...)
            {
                os.precision(old_precision);
                os.width(old_width);
                throw;
            }
            os.precision(old_precision);
            os.width(old_width);
            os << ' ';
        }
        os << "|\n";
    }
    return os;
}

template <std::size_t N>
using Row_vector = Matrix<N, 1>;

template <std::size_t N>
using Column_vector = Matrix<1, N>;

template <unsigned Derivative_count = 0, std::size_t X, std::size_t Y>
constexpr Matrix<X, Y> transfer_function(const Matrix<X, Y> &input) noexcept
{
    Matrix<X, Y> retval;
    for(std::size_t y = 0; y < Y; y++)
        for(std::size_t x = 0; x < X; x++)
            retval(x, y) = transfer_function<Derivative_count>(input(x, y));
    return retval;
}

template <std::size_t Input_size,
          std::size_t Output_size,
          std::size_t Hidden_layer_size = 2 * Input_size + 1>
struct Neural_net
{
    typedef Row_vector<Input_size> Input_vector;
    typedef Matrix<Hidden_layer_size, Input_size> Input_to_hidden_weights;
    typedef Row_vector<Hidden_layer_size> Hidden_vector;
    typedef Matrix<Output_size, Hidden_layer_size> Hidden_to_output_weights;
    typedef Row_vector<Output_size> Output_vector;
    Input_to_hidden_weights input_to_hidden_weights;
    Hidden_to_output_weights hidden_to_output_weights;
    template <typename Random_engine>
    void initialize_to_random(Random_engine &re)
    {
        std::uniform_real_distribution<Number_type> dist(-0.2_n, 0.2_n);
        for(std::size_t i = 0; i < Input_size; i++)
            for(std::size_t j = 0; j < Hidden_layer_size; j++)
                input_to_hidden_weights(j, i) = dist(re);
        for(std::size_t i = 0; i < Hidden_layer_size; i++)
            for(std::size_t j = 0; j < Output_size; j++)
                hidden_to_output_weights(j, i) = dist(re);
    }
    constexpr Output_vector evaluate(const Input_vector &input) const noexcept
    {
        Hidden_vector hidden = transfer_function(input_to_hidden_weights * input);
        return transfer_function(hidden_to_output_weights * hidden);
    }
    struct Learn_results
    {
        Number_type initial_squared_error;
    };
    constexpr Learn_results learn(const Input_vector &input,
                                  const Output_vector &correct_output,
                                  Number_type step_size) noexcept
    {
        Learn_results learn_results{};
        Hidden_vector hidden = transfer_function(input_to_hidden_weights * input);
        Output_vector initial_output = transfer_function(hidden_to_output_weights * hidden);
        Output_vector output_difference = initial_output - correct_output;
        learn_results.initial_squared_error = output_difference.get_transpose() * output_difference;
#warning finish implementing
        return learn_results;
    }
};
}
}

#endif // NEURAL_NEURAL_H_
