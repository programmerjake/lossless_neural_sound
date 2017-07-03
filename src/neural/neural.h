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
typedef double Number_type;
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
        typename std::enable_if<std::is_void<T2>::value && (Width == 1) != (Height == 1), T>::type &
        operator[](std::size_t index) noexcept
    {
        if(Height == 1)
            return (*this)(0, index);
        return (*this)(index, 0);
    }
    template <typename T2 = void>
    constexpr const typename std::enable_if<std::is_void<T2>::value
                                                && (Width == 1) != (Height == 1),
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
        return Matrix(0_n);
    }
    static constexpr Matrix identity() noexcept
    {
        Matrix retval = zero();
        for(std::size_t i = 0; i < Width && i < Height; i++)
            retval(i, i) = 1_n;
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
    friend std::ostream &operator<<(std::ostream &os, const Matrix &v)
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

template <unsigned Derivative_count = 0, std::size_t Width, std::size_t Height, typename T>
constexpr Matrix<Width, Height, decltype(transfer_function<Derivative_count>(std::declval<T>()))>
    transfer_function(const Matrix<Width, Height, T> &input) noexcept
{
    Matrix<Width, Height, decltype(transfer_function<Derivative_count>(std::declval<T>()))> retval;
    for(std::size_t i = 0; i < Height; i++)
        for(std::size_t j = 0; j < Width; j++)
            retval(i, j) = transfer_function<Derivative_count>(input(i, j));
    return retval;
}

template <std::size_t Input_size,
          std::size_t Output_size,
          std::size_t Hidden_layer_size = 2 * Input_size + 1>
struct Neural_net
{
    typedef Row_vector<Input_size, Number_type> Input_vector;
    typedef Matrix<Hidden_layer_size, Input_size, Number_type> Input_to_hidden_weights;
    typedef Row_vector<Hidden_layer_size, Number_type> Hidden_vector;
    typedef Matrix<Output_size, Hidden_layer_size, Number_type> Hidden_to_output_weights;
    typedef Row_vector<Output_size, Number_type> Output_vector;
    Input_to_hidden_weights input_to_hidden_weights;
    Hidden_to_output_weights hidden_to_output_weights;
    template <typename Random_engine>
    void initialize_to_random(Random_engine &re)
    {
        std::uniform_real_distribution<Number_type> dist(-0.2_n, 0.2_n);
        for(std::size_t i = 0; i < Input_size; i++)
            for(std::size_t j = 0; j < Hidden_layer_size; j++)
                input_to_hidden_weights(i, j) = dist(re);
        for(std::size_t i = 0; i < Hidden_layer_size; i++)
            for(std::size_t j = 0; j < Output_size; j++)
                hidden_to_output_weights(i, j) = dist(re);
    }
    template <typename Input_number_type, typename Input_to_hidden_number_type>
    static constexpr auto input_to_hidden(
        const Row_vector<Input_size, Input_number_type> &input,
        const Matrix<Hidden_layer_size, Input_size, Input_to_hidden_number_type>
            &input_to_hidden_weights) noexcept
    {
        return transfer_function(input * input_to_hidden_weights);
    }
    constexpr Hidden_vector input_to_hidden(const Input_vector &input) const noexcept
    {
        return input_to_hidden(input, input_to_hidden_weights);
    }
    template <typename Hidden_number_type, typename Hidden_to_output_number_type>
    static constexpr auto hidden_to_output(
        const Row_vector<Hidden_layer_size, Hidden_number_type> &hidden,
        const Matrix<Output_size, Hidden_layer_size, Hidden_to_output_number_type>
            &hidden_to_output_weights) noexcept
    {
        return transfer_function(hidden * hidden_to_output_weights);
    }
    constexpr Output_vector hidden_to_output(const Hidden_vector &hidden) const noexcept
    {
        return hidden_to_output(hidden, hidden_to_output_weights);
    }
    template <typename Input_number_type,
              typename Input_to_hidden_number_type,
              typename Hidden_to_output_number_type>
    static constexpr auto evaluate(
        const Row_vector<Input_size, Input_number_type> &input,
        const Matrix<Hidden_layer_size, Input_size, Input_to_hidden_number_type>
            &input_to_hidden_weights,
        const Matrix<Output_size, Hidden_layer_size, Hidden_to_output_number_type>
            &hidden_to_output_weights) noexcept
    {
        auto hidden = input_to_hidden(input, input_to_hidden_weights);
        return hidden_to_output(hidden, hidden_to_output_weights);
    }
    constexpr Output_vector evaluate(const Input_vector &input) const noexcept
    {
        return evaluate(input, input_to_hidden_weights, hidden_to_output_weights);
    }
    struct Learn_results
    {
        Number_type initial_squared_error;

        // derivative of initial_squared_error with respect to each element of
        // hidden_to_output_weights
        Hidden_to_output_weights hidden_to_output_weight_derivatives;

        // derivative of initial_squared_error with respect to each element of
        // input_to_hidden_weights
        Input_to_hidden_weights input_to_hidden_weight_derivatives;
    };
    constexpr Learn_results learn(const Input_vector &input,
                                  const Output_vector &correct_output,
                                  Number_type step_size) noexcept
    {
        Learn_results learn_results{};
        Hidden_vector hidden_input = input * input_to_hidden_weights;
        Hidden_vector hidden = transfer_function(hidden_input);
        Output_vector output_input = hidden * hidden_to_output_weights;
        Output_vector output = transfer_function(output_input);
        Output_vector output_difference = output - correct_output;
        learn_results.initial_squared_error = output_difference * output_difference.get_transpose();

        Output_vector dtf_output = transfer_function<1>(output_input);
        Output_vector dtf_output_times_output_difference_times_2 =
            2_n * dtf_output.elementwise_product(output_difference);
        learn_results.hidden_to_output_weight_derivatives =
            hidden.get_transpose() * dtf_output_times_output_difference_times_2;
        learn_results.input_to_hidden_weight_derivatives =
            input.get_transpose()
            * transfer_function<1>(hidden_input)
                  .elementwise_product(
                      (hidden_to_output_weights
                       * dtf_output_times_output_difference_times_2.get_transpose())
                          .get_transpose());
        input_to_hidden_weights -= step_size * learn_results.input_to_hidden_weight_derivatives;
        hidden_to_output_weights -= step_size * learn_results.hidden_to_output_weight_derivatives;
        return learn_results;
    }
};
}
}

#endif // NEURAL_NEURAL_H_


