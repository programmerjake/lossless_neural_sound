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

#include "expression.h"
#include "math/matrix.h"
#include <iostream>

using namespace lossless_neural_sound;
using namespace expression_compiler;

namespace
{
template <std::size_t Width, std::size_t Height>
using Expression_matrix = math::Matrix<Width, Height, expression::Expression>;

template <std::size_t Size>
using Expression_row_vector = math::Row_vector<Size, expression::Expression>;

template <std::size_t Size>
using Expression_column_vector = math::Column_vector<Size, expression::Expression>;

template <std::size_t Width, std::size_t Height, typename T = float>
inline Expression_matrix<Width, Height> make_matrix_variable(
    Arena &arena,
    expression::Nodes &nodes,
    std::string name,
    const math::Matrix<Width, Height, T> * = nullptr)
{
    auto matrix_variable =
        nodes.intern(arena, expression::Matrix_variable(std::move(name), {{Width, Height}}));
    Expression_matrix<Width, Height> retval;
    for(std::size_t i = 0; i < Height; i++)
        for(std::size_t j = 0; j < Width; j++)
            retval(i, j) = expression::Expression(
                arena,
                nodes,
                nodes.intern(arena, expression::Element_variable(matrix_variable, {{i, j}})));
    return retval;
}

template <std::size_t Size, typename T = float>
inline Expression_row_vector<Size> make_row_vector_variable(
    Arena &arena,
    expression::Nodes &nodes,
    std::string name,
    const math::Row_vector<Size, T> * = nullptr)
{
    return make_matrix_variable(
        arena, nodes, std::move(name), static_cast<const math::Row_vector<Size, float> *>(nullptr));
}

template <std::size_t Size, typename T = float>
inline Expression_column_vector<Size> make_column_vector_variable(
    Arena &arena,
    expression::Nodes &nodes,
    std::string name,
    const math::Column_vector<Size, T> * = nullptr)
{
    return make_matrix_variable(arena,
                                nodes,
                                std::move(name),
                                static_cast<const math::Column_vector<Size, float> *>(nullptr));
}

template <std::size_t Width, std::size_t Height>
inline Expression_matrix<Width, Height> get_derivative(expression::Expression value,
                                                       Expression_matrix<Width, Height> variables)
{
    for(std::size_t i = 0; i < Height; i++)
        for(std::size_t j = 0; j < Width; j++)
            variables(i, j) = value.get_derivative(variables(i, j));
    return variables;
}

template <std::size_t Width, std::size_t Height>
inline Expression_matrix<Width, Height> get_derivative(Expression_matrix<Width, Height> values,
                                                       expression::Expression variable)
{
    for(std::size_t i = 0; i < Height; i++)
        for(std::size_t j = 0; j < Width; j++)
            values(i, j) = values(i, j).get_derivative(variable);
    return values;
}

template <std::size_t derivative_count = 0, std::size_t Width, std::size_t Height>
inline Expression_matrix<Width, Height> transfer_function(Arena &arena,
                                                          expression::Nodes &nodes,
                                                          Expression_matrix<Width, Height> m)
{
    for(std::size_t i = 0; i < Height; i++)
        for(std::size_t j = 0; j < Width; j++)
            m(i, j) = transfer_function<derivative_count>(arena, nodes, m(i, j));
    return m;
}

template <std::size_t Width, std::size_t Height>
inline void write_code(std::ostream &os,
                       expression::Expression_node::Code_writing_state &code_writing_state,
                       Arena &arena,
                       expression::Nodes &nodes,
                       const Expression_matrix<Width, Height> &values)
{
    for(std::size_t i = 0; i < Height; i++)
    {
        for(std::size_t j = 0; j < Width; j++)
        {
            values(i, j).get(arena, nodes)->write_code(os, code_writing_state);
        }
    }
}

template <std::size_t Width, std::size_t Height>
inline void write_variable_assignments(
    std::ostream &os,
    expression::Expression_node::Code_writing_state &code_writing_state,
    Arena &arena,
    expression::Nodes &nodes,
    const Expression_matrix<Width, Height> &names,
    const Expression_matrix<Width, Height> &values)
{
    for(std::size_t i = 0; i < Height; i++)
    {
        for(std::size_t j = 0; j < Width; j++)
        {
            auto *variable = names(i, j).get(arena, nodes);
            assert(dynamic_cast<const expression::Variable *>(variable));
            auto name =
                static_cast<const expression::Variable *>(variable)->get_variable_code_name();
            code_writing_state.write_variable_assignment(os, name, values(i, j).get(arena, nodes));
        }
    }
}

template <std::size_t Width, std::size_t Height>
inline void write_variable_assignments(
    std::ostream &os,
    expression::Expression_node::Code_writing_state &code_writing_state,
    Arena &arena,
    expression::Nodes &nodes,
    std::string matrix_name,
    const Expression_matrix<Width, Height> &values)
{
    write_variable_assignments(os,
                               code_writing_state,
                               arena,
                               nodes,
                               make_matrix_variable(arena, nodes, std::move(matrix_name), &values),
                               values);
}
}

int main(int argc, char **argv)
{
    Arena arena;
    expression::Nodes nodes;
    constexpr std::size_t input_size = 16;
    typedef Expression_row_vector<input_size> Input_vector;
    constexpr std::size_t hidden1_size = 16;
    constexpr std::size_t hidden2_size = 16;
    constexpr std::size_t output_size = 16;
    typedef Expression_row_vector<output_size> Output_vector;
    const auto input =
        make_row_vector_variable(arena, nodes, "input", static_cast<const Input_vector *>(nullptr));
    const auto correct_output = make_row_vector_variable(
        arena, nodes, "correct_output", static_cast<const Output_vector *>(nullptr));
    const auto t1 = make_matrix_variable<hidden1_size, input_size>(arena, nodes, "t1");
    const auto t2 = make_matrix_variable<hidden2_size, hidden1_size>(arena, nodes, "t2");
    const auto t3 = make_matrix_variable<output_size, hidden2_size>(arena, nodes, "t3");
    const auto hidden1 = transfer_function(arena, nodes, input * t1);
    const auto hidden2 = transfer_function(arena, nodes, hidden1 * t2);
    const auto output = transfer_function(arena, nodes, hidden2 * t3);
    const auto output_difference = output - correct_output;
    const expression::Expression error_squared =
        output_difference * output_difference.get_transpose();
    const auto t1_derivatives = get_derivative(error_squared, t1);
    const auto t2_derivatives = get_derivative(error_squared, t2);
    const auto t3_derivatives = get_derivative(error_squared, t3);
    expression::Expression_node::Code_writing_options code_writing_options;
    code_writing_options.indent = "    ";
    code_writing_options.value_type = "float";
    std::cout << "{\n";
    {
        expression::Expression_node::Code_writing_state code_writing_state(code_writing_options);
        write_code(std::cout, code_writing_state, arena, nodes, output);
        write_variable_assignments(std::cout, code_writing_state, arena, nodes, "output", output);
    }
    std::cout << "}\n{\n";
    {
        expression::Expression_node::Code_writing_state code_writing_state(code_writing_options);
        write_code(std::cout, code_writing_state, arena, nodes, t1_derivatives);
        write_code(std::cout, code_writing_state, arena, nodes, t2_derivatives);
        write_code(std::cout, code_writing_state, arena, nodes, t3_derivatives);
        write_variable_assignments(
            std::cout, code_writing_state, arena, nodes, "t1_derivatives", t1_derivatives);
        write_variable_assignments(
            std::cout, code_writing_state, arena, nodes, "t2_derivatives", t2_derivatives);
        write_variable_assignments(
            std::cout, code_writing_state, arena, nodes, "t3_derivatives", t3_derivatives);
    }
    std::cout << "}\n";
}
