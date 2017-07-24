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
#include <fstream>
#include <vector>

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

constexpr std::size_t constexpr_pow(std::size_t base, std::size_t exponent) noexcept
{
    std::size_t retval = 1;
    for(std::size_t i = 0; i < exponent; i++)
    {
        std::size_t product = retval * base;
        assert(product / base == retval); // assert we didn't overflow
        retval = product;
    }
    return retval;
}

constexpr const char *output_type_name = "float";

struct Output_type_declaration
{
    std::string type;
    std::string variable_prefix;
    std::string variable_suffix;
    std::string operator()(std::string variable_name) const
    {
        constexpr const char *separator_string = " ";
        constexpr std::size_t separator_length = 1;
        variable_name.reserve(type.size() + separator_length + variable_prefix.size()
                              + variable_name.size()
                              + variable_suffix.size());
        return type + (separator_string + (variable_prefix + std::move(variable_name)))
               + variable_suffix;
    }
};

template <typename T>
struct Get_output_type_declaration
{
    Output_type_declaration operator()() const = delete;
};

template <typename T>
struct Get_output_type_declaration<const T> : public Get_output_type_declaration<T>
{
};

template <>
struct Get_output_type_declaration<expression::Expression>
{
    Output_type_declaration operator()() const
    {
        Output_type_declaration retval;
        retval.type = output_type_name;
        return retval;
    }
};

static constexpr util::Constexpr_array<char, std::numeric_limits<std::size_t>::digits10 + 1>
    constexpr_to_string_constant(std::size_t value) noexcept
{
    if(value < 10)
        return {{static_cast<char>(value + '0'), '\0'}};
    if(value < 100)
        return {{static_cast<char>(value / 10 + '0'), static_cast<char>(value % 10 + '0'), '\0'}};
    auto retval = constexpr_to_string_constant(value / 100);
    value %= 100;
    char *ptr = retval.data();
    while(*ptr)
        ptr++;
    *ptr++ = value / 10 + '0';
    *ptr++ = value % 10 + '0';
    *ptr = '\0';
    return retval;
}

template <typename T, std::size_t N>
struct Get_output_type_declaration<T[N]>
{
    Output_type_declaration operator()() const
    {
        Output_type_declaration retval = Get_output_type_declaration<T>()();
        constexpr auto number_str = constexpr_to_string_constant(N);
        retval.variable_suffix += '[';
        retval.variable_suffix += number_str.data();
        retval.variable_suffix += ']';
        return retval;
    }
};

template <typename T, std::size_t Width, std::size_t Height>
struct Get_output_type_declaration<math::Matrix<Width, Height, T>>
{
    Output_type_declaration operator()() const
    {
        constexpr auto width_str = constexpr_to_string_constant(Width);
        constexpr auto height_str = constexpr_to_string_constant(Height);
        Output_type_declaration retval = Get_output_type_declaration<T>()();
        retval.type =
            "math::Matrix<"
            + (width_str.data() + (", " + (height_str.data() + (", " + std::move(retval.type)))))
            + '>';
        return retval;
    }
};
}

int main(int argc, char **argv)
{
    Arena arena;
    expression::Nodes nodes;
    constexpr std::size_t layer_count = 3;
    constexpr std::size_t unit_size = 8;
    static_assert(layer_count >= 2, "");
    constexpr std::size_t input_size = constexpr_pow(unit_size, layer_count - 1);
    typedef Expression_row_vector<input_size> Input_vector;
    constexpr std::size_t unit_count = input_size / unit_size;
    typedef Expression_row_vector<unit_size> Unit_input_output_vector;
    constexpr std::size_t unit_hidden_size = 2 * unit_size + 1;
    typedef Expression_matrix<unit_hidden_size, unit_size> Unit_input_to_hidden_matrix;
    typedef Expression_matrix<unit_size, unit_hidden_size> Unit_hidden_to_output_matrix;
    constexpr std::size_t output_size = 1;
    typedef Expression_row_vector<output_size> Output_vector;
    const auto input =
        make_row_vector_variable(arena, nodes, "input", static_cast<const Input_vector *>(nullptr));
    const auto correct_output = make_row_vector_variable(
        arena, nodes, "correct_output", static_cast<const Output_vector *>(nullptr));
    Unit_input_to_hidden_matrix unit_input_to_hidden_matrixes[layer_count][unit_count];
    Unit_hidden_to_output_matrix unit_hidden_to_output_matrixes[layer_count][unit_count];
    auto layer_output = input;
    for(std::size_t layer = 0; layer < layer_count; layer++)
    {
        static_assert(input_size % unit_size == 0, "");
        auto layer_input = layer_output;
        for(std::size_t unit = 0; unit < unit_count; unit++)
        {
            Unit_input_output_vector unit_input;
            for(std::size_t i = 0; i < unit_size; i++)
                unit_input[i] = layer_input[unit * unit_size + i]; // load with stride 1
            auto &unit_input_to_hidden_matrix = unit_input_to_hidden_matrixes[layer][unit];
            auto &unit_hidden_to_output_matrix = unit_hidden_to_output_matrixes[layer][unit];
            {
                std::ostringstream ss;
                ss << "input_to_hidden[" << layer << "][" << unit << "]";
                unit_input_to_hidden_matrix =
                    make_matrix_variable(arena, nodes, ss.str(), &unit_input_to_hidden_matrix);
            }
            {
                std::ostringstream ss;
                ss << "hidden_to_output[" << layer << "][" << unit << "]";
                unit_hidden_to_output_matrix =
                    make_matrix_variable(arena, nodes, ss.str(), &unit_hidden_to_output_matrix);
            }
            auto unit_hidden =
                transfer_function(arena, nodes, unit_input * unit_input_to_hidden_matrix);
            auto unit_output =
                transfer_function(arena, nodes, unit_hidden * unit_hidden_to_output_matrix);
            for(std::size_t i = 0; i < unit_size; i++)
                layer_output[unit + i * unit_count] =
                    unit_output[i]; // apply butterfly permutation by storing with stride unit_count
        }
    }
    static_assert(output_size <= unit_size * unit_count, "");
    Output_vector output;
    for(std::size_t i = 0; i < output_size; i++)
        output[i] = layer_output[i];
    const auto output_difference = output - correct_output;
    const expression::Expression error_squared =
        output_difference * output_difference.get_transpose();
    expression::Expression_node::Code_writing_options code_writing_options;
    code_writing_options.indent = "    ";
    code_writing_options.value_type = output_type_name;
    std::string function_str =
        "eval(" + Get_output_type_declaration<decltype(input)>()()("input") + ",\n    "
        + Get_output_type_declaration<decltype(unit_input_to_hidden_matrixes)>()()(
              "unit_input_to_hidden_matrixes")
        + ",\n    " + Get_output_type_declaration<decltype(unit_hidden_to_output_matrixes)>()()(
                          "unit_hidden_to_output_matrixes")
        + ")";
    std::ofstream os;
    os.open("output_eval.h");
    os << "constexpr " << Get_output_type_declaration<decltype(output)>()()(std::move(function_str));
    os << "\n{\n    " << Get_output_type_declaration<decltype(output)>()()("output") << ";\n";
    {
        expression::Expression_node::Code_writing_state code_writing_state(code_writing_options);
        write_code(os, code_writing_state, arena, nodes, output);
        write_variable_assignments(os, code_writing_state, arena, nodes, "output", output);
    }
    os << R"(    return output;
}
)";
    os.close();
    std::cout << "wrote eval" << std::endl;
    Unit_input_to_hidden_matrix unit_input_to_hidden_derivative_matrixes[layer_count][unit_count];
    Unit_hidden_to_output_matrix unit_hidden_to_output_derivative_matrixes[layer_count][unit_count];
    for(std::size_t layer = 0; layer < layer_count; layer++)
    {
        for(std::size_t unit = 0; unit < unit_count; unit++)
        {
            unit_input_to_hidden_derivative_matrixes[layer][unit] =
                get_derivative(error_squared, unit_input_to_hidden_matrixes[layer][unit]);
            unit_hidden_to_output_derivative_matrixes[layer][unit] =
                get_derivative(error_squared, unit_hidden_to_output_matrixes[layer][unit]);
        }
    }
    os.open("output_learn.h");
    os << "{\n";
    {
        expression::Expression_node::Code_writing_state code_writing_state(code_writing_options);
        for(std::size_t layer = 0; layer < layer_count; layer++)
        {
            for(std::size_t unit = 0; unit < unit_count; unit++)
            {
                write_code(os,
                           code_writing_state,
                           arena,
                           nodes,
                           unit_input_to_hidden_derivative_matrixes[layer][unit]);
                write_code(os,
                           code_writing_state,
                           arena,
                           nodes,
                           unit_hidden_to_output_derivative_matrixes[layer][unit]);
            }
        }
        for(std::size_t layer = 0; layer < layer_count; layer++)
        {
            for(std::size_t unit = 0; unit < unit_count; unit++)
            {
                {
                    std::ostringstream ss;
                    ss << "input_to_hidden_derivatives[" << layer << "][" << unit << "]";
                    write_variable_assignments(
                        os,
                        code_writing_state,
                        arena,
                        nodes,
                        ss.str(),
                        unit_input_to_hidden_derivative_matrixes[layer][unit]);
                }
                {
                    std::ostringstream ss;
                    ss << "hidden_to_output_derivatives[" << layer << "][" << unit << "]";
                    write_variable_assignments(
                        os,
                        code_writing_state,
                        arena,
                        nodes,
                        ss.str(),
                        unit_hidden_to_output_derivative_matrixes[layer][unit]);
                }
            }
        }
    }
    os << "}" << std::endl;
    os.close();
    std::cout << "wrote learn" << std::endl;
}
