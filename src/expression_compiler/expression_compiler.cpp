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
#include <iostream>

int main(int argc, char **argv)
{
    using namespace lossless_neural_sound;
    using namespace expression_compiler;
    Arena arena;
    expression::Nodes nodes;
    auto matrix_variable = nodes.intern(arena, expression::Matrix_variable("m", {{2, 3}}));
    auto variable = nodes.intern(arena, expression::Element_variable(matrix_variable, {{1, 2}}));
    auto expr = expression::Expression(arena, nodes, variable);
    expr.get()->dump(std::cout);
    expr *= 2;
    expr.get()->dump(std::cout);
    expr += 1;
    expr.get()->dump(std::cout);
    expr = expression::Expression::transfer_function(expr);
    expr *= expr;
    expr.get()->dump(std::cout);
    expr.get()->write_code(std::cout);
    expr = expr.get_derivative(variable);
    expr.get()->dump(std::cout);
    expr.get()->write_code(std::cout);
}
