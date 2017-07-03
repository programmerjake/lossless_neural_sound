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
#include <iostream>
#include "neural/neural.h"
#include <random>
#include <cmath>
#include <vector>

int main(int argc, char **argv)
{
    using namespace lossless_neural_sound;
    static constexpr std::size_t input_size = 20;
    static constexpr std::size_t output_size = 20;
    typedef neural::Neural_net<input_size, output_size> Neural_net;
    Neural_net neural_net;
    std::default_random_engine re;
    neural_net.initialize_to_random(re);
    constexpr std::size_t learn_step_count = 10000;
    for(std::size_t learn_step = 0; learn_step <= learn_step_count; learn_step++)
    {
        for(std::size_t input_index = 0; input_index < output_size; input_index++)
        {
            re.seed(input_index + 1);
            re.discard(20);
            Neural_net::Input_vector input(0);
            for(std::size_t i = 0; i < input_size; i++)
                input[i] = std::uniform_real_distribution<neural::Number_type>(-1, 1)(re);
            bool display = learn_step % (learn_step_count / 10) == 0;
            neural::Number_type step_size = 0.1;
            if(display)
            {
                std::cout << "input:\n" << input;
                auto output = neural_net.evaluate(input);
                std::cout << "output:\n" << output;
            }
            Neural_net::Output_vector correct_output(0);
            correct_output[input_index] = 0.25;
            auto learn_results =
                neural_net.learn(input, correct_output, step_size);
            if(display)
            {
                std::cout << "step_size: " << step_size
                          << "\n";
                std::cout << "initial_squared_error: " << learn_results.initial_squared_error
                          << "\n";
                std::cout << std::endl;
            }
        }
    }
}
