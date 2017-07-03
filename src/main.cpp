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
#include "util/scheduler.h"
#include "util/constexpr_array.h"
#include "audio/audio.h"
#include <random>
#include <cmath>
#include <mutex>
#include <vector>
#include <chrono>
#include <fstream>
#include <typeinfo>

int main(int argc, char **argv)
{
    using namespace lossless_neural_sound;
#if 0
    util::Scheduler scheduler;
    static constexpr std::size_t input_size = 64;
    static constexpr std::size_t output_size = 50;
    typedef neural::Neural_net<input_size, output_size> Neural_net;
    Neural_net neural_net;
    {
        std::default_random_engine re;
        neural_net.initialize_to_random(re);
    }
    constexpr std::size_t learn_step_count = 10000;
    constexpr std::size_t input_count = output_size;
    std::size_t input_partitions = scheduler.get_thread_count() * 2;
    std::vector<std::future<void>> futures;
    futures.resize(input_partitions);
    for(std::size_t learn_step_index = 0; learn_step_index <= learn_step_count; learn_step_index++)
    {
        bool display = learn_step_index % (learn_step_count / 10) == 0;
        Neural_net::Learn_step total_learn_step;
        std::mutex total_learn_step_lock;
        for(std::size_t input_partition = 0; input_partition < input_partitions; input_partition++)
        {
            auto task = [&, input_partition]() -> void
            {
                std::size_t start_input_index = input_partition * input_count / input_partitions;
                std::size_t end_input_index =
                    (input_partition + 1) * input_count / input_partitions;
                Neural_net::Learn_step partition_total_learn_step;
                for(std::size_t input_index = start_input_index; input_index < end_input_index;
                    input_index++)
                {
                    std::default_random_engine re(input_index + 1);
                    re.discard(20);
                    Neural_net::Input_vector input(0);
                    for(std::size_t i = 0; i < input_size; i++)
                        input[i] = std::uniform_real_distribution<neural::Number_type>(-1, 1)(re);
                    Neural_net::Output_vector correct_output(0);
                    correct_output[input_index % output_size] = 0.25;
                    partition_total_learn_step += neural_net.get_learn_step(input, correct_output);
                }
                std::unique_lock<std::mutex> lock(total_learn_step_lock);
                total_learn_step += partition_total_learn_step;
            };
            futures[input_partition] = scheduler.queue_task(task);
        }
        for(auto &future : futures)
            future.get();
        neural_net.apply_learn_step(
            total_learn_step,
            static_cast<neural::Number_type>(learn_step_count - learn_step_index) / learn_step_count
                / input_count);
        if(display)
        {
            std::cout << "total squared error: " << total_learn_step.squared_error << "\n";
            std::cout << "rms error per output: "
                      << std::sqrt(total_learn_step.squared_error / input_count / output_size)
                      << "\n";
            std::cout << std::endl;
        }
    }
    auto elapsed_time = std::chrono::steady_clock::duration::zero();
    auto target_duration = std::chrono::seconds(2);
    std::size_t evaluate_count = 1;
    while(elapsed_time < target_duration)
    {
        evaluate_count *= 2;
        auto start_time = std::chrono::steady_clock::now();
        for(std::size_t i = 0; i < evaluate_count; i++)
        {
            Neural_net::Input_vector input(0);
            asm volatile("" ::"r"(&input) : "memory");
            auto result = neural_net.evaluate(input);
            asm volatile("" ::"r"(&result) : "memory");
        }
        auto end_time = std::chrono::steady_clock::now();
        elapsed_time = end_time - start_time;
    }
    std::cout << evaluate_count
                     / std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_time)
                           .count()
              << " evaluations per second" << std::endl;
#else
    try
    {
        auto input_stream =
            std::unique_ptr<audio::Input_stream>(new audio::File_input_stream("Fluids.flac"));
        std::unique_ptr<audio::Audio_reader> audio_reader;
        for(auto *format : audio::Audio_formats::get())
        {
            std::cerr << "trying " << format->name << std::endl;
            try
            {
                audio_reader = format->create_reader(std::move(input_stream));
            }
            catch(audio::Audio_error &e)
            {
                if(e.code() == audio::Audio_error_code::format_does_not_match)
                {
                    dynamic_cast<audio::Rewindable_input_stream &>(*input_stream).rewind();
                    continue;
                }
                throw;
            }
            break;
        }
        if(!audio_reader)
        {
            std::cerr << "can't read: all formats failed" << std::endl;
            return 1;
        }
        std::cout << "channel_count: "
                  << static_cast<std::size_t>(audio_reader->get_channel_count()) << "\n";
        std::cout << "sample_rate: " << static_cast<std::size_t>(audio_reader->get_sample_rate())
                  << "\n";
        std::ofstream os("output.bin", std::ios::binary);
        std::vector<float> buffer(audio_reader->get_ideal_buffer_size_in_samples());
        while(true)
        {
            auto read_count = audio_reader->read_samples(buffer.data(), buffer.size());
            if(read_count == 0)
                break;
            os.write(reinterpret_cast<const char *>(buffer.data()), read_count * sizeof(buffer[0]));
        }
    }
    catch(std::exception &e)
    {
        std::cerr << "caught: " << typeid(e).name() << ":\nwhat: " << e.what() << std::endl;
        return 1;
    }
#endif
}
