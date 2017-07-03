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
#ifndef UTIL_SCHEDULER_H_
#define UTIL_SCHEDULER_H_

#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <future>
#include <exception>

namespace lossless_neural_sound
{
namespace util
{
class Scheduler
{
    Scheduler(const Scheduler &) = delete;
    Scheduler &operator=(const Scheduler &) = delete;

private:
    struct Task
    {
        typedef void (*Run_function)(void *state);
        typedef void (*Destroy_function)(void *state);
        Run_function run_function;
        Destroy_function destroy_function;
        void *state;
        constexpr Task() noexcept : run_function(nullptr), destroy_function(nullptr), state(nullptr)
        {
        }
        constexpr Task(Run_function run_function,
                       Destroy_function destroy_function,
                       void *state) noexcept : run_function(run_function),
                                               destroy_function(destroy_function),
                                               state(state)
        {
        }
        Task(Task &&rt) noexcept : run_function(rt.run_function),
                                   destroy_function(rt.destroy_function),
                                   state(rt.state)
        {
            rt.run_function = nullptr;
            rt.destroy_function = nullptr;
            rt.state = nullptr;
        }
        Task &operator=(Task rt) noexcept
        {
            using std::swap;
            swap(run_function, rt.run_function);
            swap(destroy_function, rt.destroy_function);
            swap(state, rt.state);
            return *this;
        }
        ~Task()
        {
            if(destroy_function)
                destroy_function(state);
        }
        void run() noexcept
        {
            run_function(state);
        }
    };

private:
    std::mutex state_lock;
    std::condition_variable state_cond;
    std::deque<Task> tasks;
    bool quitting;
    std::deque<std::thread> threads;

private:
    void thread_fn() noexcept
    {
        std::unique_lock<std::mutex> lock(state_lock);
        while(!quitting)
        {
            if(tasks.empty())
            {
                state_cond.wait(lock);
                continue;
            }
            {
                Task task = std::move(tasks.front());
                tasks.pop_front();
                lock.unlock();
                task.run();
            }
            lock.lock();
        }
    }

public:
    Scheduler()
        : Scheduler([]() -> std::size_t
                    {
                        std::size_t retval = std::thread::hardware_concurrency();
                        if(retval == 0)
                            return 1;
                        return retval;
                    }())
    {
    }
    explicit Scheduler(std::size_t thread_count)
        : state_lock(), state_cond(), tasks(), quitting(false), threads()
    {
        for(std::size_t i = 0; i < thread_count; i++)
            threads.push_back(std::thread(&Scheduler::thread_fn, this));
    }
    ~Scheduler()
    {
        std::unique_lock<std::mutex> lock(state_lock);
        quitting = true;
        state_cond.notify_all();
        lock.unlock();
        for(auto &thread : threads)
            thread.join();
    }
    template <typename Fn>
    std::future<void> queue_task(Fn fn)
    {
        struct State
        {
            std::promise<void> promise;
            Fn fn;
            explicit State(Fn fn) : promise(), fn(std::move(fn))
            {
            }
            void run() noexcept
            {
                try
                {
                    fn();
                }
                catch(...)
                {
                    promise.set_exception(std::current_exception());
                    return;
                }
                promise.set_value();
            }
            static void run(void *state) noexcept
            {
                static_cast<State *>(state)->run();
            }
            static void destroy(void *state) noexcept
            {
                delete static_cast<State *>(state);
            }
        };
        State *state = new State(std::move(fn));
        Task task(&State::run, &State::destroy, static_cast<void *>(state));
        std::future<void> retval = state->promise.get_future();
        std::unique_lock<std::mutex> lock(state_lock);
        if(tasks.empty())
            state_cond.notify_all();
        tasks.push_back(std::move(task));
        lock.unlock();
        return retval;
    }
    std::size_t get_thread_count() const noexcept
    {
        return threads.size();
    }
};
}
}

#endif /* UTIL_SCHEDULER_H_ */
