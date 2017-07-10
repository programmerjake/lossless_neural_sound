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
#ifndef EXPRESSION_COMPILER_ARENA_H_
#define EXPRESSION_COMPILER_ARENA_H_

#include <deque>
#include <cstddef>
#include <memory>
#include <utility>
#include <cassert>
#include <type_traits>

namespace lossless_neural_sound
{
namespace expression_compiler
{
class Arena final
{
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;

private:
    struct Node_base
    {
        virtual ~Node_base() = default;
    };
    template <typename T>
    struct Node final : public Node_base
    {
        T value;
        template <typename... Args>
        explicit Node(Args &&... args)
            : value(std::forward<Args>(args)...)
        {
        }
    };
    static constexpr std::size_t memory_block_alignment = alignof(std::max_align_t);
    static constexpr std::size_t memory_block_size = 0x1000;
    static constexpr std::size_t big_memory_block_start_size = memory_block_size / 4;
    static_assert(memory_block_size >= memory_block_alignment
                      && memory_block_size % memory_block_alignment == 0,
                  "");
    struct Memory_block
    {
        alignas(memory_block_alignment) unsigned char bytes[memory_block_size] = {};
        std::size_t used = 0;
        template <std::size_t size, std::size_t alignment>
        void *allocate() noexcept
        {
            std::size_t start_index =
                (used + (alignment - 1)) & ~static_cast<std::size_t>(alignment - 1);
            if(start_index + size > memory_block_size)
                return nullptr;
            void *retval = bytes + start_index;
            used = start_index + size;
            return retval;
        }
    };
    struct Destroy_only
    {
        void operator()(Node_base *node) noexcept
        {
            node->~Node_base();
        }
    };

private:
    std::deque<Memory_block> memory_blocks;
    std::deque<std::unique_ptr<Node_base, Destroy_only>> small_nodes;
    std::deque<std::unique_ptr<Node_base>> big_nodes;

private:
    static constexpr bool is_nonzero_power_of_2(std::size_t v) noexcept
    {
        return v != 0 && (v & (v - 1U)) == 0;
    }
    template <typename T>
    void *allocate()
    {
        static_assert(alignof(T) <= memory_block_alignment, "");
        static_assert(is_nonzero_power_of_2(alignof(T)), "");
        assert(sizeof(T) < big_memory_block_start_size);
        void *retval = memory_blocks.empty() ?
                           nullptr :
                           memory_blocks.back().allocate<sizeof(T), alignof(T)>();
        if(!retval)
        {
            memory_blocks.emplace_back();
            retval = memory_blocks.back().allocate<sizeof(T), alignof(T)>();
        }
        assert(retval);
        return retval;
    }

public:
    Arena() noexcept
    {
    }
    ~Arena()
    {
        small_nodes.clear(); // must clear before memory is deallocated by memory_blocks
    }
    template <typename T, typename... Args>
    T *create(Args &&... args)
    {
        if(sizeof(Node<T>) >= big_memory_block_start_size)
        {
            big_nodes.emplace_back(); // allocate slot first
            auto &retval = big_nodes.back();
            retval.reset(new Node<T>(std::forward<Args>(args)...));
            return std::addressof(static_cast<Node<T> *>(retval.get())->value);
        }
        if(std::is_trivially_destructible<T>::value)
        {
            // don't need to keep in small_nodes because we don't need to call the destructor
            return new(allocate<T>()) T(std::forward<Args>(args)...);
        }
        small_nodes.emplace_back(); // allocate slot first
        auto &retval = small_nodes.back();
        retval.reset(new(allocate<Node<T>>()) Node<T>(std::forward<Args>(args)...));
        return std::addressof(static_cast<Node<T> *>(retval.get())->value);
    }
};
}
}

#endif /* EXPRESSION_COMPILER_ARENA_H_ */
