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
#ifndef EXPRESSION_COMPILER_EXPRESSION_H_
#define EXPRESSION_COMPILER_EXPRESSION_H_

#include <vector>
#include <string>
#include <array>
#include "arena.h"
#include "util/array_stack.h"
#include <utility>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>

namespace lossless_neural_sound
{
namespace expression_compiler
{
namespace expression
{
struct Node
{
    virtual ~Node() = default;
    virtual std::size_t hash() const noexcept = 0;
    virtual bool same(const Node *other) const noexcept = 0;
    virtual Node *reallocate(Arena &arena) const & = 0;
    virtual Node *reallocate(Arena &arena)&& = 0;
};

class Nodes final
{
private:
    std::unordered_multimap<std::size_t, const Node *> node_set;

private:
    struct Find_results
    {
        std::size_t hash;
        const Node *node;
    };
    static Node &&to_node_reference(Node &&node) noexcept
    {
        return std::move(node);
    }
    static const Node &to_node_reference(const Node &node) noexcept
    {
        return std::move(node);
    }

public:
    template <typename T>
    const T *intern(Arena &arena, T &&node)
    {
        auto &&node_reference = to_node_reference(std::forward<T>(node));
        std::size_t hash = node_reference.hash();
        auto equal_range = node_set.equal_range(hash);
        for(auto iter = std::get<0>(equal_range); iter != std::get<1>(equal_range); ++iter)
            if(node_reference.same(std::get<1>(*iter)))
                return static_cast<const T *>(std::get<1>(*iter));
        const Node *retval =
            std::forward<decltype(node_reference)>(node_reference).reallocate(arena);
        node_set.emplace(hash, retval);
        return static_cast<const T *>(retval);
    }
};

struct Matrix_variable final : public Node
{
    static constexpr std::size_t dimension_count = 2;
    std::string name;
    std::array<std::size_t, dimension_count> sizes;
    explicit Matrix_variable(std::string name, std::array<std::size_t, dimension_count> sizes)
        : name(std::move(name)), sizes(sizes)
    {
    }
    virtual std::size_t hash() const noexcept override
    {
        std::size_t retval = std::hash<std::string>()(name);
        for(std::size_t i : sizes)
            retval = retval * 0x12345UL ^ i;
        return retval;
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Matrix_variable *>(other);
        if(rt && rt->name == name && rt->sizes == sizes)
            return true;
        return false;
    }
    virtual Matrix_variable *reallocate(Arena &arena) const &override
    {
        return arena.create<Matrix_variable>(*this);
    }
    virtual Matrix_variable *reallocate(Arena &arena) && override
    {
        return arena.create<Matrix_variable>(std::move(*this));
    }
};

struct Variable;

struct Expression_node : public Node
{
    enum class Kind : unsigned
    {
        Constant = 0, // Constant must be first
        Element_variable,
        Sum,
        Product,
        Transfer_function,
    };
    struct Dump_state
    {
        std::size_t indent_depth = 0;
        void write_indent(std::ostream &os) const
        {
            for(std::size_t i = 0; i < indent_depth; i++)
                os << "    ";
        }
    };
    struct Code_writing_options
    {
        std::string value_type = "double";
        std::string indent = "";
    };
    struct Code_writing_state
    {
        Code_writing_options options;
        std::size_t next_temporary_variable_index = 0;
        std::string make_temporary()
        {
            std::ostringstream ss;
            ss << "temp_" << next_temporary_variable_index++;
            return ss.str();
        }
        struct State
        {
            bool written = false;
            std::string cached_variable_name{};
            const std::string &get_variable_name(const Expression_node *node,
                                                 Code_writing_state &code_writing_state);
        };
        std::unordered_map<const Expression_node *, State> states;
        State &get_state(const Expression_node *node)
        {
            return states[node];
        }
        const std::string &get_variable_name(const Expression_node *node)
        {
            return get_state(node).get_variable_name(node, *this);
        }
        void write_variable_assignment(std::ostream &os,
                                       const std::string &variable,
                                       const Expression_node *value)
        {
            os << options.indent << variable << " = " << get_variable_name(value) << ";\n";
        }
        explicit Code_writing_state(Code_writing_options options) : options(std::move(options))
        {
        }
    };
    struct Expression_writing_state
    {
        enum class Precedence
        {
            Add,
            Times,
            Atom,
        };
        Precedence current_precedence = Precedence();
        struct Parenthesis_writer
        {
            Parenthesis_writer(const Parenthesis_writer &) = delete;
            Parenthesis_writer &operator=(const Parenthesis_writer &) = delete;
            Expression_writing_state &state;
            std::ostream &os;
            Precedence old_precedence;
            bool need_parenthesis;
            Parenthesis_writer(Expression_writing_state &state,
                               std::ostream &os,
                               Precedence precedence,
                               bool can_write_parenthesis = true)
                : state(state),
                  os(os),
                  old_precedence(state.current_precedence),
                  need_parenthesis(false)
            {
                state.current_precedence = precedence;
                if(state.current_precedence < old_precedence && can_write_parenthesis)
                {
                    need_parenthesis = true;
                    os << "(";
                }
            }
            ~Parenthesis_writer() noexcept(false)
            {
                if(need_parenthesis)
                    os << ")";
                state.current_precedence = old_precedence;
            }
        };
    };
    void dump(std::ostream &os) const
    {
        Dump_state state;
        dump(os, state);
    }
    void write_code(std::ostream &os) const
    {
        Code_writing_state state({});
        write_code(os, state);
    }
    void write_code(std::ostream &os, Code_writing_options options) const
    {
        Code_writing_state state(std::move(options));
        write_code(os, state);
    }
    void write_expression(std::ostream &os) const
    {
        Expression_writing_state state;
        write_expression(os, state);
    }
    virtual Kind get_kind() const noexcept = 0;
    virtual void dump(std::ostream &os, Dump_state &state) const = 0;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const = 0;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const = 0;
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const = 0;
    struct Get_constant_factor_result
    {
        double constant_factor;
        const Expression_node *rest;
    };
    virtual Get_constant_factor_result get_constant_factor(Arena &arena, Nodes &nodes) const
    {
        static_cast<void>(arena);
        static_cast<void>(nodes);
        return {1, this};
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept = 0;
};

struct Constant final : public Expression_node
{
    double value;
    explicit Constant(double value) : value(value)
    {
    }
    Constant &operator+=(const Constant &rt) noexcept
    {
        value += rt.value;
        return *this;
    }
    Constant operator+(const Constant &rt) const
    {
        return Constant(value + rt.value);
    }
    Constant &operator-=(const Constant &rt) noexcept
    {
        value -= rt.value;
        return *this;
    }
    Constant operator-(const Constant &rt) const
    {
        return Constant(value - rt.value);
    }
    Constant operator-() const
    {
        return Constant(-value);
    }
    Constant &operator*=(const Constant &rt) noexcept
    {
        value *= rt.value;
        return *this;
    }
    Constant operator*(const Constant &rt) const
    {
        return Constant(value * rt.value);
    }
    Constant operator/(const Constant &rt) const
    {
        return Constant(value / rt.value);
    }
    Constant &operator/=(const Constant &rt)
    {
        value /= rt.value;
        return *this;
    }
    bool operator==(const Constant &rt) const noexcept
    {
        return value == rt.value;
    }
    bool operator!=(const Constant &rt) const noexcept
    {
        return value != rt.value;
    }
    bool operator<(const Constant &rt) const noexcept
    {
        return value < rt.value;
    }
    bool operator>(const Constant &rt) const noexcept
    {
        return value > rt.value;
    }
    bool operator<=(const Constant &rt) const noexcept
    {
        return value <= rt.value;
    }
    bool operator>=(const Constant &rt) const noexcept
    {
        return value >= rt.value;
    }
    virtual std::size_t hash() const noexcept override
    {
        return std::hash<double>()(value);
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Constant *>(other);
        if(rt && value == rt->value)
            return true;
        return false;
    }
    virtual Constant *reallocate(Arena &arena) const &override
    {
        return arena.create<Constant>(*this);
    }
    virtual Constant *reallocate(Arena &arena) && override
    {
        return arena.create<Constant>(std::move(*this));
    }
    virtual Kind get_kind() const noexcept override
    {
        return Kind::Constant;
    }
    using Expression_node::dump;
    virtual void dump(std::ostream &os, Dump_state &state) const override
    {
        state.write_indent(os);
        os << value << "\n";
    }
    static void write_value(std::ostream &os,
                            double value,
                            const char *nan_string = "NaN",
                            const char *infinity_string = "Infinity")
    {
        if(std::isnan(value))
        {
            os << nan_string;
            return;
        }
        if(value == 0)
        {
            if(std::signbit(value))
                os << '-';
            os << '0';
            return;
        }
        if(value < 0)
        {
            os << '-';
            value = -value;
        }
        if(std::isinf(value))
        {
            os << infinity_string;
            return;
        }
        constexpr std::size_t buffer_size = 64; // always big enough
        char buffer[buffer_size];
        for(int precision = 0; precision < 20; precision++)
        {
            std::snprintf(buffer, sizeof(buffer), "%0.*g", precision, value);
            double read_value = std::atof(buffer);
            if(read_value == value)
                break;
        }
        os << buffer;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        os << state.options.indent << "const " << state.options.value_type << " "
           << node_state.get_variable_name(this, state) << " = ";
        write_value(os, value);
        os << ";\n";
    }
    using Expression_node::write_expression;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const override
    {
        Expression_writing_state::Parenthesis_writer pw(
            state, os, Expression_writing_state::Precedence::Atom);
        write_value(os, value);
    }
    virtual const Expression_node *get_derivative(
        Arena &arena, Nodes &nodes, [[gnu::unused]] const Variable *variable) const override
    {
        return nodes.intern(arena, Constant(0));
    }
    virtual Get_constant_factor_result get_constant_factor(Arena &arena,
                                                           Nodes &nodes) const override
    {
        return {value, nodes.intern(arena, Constant(1))};
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *constant = dynamic_cast<const Constant *>(rt))
        {
            if(std::isnan(constant->value))
                return std::isnan(value) ? 0 : 1;
            if(std::isnan(value))
                return -1;
            if(value < constant->value)
                return -1;
            if(value > constant->value)
                return 1;
            return 0;
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Constant < rt_kind)
            return -1;
        if(Kind::Constant > rt_kind)
            return 1;
        return 0;
    }
};

struct Variable : public Expression_node
{
    virtual std::string get_variable_code_name() const = 0;
    using Expression_node::write_code;
    virtual void write_code([[gnu::unused]] std::ostream &os,
                            [[gnu::unused]] Code_writing_state &state) const override
    {
    }
    virtual const Expression_node *get_derivative(
        Arena &arena, Nodes &nodes, [[gnu::unused]] const Variable *variable) const override final
    {
        return nodes.intern(arena, Constant(same(variable) ? 1 : 0));
    }
};

struct Element_variable final : public Variable
{
    const Matrix_variable *matrix_variable;
    std::array<std::size_t, Matrix_variable::dimension_count>
        indexes; // in reverse order of Matrix_variable::sizes
    explicit Element_variable(const Matrix_variable *matrix_variable,
                              std::array<std::size_t, Matrix_variable::dimension_count> indexes)
        : matrix_variable(matrix_variable), indexes(indexes)
    {
    }
    void write_indexes(std::ostream &os) const
    {
        std::size_t non_unit_dimension_count = 0;
        std::size_t non_unit_dimension_index = 0;
        for(std::size_t i = 0; i < indexes.size(); i++)
        {
            if(matrix_variable->sizes[i] != 1)
            {
                non_unit_dimension_count++;
                non_unit_dimension_index = i;
            }
        }
        if(non_unit_dimension_count == 1)
        {
            os << "[" << indexes[indexes.size() - 1 - non_unit_dimension_index] << "]";
        }
        else
        {
            auto separator = "";
            os << "(";
            for(std::size_t i : indexes)
            {
                os << separator << i;
                separator = ", ";
            }
            os << ")";
        }
    }
    virtual std::string get_variable_code_name() const override
    {
        std::ostringstream ss;
        ss << matrix_variable->name;
        write_indexes(ss);
        return ss.str();
    }
    virtual Kind get_kind() const noexcept override
    {
        return Kind::Element_variable;
    }
    using Expression_node::dump;
    virtual void dump(std::ostream &os, Dump_state &state) const override
    {
        state.write_indent(os);
        os << matrix_variable->name;
        write_indexes(os);
        os << "\n";
    }
    using Expression_node::write_code;
    virtual void write_code([[gnu::unused]] std::ostream &os,
                            [[gnu::unused]] Code_writing_state &state) const override
    {
    }
    using Expression_node::write_expression;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const override
    {
        Expression_writing_state::Parenthesis_writer pw(
            state, os, Expression_writing_state::Precedence::Atom);
        os << matrix_variable->name;
        write_indexes(os);
    }
    virtual std::size_t hash() const noexcept override
    {
        std::size_t retval = std::hash<const Matrix_variable *>()(matrix_variable);
        for(std::size_t i : indexes)
            retval = retval * 0x12345UL ^ i;
        return retval;
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Element_variable *>(other);
        if(rt && rt->matrix_variable == matrix_variable && rt->indexes == indexes)
            return true;
        return false;
    }
    virtual Element_variable *reallocate(Arena &arena) const &override
    {
        return arena.create<Element_variable>(*this);
    }
    virtual Element_variable *reallocate(Arena &arena) && override
    {
        return arena.create<Element_variable>(std::move(*this));
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *variable = dynamic_cast<const Element_variable *>(rt))
        {
            int name_compare = matrix_variable->name.compare(variable->matrix_variable->name);
            if(name_compare < 0)
                return -1;
            if(name_compare > 0)
                return 1;
            for(std::size_t i = 0; i < indexes.size(); i++)
            {
                if(indexes[i] < variable->indexes[i])
                    return -1;
                if(indexes[i] > variable->indexes[i])
                    return 1;
            }
            return 0;
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Element_variable < rt_kind)
            return -1;
        if(Kind::Element_variable > rt_kind)
            return 1;
        return 0;
    }
};

inline const std::string &Expression_node::Code_writing_state::State::get_variable_name(
    const Expression_node *node, Code_writing_state &code_writing_state)
{
    if(cached_variable_name.empty())
    {
        auto *variable = dynamic_cast<const Variable *>(node);
        if(variable)
            cached_variable_name = variable->get_variable_code_name();
        else
            cached_variable_name = code_writing_state.make_temporary();
    }
    return cached_variable_name;
}

template <typename Derived_class>
class Balanced_tree : public Expression_node
{
public:
    const Expression_node *left_node;
    const Expression_node *right_node;

protected:
    explicit Balanced_tree(const Expression_node *left_node, const Expression_node *right_node)
        : left_node(left_node), right_node(right_node)
    {
    }

public:
    virtual std::size_t hash() const noexcept override
    {
        return std::hash<const Expression_node *>()(left_node) * 0x12345UL
               ^ std::hash<const Expression_node *>()(right_node);
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Derived_class *>(other);
        if(rt && left_node == rt->left_node && right_node == rt->right_node)
            return true;
        return false;
    }

protected:
    template <typename Fn>
    void visit_leaves(Fn fn) const
    {
        if(auto *node = dynamic_cast<const Balanced_tree *>(left_node))
            node->visit_leaves(fn);
        else
            fn(left_node);
        if(auto *node = dynamic_cast<const Balanced_tree *>(right_node))
            node->visit_leaves(fn);
        else
            fn(right_node);
    }
    template <typename... Args>
    static const Expression_node *make_tree(Arena &arena,
                                            Nodes &nodes,
                                            const Expression_node *const *leaves,
                                            std::size_t leaf_count,
                                            const Args &... args)
    {
        assert(leaf_count != 0);
        if(leaf_count == 1)
            return leaves[0];
        std::size_t split_index = leaf_count / 2;
        const Expression_node *left_node = make_tree(arena, nodes, leaves, split_index, args...);
        const Expression_node *right_node =
            make_tree(arena, nodes, leaves + split_index, leaf_count - split_index, args...);
        return nodes.intern(arena, Derived_class(left_node, right_node, args...));
    }

protected:
    class Leaf_iterator
    {
    private:
        // tree is always balanced, so this is the max tree depth for
        // std::numeric_limits<std::size_t>::max() nodes:
        static constexpr std::size_t max_tree_depth = std::numeric_limits<std::size_t>::digits + 1;
        typedef util::Array_stack<const Expression_node *, max_tree_depth> Stack_type;
        Stack_type stack;

    private:
        void find_leaf() noexcept
        {
            while(auto *node = dynamic_cast<const Balanced_tree *>(stack.top()))
            {
                stack.top() = node->right_node;
                stack.push(node->left_node);
            }
        }

    public:
        constexpr Leaf_iterator() noexcept : stack()
        {
        }
        explicit Leaf_iterator(const Balanced_tree *node) noexcept : stack()
        {
            assert(node);
            stack.push(node);
            find_leaf();
        }
        constexpr bool operator==(const Leaf_iterator &rt) const noexcept
        {
            if(stack.empty() && rt.stack.empty())
                return true;
            return false;
        }
        constexpr bool operator!=(const Leaf_iterator &rt) const noexcept
        {
            return !operator==(rt);
        }
        const Expression_node *operator*() const noexcept
        {
            return stack.top();
        }
        Leaf_iterator &operator++() noexcept
        {
            stack.pop();
            if(!stack.empty())
                find_leaf();
            return *this;
        }
        Leaf_iterator operator++(int) noexcept
        {
            auto retval = *this;
            operator++();
            return retval;
        }
    };

protected:
    int structurally_compare_tree(const Balanced_tree *rt) const noexcept
    {
        assert(rt);
        auto left_iterator = Leaf_iterator(this);
        auto right_iterator = Leaf_iterator(rt);
        for(; left_iterator != Leaf_iterator() && right_iterator != Leaf_iterator();
            ++left_iterator, ++right_iterator)
        {
            int compare_result = (*left_iterator)->structurally_compare(*right_iterator);
            if(compare_result != 0)
                return compare_result;
        }
        bool left_at_end = left_iterator == Leaf_iterator();
        bool right_at_end = right_iterator == Leaf_iterator();
        if(left_at_end && !right_at_end)
            return -1;
        if(!left_at_end && right_at_end)
            return 1;
        return 0;
    }
};

struct Sum final : public Balanced_tree<Sum>
{
    friend class Balanced_tree<Sum>;

private:
    explicit Sum(const Expression_node *left_node, const Expression_node *right_node)
        : Balanced_tree(left_node, right_node)
    {
    }

public:
    virtual Sum *reallocate(Arena &arena) const &override
    {
        return arena.create<Sum>(*this);
    }
    virtual Sum *reallocate(Arena &arena) && override
    {
        return arena.create<Sum>(std::move(*this));
    }
    virtual Kind get_kind() const noexcept override
    {
        return Kind::Sum;
    }
    using Expression_node::dump;
    virtual void dump(std::ostream &os, Dump_state &state) const override
    {
        state.write_indent(os);
        os << "Sum:\n";
        state.indent_depth++;
        visit_leaves([&](const Expression_node *term)
                     {
                         term->dump(os, state);
                     });
        state.indent_depth--;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        left_node->write_code(os, state);
        right_node->write_code(os, state);
        os << state.options.indent << state.options.value_type << " "
           << node_state.get_variable_name(this, state) << " = "
           << state.get_variable_name(left_node) << " + " << state.get_variable_name(right_node)
           << ";\n";
    }
    using Expression_node::write_expression;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const override
    {
        Expression_writing_state::Parenthesis_writer pw(
            state, os, Expression_writing_state::Precedence::Add);
        auto separator = "";
        visit_leaves([&](const Expression_node *term)
                     {
                         os << separator;
                         separator = " + ";
                         term->write_expression(os, state);
                     });
    }
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const override
    {
        std::vector<const Expression_node *> new_terms;
        visit_leaves([&](const Expression_node *term)
                     {
                         new_terms.push_back(term->get_derivative(arena, nodes, variable));
                     });
        return make(arena, nodes, new_terms.data(), new_terms.size());
    }

private:
    static void make_helper(Arena &arena,
                            Nodes &nodes,
                            double &constant_term,
                            std::unordered_map<const Expression_node *, double> &terms_map,
                            const Expression_node *term)
    {
        if(auto *constant = dynamic_cast<const Constant *>(term))
            constant_term += constant->value;
        else
        {
            auto get_constant_factor_result = term->get_constant_factor(arena, nodes);
            terms_map[get_constant_factor_result.rest] +=
                get_constant_factor_result.constant_factor;
        }
    }

public:
    static const Expression_node *make(Arena &arena,
                                       Nodes &nodes,
                                       const Expression_node *const *terms,
                                       std::size_t term_count);
    static const Expression_node *make(Arena &arena,
                                       Nodes &nodes,
                                       std::initializer_list<const Expression_node *> terms)
    {
        return make(arena, nodes, terms.begin(), terms.size());
    }

public:
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *sum = dynamic_cast<const Sum *>(rt))
        {
            return structurally_compare_tree(sum);
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Sum < rt_kind)
            return -1;
        if(Kind::Sum > rt_kind)
            return 1;
        return 0;
    }
};

struct Product final : public Balanced_tree<Product>
{
    friend class Balanced_tree<Product>;

private:
    explicit Product(const Expression_node *left_node, const Expression_node *right_node)
        : Balanced_tree(left_node, right_node)
    {
    }

public:
    virtual Product *reallocate(Arena &arena) const &override
    {
        return arena.create<Product>(*this);
    }
    virtual Product *reallocate(Arena &arena) && override
    {
        return arena.create<Product>(std::move(*this));
    }
    virtual Kind get_kind() const noexcept override
    {
        return Kind::Product;
    }
    using Expression_node::dump;
    virtual void dump(std::ostream &os, Dump_state &state) const override
    {
        state.write_indent(os);
        os << "Product:\n";
        state.indent_depth++;
        visit_leaves([&](const Expression_node *factor)
                     {
                         factor->dump(os, state);
                     });
        state.indent_depth--;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        left_node->write_code(os, state);
        right_node->write_code(os, state);
        os << state.options.indent << state.options.value_type << " "
           << node_state.get_variable_name(this, state) << " = "
           << state.get_variable_name(left_node) << " * " << state.get_variable_name(right_node)
           << ";\n";
    }
    using Expression_node::write_expression;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const override
    {
        Expression_writing_state::Parenthesis_writer pw(
            state, os, Expression_writing_state::Precedence::Times);
        auto separator = "";
        visit_leaves([&](const Expression_node *factor)
                     {
                         os << separator;
                         separator = " * ";
                         factor->write_expression(os, state);
                     });
    }
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const override
    {
        std::vector<const Expression_node *> factors;
        visit_leaves([&](const Expression_node *factor)
                     {
                         factors.push_back(factor);
                     });
        std::vector<const Expression_node *> terms;
        terms.reserve(factors.size());
        auto current_factors = factors;
        for(std::size_t i = 0; i < factors.size(); i++)
        {
            current_factors[i] = factors[i]->get_derivative(arena, nodes, variable);
            terms.push_back(make(arena, nodes, current_factors.data(), current_factors.size()));
            current_factors[i] = factors[i];
        }
        return Sum::make(arena, nodes, terms.data(), terms.size());
    }

private:
    static void make_helper(double &constant_factor,
                            std::vector<const Expression_node *> &factors_list,
                            const Expression_node *factor)
    {
        if(auto *constant = dynamic_cast<const Constant *>(factor))
            constant_factor *= constant->value;
        else
            factors_list.push_back(factor);
    }

public:
    static const Expression_node *make(Arena &arena,
                                       Nodes &nodes,
                                       const Expression_node *const *factors,
                                       std::size_t factor_count)
    {
        std::vector<const Expression_node *> factors_list;
        double constant_factor = 1;
        for(std::size_t i = 0; i < factor_count; i++)
        {
            if(auto *product = dynamic_cast<const Product *>(factors[i]))
            {
                product->visit_leaves([&](const Expression_node *factor)
                                      {
                                          make_helper(constant_factor, factors_list, factor);
                                      });
            }
            else
            {
                make_helper(constant_factor, factors_list, factors[i]);
            }
        }
        if(constant_factor == 0)
            return nodes.intern(arena, Constant(0));
        if(constant_factor != 1)
            factors_list.insert(factors_list.begin(),
                                nodes.intern(arena, Constant(constant_factor)));
        if(factors_list.empty())
            return nodes.intern(arena, Constant(1));
        if(factors_list.size() == 1)
            return factors_list[0];
        std::sort(factors_list.begin(),
                  factors_list.end(),
                  [](const Expression_node *a, const Expression_node *b)
                  {
                      assert(a && b);
                      return a->structurally_compare(b) < 0;
                  });
        return make_tree(arena, nodes, factors_list.data(), factors_list.size());
    }
    static const Expression_node *make(Arena &arena,
                                       Nodes &nodes,
                                       std::initializer_list<const Expression_node *> factors)
    {
        return make(arena, nodes, factors.begin(), factors.size());
    }
    virtual Get_constant_factor_result get_constant_factor(Arena &arena,
                                                           Nodes &nodes) const override
    {
        auto *front = left_node;
        while(auto *node = dynamic_cast<const Product *>(front))
            front = node->left_node;
        if(auto *constant = dynamic_cast<const Constant *>(front))
        {
            std::vector<const Expression_node *> factors;
            visit_leaves([&](const Expression_node *factor)
                         {
                             factors.push_back(factor);
                         });
            assert(factors.size() >= 2);
            if(factors.size() == 2)
                return {constant->value, factors.back()};
            factors.erase(factors.begin());
            return {constant->value, make_tree(arena, nodes, factors.data(), factors.size())};
        }
        return {1, this};
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *product = dynamic_cast<const Product *>(rt))
        {
            return structurally_compare_tree(product);
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Product < rt_kind)
            return -1;
        if(Kind::Product > rt_kind)
            return 1;
        return 0;
    }
};

inline const Expression_node *Sum::make(Arena &arena,
                                        Nodes &nodes,
                                        const Expression_node *const *terms,
                                        std::size_t term_count)
{
    std::unordered_map<const Expression_node *, double> terms_map;
    double constant_term = 0;
    for(std::size_t i = 0; i < term_count; i++)
    {
        if(auto *sum = dynamic_cast<const Sum *>(terms[i]))
        {
            sum->visit_leaves([&](const Expression_node *term)
                              {
                                  make_helper(arena, nodes, constant_term, terms_map, term);
                              });
        }
        else
        {
            make_helper(arena, nodes, constant_term, terms_map, terms[i]);
        }
    }
    std::vector<const Expression_node *> new_terms;
    new_terms.reserve(terms_map.size() + (constant_term != 0 ? 1 : 0));
    if(constant_term != 0)
        new_terms.push_back(nodes.intern(arena, Constant(constant_term)));
    for(auto &terms_map_entry : terms_map)
    {
        if(std::get<1>(terms_map_entry) != 0)
        {
            new_terms.push_back(
                Product::make(arena,
                              nodes,
                              {nodes.intern(arena, Constant(std::get<1>(terms_map_entry))),
                               std::get<0>(terms_map_entry)}));
        }
    }
    if(new_terms.empty())
        return nodes.intern(arena, Constant(0));
    if(new_terms.size() == 1)
        return new_terms[0];
    std::sort(new_terms.begin(),
              new_terms.end(),
              [](const Expression_node *a, const Expression_node *b)
              {
                  assert(a && b);
                  return a->structurally_compare(b) < 0;
              });
    return make_tree(arena, nodes, new_terms.data(), new_terms.size());
}


struct Transfer_function final : public Expression_node
{
    const Expression_node *arg;
    std::size_t derivative_count;
    explicit Transfer_function(const Expression_node *arg, std::size_t derivative_count = 0)
        : arg(arg), derivative_count(derivative_count)
    {
    }
    virtual std::size_t hash() const noexcept override
    {
        return derivative_count ^ 0x12345UL * std::hash<const Expression_node *>()(arg);
    }
    virtual bool same(const Node *other) const noexcept override
    {
        if(auto *rt = dynamic_cast<const Transfer_function *>(other))
            return arg == rt->arg && derivative_count == rt->derivative_count;
        return false;
    }
    virtual Transfer_function *reallocate(Arena &arena) const &override
    {
        return arena.create<Transfer_function>(*this);
    }
    virtual Transfer_function *reallocate(Arena &arena) && override
    {
        return arena.create<Transfer_function>(std::move(*this));
    }
    virtual Kind get_kind() const noexcept override
    {
        return Kind::Transfer_function;
    }
    using Expression_node::dump;
    virtual void dump(std::ostream &os, Dump_state &state) const override
    {
        state.write_indent(os);
        os << "Transfer_function: derivative_count=" << derivative_count << "\n";
        state.indent_depth++;
        arg->dump(os, state);
        state.indent_depth--;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        arg->write_code(os, state);
        os << state.options.indent << state.options.value_type << " "
           << node_state.get_variable_name(this, state) << " = transfer_function";
        if(derivative_count != 0)
            os << "<" << derivative_count << ">";
        os << "(" << state.get_variable_name(arg) << ");\n";
    }
    using Expression_node::write_expression;
    virtual void write_expression(std::ostream &os, Expression_writing_state &state) const override
    {
        Expression_writing_state::Parenthesis_writer pw(
            state, os, Expression_writing_state::Precedence(), false);
        os << "transfer_function";
        if(derivative_count != 0)
            os << "<" << derivative_count << ">";
        os << "(";
        arg->write_expression(os, state);
        os << ")";
    }
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const override
    {
        return Product::make(arena,
                             nodes,
                             {nodes.intern(arena, Transfer_function(arg, derivative_count + 1)),
                              arg->get_derivative(arena, nodes, variable)});
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *fn = dynamic_cast<const Transfer_function *>(rt))
        {
            if(derivative_count < fn->derivative_count)
                return -1;
            if(derivative_count > fn->derivative_count)
                return 1;
            return arg->structurally_compare(fn->arg);
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Transfer_function < rt_kind)
            return -1;
        if(Kind::Transfer_function > rt_kind)
            return 1;
        return 0;
    }
};

class Expression;

template <std::size_t derivative_count = 0>
Expression transfer_function(Arena &arena, Nodes &nodes, Expression e);

class Expression
{
    template <std::size_t derivative_count>
    friend Expression transfer_function(Arena &arena, Nodes &nodes, Expression e);

private:
    Arena *arena;
    Nodes *nodes;
    const Expression_node *node;
    double constant_value;

private:
    bool is_compatible(const Expression &rt) const noexcept
    {
        return arena == rt.arena && nodes == rt.nodes;
    }

public:
    Expression(double constant_value = 0) noexcept : arena(nullptr),
                                                     nodes(nullptr),
                                                     node(nullptr),
                                                     constant_value(constant_value)
    {
    }
    explicit Expression(Arena &arena, Nodes &nodes, double value)
        : arena(&arena), nodes(&nodes), node(nodes.intern(arena, Constant(value)))
    {
    }
    explicit Expression(Arena &arena, Nodes &nodes, const Expression_node *node) noexcept
        : arena(&arena),
          nodes(&nodes),
          node(node)
    {
    }
    Expression operator-() const
    {
        return -1 * *this;
    }
    const Expression &operator+() const
    {
        return *this;
    }
    friend Expression operator+(double l, Expression r)
    {
        return Expression(l) + r;
    }
    friend Expression operator+(Expression l, Expression r)
    {
        if(!l.arena && !r.arena)
            return l.constant_value + r.constant_value;
        if(!l.arena)
            l.get(*r.arena, *r.nodes);
        else if(!r.arena)
            r.get(*l.arena, *l.nodes);
        assert(l.is_compatible(r));
        return Expression(*l.arena, *l.nodes, Sum::make(*l.arena, *l.nodes, {l.node, r.node}));
    }
    friend Expression operator+(Expression l, double r)
    {
        return l + Expression(r);
    }
    Expression &operator+=(Expression r)
    {
        return operator=(*this + r);
    }
    Expression &operator+=(double r)
    {
        return operator=(*this + r);
    }
    friend Expression operator-(double l, Expression r)
    {
        return Expression(l) - r;
    }
    friend Expression operator-(Expression l, Expression r)
    {
        return l + -r;
    }
    friend Expression operator-(Expression l, double r)
    {
        return l - Expression(r);
    }
    Expression &operator-=(Expression r)
    {
        return operator=(*this - r);
    }
    Expression &operator-=(double r)
    {
        return operator=(*this - r);
    }
    friend Expression operator*(Expression l, Expression r)
    {
        if(!l.arena && !r.arena)
            return l.constant_value * r.constant_value;
        if(!l.arena)
            l.get(*r.arena, *r.nodes);
        else if(!r.arena)
            r.get(*l.arena, *l.nodes);
        assert(l.is_compatible(r));
        return Expression(*l.arena, *l.nodes, Product::make(*l.arena, *l.nodes, {l.node, r.node}));
    }
    friend Expression operator*(double l, Expression r)
    {
        return Expression(l) * r;
    }
    friend Expression operator*(Expression l, double r)
    {
        return l * Expression(r);
    }
    Expression &operator*=(Expression r)
    {
        return operator=(*this * r);
    }
    Expression &operator*=(double r)
    {
        return operator=(*this * r);
    }
    const Expression_node *get(Arena &arena, Nodes &nodes) const
    {
        return Expression(*this).get(arena, nodes);
    }
    const Expression_node *get(Arena &arena, Nodes &nodes)
    {
        if(!node)
            *this = Expression(arena, nodes, constant_value);
        return node;
    }
    Expression get_derivative(const Variable *variable) const
    {
        if(!node)
            return Expression(0);
        return Expression(*arena, *nodes, node->get_derivative(*arena, *nodes, variable));
    }
    Expression get_derivative(const Expression &variable) const
    {
        assert(dynamic_cast<const Variable *>(variable.node));
        return get_derivative(static_cast<const Variable *>(variable.node));
    }
    friend std::ostream &operator<<(std::ostream &os, const Expression &e)
    {
        if(!e.node)
            Constant::write_value(os, e.constant_value);
        else
            e.node->write_expression(os);
        return os;
    }
};

template <std::size_t derivative_count>
inline Expression transfer_function(Arena &arena, Nodes &nodes, Expression e)
{
    e.get(arena, nodes);
    e.node = e.nodes->intern(*e.arena, Transfer_function(e.node, derivative_count));
    return e;
}
}
}
}

#endif /* EXPRESSION_COMPILER_EXPRESSION_H_ */
