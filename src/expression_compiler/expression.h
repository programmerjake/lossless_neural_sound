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
#include <utility>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

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
    struct Code_writing_state
    {
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
            std::string variable_name;
            explicit State(const Expression_node *node, Code_writing_state *code_writing_state);
        };
        std::unordered_map<const Expression_node *, State> states;
        State &get_state(const Expression_node *node)
        {
            auto iter = states.find(node);
            if(iter == states.end())
                return std::get<1>(*std::get<0>(states.emplace(node, State(node, this))));
            return std::get<1>(*iter);
        }
    };
    void dump(std::ostream &os) const
    {
        Dump_state state;
        dump(os, state);
    }
    void write_code(std::ostream &os) const
    {
        Code_writing_state state;
        write_code(os, state);
    }
    virtual Kind get_kind() const noexcept = 0;
    virtual void dump(std::ostream &os, Dump_state &state) const = 0;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const = 0;
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
        os << "constexpr double " << node_state.variable_name << " = ";
        write_value(os, value);
        os << ";\n";
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
    std::array<std::size_t, Matrix_variable::dimension_count> indexes;
    explicit Element_variable(const Matrix_variable *matrix_variable,
                              std::array<std::size_t, Matrix_variable::dimension_count> indexes)
        : matrix_variable(matrix_variable), indexes(indexes)
    {
    }
    virtual std::string get_variable_code_name() const override
    {
        std::ostringstream ss;
        ss << matrix_variable->name;
        for(std::size_t i : indexes)
            ss << "[" << i << "]";
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
        for(std::size_t i : indexes)
            os << "[" << i << "]";
        os << "\n";
    }
    using Expression_node::write_code;
    virtual void write_code([[gnu::unused]] std::ostream &os,
                            [[gnu::unused]] Code_writing_state &state) const override
    {
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

inline Expression_node::Code_writing_state::State::State(const Expression_node *node,
                                                         Code_writing_state *code_writing_state)
{
    auto *variable = dynamic_cast<const Variable *>(node);
    if(variable)
        variable_name = variable->get_variable_code_name();
    else
        variable_name = code_writing_state->make_temporary();
}

struct Sum final : public Expression_node
{
    std::vector<const Expression_node *> terms;

private:
    explicit Sum(std::vector<const Expression_node *> terms) : terms(std::move(terms))
    {
        assert(this->terms.size() >= 2);
    }

public:
    virtual std::size_t hash() const noexcept override
    {
        std::size_t retval = terms.size();
        for(auto *term : terms)
            retval = retval * 0x12345UL ^ std::hash<const Expression_node *>()(term);
        return retval;
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Sum *>(other);
        if(rt && terms.size() == rt->terms.size())
        {
            for(std::size_t i = 0; i < terms.size(); i++)
                if(terms[i] != rt->terms[i])
                    return false;
            return true;
        }
        return false;
    }
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
        for(auto *term : terms)
            term->dump(os, state);
        state.indent_depth--;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        for(auto *term : terms)
            term->write_code(os, state);
        os << "double " << node_state.variable_name << " = ";
        auto separator = "";
        for(auto *term : terms)
        {
            os << separator;
            separator = " + ";
            os << state.get_state(term).variable_name;
        }
        os << ";\n";
    }
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const override
    {
        auto new_terms = terms;
        for(auto &term : new_terms)
            term = term->get_derivative(arena, nodes, variable);
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
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *sum = dynamic_cast<const Sum *>(rt))
        {
            for(std::size_t i = 0; i < terms.size() && i < sum->terms.size(); i++)
            {
                int compare_result = terms[i]->structurally_compare(sum->terms[i]);
                if(compare_result != 0)
                    return compare_result;
            }
            if(terms.size() < sum->terms.size())
                return -1;
            if(terms.size() > sum->terms.size())
                return 1;
            return 0;
        }
        auto rt_kind = rt->get_kind();
        if(Kind::Sum < rt_kind)
            return -1;
        if(Kind::Sum > rt_kind)
            return 1;
        return 0;
    }
};

struct Product final : public Expression_node
{
    std::vector<const Expression_node *> factors;

private:
    explicit Product(std::vector<const Expression_node *> factors) : factors(std::move(factors))
    {
        assert(this->factors.size() >= 2);
    }

public:
    virtual std::size_t hash() const noexcept override
    {
        std::size_t retval = factors.size();
        for(auto *factor : factors)
            retval = retval * 0x12345UL ^ std::hash<const Expression_node *>()(factor);
        return retval;
    }
    virtual bool same(const Node *other) const noexcept override
    {
        auto *rt = dynamic_cast<const Product *>(other);
        if(rt && factors.size() == rt->factors.size())
        {
            for(std::size_t i = 0; i < factors.size(); i++)
                if(factors[i] != rt->factors[i])
                    return false;
            return true;
        }
        return false;
    }
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
        for(auto *factor : factors)
            factor->dump(os, state);
        state.indent_depth--;
    }
    using Expression_node::write_code;
    virtual void write_code(std::ostream &os, Code_writing_state &state) const override
    {
        auto &node_state = state.get_state(this);
        if(node_state.written)
            return;
        node_state.written = true;
        for(auto *factor : factors)
            factor->write_code(os, state);
        os << "double " << node_state.variable_name << " = ";
        auto separator = "";
        for(auto *factor : factors)
        {
            os << separator;
            separator = " * ";
            os << state.get_state(factor).variable_name;
        }
        os << ";\n";
    }
    virtual const Expression_node *get_derivative(Arena &arena,
                                                  Nodes &nodes,
                                                  const Variable *variable) const override
    {
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
        std::size_t factors_list_size = 0;
        for(std::size_t i = 0; i < factor_count; i++)
            if(auto *product = dynamic_cast<const Product *>(factors[i]))
                factors_list_size += product->factors.size();
            else
                factors_list_size++;
        std::vector<const Expression_node *> factors_list;
        factors_list.reserve(factors_list_size);
        double constant_factor = 1;
        for(std::size_t i = 0; i < factor_count; i++)
            if(auto *product = dynamic_cast<const Product *>(factors[i]))
                for(auto *factor : product->factors)
                    make_helper(constant_factor, factors_list, factor);
            else
                make_helper(constant_factor, factors_list, factors[i]);
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
        return nodes.intern(arena, Product(std::move(factors_list)));
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
        if(auto *constant = dynamic_cast<const Constant *>(factors.front()))
        {
            assert(factors.size() >= 2);
            if(factors.size() == 2)
                return {constant->value, factors.back()};
            std::vector<const Expression_node *> rest;
            rest.reserve(factors.size() - 1);
            for(std::size_t i = 1; i < factors.size(); i++)
                rest.push_back(factors[i]);
            return {constant->value, nodes.intern(arena, Product(std::move(rest)))};
        }
        return {1, this};
    }
    virtual int structurally_compare(const Expression_node *rt) const noexcept override
    {
        assert(rt);
        if(auto *product = dynamic_cast<const Product *>(rt))
        {
            for(std::size_t i = 0; i < factors.size() && i < product->factors.size(); i++)
            {
                int compare_result = factors[i]->structurally_compare(product->factors[i]);
                if(compare_result != 0)
                    return compare_result;
            }
            if(factors.size() < product->factors.size())
                return -1;
            if(factors.size() > product->factors.size())
                return 1;
            return 0;
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
            for(auto &term : sum->terms)
                make_helper(arena, nodes, constant_term, terms_map, term);
        else
            make_helper(arena, nodes, constant_term, terms_map, terms[i]);
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
    return nodes.intern(arena, Sum(std::move(new_terms)));
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
        os << "double " << node_state.variable_name << " = transfer_function<" << derivative_count
           << ">(" << state.get_state(arg).variable_name << ");\n";
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

class Expression
{
private:
    Arena *arena;
    Nodes *nodes;
    const Expression_node *node;

private:
    bool is_compatible(const Expression &rt) const noexcept
    {
        return arena == rt.arena && nodes == rt.nodes;
    }

public:
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
        return Expression(*r.arena, *r.nodes, l) + r;
    }
    friend Expression operator+(Expression l, Expression r)
    {
        assert(l.is_compatible(r));
        return Expression(*l.arena, *l.nodes, Sum::make(*l.arena, *l.nodes, {l.node, r.node}));
    }
    friend Expression operator+(Expression l, double r)
    {
        return l + Expression(*l.arena, *l.nodes, r);
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
        return Expression(*r.arena, *r.nodes, l) - r;
    }
    friend Expression operator-(Expression l, Expression r)
    {
        return l + -r;
    }
    friend Expression operator-(Expression l, double r)
    {
        return l - Expression(*l.arena, *l.nodes, r);
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
        assert(l.is_compatible(r));
        return Expression(*l.arena, *l.nodes, Product::make(*l.arena, *l.nodes, {l.node, r.node}));
    }
    friend Expression operator*(double l, Expression r)
    {
        return Expression(*r.arena, *r.nodes, l) * r;
    }
    friend Expression operator*(Expression l, double r)
    {
        return l * Expression(*l.arena, *l.nodes, r);
    }
    Expression &operator*=(Expression r)
    {
        return operator=(*this * r);
    }
    Expression &operator*=(double r)
    {
        return operator=(*this * r);
    }
    template <std::size_t derivative_count = 0>
    static Expression transfer_function(Expression e)
    {
        e.node = e.nodes->intern(*e.arena, Transfer_function(e.node, derivative_count));
        return e;
    }
    const Expression_node *get() const noexcept
    {
        return node;
    }
    Expression get_derivative(const Variable *variable) const
    {
        return Expression(*arena, *nodes, node->get_derivative(*arena, *nodes, variable));
    }
};
}
}
}

#endif /* EXPRESSION_COMPILER_EXPRESSION_H_ */
