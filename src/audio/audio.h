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
#ifndef AUDIO_AUDIO_H_
#define AUDIO_AUDIO_H_

#include <cstdint>
#include <memory>
#include <system_error>
#include <string>
#include <vector>
#include <fstream>

namespace lossless_neural_sound
{
namespace audio
{
typedef float Audio_value_type;
typedef std::uint64_t Sample_count_type;
constexpr Sample_count_type unknown_sample_count = -1;

struct Audio_error : public std::system_error
{
    Audio_error() : system_error()
    {
    }
    Audio_error(std::error_code ec) : system_error(ec)
    {
    }
    Audio_error(std::error_code ec, const std::string &what) : system_error(ec, what)
    {
    }
    Audio_error(std::error_code ec, const char *what) : system_error(ec, what)
    {
    }
    Audio_error(int error, const std::error_category &category) : system_error(error, category)
    {
    }
    Audio_error(int error, const std::error_category &category, const std::string &what)
        : system_error(error, category, what)
    {
    }
    Audio_error(int error, const std::error_category &category, const char *what)
        : system_error(error, category, what)
    {
    }
};

enum class Audio_error_code : int
{
    format_does_not_match = 1,
    corrupt_file,
    unsupported,
};

class Audio_error_category;

const Audio_error_category &audio_error_category() noexcept;

class Audio_error_category final : public std::error_category
{
    friend const Audio_error_category &audio_error_category() noexcept;

private:
    constexpr Audio_error_category() = default;
    ~Audio_error_category() = default;

public:
    virtual const char *name() const noexcept override;
    virtual std::string message(int condition) const override;
};

inline std::error_code make_error_code(Audio_error_code v) noexcept
{
    return std::error_code(static_cast<int>(v), audio_error_category());
}

inline std::error_condition make_error_condition(Audio_error_code v) noexcept
{
    return std::error_condition(static_cast<int>(v), audio_error_category());
}

struct Input_stream
{
protected:
    Input_stream(const Input_stream &) = default;
    Input_stream &operator=(const Input_stream &) = default;
    Input_stream(Input_stream &&) noexcept = default;
    Input_stream &operator=(Input_stream &&) noexcept = default;

public:
    Input_stream() noexcept = default;
    virtual ~Input_stream() = default;
    virtual std::size_t read(unsigned char *buffer, std::size_t buffer_size) = 0;
};

class Shared_input_stream final : public Input_stream
{
private:
    std::shared_ptr<Input_stream> input_stream;

public:
    explicit Shared_input_stream(std::shared_ptr<Input_stream> input_stream) noexcept
        : input_stream(std::move(input_stream))
    {
    }
    explicit Shared_input_stream(std::unique_ptr<Input_stream> input_stream)
        : input_stream(std::move(input_stream))
    {
    }
    Shared_input_stream(const Shared_input_stream &) noexcept = default;
    Shared_input_stream(Shared_input_stream &&) noexcept = default;
    Shared_input_stream &operator=(const Shared_input_stream &) noexcept = default;
    Shared_input_stream &operator=(Shared_input_stream &&) noexcept = default;
    virtual std::size_t read(unsigned char *buffer, std::size_t buffer_size) override
    {
        return input_stream->read(buffer, buffer_size);
    }
};

struct Rewindable_input_stream : public Input_stream
{
protected:
    Rewindable_input_stream(const Rewindable_input_stream &) = default;
    Rewindable_input_stream &operator=(const Rewindable_input_stream &) = default;
    Rewindable_input_stream(Rewindable_input_stream &&) noexcept = default;
    Rewindable_input_stream &operator=(Rewindable_input_stream &&) noexcept = default;

public:
    Rewindable_input_stream() noexcept = default;
    virtual void rewind() = 0;
};

class File_input_stream final : public Rewindable_input_stream
{
private:
    std::ifstream is;

public:
    explicit File_input_stream(const char *file_name) : is()
    {
        is.exceptions(std::ios::badbit);
        is.open(file_name, std::ios::binary);
        if(!is)
            throw Audio_error(make_error_code(std::errc::no_such_file_or_directory), "open failed");
    }
    explicit File_input_stream(const std::string &file_name) : File_input_stream(file_name.c_str())
    {
    }
    virtual std::size_t read(unsigned char *buffer, std::size_t buffer_size) override
    {
        std::size_t retval = 0;
        while(buffer_size > 0 && is.peek() != std::char_traits<char>::eof())
        {
            *buffer++ = is.get();
            buffer_size--;
            retval++;
        }
        return retval;
    }
    virtual void rewind() override
    {
        is.seekg(0, std::ios::beg);
    }
};

enum class Channel_count : unsigned
{
    mono = 1,
    stereo = 2,
    // TODO: name more channel counts
};

class Audio_reader
{
private:
    const float sample_rate;
    const Channel_count channel_count;
    const std::size_t ideal_buffer_size_in_samples;

public:
    Audio_reader(float sample_rate,
                 Channel_count channel_count,
                 std::size_t ideal_buffer_size_in_samples) noexcept
        : sample_rate(sample_rate),
          channel_count(channel_count),
          ideal_buffer_size_in_samples(ideal_buffer_size_in_samples)
    {
    }
    virtual ~Audio_reader() noexcept = default;
    float get_sample_rate() const noexcept
    {
        return sample_rate;
    }
    Channel_count get_channel_count() const noexcept
    {
        return channel_count;
    }
    std::size_t get_ideal_buffer_size_in_samples() const noexcept
    {
        return ideal_buffer_size_in_samples;
    }
    virtual std::size_t read_samples(Audio_value_type *values, std::size_t values_size) = 0;
};

class Reformatting_audio_reader final : public Audio_reader
{
private:
    std::unique_ptr<Audio_reader> base_audio_reader;
    std::vector<float> input_buffer;
    static constexpr std::size_t buffer_size_in_samples = 0x1000;

public:
    Reformatting_audio_reader(std::unique_ptr<Audio_reader> base_audio_reader,
                              float sample_rate,
                              Channel_count channel_count)
        : Audio_reader(sample_rate,
                       channel_count,
                       static_cast<unsigned>(channel_count) * buffer_size_in_samples),
          base_audio_reader(std::move(base_audio_reader)),
          input_buffer()
    {
        if(!is_supported_channel_count(this->base_audio_reader->get_channel_count())
           || !is_supported_channel_count(get_channel_count()))
            throw Audio_error(make_error_code(Audio_error_code::unsupported),
                              "unsupported channel count");
    }
    static constexpr bool is_supported_channel_count(Channel_count channel_count) noexcept
    {
        switch(channel_count)
        {
        case Channel_count::mono:
        case Channel_count::stereo:
            return true;
        }
        return false;
    }
    /** returns the number of read values */
    virtual std::size_t read_samples(Audio_value_type *values, std::size_t values_size) override;
    std::size_t read_all_samples(Audio_value_type *values, std::size_t values_size)
    {
        std::size_t retval = 0;
        while(values_size >= static_cast<std::size_t>(get_channel_count()))
        {
            auto read_count = read_samples(values, values_size);
            if(read_count == 0)
                return retval;
            values += read_count;
            values_size -= read_count;
            retval += read_count;
        }
        return retval;
    }
};

struct Audio_format
{
    const char *name;
    /** leaves input_stream with the original input stream on exception */
    std::unique_ptr<Audio_reader> (*create_reader)(std::unique_ptr<Input_stream> &&input_stream);
    constexpr bool is_supported() const noexcept
    {
        return create_reader != nullptr;
    }
    constexpr operator bool() const noexcept
    {
        return is_supported();
    }
};

class Audio_formats
{
private:
    const Audio_format *const *formats_array_pointer;
    std::size_t formats_array_size;
    constexpr Audio_formats(const Audio_format *const *formats_array_pointer,
                            std::size_t formats_array_size) noexcept
        : formats_array_pointer(formats_array_pointer),
          formats_array_size(formats_array_size)
    {
    }

public:
    constexpr Audio_formats() noexcept : formats_array_pointer(nullptr), formats_array_size(0)
    {
    }
    constexpr std::size_t size() const noexcept
    {
        return formats_array_size;
    }
    typedef const Audio_format *const *iterator;
    typedef const Audio_format *const *const_iterator;
    constexpr const_iterator begin() const noexcept
    {
        return formats_array_pointer;
    }
    constexpr const_iterator end() const noexcept
    {
        return formats_array_pointer + formats_array_size;
    }
    static Audio_formats get() noexcept;
};
}
}

namespace std
{
template <>
struct is_error_condition_enum<lossless_neural_sound::audio::Audio_error_code> : public true_type
{
};
}

#endif /* AUDIO_AUDIO_H_ */
