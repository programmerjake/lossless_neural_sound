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
#include "audio.h"
#include "flac.h"
#include "audio_config.h"

namespace lossless_neural_sound
{
namespace audio
{
const Audio_error_category &audio_error_category() noexcept
{
    static const Audio_error_category retval;
    return retval;
}

const char *Audio_error_category::name() const noexcept
{
    return "audio";
}

std::string Audio_error_category::message(int condition) const
{
    switch(static_cast<Audio_error_code>(condition))
    {
    case Audio_error_code::format_does_not_match:
        return "format does not match";
    case Audio_error_code::corrupt_file:
        return "corrupt file";
    case Audio_error_code::unsupported:
        return "unsupported";
    }
    return "unknown error";
}

Audio_formats Audio_formats::get() noexcept
{
    static constexpr const Audio_format *audio_formats[] =
    {
#if AUDIO_HAS_FLAC
        &Flac::audio_format,
#endif
        nullptr,
    };
    constexpr std::size_t terminating_null_format_size = 1;
    return Audio_formats(
        audio_formats,
        sizeof(audio_formats) / sizeof(audio_formats[0]) - terminating_null_format_size);
}
}
}
