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
#include "flac.h"
#include "audio_config.h"
#include <exception>
#include <utility>
#include <cstdint>
#include <new>
#include <type_traits>
#if AUDIO_HAS_FLAC
#include <FLAC/stream_decoder.h>
#endif

namespace lossless_neural_sound
{
namespace audio
{
#if AUDIO_HAS_FLAC
class Flac::Flac_audio_reader final : public Audio_reader
{
    Flac_audio_reader(const Flac_audio_reader &) = delete;
    Flac_audio_reader &operator=(const Flac_audio_reader &) = delete;

private:
    struct Stream_decoder_error_status_error_category final : public std::error_category
    {
        enum class Value : int
        {
            lost_sync = 1 + FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC,
            bad_header = 1 + FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER,
            frame_crc_mismatch = 1 + FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH,
            unparseable_stream = 1 + FLAC__STREAM_DECODER_ERROR_STATUS_UNPARSEABLE_STREAM
        };
        const char *name() const noexcept override
        {
            return "FLAC__StreamDecoderErrorStatus";
        }
        std::string message(int code) const override
        {
            switch(static_cast<Value>(code))
            {
            case Value::lost_sync:
                return FLAC__StreamDecoderErrorStatusString
                    [FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC];
            case Value::bad_header:
                return FLAC__StreamDecoderErrorStatusString
                    [FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER];
            case Value::frame_crc_mismatch:
                return FLAC__StreamDecoderErrorStatusString
                    [FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH];
            case Value::unparseable_stream:
                return FLAC__StreamDecoderErrorStatusString
                    [FLAC__STREAM_DECODER_ERROR_STATUS_UNPARSEABLE_STREAM];
            }
            return "unknown";
        }
        std::error_condition default_error_condition(int code) const noexcept override
        {
            switch(static_cast<Value>(code))
            {
            case Value::lost_sync:
            case Value::bad_header:
            case Value::frame_crc_mismatch:
            case Value::unparseable_stream:
                return Audio_error_code::corrupt_file;
            }
            return std::error_condition(code, *this);
        }
        static std::error_code from_flac_error(FLAC__StreamDecoderErrorStatus v) noexcept
        {
            return std::error_code(static_cast<int>(v) + 1, get());
        }
        static const Stream_decoder_error_status_error_category &get() noexcept
        {
            static const Stream_decoder_error_status_error_category retval;
            return retval;
        }
    };
    struct Stream_decoder_init_status_error_category final : public std::error_category
    {
        const char *name() const noexcept override
        {
            return "FLAC__StreamDecoderInitStatus";
        }
        std::string message(int code) const override
        {
            switch(static_cast<FLAC__StreamDecoderInitStatus>(code))
            {
            case FLAC__STREAM_DECODER_INIT_STATUS_OK:
                return FLAC__StreamDecoderInitStatusString[FLAC__STREAM_DECODER_INIT_STATUS_OK];
            case FLAC__STREAM_DECODER_INIT_STATUS_UNSUPPORTED_CONTAINER:
                return FLAC__StreamDecoderInitStatusString
                    [FLAC__STREAM_DECODER_INIT_STATUS_UNSUPPORTED_CONTAINER];
            case FLAC__STREAM_DECODER_INIT_STATUS_INVALID_CALLBACKS:
                return FLAC__StreamDecoderInitStatusString
                    [FLAC__STREAM_DECODER_INIT_STATUS_INVALID_CALLBACKS];
            case FLAC__STREAM_DECODER_INIT_STATUS_MEMORY_ALLOCATION_ERROR:
                return FLAC__StreamDecoderInitStatusString
                    [FLAC__STREAM_DECODER_INIT_STATUS_MEMORY_ALLOCATION_ERROR];
            case FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE:
                return FLAC__StreamDecoderInitStatusString
                    [FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE];
            case FLAC__STREAM_DECODER_INIT_STATUS_ALREADY_INITIALIZED:
                return FLAC__StreamDecoderInitStatusString
                    [FLAC__STREAM_DECODER_INIT_STATUS_ALREADY_INITIALIZED];
            }
            return "unknown";
        }
        std::error_condition default_error_condition(int code) const noexcept override
        {
            switch(static_cast<FLAC__StreamDecoderInitStatus>(code))
            {
            case FLAC__STREAM_DECODER_INIT_STATUS_OK:
                return std::error_condition();
            case FLAC__STREAM_DECODER_INIT_STATUS_UNSUPPORTED_CONTAINER:
                return Audio_error_code::unsupported;
            case FLAC__STREAM_DECODER_INIT_STATUS_INVALID_CALLBACKS:
                return std::errc::invalid_argument;
            case FLAC__STREAM_DECODER_INIT_STATUS_MEMORY_ALLOCATION_ERROR:
                return std::errc::not_enough_memory;
            case FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE:
                return std::errc::io_error;
            case FLAC__STREAM_DECODER_INIT_STATUS_ALREADY_INITIALIZED:
                return std::errc::invalid_argument;
            }
            return std::error_condition(code, *this);
        }
        static std::error_code from_flac_error(FLAC__StreamDecoderInitStatus v) noexcept
        {
            return std::error_code(static_cast<int>(v), get());
        }
        static const Stream_decoder_init_status_error_category &get() noexcept
        {
            static const Stream_decoder_init_status_error_category retval;
            return retval;
        }
    };
    struct Flac_audio_error : public Audio_error
    {
        Flac_audio_error() : Audio_error()
        {
        }
        Flac_audio_error(std::error_code ec) : Audio_error(ec)
        {
        }
        Flac_audio_error(std::error_code ec, const std::string &what) : Audio_error(ec, what)
        {
        }
        Flac_audio_error(std::error_code ec, const char *what) : Audio_error(ec, what)
        {
        }
        Flac_audio_error(int error, const std::error_category &category)
            : Audio_error(error, category)
        {
        }
        Flac_audio_error(int error, const std::error_category &category, const std::string &what)
            : Audio_error(error, category, what)
        {
        }
        Flac_audio_error(int error, const std::error_category &category, const char *what)
            : Audio_error(error, category, what)
        {
        }
        explicit Flac_audio_error(FLAC__StreamDecoderErrorStatus v)
            : Audio_error(Stream_decoder_error_status_error_category::from_flac_error(v))
        {
        }
    };
    struct Stream_decoder
    {
        FLAC__StreamDecoder *value;
        Stream_decoder() noexcept : value(nullptr)
        {
        }
        explicit Stream_decoder(FLAC__StreamDecoder *value) noexcept : value(value)
        {
        }
        Stream_decoder(Stream_decoder &&rt) noexcept : value(rt.value)
        {
            rt.value = nullptr;
        }
        ~Stream_decoder() noexcept
        {
            if(value)
                FLAC__stream_decoder_delete(value);
        }
        Stream_decoder &operator=(Stream_decoder rt) noexcept
        {
            using std::swap;
            swap(value, rt.value);
            return *this;
        }
        FLAC__StreamDecoder *get() const noexcept
        {
            return value;
        }
        explicit operator bool() const noexcept
        {
            return value != nullptr;
        }
    };
    struct State
    {
        std::unique_ptr<Input_stream> input_stream;
        Stream_decoder stream_decoder;
        std::exception_ptr exception;
        std::vector<float> output_buffer{};
        std::size_t channel_count{};
        unsigned sample_rate{};
        std::size_t max_block_size{};
        bool got_metadata = false;
        std::size_t read_position = 0;
        FLAC__StreamDecoderReadStatus read_callback(FLAC__byte *buffer, std::size_t *bytes)
        {
            static_assert(std::is_same<FLAC__byte, unsigned char>::value, "");
            if(*bytes == 0)
                return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
            *bytes = input_stream->read(buffer, *bytes);
            if(*bytes == 0)
                return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
            return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
        }
        FLAC__StreamDecoderWriteStatus write_callback(const FLAC__Frame *frame,
                                                      const FLAC__int32 *const *buffer)
        {
            if(frame->header.channels != channel_count)
                throw Flac_audio_error(make_error_code(Audio_error_code::unsupported),
                                       "dynamic channel count change is not supported");
            if(frame->header.sample_rate != sample_rate)
                throw Flac_audio_error(make_error_code(Audio_error_code::unsupported),
                                       "dynamic sample rate change is not supported");
            if(frame->header.bits_per_sample < 1 || frame->header.bits_per_sample > 32)
                throw Flac_audio_error(make_error_code(Audio_error_code::corrupt_file),
                                       "bits per sample is out of range");
            std::size_t block_size = frame->header.blocksize;
            if(block_size == 0)
                throw Flac_audio_error(make_error_code(Audio_error_code::corrupt_file),
                                       "block size is 0");
            std::size_t output_buffer_original_size = output_buffer.size();
            output_buffer.resize(output_buffer_original_size + channel_count * block_size);
            float scale_factor =
                static_cast<float>(0x1'0000'0000ULL >> frame->header.bits_per_sample)
                * static_cast<float>(1.0 / 0x8000'0000UL);
            for(std::size_t channel = 0; channel < channel_count; channel++)
            {
                auto *channel_buffer = buffer[channel];
                if(block_size != 0)
                {
                    auto *output = output_buffer.data() + output_buffer_original_size + channel;
                    *output = channel_buffer[0] * scale_factor;
                    for(std::size_t sample = 1; sample < block_size; sample++)
                    {
                        output += channel_count;
                        *output = channel_buffer[sample] * scale_factor;
                    }
                }
            }
            return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
        }
        void metadata_callback(const FLAC__StreamMetadata *metadata) noexcept
        {
            if(metadata->type == FLAC__METADATA_TYPE_STREAMINFO)
            {
                sample_rate = metadata->data.stream_info.sample_rate;
                channel_count = metadata->data.stream_info.channels;
                max_block_size = metadata->data.stream_info.max_blocksize;
                got_metadata = true;
            }
        }
        void error_callback(FLAC__StreamDecoderErrorStatus status)
        {
            throw Flac_audio_error(status);
        }
    };

private:
    std::unique_ptr<State> state;

private:
    explicit Flac_audio_reader(float sample_rate,
                               Channel_count channel_count,
                               std::size_t ideal_buffer_size_in_samples) noexcept
        : Audio_reader(sample_rate, channel_count, ideal_buffer_size_in_samples),
          state()
    {
    }
    static FLAC__StreamDecoderReadStatus read_callback(
        [[gnu::unused]] const FLAC__StreamDecoder *decoder,
        FLAC__byte *buffer,
        std::size_t *bytes,
        void *client_data) noexcept
    {
        auto *state = static_cast<State *>(client_data);
        try
        {
            return state->read_callback(buffer, bytes);
        }
        catch(...)
        {
            state->exception = std::current_exception();
            return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
        }
    }
    static FLAC__StreamDecoderWriteStatus write_callback(
        [[gnu::unused]] const FLAC__StreamDecoder *decoder,
        const FLAC__Frame *frame,
        const FLAC__int32 *const *buffer,
        void *client_data) noexcept
    {
        auto *state = static_cast<State *>(client_data);
        try
        {
            return state->write_callback(frame, buffer);
        }
        catch(...)
        {
            state->exception = std::current_exception();
            return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
        }
    }
    static void metadata_callback([[gnu::unused]] const FLAC__StreamDecoder *decoder,
                                  const FLAC__StreamMetadata *metadata,
                                  void *client_data) noexcept
    {
        static_cast<State *>(client_data)->metadata_callback(metadata);
    }
    static void error_callback([[gnu::unused]] const FLAC__StreamDecoder *decoder,
                               FLAC__StreamDecoderErrorStatus status,
                               void *client_data) noexcept
    {
        auto *state = static_cast<State *>(client_data);
        try
        {
            state->error_callback(status);
        }
        catch(...)
        {
            state->exception = std::current_exception();
        }
    }

public:
    static std::unique_ptr<Audio_reader> create_reader(std::unique_ptr<Input_stream> &&input_stream)
    {
        auto state = std::unique_ptr<State>(new State);
        state->input_stream = std::move(input_stream);
        try
        {
            state->stream_decoder = Stream_decoder(FLAC__stream_decoder_new());
            if(!state->stream_decoder)
                throw std::bad_alloc();
            constexpr auto seek_callback = nullptr;
            constexpr auto tell_callback = nullptr;
            constexpr auto length_callback = nullptr;
            constexpr auto eof_callback = nullptr;
            auto init_result = FLAC__stream_decoder_init_stream(state->stream_decoder.get(),
                                                                read_callback,
                                                                seek_callback,
                                                                tell_callback,
                                                                length_callback,
                                                                eof_callback,
                                                                write_callback,
                                                                metadata_callback,
                                                                error_callback,
                                                                state.get());
            if(state->exception)
                std::rethrow_exception(state->exception);
            switch(init_result)
            {
            case FLAC__STREAM_DECODER_INIT_STATUS_OK:
                break;
            case FLAC__STREAM_DECODER_INIT_STATUS_MEMORY_ALLOCATION_ERROR:
                throw std::bad_alloc();
            default:
                throw Flac_audio_error(
                    Stream_decoder_init_status_error_category::from_flac_error(init_result),
                    "FLAC__stream_decoder_init_ogg_stream");
            }
            bool read_metadata_succeded =
                FLAC__stream_decoder_process_until_end_of_metadata(state->stream_decoder.get());
            if(state->exception)
                std::rethrow_exception(state->exception);
            if(!read_metadata_succeded)
            {
                auto decoder_state = FLAC__stream_decoder_get_state(state->stream_decoder.get());
                switch(decoder_state)
                {
                case FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR:
                    throw std::bad_alloc();
                case FLAC__STREAM_DECODER_END_OF_STREAM:
                case FLAC__STREAM_DECODER_OGG_ERROR:
                    throw Flac_audio_error(make_error_code(Audio_error_code::corrupt_file),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                case FLAC__STREAM_DECODER_SEEK_ERROR:
                    throw Flac_audio_error(make_error_code(std::errc::invalid_seek),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                case FLAC__STREAM_DECODER_SEARCH_FOR_METADATA:
                case FLAC__STREAM_DECODER_READ_METADATA:
                case FLAC__STREAM_DECODER_SEARCH_FOR_FRAME_SYNC:
                case FLAC__STREAM_DECODER_READ_FRAME:
                case FLAC__STREAM_DECODER_ABORTED:
                case FLAC__STREAM_DECODER_UNINITIALIZED:
                    throw Flac_audio_error(make_error_code(std::errc::invalid_argument),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                }
                throw Flac_audio_error(make_error_code(std::errc::invalid_argument),
                                       "FLAC__stream_decoder_process_until_end_of_metadata");
            }
            if(state->channel_count == 0)
                throw Flac_audio_error(make_error_code(Audio_error_code::unsupported),
                                       "channel_count is 0");
            if(state->channel_count > 0x1000U)
                throw Flac_audio_error(make_error_code(Audio_error_code::unsupported),
                                       "channel_count is too big");
            if(state->sample_rate == 0)
                throw Flac_audio_error(make_error_code(Audio_error_code::unsupported),
                                       "sample_rate is 0");
            std::size_t ideal_buffer_size_in_samples = state->max_block_size * state->channel_count;
            auto retval = std::unique_ptr<Audio_reader>(
                new Flac_audio_reader(state->sample_rate,
                                      static_cast<Channel_count>(state->channel_count),
                                      ideal_buffer_size_in_samples));
            static_cast<Flac_audio_reader *>(retval.get())->state =
                std::move(state); // move after we can't throw anymore
            return retval;
        }
        catch(...)
        {
            // move back on exception, so caller can reuse the input stream
            input_stream = std::move(state->input_stream);
            throw;
        }
    }
    virtual std::size_t read_samples(Audio_value_type *values, std::size_t values_size) override
    {
        if(values_size < state->channel_count)
            return 0;
        while(state->read_position >= state->output_buffer.size())
        {
            state->read_position = 0;
            state->output_buffer.clear();
            bool process_result = FLAC__stream_decoder_process_single(state->stream_decoder.get());
            if(state->exception)
                std::rethrow_exception(state->exception);
            if(!process_result)
            {
                auto decoder_state = FLAC__stream_decoder_get_state(state->stream_decoder.get());
                switch(decoder_state)
                {
                case FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR:
                    throw std::bad_alloc();
                case FLAC__STREAM_DECODER_END_OF_STREAM:
                    return 0;
                case FLAC__STREAM_DECODER_OGG_ERROR:
                    throw Flac_audio_error(make_error_code(Audio_error_code::corrupt_file),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                case FLAC__STREAM_DECODER_SEEK_ERROR:
                    throw Flac_audio_error(make_error_code(std::errc::invalid_seek),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                case FLAC__STREAM_DECODER_SEARCH_FOR_METADATA:
                case FLAC__STREAM_DECODER_READ_METADATA:
                case FLAC__STREAM_DECODER_SEARCH_FOR_FRAME_SYNC:
                case FLAC__STREAM_DECODER_READ_FRAME:
                case FLAC__STREAM_DECODER_ABORTED:
                case FLAC__STREAM_DECODER_UNINITIALIZED:
                    throw Flac_audio_error(make_error_code(std::errc::invalid_argument),
                                           "FLAC__stream_decoder_process_until_end_of_metadata");
                }
                throw Flac_audio_error(make_error_code(std::errc::invalid_argument),
                                       "FLAC__stream_decoder_process_until_end_of_metadata");
            }
            auto decoder_state = FLAC__stream_decoder_get_state(state->stream_decoder.get());
            switch(decoder_state)
            {
            case FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR:
                throw std::bad_alloc();
            case FLAC__STREAM_DECODER_END_OF_STREAM:
                return 0;
            case FLAC__STREAM_DECODER_OGG_ERROR:
                throw Flac_audio_error(make_error_code(Audio_error_code::corrupt_file),
                                       "FLAC__stream_decoder_process_until_end_of_metadata");
            case FLAC__STREAM_DECODER_SEEK_ERROR:
                throw Flac_audio_error(make_error_code(std::errc::invalid_seek),
                                       "FLAC__stream_decoder_process_until_end_of_metadata");
            case FLAC__STREAM_DECODER_SEARCH_FOR_METADATA:
            case FLAC__STREAM_DECODER_READ_METADATA:
            case FLAC__STREAM_DECODER_SEARCH_FOR_FRAME_SYNC:
            case FLAC__STREAM_DECODER_READ_FRAME:
                break;
            case FLAC__STREAM_DECODER_ABORTED:
            case FLAC__STREAM_DECODER_UNINITIALIZED:
                throw Flac_audio_error(make_error_code(std::errc::invalid_argument),
                                       "FLAC__stream_decoder_process_until_end_of_metadata");
            }
        }
        static_assert(std::is_same<float, Audio_value_type>::value, "");
        std::size_t retval = 0;
        while(state->read_position < state->output_buffer.size())
        {
            if(values_size < state->channel_count)
                return retval;
            for(std::size_t i = 0; i < state->channel_count; i++, values_size--, retval++)
                *values++ = state->output_buffer[state->read_position++];
        }
        return retval;
    }
};
#endif

const Audio_format Flac::audio_format = {
    "FLAC",
#if AUDIO_HAS_FLAC
    &Flac_audio_reader::create_reader,
#else
    nullptr
#endif
};
}
}
