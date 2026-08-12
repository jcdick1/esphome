#include "micro_wake_word.h"

#ifdef USE_ESP32

#include "esphome/core/application.h"
#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

#include "esphome/components/audio/audio_transfer_buffer.h"

#ifdef USE_OTA
#include "esphome/components/ota/ota_backend.h"
#endif

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
#include <esp_http_client.h>
#include <cstring>
#endif

namespace esphome::micro_wake_word {

static const char *const TAG = "micro_wake_word";

static const ssize_t DETECTION_QUEUE_LENGTH = 5;

static const size_t DATA_TIMEOUT_MS = 50;

static const uint32_t RING_BUFFER_DURATION_MS = 120;

#ifdef CONFIG_IDF_TARGET_ESP32P4
// ESP32-P4 PIE-optimized esp-nn kernels (e.g. depthwise_conv_s8_ch1_pie) require
// significantly more stack than other variants, causing stack protection faults at 3072.
static const uint32_t INFERENCE_TASK_STACK_SIZE = 8192;
#else
static const uint32_t INFERENCE_TASK_STACK_SIZE = 3072;
#endif
static const UBaseType_t INFERENCE_TASK_PRIORITY = 3;

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
// Only one capture is ever in flight, so the queue exists to move work off the inference task rather than to buffer.
static const ssize_t CAPTURE_QUEUE_LENGTH = 1;
// The HTTP client needs considerably more stack than the inference task
static const uint32_t CAPTURE_TASK_STACK_SIZE = 6144;
// Below the inference task: uploading must never delay feeding the frontend
static const UBaseType_t CAPTURE_TASK_PRIORITY = 1;
static const size_t CAPTURE_WRITE_CHUNK_BYTES = 2048;
static const int CAPTURE_HTTP_TIMEOUT_MS = 10000;
static const size_t WAV_HEADER_BYTES = 44;
// A sustained near miss keeps clearing the capture cutoff on every new probability, so captures are rate limited
static const uint32_t CAPTURE_MIN_INTERVAL_MS = 2000;
#endif

enum EventGroupBits : uint32_t {
  COMMAND_STOP = (1 << 0),               // Signals the inference task should stop
  COMMAND_RESET_RING_BUFFER = (1 << 1),  // Signals the inference task to discard buffered audio

  TASK_STARTING = (1 << 3),
  TASK_RUNNING = (1 << 4),
  TASK_STOPPING = (1 << 5),
  TASK_STOPPED = (1 << 6),

  ERROR_MEMORY = (1 << 9),
  ERROR_INFERENCE = (1 << 10),

  WARNING_FULL_RING_BUFFER = (1 << 13),

  ERROR_BITS = ERROR_MEMORY | ERROR_INFERENCE,
  ALL_BITS = 0xfffff,  // 24 total bits available in an event group
};

float MicroWakeWord::get_setup_priority() const { return setup_priority::AFTER_CONNECTION; }

static const LogString *micro_wake_word_state_to_string(State state) {
  switch (state) {
    case State::STARTING:
      return LOG_STR("STARTING");
    case State::DETECTING_WAKE_WORD:
      return LOG_STR("DETECTING_WAKE_WORD");
    case State::STOPPING:
      return LOG_STR("STOPPING");
    case State::STOPPED:
      return LOG_STR("STOPPED");
    default:
      return LOG_STR("UNKNOWN");
  }
}

void MicroWakeWord::dump_config() {
  ESP_LOGCONFIG(TAG, "microWakeWord:");
  ESP_LOGCONFIG(TAG, "  models:");
  for (auto &model : this->wake_word_models_) {
    model->log_model_config();
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  this->vad_model_->log_model_config();
#endif
}

void MicroWakeWord::setup() {
  this->frontend_config_.window.size_ms = FEATURE_DURATION_MS;
  this->frontend_config_.window.step_size_ms = this->features_step_size_;
  this->frontend_config_.filterbank.num_channels = PREPROCESSOR_FEATURE_SIZE;
  this->frontend_config_.filterbank.lower_band_limit = FILTERBANK_LOWER_BAND_LIMIT;
  this->frontend_config_.filterbank.upper_band_limit = FILTERBANK_UPPER_BAND_LIMIT;
  this->frontend_config_.noise_reduction.smoothing_bits = NOISE_REDUCTION_SMOOTHING_BITS;
  this->frontend_config_.noise_reduction.even_smoothing = NOISE_REDUCTION_EVEN_SMOOTHING;
  this->frontend_config_.noise_reduction.odd_smoothing = NOISE_REDUCTION_ODD_SMOOTHING;
  this->frontend_config_.noise_reduction.min_signal_remaining = NOISE_REDUCTION_MIN_SIGNAL_REMAINING;
  this->frontend_config_.pcan_gain_control.enable_pcan = PCAN_GAIN_CONTROL_ENABLE_PCAN;
  this->frontend_config_.pcan_gain_control.strength = PCAN_GAIN_CONTROL_STRENGTH;
  this->frontend_config_.pcan_gain_control.offset = PCAN_GAIN_CONTROL_OFFSET;
  this->frontend_config_.pcan_gain_control.gain_bits = PCAN_GAIN_CONTROL_GAIN_BITS;
  this->frontend_config_.log_scale.enable_log = LOG_SCALE_ENABLE_LOG;
  this->frontend_config_.log_scale.scale_shift = LOG_SCALE_SCALE_SHIFT;

  this->event_group_ = xEventGroupCreate();
  if (this->event_group_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create event group");
    this->mark_failed();
    return;
  }

  this->detection_queue_ = xQueueCreate(DETECTION_QUEUE_LENGTH, sizeof(DetectionEvent));
  if (this->detection_queue_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create detection event queue");
    this->mark_failed();
    return;
  }

  this->microphone_source_->add_data_callback([this](const std::vector<uint8_t> &data) {
    if (this->state_ == State::STOPPED) {
      return;
    }
    std::shared_ptr<ring_buffer::RingBuffer> temp_ring_buffer = this->ring_buffer_.lock();
    if (this->ring_buffer_.use_count() > 1) {
      // Producer-only write: never touches consumer state. If the buffer is full, ask the inference task
      // to drain it - reset() is a consumer operation and must run on the inference task's thread.
      // Disable partial writes so audio chunks are either fully accepted or rejected and handled below.
      if (temp_ring_buffer->write_without_replacement(data.data(), data.size(), 0, false) == 0) {
        xEventGroupSetBits(this->event_group_,
                           EventGroupBits::WARNING_FULL_RING_BUFFER | EventGroupBits::COMMAND_RESET_RING_BUFFER);
      }
    }
  });

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
  if (this->capture_url_ != nullptr) {
    if (!this->setup_capture_()) {
      // Capture is a diagnostic aid, so a failure here leaves wake word detection running as normal
      ESP_LOGW(TAG, "Could not set up trigger window capture; continuing without it");
    }
  }
#endif

#ifdef USE_OTA_STATE_LISTENER
  ota::get_global_ota_callback()->add_global_state_listener(this);
#endif
}

#ifdef USE_OTA_STATE_LISTENER
void MicroWakeWord::on_ota_global_state(ota::OTAState state, float progress, uint8_t error, ota::OTAComponent *comp) {
  if (state == ota::OTA_STARTED) {
    this->suspend_task_();
  } else if (state == ota::OTA_ERROR) {
    this->resume_task_();
  }
}
#endif

void MicroWakeWord::inference_task(void *params) {
  MicroWakeWord *this_mww = (MicroWakeWord *) params;

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STARTING);

  {  // Ensures any C++ objects fall out of scope to deallocate before deleting the task

    const auto &stream_info = this_mww->microphone_source_->get_audio_stream_info();
    const size_t bytes_per_frame = stream_info.frames_to_bytes(1);
    const size_t max_fill_bytes = stream_info.ms_to_bytes(this_mww->features_step_size_);
    std::unique_ptr<audio::RingBufferAudioSource> audio_source;
    int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE];

    if (!(xEventGroupGetBits(this_mww->event_group_) & ERROR_BITS)) {
      // Round ring buffer size down to a frame multiple so the wrap boundary never splits an int16 sample.
      const size_t ring_buffer_size =
          (stream_info.ms_to_bytes(RING_BUFFER_DURATION_MS) / bytes_per_frame) * bytes_per_frame;
      std::shared_ptr<ring_buffer::RingBuffer> temp_ring_buffer = ring_buffer::RingBuffer::create(ring_buffer_size);
      if (temp_ring_buffer == nullptr) {
        xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_MEMORY);
      } else {
        audio_source = audio::RingBufferAudioSource::create(temp_ring_buffer, max_fill_bytes,
                                                            static_cast<uint8_t>(bytes_per_frame));
        if (audio_source == nullptr) {
          xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_MEMORY);
        } else {
          this_mww->ring_buffer_ = temp_ring_buffer;
        }
      }
    }

    if (!(xEventGroupGetBits(this_mww->event_group_) & ERROR_BITS)) {
      this_mww->microphone_source_->start();
      xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_RUNNING);

      while (!(xEventGroupGetBits(this_mww->event_group_) & (COMMAND_STOP | ERROR_BITS))) {
        if (xEventGroupGetBits(this_mww->event_group_) & EventGroupBits::COMMAND_RESET_RING_BUFFER) {
          // Producer asked us to drain; run the consumer-side reset from this thread.
          audio_source->clear_buffered_data();
          xEventGroupClearBits(this_mww->event_group_, EventGroupBits::COMMAND_RESET_RING_BUFFER);
        }

        audio_source->fill(pdMS_TO_TICKS(DATA_TIMEOUT_MS), false);

        // The frontend buffers samples internally and only emits a feature once it has a full window, so we can
        // hand it whatever the source exposes. The frontend consumes at least one sample per call, so available()
        // strictly decreases and this loop always terminates.
        while (audio_source->available() >= sizeof(int16_t)) {
          const size_t samples_available = audio_source->available() / sizeof(int16_t);
          const int16_t *audio_data = reinterpret_cast<const int16_t *>(audio_source->data());

          size_t processed_samples = 0;
          const bool feature_generated =
              this_mww->generate_features_(audio_data, samples_available, features_buffer, &processed_samples);
#ifdef USE_MICRO_WAKE_WORD_CAPTURE
          // Mirror exactly what the frontend consumed, before consume() releases it back to the source
          this_mww->capture_write_(audio_data, processed_samples);
#endif
          audio_source->consume(processed_samples * sizeof(int16_t));

          if (feature_generated) {
            if (!this_mww->update_model_probabilities_(features_buffer)) {
              xEventGroupSetBits(this_mww->event_group_, EventGroupBits::ERROR_INFERENCE);
              break;
            }

            // Process each model's probabilities and possibly send a Detection Event to the queue
            this_mww->process_probabilities_();
          }
        }
      }
    }
  }

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STOPPING);

  this_mww->unload_models_();
  this_mww->microphone_source_->stop();
  FrontendFreeStateContents(&this_mww->frontend_state_);

  xEventGroupSetBits(this_mww->event_group_, EventGroupBits::TASK_STOPPED);
  vTaskSuspend(nullptr);  // Suspend this task indefinitely until the loop method deletes it
}

std::vector<WakeWordModel *> MicroWakeWord::get_wake_words() {
  std::vector<WakeWordModel *> external_wake_word_models;
  for (auto *model : this->wake_word_models_) {
    if (!model->get_internal_only()) {
      external_wake_word_models.push_back(model);
    }
  }
  return external_wake_word_models;
}

void MicroWakeWord::add_wake_word_model(WakeWordModel *model) { this->wake_word_models_.push_back(model); }

#ifdef USE_MICRO_WAKE_WORD_VAD
void MicroWakeWord::add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                                  size_t tensor_arena_size) {
  this->vad_model_ = make_unique<VADModel>(model_start, probability_cutoff, sliding_window_size, tensor_arena_size);
}
#endif

void MicroWakeWord::suspend_task_() {
  if (this->inference_task_.is_created()) {
    vTaskSuspend(this->inference_task_.get_handle());
  }
}

void MicroWakeWord::resume_task_() {
  if (this->inference_task_.is_created()) {
    vTaskResume(this->inference_task_.get_handle());
  }
}

void MicroWakeWord::loop() {
  uint32_t event_group_bits = xEventGroupGetBits(this->event_group_);

  if (event_group_bits & EventGroupBits::ERROR_MEMORY) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::ERROR_MEMORY);
    ESP_LOGE(TAG, "Encountered an error allocating buffers");
  }

  if (event_group_bits & EventGroupBits::ERROR_INFERENCE) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::ERROR_INFERENCE);
    ESP_LOGE(TAG, "Encountered an error while performing an inference");
  }

  if (event_group_bits & EventGroupBits::WARNING_FULL_RING_BUFFER) {
    xEventGroupClearBits(this->event_group_, EventGroupBits::WARNING_FULL_RING_BUFFER);
    ESP_LOGW(TAG, "Not enough free bytes in ring buffer to store incoming audio data. Resetting the ring buffer. Wake "
                  "word detection accuracy will temporarily be reduced.");
  }

  if (event_group_bits & EventGroupBits::TASK_STARTING) {
    ESP_LOGD(TAG, "Inference task has started, attempting to allocate memory for buffers");
    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_STARTING);
  }

  if (event_group_bits & EventGroupBits::TASK_RUNNING) {
    ESP_LOGD(TAG, "Inference task is running");

    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_RUNNING);
    this->set_state_(State::DETECTING_WAKE_WORD);
  }

  if (event_group_bits & EventGroupBits::TASK_STOPPING) {
    ESP_LOGD(TAG, "Inference task is stopping, deallocating buffers");
    xEventGroupClearBits(this->event_group_, EventGroupBits::TASK_STOPPING);
  }

  if ((event_group_bits & EventGroupBits::TASK_STOPPED)) {
    ESP_LOGD(TAG, "Inference task is finished, freeing task resources");
    this->inference_task_.deallocate();
    xEventGroupClearBits(this->event_group_, ALL_BITS);
    xQueueReset(this->detection_queue_);
    this->set_state_(State::STOPPED);
  }

  if ((this->pending_start_) && (this->state_ == State::STOPPED)) {
    this->set_state_(State::STARTING);
    this->pending_start_ = false;
  }

  if ((this->pending_stop_) && (this->state_ == State::DETECTING_WAKE_WORD)) {
    this->set_state_(State::STOPPING);
    this->pending_stop_ = false;
  }

  switch (this->state_) {
    case State::STARTING:
      if (!this->inference_task_.is_created() && !this->status_has_error()) {
        // Setup preprocesor feature generator. If done in the task, it would lock the task to its initial core, as it
        // uses floating point operations.
        if (!FrontendPopulateState(&this->frontend_config_, &this->frontend_state_,
                                   this->microphone_source_->get_audio_stream_info().get_sample_rate())) {
          this->status_momentary_error("frontend_alloc", 1000);
          return;
        }

        if (!this->inference_task_.create(MicroWakeWord::inference_task, "mww", INFERENCE_TASK_STACK_SIZE,
                                          (void *) this, INFERENCE_TASK_PRIORITY, this->task_stack_in_psram_)) {
          FrontendFreeStateContents(&this->frontend_state_);  // Deallocate frontend state
          this->status_momentary_error("task_start", 1000);
        }
      }
      break;
    case State::DETECTING_WAKE_WORD: {
      DetectionEvent detection_event;
      while (xQueueReceive(this->detection_queue_, &detection_event, 0)) {
        if (detection_event.blocked_by_vad) {
          ESP_LOGD(TAG, "Wake word model predicts '%s', but VAD model doesn't.", detection_event.wake_word->c_str());
        } else {
          constexpr float uint8_to_float_divisor =
              255.0f;  // Converting a quantized uint8 probability to floating point
          ESP_LOGD(TAG, "Detected '%s' with sliding average probability is %.2f and max probability is %.2f",
                   detection_event.wake_word->c_str(), (detection_event.average_probability / uint8_to_float_divisor),
                   (detection_event.max_probability / uint8_to_float_divisor));
          this->wake_word_detected_trigger_.trigger(*detection_event.wake_word);
          if (this->stop_after_detection_) {
            this->stop();
          }
        }
      }
      break;
    }
    case State::STOPPING:
      xEventGroupSetBits(this->event_group_, EventGroupBits::COMMAND_STOP);
      break;
    case State::STOPPED:
      break;
  }
}

void MicroWakeWord::start() {
  if (!this->is_ready()) {
    ESP_LOGW(TAG, "Wake word detection can't start as the component hasn't been setup yet");
    return;
  }

  if (this->is_failed()) {
    ESP_LOGW(TAG, "Wake word component is marked as failed. Please check setup logs");
    return;
  }

  if (this->is_running()) {
    ESP_LOGW(TAG, "Wake word detection is already running");
    return;
  }

  ESP_LOGD(TAG, "Starting wake word detection");

  this->pending_start_ = true;
  this->pending_stop_ = false;
}

void MicroWakeWord::stop() {
  if (this->state_ == STOPPED)
    return;

  ESP_LOGD(TAG, "Stopping wake word detection");

  this->pending_start_ = false;
  this->pending_stop_ = true;
}

void MicroWakeWord::set_state_(State state) {
  if (this->state_ != state) {
    ESP_LOGD(TAG, "State changed from %s to %s", LOG_STR_ARG(micro_wake_word_state_to_string(this->state_)),
             LOG_STR_ARG(micro_wake_word_state_to_string(state)));
    this->state_ = state;
  }
}

bool MicroWakeWord::generate_features_(const int16_t *audio_buffer, size_t samples_available,
                                       int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE], size_t *processed_samples) {
  *processed_samples = 0;
  struct FrontendOutput frontend_output =
      FrontendProcessSamples(&this->frontend_state_, audio_buffer, samples_available, processed_samples);

  if (frontend_output.size == 0) {
    return false;
  }

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // These scaling values are set to match the TFLite audio frontend int8 output.
    // The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
    // range. In training, these are then arbitrarily divided by 25.6 to get
    // float values in the rough range of 0.0 to 26.0. This scaling is performed
    // for historical reasons, to match up with the output of other feature
    // generators.
    // The process is then further complicated when we quantize the model. This
    // means we have to scale the 0.0 to 26.0 real values to the -128 (INT8_MIN)
    // to 127 (INT8_MAX) signed integer numbers.
    // All this means that to get matching values from our integer feature
    // output into the tensor input, we have to perform:
    // input = (((feature / 25.6) / 26.0) * 256) - 128
    // To simplify this and perform it in 32-bit integer math, we rearrange to:
    // input = (feature * 256) / (25.6 * 26.0) - 128
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = 666;  // 666 = 25.6 * 26.0 after rounding
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;

    value += INT8_MIN;  // Adds a -128; i.e., subtracts 128
    features_buffer[i] = static_cast<int8_t>(clamp<int32_t>(value, INT8_MIN, INT8_MAX));
  }

  return true;
}

void MicroWakeWord::process_probabilities_() {
#ifdef USE_MICRO_WAKE_WORD_VAD
  DetectionEvent vad_state = this->vad_model_->determine_detected();

  this->vad_state_ = vad_state.detected;  // atomic write, so thread safe
#endif

  for (auto &model : this->wake_word_models_) {
    if (model->get_unprocessed_probability_status()) {
      // Only detect wake words if there is a new probability since the last check
      DetectionEvent wake_word_state = model->determine_detected();

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
      // Capture the audio behind anything that fired, plus anything that came close enough to be worth training
      // against. capture_enqueue_ rate limits, so a sustained near miss can't flood the upload task.
      if (wake_word_state.detected || ((this->capture_probability_cutoff_ > 0) &&
                                       (wake_word_state.average_probability >= this->capture_probability_cutoff_))) {
#ifdef USE_MICRO_WAKE_WORD_VAD
        wake_word_state.blocked_by_vad = wake_word_state.detected && !vad_state.detected;
#endif
        this->capture_enqueue_(wake_word_state);
      }
#endif

      if (wake_word_state.detected) {
#ifdef USE_MICRO_WAKE_WORD_VAD
        if (vad_state.detected) {
#endif
          xQueueSend(this->detection_queue_, &wake_word_state, portMAX_DELAY);

          // Wake main loop immediately to process wake word detection
          App.wake_loop_threadsafe();

          model->reset_probabilities();
#ifdef USE_MICRO_WAKE_WORD_VAD
        } else {
          wake_word_state.blocked_by_vad = true;
          xQueueSend(this->detection_queue_, &wake_word_state, portMAX_DELAY);
        }
#endif
      }
    }
  }
}

#ifdef USE_MICRO_WAKE_WORD_CAPTURE

/// @brief Writes a 44 byte canonical PCM WAV header. ESP32 is little endian, so the length and rate fields can be
/// copied in directly.
static void build_wav_header(uint8_t *header, uint32_t sample_rate, uint32_t data_bytes) {
  const uint32_t riff_chunk_size = 36 + data_bytes;
  const uint32_t fmt_chunk_size = 16;
  const uint16_t audio_format = 1;  // PCM
  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint16_t block_align = num_channels * (bits_per_sample / 8);
  const uint32_t byte_rate = sample_rate * block_align;

  memcpy(header, "RIFF", 4);
  memcpy(header + 4, &riff_chunk_size, 4);
  memcpy(header + 8, "WAVEfmt ", 8);
  memcpy(header + 16, &fmt_chunk_size, 4);
  memcpy(header + 20, &audio_format, 2);
  memcpy(header + 22, &num_channels, 2);
  memcpy(header + 24, &sample_rate, 4);
  memcpy(header + 28, &byte_rate, 4);
  memcpy(header + 32, &block_align, 2);
  memcpy(header + 34, &bits_per_sample, 2);
  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &data_bytes, 4);
}

bool MicroWakeWord::setup_capture_() {
  const uint32_t sample_rate = this->microphone_source_->get_audio_stream_info().get_sample_rate();
  this->capture_ring_samples_ = (sample_rate / 1000) * this->capture_duration_ms_;

  // External RAM only: these buffers are far too large to take from internal RAM, and neither is touched from an ISR
  RAMAllocator<int16_t> allocator(RAMAllocator<int16_t>::ALLOC_EXTERNAL);
  this->capture_ring_ = allocator.allocate(this->capture_ring_samples_);
  this->capture_snapshot_ = allocator.allocate(this->capture_ring_samples_);
  if ((this->capture_ring_ == nullptr) || (this->capture_snapshot_ == nullptr)) {
    ESP_LOGE(TAG, "Failed to allocate capture buffers for %u ms of audio",
             static_cast<unsigned>(this->capture_duration_ms_));
    return false;
  }

  this->capture_queue_ = xQueueCreate(CAPTURE_QUEUE_LENGTH, sizeof(CaptureEvent));
  if (this->capture_queue_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create capture queue");
    return false;
  }

  if (!this->capture_task_.create(MicroWakeWord::capture_task, "mww_capture", CAPTURE_TASK_STACK_SIZE, this,
                                  CAPTURE_TASK_PRIORITY, true)) {
    ESP_LOGE(TAG, "Failed to create capture upload task");
    return false;
  }

  ESP_LOGCONFIG(TAG, "Capturing %u ms trigger windows to %s", static_cast<unsigned>(this->capture_duration_ms_),
                this->capture_url_);
  if (this->capture_probability_cutoff_ > 0) {
    ESP_LOGCONFIG(TAG, "  Also capturing near misses at or above probability %.2f",
                  this->capture_probability_cutoff_ / 255.0f);
  }
  return true;
}

void MicroWakeWord::capture_write_(const int16_t *data, size_t samples) {
  if ((this->capture_ring_ == nullptr) || (samples == 0)) {
    return;
  }

  if (samples >= this->capture_ring_samples_) {
    // The chunk is larger than the whole ring, so only its newest samples can survive
    data += samples - this->capture_ring_samples_;
    samples = this->capture_ring_samples_;
    this->capture_ring_write_ = 0;
    this->capture_ring_wrapped_ = true;
  }

  const size_t until_end = this->capture_ring_samples_ - this->capture_ring_write_;
  const size_t first = std::min(samples, until_end);
  memcpy(this->capture_ring_ + this->capture_ring_write_, data, first * sizeof(int16_t));
  this->capture_ring_write_ += first;

  if (this->capture_ring_write_ >= this->capture_ring_samples_) {
    this->capture_ring_write_ = 0;
    this->capture_ring_wrapped_ = true;
  }

  const size_t remaining = samples - first;
  if (remaining > 0) {
    memcpy(this->capture_ring_, data + first, remaining * sizeof(int16_t));
    this->capture_ring_write_ = remaining;
  }
}

void MicroWakeWord::capture_enqueue_(const DetectionEvent &detection_event) {
  if ((this->capture_ring_ == nullptr) || (this->capture_queue_ == nullptr)) {
    return;
  }

  const uint32_t now = millis();
  if ((this->last_capture_ms_ != 0) && ((now - this->last_capture_ms_) < CAPTURE_MIN_INTERVAL_MS)) {
    return;
  }

  if (this->capture_in_flight_) {
    ESP_LOGD(TAG, "Previous capture is still uploading; skipping this one");
    return;
  }

  const size_t available = this->capture_ring_wrapped_ ? this->capture_ring_samples_ : this->capture_ring_write_;
  if (available == 0) {
    return;
  }

  // Unroll oldest to newest so the snapshot is plain PCM ending at the moment the model fired
  if (this->capture_ring_wrapped_) {
    const size_t tail = this->capture_ring_samples_ - this->capture_ring_write_;
    memcpy(this->capture_snapshot_, this->capture_ring_ + this->capture_ring_write_, tail * sizeof(int16_t));
    memcpy(this->capture_snapshot_ + tail, this->capture_ring_, this->capture_ring_write_ * sizeof(int16_t));
  } else {
    memcpy(this->capture_snapshot_, this->capture_ring_, available * sizeof(int16_t));
  }

  CaptureEvent capture_event{};
  const char *wake_word =
      (detection_event.wake_word != nullptr) ? detection_event.wake_word->c_str() : "unknown";
  strncpy(capture_event.wake_word, wake_word, sizeof(capture_event.wake_word) - 1);
  capture_event.average_probability = detection_event.average_probability;
  capture_event.max_probability = detection_event.max_probability;
  capture_event.detected = detection_event.detected;
  capture_event.blocked_by_vad = detection_event.blocked_by_vad;
  capture_event.samples = available;
  capture_event.sample_rate = this->microphone_source_->get_audio_stream_info().get_sample_rate();

  this->capture_in_flight_ = true;
  if (xQueueSend(this->capture_queue_, &capture_event, 0) != pdTRUE) {
    this->capture_in_flight_ = false;
    return;
  }
  this->last_capture_ms_ = now;
}

void MicroWakeWord::capture_task(void *params) {
  MicroWakeWord *this_mww = (MicroWakeWord *) params;
  CaptureEvent capture_event;

  while (true) {
    if (xQueueReceive(this_mww->capture_queue_, &capture_event, portMAX_DELAY) == pdTRUE) {
      this_mww->capture_upload_(capture_event);
      // Released only once the snapshot buffer is free for the inference task to overwrite
      this_mww->capture_in_flight_ = false;
    }
  }
}

void MicroWakeWord::capture_upload_(const CaptureEvent &capture_event) {
  const uint32_t data_bytes = capture_event.samples * sizeof(int16_t);

  // Metadata rides in the query string so the receiver can name and file the clip without decoding the body
  const std::string url = str_sprintf("%s?wake_word=%s&avg=%u&max=%u&detected=%u&vad_blocked=%u", this->capture_url_,
                                      capture_event.wake_word, capture_event.average_probability,
                                      capture_event.max_probability, capture_event.detected ? 1 : 0,
                                      capture_event.blocked_by_vad ? 1 : 0);

  esp_http_client_config_t config = {};
  config.url = url.c_str();
  config.method = HTTP_METHOD_POST;
  config.timeout_ms = CAPTURE_HTTP_TIMEOUT_MS;

  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (client == nullptr) {
    ESP_LOGW(TAG, "Failed to initialize HTTP client for capture upload");
    return;
  }

  esp_http_client_set_header(client, "Content-Type", "audio/wav");

  esp_err_t err = esp_http_client_open(client, WAV_HEADER_BYTES + data_bytes);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "Capture upload could not connect: %s", esp_err_to_name(err));
    esp_http_client_cleanup(client);
    return;
  }

  uint8_t header[WAV_HEADER_BYTES];
  build_wav_header(header, capture_event.sample_rate, data_bytes);

  bool write_ok = (esp_http_client_write(client, reinterpret_cast<const char *>(header), WAV_HEADER_BYTES) ==
                   static_cast<int>(WAV_HEADER_BYTES));

  const uint8_t *pcm = reinterpret_cast<const uint8_t *>(this->capture_snapshot_);
  size_t remaining = data_bytes;
  while (write_ok && (remaining > 0)) {
    const size_t chunk = std::min(remaining, CAPTURE_WRITE_CHUNK_BYTES);
    const int written = esp_http_client_write(client, reinterpret_cast<const char *>(pcm), chunk);
    if (written <= 0) {
      write_ok = false;
      break;
    }
    pcm += written;
    remaining -= written;
  }

  if (write_ok) {
    esp_http_client_fetch_headers(client);
    const int status = esp_http_client_get_status_code(client);
    if ((status >= 200) && (status < 300)) {
      ESP_LOGD(TAG, "Uploaded capture for '%s' (average probability %.2f, detected %s)", capture_event.wake_word,
               capture_event.average_probability / 255.0f, YESNO(capture_event.detected));
    } else {
      ESP_LOGW(TAG, "Capture upload rejected with status %d", status);
    }
  } else {
    ESP_LOGW(TAG, "Capture upload failed while sending audio");
  }

  esp_http_client_close(client);
  esp_http_client_cleanup(client);
}

#endif  // USE_MICRO_WAKE_WORD_CAPTURE

void MicroWakeWord::unload_models_() {
  for (auto &model : this->wake_word_models_) {
    model->unload_model();
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  this->vad_model_->unload_model();
#endif
}

bool MicroWakeWord::update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]) {
  bool success = true;

  for (auto &model : this->wake_word_models_) {
    // Perform inference
    success = success & model->perform_streaming_inference(audio_features);
  }
#ifdef USE_MICRO_WAKE_WORD_VAD
  success = success & this->vad_model_->perform_streaming_inference(audio_features);
#endif

  return success;
}

}  // namespace esphome::micro_wake_word

#endif  // USE_ESP32
