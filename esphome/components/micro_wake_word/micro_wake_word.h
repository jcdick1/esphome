#pragma once

#ifdef USE_ESP32

#include "preprocessor_settings.h"
#include "streaming_model.h"

#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/ring_buffer/ring_buffer.h"

#include "esphome/core/automation.h"
#include "esphome/core/component.h"
#include "esphome/core/defines.h"
#include "esphome/core/static_task.h"

#ifdef USE_OTA_STATE_LISTENER
#include "esphome/components/ota/ota_backend.h"
#endif

#include <freertos/event_groups.h>

#include <frontend.h>
#include <frontend_util.h>

namespace esphome::micro_wake_word {

enum State {
  STARTING,
  DETECTING_WAKE_WORD,
  STOPPING,
  STOPPED,
};

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
// Metadata describing a snapshot of the audio that immediately preceded a model firing. Queued from the inference
// task and consumed by the upload task, so it must stay trivially copyable.
struct CaptureEvent {
  char wake_word[32];
  uint8_t average_probability;
  uint8_t max_probability;
  bool detected;         // False when only the capture cutoff was crossed, not the detection cutoff
  bool blocked_by_vad;   // Detection cutoff was crossed, but the VAD model vetoed it
  uint32_t samples;      // Valid samples in the snapshot buffer
  uint32_t sample_rate;
};
#endif

class MicroWakeWord final : public Component
#ifdef USE_OTA_STATE_LISTENER
    ,
                            public ota::OTAGlobalStateListener
#endif
{
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;
  void dump_config() override;

#ifdef USE_OTA_STATE_LISTENER
  void on_ota_global_state(ota::OTAState state, float progress, uint8_t error, ota::OTAComponent *comp) override;
#endif

  void start();
  void stop();

  bool is_running() const { return this->state_ != State::STOPPED; }

  void set_features_step_size(uint8_t step_size) { this->features_step_size_ = step_size; }

  void set_microphone_source(microphone::MicrophoneSource *microphone_source) {
    this->microphone_source_ = microphone_source;
  }

  void set_stop_after_detection(bool stop_after_detection) { this->stop_after_detection_ = stop_after_detection; }

  void set_task_stack_in_psram(bool task_stack_in_psram) { this->task_stack_in_psram_ = task_stack_in_psram; }

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
  void set_capture_url(const char *capture_url) { this->capture_url_ = capture_url; }
  void set_capture_duration_ms(uint32_t capture_duration_ms) { this->capture_duration_ms_ = capture_duration_ms; }
  void set_capture_probability_cutoff(uint8_t capture_probability_cutoff) {
    this->capture_probability_cutoff_ = capture_probability_cutoff;
  }
#endif

  Trigger<std::string> *get_wake_word_detected_trigger() { return &this->wake_word_detected_trigger_; }

  void add_wake_word_model(WakeWordModel *model);

#ifdef USE_MICRO_WAKE_WORD_VAD
  void add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                     size_t tensor_arena_size);

  // Intended for the voice assistant component to fetch VAD status
  bool get_vad_state() { return this->vad_state_; }
#endif

  // Intended for the voice assistant component to access which wake words are available
  // Since these are pointers to the WakeWordModel objects, the voice assistant component can enable or disable them
  std::vector<WakeWordModel *> get_wake_words();

 protected:
  microphone::MicrophoneSource *microphone_source_{nullptr};
  Trigger<std::string> wake_word_detected_trigger_;
  State state_{State::STOPPED};

  std::weak_ptr<ring_buffer::RingBuffer> ring_buffer_;
  std::vector<WakeWordModel *> wake_word_models_;

#ifdef USE_MICRO_WAKE_WORD_VAD
  std::unique_ptr<VADModel> vad_model_;
  bool vad_state_{false};
#endif

  bool pending_start_{false};
  bool pending_stop_{false};

  bool stop_after_detection_;

  bool task_stack_in_psram_{false};

  uint8_t features_step_size_;

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
  const char *capture_url_{nullptr};
  uint32_t capture_duration_ms_{3000};
  // Quantized probability cutoff mapping 0.0 - 1.0 to 0 - 255. Zero disables sub-detection captures; any nonzero
  // value captures whenever a model's sliding window average reaches it, even if the model didn't fire.
  uint8_t capture_probability_cutoff_{0};

  // Circular buffer of the raw samples the frontend has consumed, so a detection can be traced backwards to the
  // audio that caused it. Written only by the inference task.
  int16_t *capture_ring_{nullptr};
  size_t capture_ring_samples_{0};
  size_t capture_ring_write_{0};
  bool capture_ring_wrapped_{false};

  // Unrolled oldest-to-newest copy of the ring, handed to the upload task
  int16_t *capture_snapshot_{nullptr};
  volatile bool capture_in_flight_{false};

  QueueHandle_t capture_queue_{nullptr};
  StaticTask capture_task_;
  uint32_t last_capture_ms_{0};
#endif

  // Audio frontend handles generating spectrogram features
  struct FrontendConfig frontend_config_;
  struct FrontendState frontend_state_;

  // Handles managing the stop/state of the inference task
  EventGroupHandle_t event_group_;

  // Used to send messages about the models' states to the main loop
  QueueHandle_t detection_queue_;

  StaticTask inference_task_;

  static void inference_task(void *params);

  /// @brief Suspends the inference task
  void suspend_task_();
  /// @brief Resumes the inference task
  void resume_task_();

  void set_state_(State state);

  /// @brief Generates a spectrogram feature from an input buffer of audio samples. The frontend buffers samples
  /// internally, so callers may stream arbitrary-sized chunks; a feature is only emitted once enough samples have
  /// accumulated to fill a full analysis window.
  /// @param audio_buffer (const int16_t *) Buffer containing input audio samples
  /// @param samples_available (size_t) Number of samples available in the input buffer
  /// @param features_buffer (int8_t *) Buffer to store the generated feature, valid only when the return value is true
  /// @param processed_samples (size_t *) Set to the number of samples consumed from the input buffer
  /// @return True if a new feature was generated; false if more samples are required
  bool generate_features_(const int16_t *audio_buffer, size_t samples_available,
                          int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE], size_t *processed_samples);

  /// @brief Processes any new probabilities for each model. If any wake word is detected, it will send a DetectionEvent
  /// to the detection_queue_.
  void process_probabilities_();

#ifdef USE_MICRO_WAKE_WORD_CAPTURE
  /// @brief Allocates the capture ring and snapshot buffers in external RAM and starts the upload task.
  /// @return True if capture is ready, false if it could not be set up (capture is then silently inactive)
  bool setup_capture_();

  /// @brief Appends samples to the capture ring. Must only be called from the inference task.
  /// @param data (const int16_t *) Buffer of samples the frontend has just consumed
  /// @param samples (size_t) Number of samples to append
  void capture_write_(const int16_t *data, size_t samples);

  /// @brief Unrolls the capture ring into the snapshot buffer and queues it for upload. Must only be called from the
  /// inference task. Silently skips if a previous upload is still in flight.
  /// @param detection_event (const DetectionEvent &) The event that prompted the capture
  void capture_enqueue_(const DetectionEvent &detection_event);

  /// @brief POSTs the snapshot buffer to capture_url_ as a WAV file. Runs on the upload task.
  void capture_upload_(const CaptureEvent &capture_event);

  static void capture_task(void *params);
#endif

  /// @brief Deletes each model's TFLite interpreters and frees tensor arena memory.
  void unload_models_();

  /// @brief Runs an inference with each model using the new spectrogram features
  /// @param audio_features (int8_t *) Buffer containing new spectrogram features
  /// @return True if successful, false if any errors were encountered
  bool update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]);
};

}  // namespace esphome::micro_wake_word

#endif  // USE_ESP32
