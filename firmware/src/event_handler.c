#include "event_handler.h"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include "modules/led_controller.h"
#include "ble/ble_manager.h"
#include "ble/heart_service.h"

LOG_MODULE_REGISTER(event_handler);

K_MSGQ_DEFINE(_event_queue, sizeof(AppEvent), 8, 4);
K_SEM_DEFINE(_event_sem, 0, 1);

typedef enum {
    STATE_IDLE,
    STATE_ADVERTISING,
    STATE_CONNECTED,
    STATE_STREAMING
} AppState;

static AppState app_state = STATE_IDLE;
static AudioStream *audio_stream_ptr = NULL;

void event_handler_set_audio_stream(AudioStream *audio_stream) {
    audio_stream_ptr = audio_stream;
}

void event_handler_post(AppEvent evt)
{
    k_msgq_put(&_event_queue, &evt, K_NO_WAIT);
    k_sem_give(&_event_sem);
}

void _send_demo_heartbeat_packet()
{
    static float counter = 0.0f;
    struct heart_packet pkt = {
        .rms = 10.0f + counter,
        .centroid = 200.0f + counter,
        .timestamp_ms = k_uptime_get()
    };

    counter =  counter + 1.0f;

    int err = bt_heart_service_notify_packet(&pkt);
    if (err) {
        printk("Failed to send packet: %d\n", err);
    } else {
        printk("Demo heartbeat packet sent!\n");
    }
}

static void handle_event(AppEvent evt)
{
    int ret;
    switch (app_state) {
        case STATE_IDLE:
            if (evt.type == EVENT_BUTTON_0_PRESS && audio_stream_ptr != NULL) {
                led_controller_start_blinking(K_MSEC(500));
                ble_advertise();
                app_state = STATE_ADVERTISING;
            }
            break;
            // if (evt.type == EVENT_BUTTON_0_PRESS && audio_stream_ptr != NULL) {
            //     led_controller_stop_blinking();
            //     led_controller_on();
            //     int ret = open_wav_for_write(&audio_stream_ptr->wav_config);
            //     capture_audio(audio_stream_ptr);
            //     led_controller_off();
            //     //ble_start_advertising();
            //     app_state = STATE_STREAMING;
            // }
            // break;
        case STATE_ADVERTISING:
            if (evt.type == EVENT_BLE_CONNECTED) {
                led_controller_stop_blinking();
                led_controller_on();
                //led_set_mode(LED_OFF);
                app_state = STATE_CONNECTED;
            }
            break;

        case STATE_CONNECTED:
            //Send Dummy BLE Data
            if (evt.type == EVENT_BUTTON_0_PRESS) {
                _send_demo_heartbeat_packet();
                ret = bt_heart_service_notify_alert(0x02);
                if(ret!=0) LOG_ERR("Alert Failed to send");
            }
            //Record Audio to SD Card and then stream over BLE
            if (evt.type == EVENT_BLE_RECORD) {
                //Record Audio
                led_controller_start_blinking(K_MSEC(150));
                static char filename[MAX_FILENAME_LEN];
                generate_filename(filename, sizeof(filename));
                audio_stream_ptr->wav_config.file_name = filename;
                ret = open_wav_for_write(&audio_stream_ptr->wav_config);
                capture_audio(audio_stream_ptr);

                #if !IS_ENABLED(CONFIG_HEART_PATCH_DSP_MODE)
                //Send Via BLE
                const int16_t *buf = get_audio_buffer();
                size_t len_samples = get_audio_buffer_length();

                if (buf && len_samples > 0) {
                    size_t len_bytes = len_samples * sizeof(int16_t);
                    int err = transmit_audio_buffer((const uint8_t *)buf, len_bytes);
                    if (err) {
                        LOG_ERR("Failed to transmit audio buffer: %d", err);
                    } else {
                        LOG_INF("Audio buffer transmitted successfully.");
                    }
                } else {
                    LOG_WRN("No audio buffer available to transmit.");
                }
                #endif

                led_controller_stop_blinking();
                led_controller_on();
            }
            // if (evt.type = EVENT_BLE_DISCONNECTED) {
            //     app_state = STATE_IDLE;
            //     led_controller_off();
            // }
            // if (evt.type = EVENT_BLE_TOGGLE_LED) {
            //     bool state = *(bool *)(evt.data);
            //     led_controller_set(state);
            // }
            break;

        case STATE_STREAMING:
            // if (evt.type == EVENT_AUDIO_FINISHED) {
            //     //audio_stop();
            //     //led_set_mode(LED_OFF);
            //     app_state = STATE_CONNECTED;
            // }
            led_controller_start_blinking(K_MSEC(500));
            app_state = STATE_IDLE;
            break;
        }
}

int event_handler_run(void)
{
    LOG_INF("Starting event handler loop");

    while (1) {
        k_sem_take(&_event_sem, K_FOREVER);
        AppEvent evt;
        while (k_msgq_get(&_event_queue, &evt, K_NO_WAIT) == 0) {
            handle_event(evt);
        }
    }

    return 0;
}
