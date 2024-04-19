from client.Whisper_speaker_diarization.whisper import Whisper
import wave
import pyaudio
import numpy as np


def main():
    whisper = Whisper(whisper_model="large-v2",gpu_id=2)
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    count = 1
    min_volume_threshold = 2000  # Minimum volume to start recording
    silence_duration_threshold = 2  # Duration of silence before stopping (in seconds)
    while True:
        wave_output_filename = f'recording/test{count}.wav'
        p = pyaudio.PyAudio()
        tempRate = int(p.get_device_info_by_index(0).get('defaultSampleRate'))
        print("tempRate", tempRate)
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=tempRate,
                        input=True,
                        frames_per_buffer=chunk_size)

        print("Listening! Timing starts now.")

        frames = []
        is_recording = False
        continue_recording = True
        below_threshold = False

        time_frame_count = 0
        silence_start_time = 0

        while continue_recording:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            max_volume = np.max(audio_data)
            print(f"Max volume: {max_volume}")

            if max_volume > min_volume_threshold and not is_recording:
                is_recording = True
                silence_start_time = time_frame_count
                print("Recording started.")

            if is_recording:
                if max_volume < min_volume_threshold:
                    if not below_threshold:
                        below_threshold = True
                        silence_start_time = time_frame_count
                        print("Volume has decreased, starting silence timer.")
                else:
                    below_threshold = False
                    silence_start_time = time_frame_count

                if time_frame_count > silence_start_time + silence_duration_threshold * sample_rate / chunk_size:
                    if below_threshold:
                        continue_recording = False
                        print("Silence detected, stopping recording.")
                    else:
                        below_threshold = False

            time_frame_count += 1
            if time_frame_count > sample_rate * 60 / chunk_size:  # Timeout after 1 minute
                continue_recording = False
                print("Timeout, stopping recording.")

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(wave_output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        count += 1
        whisper.speech_to_text(wave_output_filename, './../output/transcript_result.csv',"en", ['Shuo Chen'])



if __name__ == "__main__":
    main()
    """
    whisper = Whisper(whisper_model="large-v2", gpu_id=2)
    for file in os.listdir("./recording/"):
        if file.endswith(".wav"):
            wave_output_filename = os.path.join("./recording/", file)
            whisper.speech_to_text(wave_output_filename, './../output/transcript_result.csv', "en", ['Shuo Chen'])
    """