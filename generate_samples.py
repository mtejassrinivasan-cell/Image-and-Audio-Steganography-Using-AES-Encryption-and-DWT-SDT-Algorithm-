import wave
import numpy as np

def generate_tone(filename, duration, freq, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate sine wave and add a bit of noise so it sounds a bit less harsh
    audio = np.sin(2 * np.pi * freq * t) * 12000
    noise = np.random.normal(0, 1000, len(t))
    audio_data = (audio + noise).astype(np.int16)
    
    # Create stereo by duplicating channels
    stereo_data = np.column_stack((audio_data, audio_data))
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo_data.tobytes())
        
if __name__ == "__main__":
    print("Generating sample_cover.wav (60 seconds, 440 Hz)")
    # 60 seconds cover has a capacity of roughly 660 KB
    generate_tone('sample_cover.wav', 60, 440) 
    
    print("Generating sample_secret.wav (2 seconds, 880 Hz)")
    # 2 seconds secret is very tiny, roughly ~176 KB
    generate_tone('sample_secret.wav', 2, 880)

    print("Success! Created sample_cover.wav and sample_secret.wav")
