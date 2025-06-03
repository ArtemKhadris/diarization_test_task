import os
from pydub import AudioSegment, effects
import noisereduce as nr
import librosa
import soundfile as sf

def preprocess_audio(input_path, output_path, target_sr=16000):
    """
    Preprocesses audio by performing noise reduction, channel conversion, 
    resampling, and normalization.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save processed audio
        target_sr (int): Target sample rate (default: 16000Hz)
    """
    
    # Load audio file using librosa
    # - sr = None preserves original sample rate
    # - mono = True converts to single channel
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    # Apply noise reduction using noisereduce
    # - prop_decrease = 0.8 reduces noise by 80%
    # - stationary = True assumes noise is constant throughout audio
    reduced_audio = nr.reduce_noise(
        y = audio,
        sr = sr,
        prop_decrease = 0.8,
        stationary = True
    )

    # Create temporary file path for intermediate processing
    temp_wav_path = "temp_clean.wav"
    
    # Save noise-reduced audio to temporary WAV file
    # using soundfile (preserves floating point format)
    sf.write(temp_wav_path, reduced_audio, sr)

    # Load temporary file with pydub for further processing
    sound = AudioSegment.from_wav(temp_wav_path)

    # Ensure mono channel (single channel)
    sound = sound.set_channels(1)
    
    # Resample to target sample rate (default 16kHz)
    sound = sound.set_frame_rate(target_sr)
    
    # Normalize audio to consistent volume level
    sound = effects.normalize(sound)

    # Export final processed audio to output path
    sound.export(output_path, format="wav")

    # Clean up temporary file
    os.remove(temp_wav_path)
    print(f"--------------Audio saved: {output_path}")