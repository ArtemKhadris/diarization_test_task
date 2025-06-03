# This test task performs the following tasks:

## Audio preprocessing:
A mechanism for cleaning audio files from noise has been implemented

Normalization of audio recording volume

Changing the format and frequency (kHz) of an audio file

These functions are implemented in the `preprocess_audio.py file`, in which you can change the noise suppression level (the default is 80%, `prop_decrease = 0.8`) and the noise type (the default is constant, `stationary = True`). 

The function itself is launched from the `run_preprocessing.py` file, in which you must specify the path to the source file and the path to the result.

## Transcription:
A local model for speech recognition is used (in this case, `Whisper`)

Correct work with Russian speech is ensured

These functions are implemented in the `transcribe_audio.py` file, in which you can change the following parameters:

Source file language (by default, Russian, `language = "ru"`)

Process output to the terminal (by default, disabled, `verbose = False`)

Use of FP16 acceleration (by default, disabled, `fp16 = False`)

Creativity (by default, 0, `temperature = 0.0`)

Search width during decoding (by default, 5, `beam_size = 5`)

The function itself is launched from the `run_transcribe.py` file, in which you must specify the path to the source audio, the path to the result in json format, and the model type (more details in the comments to the code).

## Call diarization:
Implemented identification of different speakers in the recording

Transcribed text marked with the speaker

These functions are implemented in the `diarize_audio.py` file

The system provides advanced call processing with:

Speaker identification (diarization) using `pyannote.audio 3.1`

Accurate Russian speech recognition via `Whisper`

Automatic labeling of speakers as OPERATOR/CLIENT

Word-level alignment of transcripts with speaker timestamps

These functions are implemented in the AudioTranscriber class which requires:

HuggingFace authentication token (for pyannote models)

Processed audio file in WAV format (recommended 16kHz mono)


### Model Selection:
Adjustable Whisper model size (default: `"large-v3"` for best accuracy)

Options: `"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"`

Diarization Settings:

Fixed 2-speaker mode (optimized for call center recordings)

Automatic GPU detection (falls back to CPU if needed)


### Output Format:
`
{
  "segments": [
    {
      "start": 3.2,
      "end": 5.1,
      "text": "Добрый день, как я могу вам помочь?",
      "speaker": "SPEAKER_01",
      "speaker_type": "OPERATOR"
    }
  ]
}
`


### Usage Example:
`
transcriber = AudioTranscriber(HF_TOKEN)
result = transcriber.transcribe_with_speakers(
    "audio.wav",
    whisper_model_size="medium"  # Balance between speed/accuracy
)
`


### Configuration Tips:
For GPU acceleration: Set `fp16 = True` in `transcribe()` call

For better accuracy: Increase `beam_size` (5-7) and set `temperature = 0.0`

### To handle noisy audio: 
Pre-process with `preprocess_audio.py` first

The script automatically:

Merges short segments from same speaker

Fixes common diarization errors

Saves results with speaker-labeled timestamps


## Launch
To launch these files, you need to create an environment. Python version `3.10.6` was used when writing the code. The necessary packages are listed in the `requirements.txt` file. But in addition to them, you need to register at https://huggingface.co/ and confirm your email, get a personal token at https://huggingface.co/settings/tokens (the token will be shown once when it is created, so you need to immediately copy it into the code and specify `HF_TOKEN = "your_huggingface_token"`). 

In addition, you need to agree to the terms of use https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0 (both of these packages are necessary for the code to work). After these conditions are met, the code should work correctly.

A complete list of all installed packages in the environment is in the `requirements1.txt` file (in case any updates are released and version conflicts occur).

## Results
The results of the run_preprocessing.py and run_transcribe.py scripts will be saved to the specified path, the results of the diarize_audio.py script will be saved to the root folder of the project.

This task already has a trial source audio, for which all parameters were configured to achieve the best result. This source file is `audio/call.wav`, after processing - `audio/call_clean.wav`, transcription results - `audio/transcript.json`, and diarization results - `call_clean_diarized.json`.
