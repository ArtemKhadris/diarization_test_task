from transcribe_audio import transcribe_audio

input_file = "audio/call_clean.wav" # path to processed audio
output_file = "transcript.json" # path to json with transcription
model_size = "large-v3" # other models: "tiny", "base", "small", "medium", "large", "large-v2"

transcribe_audio(input_file, output_file, model_size)