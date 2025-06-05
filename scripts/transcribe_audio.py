import whisper
import json

def transcribe_audio(input_path: str, output_json: str, model_size: str = "large-v3", temperature: float = 0.0, beam_size: int = 5) -> None:
    """
    Transcribes audio using OpenAI's Whisper model and saves results to JSON.
    
    Args:
        input_path (str): Path to input audio file (WAV, MP3, etc.)
        output_json (str): Path to save transcription results (JSON format)
        model_size (str): Whisper model size (default: "large-v3"). Options:
            - "tiny" - Fastest, lowest accuracy
            - "base"
            - "small"
            - "medium"
            - "large"
            - "large-v2"
            - "large-v3" - Best accuracy, slowest
    
    Returns:
        None: Results are saved to the specified JSON file
    """
    
    # Validate model size input
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if model_size not in valid_models:
        raise ValueError(f"--------------Invalid model size. Choose from: {', '.join(valid_models)}")
    
    # Load the specified Whisper model
    # Note: First run will download the model (~2.5GB for large-v3)
    print(f"--------------Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    print(f"--------------Model loaded successfully")
    
    # Perform transcription with these settings:
    # - language = "ru" : Forces Russian language detection
    # - verbose = False : Suppresses progress messages in console
    # - fp16 = False    : Disables mixed precision (use when running on CPU)
    print("--------------Starting transcription...")
    result = model.transcribe(
        input_path,
        language = "ru",
        verbose = False,
        fp16 = False,
        temperature = 0.0,
        beam_size = 5
    )
    
    # Save results to JSON file with:
    # - ensure_ascii=False : Preserves Russian characters
    # - indent=2          : Pretty-prints JSON for readability
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"--------------Transcription saved to: {output_json}")
    print(f"--------------Detected language: {result['language']}")
    print(f"--------------Processing time: {result['segments'][-1]['end']:.2f}s audio processed")