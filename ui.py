import gradio as gr
import json
from pathlib import Path
from scripts.preprocess_audio import preprocess_audio
from scripts.diarize_audio import AudioTranscriber
from scripts.transcribe_audio import transcribe_audio

# Configuration
HF_TOKEN = "your_huggingface_token"  # HuggingFace token for pyannote models

# === CORE PROCESSING FUNCTIONS ===

def run_preprocessing(file):
    """
    Process audio file through noise reduction and normalization pipeline.
    
    Args:
        file (str): Path to the input audio file
        prop_decrease (float): Level of noise reduction to apply (0.1-1.0)
        
    Returns:
        str: Path to the processed audio file in WAV format
        
    Processing Steps:
    1. Takes input audio file path
    2. Generates output path in 'audio/' directory
    3. Applies noise reduction, resampling and normalization
    4. Returns path to cleaned audio file
    """
    input_path = file  # Store input file path
    base_name = Path(file).stem  # Extract filename without extension
    output_path = f"audio/{base_name}_preprocessed.wav"  # Output path
    
    # Call preprocessing function from imported module
    preprocess_audio(
        input_path = input_path,
        output_path = output_path,
        target_sr = 16000  # Target sample rate (16kHz standard for speech)
    )

    return output_path  # Return path to processed file


def run_transcription(file, model_size, temperature, beam_size):
    """
    Transcribe audio file using OpenAI's Whisper model.
    
    Args:
        file (str): Path to audio file
        model_size (str): Whisper model variant (tiny, base, small, medium, large, large-v3)
        temperature (float): Controls model randomness (0.0 = most deterministic)
        beam_size (int): Number of beams for beam search (higher = more accurate but slower)
        
    Returns:
        tuple: (formatted_transcript, json_path)
            formatted_transcript (str): Human-readable transcript with timestamps
            json_path (str): Path to full JSON results file
            
    The function:
    1. Runs Whisper transcription with specified parameters
    2. Formats results with timestamps
    3. Saves full results to JSON
    4. Returns both formatted text and JSON path
    """
    base_name = Path(file).stem  # Get filename without extension
    output_path = f"audio/{base_name}_transcript.json"  # Output JSON path
    
    # Run Whisper transcription with specified parameters
    transcribe_audio(
        input_path = file,
        output_json = output_path,
        model_size = model_size,
        temperature = temperature,
        beam_size = beam_size
    )
    
    # Load and format the results
    with open(output_path, encoding="utf-8") as f:
        result = json.load(f)  # Load JSON results
    
    # Format into human-readable text with timestamps
    formatted = ""
    for segment in result["segments"]:
        formatted += f"[{segment['start']:.2f}s — {segment['end']:.2f}s]: {segment['text']}\n"
    
    return formatted, output_path  # Return both text and JSON path


def run_full_pipeline(file):
    """
    Complete audio processing pipeline: preprocessing + transcription + speaker diarization.
    
    Args:
        file (str): Path to input audio file
        
    Returns:
        tuple: (formatted_output, json_path)
            formatted_output (str): Transcript with speaker labels and timestamps
            json_path (str): Path to full JSON results file
            
    Processing Flow:
    1. Audio preprocessing (noise reduction, normalization)
    2. Speaker diarization (identifying OPERATOR/CLIENT)
    3. Transcription with speaker-attributed segments
    4. Results formatting and JSON export
    """
    input_path = file  # Store input path
    base_name = Path(input_path).stem  # Get base filename
    clean_path = f"audio/{base_name}_clean.wav"  # Path for cleaned audio
    
    # Step 1: Audio preprocessing
    preprocess_audio(input_path, clean_path)
    
    # Step 2: Initialize transcriber and run pipeline
    transcriber = AudioTranscriber(HF_TOKEN)
    result = transcriber.transcribe_with_speakers(
        audio_path = clean_path,
        whisper_model_size = "large-v3"  # Using largest model for best accuracy
    )
    
    # Error handling
    if "error" in result:
        return f"Error: {result['error']}", None

    # Step 3: Format results with speaker labels
    formatted = ""
    for seg in result["segments"]:
        speaker_label = "OPERATOR" if seg['speaker_type'] == "OPERATOR" else "CLIENT"
        formatted += f"[{seg['start']:.2f}s — {seg['end']:.2f}s] {speaker_label}: {seg['text']}\n"
    
    # Step 4: Save complete results to JSON
    output_json = f"audio/{base_name}_result.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return formatted, output_json

# === GRADIO USER INTERFACE ===

# Create interface with soft theme and title
with gr.Blocks(title = "Call Analysis System", theme = gr.themes.Soft()) as demo:
    gr.Markdown("## Call Transcription & Diarization System")

    # Tab 1: Audio Preprocessing
    with gr.Tab("Audio Preprocessing"):
        gr.Markdown("### Noise Reduction and Audio Enhancement")
        
        # Input components
        audio_input = gr.Audio(
            label = "Input Audio",
            type = "filepath",  # Expects file path rather than raw audio
            sources = ["upload"]  # Allow file uploads
        )
        prop_slider = gr.Slider(
            minimum = 0.1, 
            maximum = 1.0, 
            value = 0.8,  # Default noise reduction level
            step = 0.05, 
            label = "Noise Reduction Level (prop_decrease)"
        )
        
        # Action button
        preprocess_btn = gr.Button("Process Audio")
        
        # Output component
        processed_output = gr.Audio(
            label = "Processed Audio",
            type = "filepath"  # Output as downloadable file
        )

        # Connect button to function
        preprocess_btn.click(
            fn = run_preprocessing,
            inputs = [audio_input, prop_slider],
            outputs = processed_output
        )

    # Tab 2: Basic Transcription
    with gr.Tab("Basic Transcription"):
        gr.Markdown("### Speech-to-Text with Whisper")
        
        # Input components
        audio_transcribe = gr.Audio(
            label = "Audio for Transcription", 
            type = "filepath"
        )
        
        # Model configuration in a row
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices = ["tiny", "base", "small", "medium", "large", "large-v3"],
                value = "large-v3",  # Default to most accurate model
                label = "Whisper Model"
            )
            temperature_slider = gr.Slider(
                minimum = 0.0,
                maximum = 1.0,
                step = 0.1,
                value = 0.0,  # Default to deterministic output
                label = "Temperature (0 = deterministic)"
            )
            beam_slider = gr.Slider(
                minimum = 1,
                maximum = 10,
                step = 1,
                value = 5,  # Default beam width
                label = "Beam Search Width"
            )
        
        # Action button
        transcribe_btn = gr.Button("Transcribe")
        
        # Output components
        transcript_output = gr.Textbox(
            lines = 20, 
            label = "Transcription Results",
            interactive = False  # Read-only
        )
        transcript_file = gr.File(
            label = "Download JSON", 
            interactive = False  # Download only
        )

        # Connect button to function
        transcribe_btn.click(
            fn = run_transcription,
            inputs = [audio_transcribe, model_dropdown, temperature_slider, beam_slider],
            outputs = [transcript_output, transcript_file]
        )

    # Tab 3: Full Diarization Pipeline
    with gr.Tab("Full Diarization"):
        gr.Markdown("### Speaker-Separated Transcription")
        
        # Input component
        audio_diar = gr.Audio(
            label = "Audio for Analysis", 
            type = "filepath"
        )
        
        # Action button
        diar_btn = gr.Button("Run Full Analysis")
        
        # Output components
        diar_output = gr.Textbox(
            lines = 20, 
            label = "Transcript with Speaker Labels",
            interactive = False
        )
        diar_file = gr.File(
            label = "Download Full Report",
            interactive = False
        )

        # Connect button to function
        diar_btn.click(
            fn = run_full_pipeline,
            inputs = [audio_diar],
            outputs = [diar_output, diar_file]
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch()