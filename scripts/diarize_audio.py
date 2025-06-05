import json
from pathlib import Path
from pyannote.audio import Pipeline
import whisper
import torch

class AudioTranscriber:
    def __init__(self, hf_token: str):
        """
        Initialize the audio transcriber with HuggingFace token.
        
        Args:
            hf_token (str): HuggingFace authentication token for pyannote models
        """
        self.hf_token = hf_token
        self.whisper_model = None
        self.diarization_pipeline = None
        
    def load_models(self, whisper_model_size: str = "large-v3") -> bool:
        """
        Load both Whisper and pyannote models.
        
        Args:
            whisper_model_size (str): Size of Whisper model to load. Options:
                - "tiny"
                - "base"
                - "small"
                - "medium"
                - "large"
                - "large-v2"
                - "large-v3"
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Load Whisper model (will download on first run)
            print("--------------Loading Whisper model...")
            self.whisper_model = whisper.load_model(whisper_model_size)
            
            # Load pyannote diarization pipeline
            print("--------------Loading diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            return True
        except Exception as e:
            print(f"--------------Model loading error: {str(e)}")
            return False

    def transcribe_with_speakers(self, audio_path: str, whisper_model_size: str = "large-v3") -> dict:
        """
        Transcribe audio with speaker diarization.
        
        Args:
            audio_path (str): Path to audio file
            whisper_model_size (str): Whisper model size (default: "large-v3")
        
        Returns:
            dict: Transcription results with speaker info or error message
        """
        if not self.load_models(whisper_model_size):
            return {"error": "Model loading failed"}
        
        try:
            # Step 1: Audio transcription with Whisper
            print("--------------Audio transcription in progress...")
            whisper_result = self.whisper_model.transcribe(
                audio_path,
                language = "ru",
                fp16 = False,
                word_timestamps = True,
                beam_size = 5,
                temperature = 0.0
            )
            
            # Step 2: Speaker diarization
            print("--------------Speaker diarization in progress...")
            diarization = self.diarization_pipeline(
                audio_path,
                min_speakers = 2,  # Minimum 2 speakers
                max_speakers = 2   # Maximum 2 speakers
            )
            
            # Step 3: Synchronize transcription with speaker info
            print("--------------Synchronizing results...")
            result = self._sync_speakers_with_transcript(whisper_result, diarization)
            
            # Step 4: Post-process to fix common diarization errors
            result = self._postprocess_diarization(result)
            
            return result
            
        except Exception as e:
            print(f"--------------Processing error: {str(e)}")
            return {"error": str(e)}

    def _sync_speakers_with_transcript(self, whisper_result: dict, diarization) -> dict:
        """
        Align Whisper transcription segments with speaker diarization results.
        
        Args:
            whisper_result (dict): Raw Whisper transcription output
            diarization: pyannote diarization pipeline output
        
        Returns:
            dict: Combined results with speaker-labeled segments
        """
        segments = []
        
        # Extract word-level timestamps from Whisper output
        word_timestamps = []
        for segment in whisper_result["segments"]:
            for word in segment["words"]:
                word_timestamps.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                })
        
        # Group words into speaker segments
        current_speaker = None
        current_text = []
        current_start = None
        last_end = None
        
        for word in word_timestamps:
            # Determine speaker for current word
            speaker = self._find_speaker(diarization, word["start"], word["end"])
            
            # Start new segment if:
            # 1. Speaker changed, or
            # 2. Pause > 0.5 seconds between words
            if (speaker != current_speaker) or (current_start and word["start"] - last_end > 0.5):
                if current_speaker is not None:  # Save previous segment
                    segments.append({
                        "start": current_start,
                        "end": last_end,
                        "text": " ".join(current_text),
                        "speaker": current_speaker,
                        "speaker_type": self._label_speaker(current_speaker)
                    })
                # Start new segment
                current_text = []
                current_speaker = speaker
                current_start = word["start"]
            
            current_text.append(word["word"])
            last_end = word["end"]
        
        # Add final segment
        if current_text:
            segments.append({
                "start": current_start,
                "end": last_end,
                "text": " ".join(current_text),
                "speaker": current_speaker,
                "speaker_type": self._label_speaker(current_speaker)
            })
        
        return {
            "metadata": {
                "model": "Whisper-large-v3",
                "diarization": "pyannote-3.1",
                "processing": "word-level alignment"
            },
            "segments": segments
        }

    def _find_speaker(self, diarization, start: float, end: float) -> str:
        """
        Identify dominant speaker for a given time interval.
        
        Args:
            diarization: pyannote diarization results
            start (float): Segment start time
            end (float): Segment end time
        
        Returns:
            str: Speaker ID or "UNKNOWN" if no speaker found
        """
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Calculate overlap between segment and speaker turn
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0:
                speakers.append((overlap, speaker))
        
        # Return speaker with maximum overlap
        return max(speakers, key = lambda x: x[0])[1] if speakers else "UNKNOWN"

    def _label_speaker(self, speaker: str) -> str:
        """
        Label speakers as OPERATOR/CLIENT based on speaker ID.
        
        Args:
            speaker (str): Speaker ID from diarization
        
        Returns:
            str: "OPERATOR", "CLIENT", or "UNKNOWN"
        """
        if speaker == "UNKNOWN":
            return speaker
        # Even-numbered speakers are operators, odd are clients
        return "OPERATOR" if int(speaker.split("_")[-1]) % 2 == 0 else "CLIENT"

    def _postprocess_diarization(self, result: dict) -> dict:
        """
        Post-process results to fix common diarization errors.
        
        Args:
            result (dict): Raw diarization results
        
        Returns:
            dict: Processed results with improved speaker consistency
        """
        segments = result["segments"]
        
        # 1. Merge consecutive segments from same speaker with short pauses
        i = 0
        while i < len(segments) - 1:
            if (segments[i]["speaker"] == segments[i+1]["speaker"] and 
                segments[i+1]["start"] - segments[i]["end"] < 0.3):
                segments[i]["end"] = segments[i+1]["end"]
                segments[i]["text"] += " " + segments[i+1]["text"]
                del segments[i+1]
            else:
                i += 1
        
        # 2. Fix single-word speaker changes that are likely errors
        for i in range(1, len(segments)-1):
            if (segments[i-1]["speaker"] == segments[i+1]["speaker"] and 
                segments[i]["speaker"] != segments[i-1]["speaker"] and
                segments[i]["end"] - segments[i]["start"] < 0.5):
                segments[i]["speaker"] = segments[i-1]["speaker"]
                segments[i]["speaker_type"] = segments[i-1]["speaker_type"]
        
        return result



if __name__ == "__main__":
    # Configuration
    HF_TOKEN = "your_huggingface_token"  # Replace with your actual token
    AUDIO_FILE = "audio/call_clean.wav"  # Path to your audio file
    
    # Initialize transcriber
    transcriber = AudioTranscriber(HF_TOKEN)
    
    # Process audio - change model size as needed
    result = transcriber.transcribe_with_speakers(
        AUDIO_FILE,
        whisper_model_size="large-v3"  # Try "medium" for faster processing
    )
    
    # Save and display results
    if "error" not in result:
        output_file = f"{Path(AUDIO_FILE).stem}_diarized.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"------------Results saved to {output_file}")
        
        # Print first 3 segments as examples
        print("\n--------------Sample segments:")
        for seg in result["segments"][:3]:
            print(f"{seg['start']:.1f}-{seg['end']:.1f} {seg['speaker_type']}: {seg['text']}")
    else:
        print("--------------Processing failed:", result["error"])