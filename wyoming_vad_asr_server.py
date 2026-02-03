#!/usr/bin/env python3
"""
Wyoming Protocol ASR Server with Silero VAD
Proper Home Assistant-compatible Wyoming implementation

Features:
- Handles Describe events for service discovery
- Integrates Silero VAD for speech detection
- Full Wyoming protocol compliance
"""

import asyncio
import argparse
import logging
import tempfile
import yaml
import time
from pathlib import Path
from typing import Any

# Wyoming protocol imports
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, Describe
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

# ASR and VAD imports
import torch
import numpy as np
import soundfile as sf

_LOGGER = logging.getLogger(__name__)

# Global configuration
CONFIG = {
    'vad_enabled': True,
    'vad_threshold': 0.7,  # More strict for faster cutoff
    'sample_rate': 16000,
    'buffer_timeout': 2.0,  # Reduced for faster processing
    'min_speech_duration': 0.3,  # Ignore very short sounds
    'max_silence_duration': 0.2,  # Cut off faster after silence
    'max_audio_length': 8.0,  # Limit max processing time
    'languages': ["hu", "en", "de", "fr", "es", "it", "pl", "nl", "cs", "sk"]
}

# Global model instances (shared across handlers for performance)
GLOBAL_VAD_PROCESSOR = None
GLOBAL_ASR_PROCESSOR = None


class VADProcessor:
    """Silero VAD processor for speech detection"""

    def __init__(self, threshold=0.5, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.vad_model = None
        self.is_loaded = False

    async def load_model(self):
        """Load Silero VAD model"""
        if self.is_loaded:
            return

        _LOGGER.info("Loading Silero VAD...")
        try:
            # Import here to handle missing dependencies gracefully
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.vad_model = load_silero_vad(onnx=True)
            self.get_speech_timestamps = get_speech_timestamps
            self.is_loaded = True
            _LOGGER.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            _LOGGER.warning(f"Failed to load Silero VAD: {e}")
            _LOGGER.warning("VAD disabled - will process all audio")
            self.vad_model = None
            self.is_loaded = False

    def has_speech(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains speech using Silero VAD"""
        if not self.vad_model or len(audio_data) == 0:
            return True  # Default to processing if VAD unavailable

        try:
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

            if not isinstance(CONFIG['min_speech_duration'], float):
                raise ValueError("Invalid min_speech_duration in configuration")
            if not isinstance(CONFIG['max_audio_length'], float):
                raise ValueError("Invalid max_audio_length in configuration")

            # Get speech timestamps with more aggressive filtering
            timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=int(CONFIG['min_speech_duration'] * 1000),  # Convert to ms
                max_speech_duration_s=CONFIG['max_audio_length']
            )

            # Filter out very short segments
            valid_timestamps = []
            for ts in timestamps:
                duration = (ts['end'] - ts['start']) / self.sample_rate
                if duration >= CONFIG['min_speech_duration']:
                    valid_timestamps.append(ts)

            has_speech = len(valid_timestamps) > 0
            if has_speech:
                total_duration = sum((ts['end'] - ts['start']) / self.sample_rate for ts in valid_timestamps)
                _LOGGER.debug(f"VAD detected {len(valid_timestamps)} speech segments ({total_duration:.2f}s total)")
            else:
                _LOGGER.debug("VAD detected no valid speech (filtered out short segments)")
            return has_speech

        except Exception as e:
            _LOGGER.warning(f"VAD error: {e}, defaulting to processing")
            return True


class ASRProcessor:
    """ASR processor supporting multiple backends"""

    def __init__(self, backend='parakeet'):
        self.backend = backend
        self.asr_model = None
        self.is_loaded = False
        self.sample_rate = 16000

    async def load_model(self):
        """Load ASR model based on backend"""
        if self.is_loaded:
            return

        _LOGGER.info(f"Loading {self.backend} ASR...")
        try:
            await self._load_parakeet()

            self.is_loaded = True
            _LOGGER.info(f"✅ {self.backend} ASR loaded successfully")

        except Exception as e:
            _LOGGER.error(f"Failed to load {self.backend} ASR: {e}")
            self.asr_model = None
            self.is_loaded = False

    async def _load_parakeet(self):
        """Load Parakeet v3 model"""
        import nemo.collections.asr as nemo_asr
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        if torch.cuda.is_available():
            self.asr_model = self.asr_model.cuda() # pyright: ignore[reportAttributeAccessIssue]
            _LOGGER.info("Parakeet loaded on GPU")
        else:
            _LOGGER.info("Parakeet loaded on CPU")

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using the selected backend"""
        if not self.asr_model:
            return "ASR model not available"

        try:
            return await self._transcribe_parakeet(audio_data)
        except Exception as e:
            _LOGGER.error(f"Transcription error: {e}")
            return f"Error: {str(e)}"

    async def _transcribe_parakeet(self, audio_data: np.ndarray) -> str:
        """Transcribe using Parakeet"""
        # Save audio to temporary file (Parakeet expects file input)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, self.sample_rate)
            temp_path = temp_file.name

        # Transcribe
        start_time = time.time()
        transcription = self.asr_model.transcribe([temp_path])[0] # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        processing_time = time.time() - start_time

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

        # Extract text from Hypothesis object
        if hasattr(transcription, 'text'):
            text = transcription.text
        else:
            text = str(transcription)

        _LOGGER.info(f"Parakeet transcribed in {processing_time:.2f}s: '{text}'")
        return text.strip()


class WyomingVADASRHandler(AsyncEventHandler):
    """Wyoming protocol event handler with VAD and ASR"""

    def __init__(self, wyoming_info: Info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info

        if not isinstance(CONFIG['vad_threshold'], float):
            raise ValueError("Invalid vad_threshold in configuration")
        if not isinstance(CONFIG['sample_rate'], int):
            raise ValueError("Invalid sample_rate in configuration")

        # Use global model instances for better performance
        global GLOBAL_VAD_PROCESSOR, GLOBAL_ASR_PROCESSOR
        if GLOBAL_VAD_PROCESSOR is None:
            GLOBAL_VAD_PROCESSOR = VADProcessor(
                threshold=CONFIG['vad_threshold'],
                sample_rate=CONFIG['sample_rate']
            )
        if GLOBAL_ASR_PROCESSOR is None:
            GLOBAL_ASR_PROCESSOR = ASRProcessor()

        self.vad_processor = GLOBAL_VAD_PROCESSOR
        self.asr_processor = GLOBAL_ASR_PROCESSOR

        # Audio buffer management
        self.audio_buffer = bytearray()
        self.sample_rate = CONFIG['sample_rate']
        self.sample_width = 2  # 16-bit
        self.channels = 1  # mono

        _LOGGER.info("Wyoming VAD ASR Handler initialized")

    async def async_setup(self):
        """Setup processors"""
        _LOGGER.info("Setting up VAD and ASR processors...")
        await self.vad_processor.load_model()
        await self.asr_processor.load_model()
        _LOGGER.info("✅ Setup complete")

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events"""

        # Handle service discovery
        if Describe.is_type(event.type):
            _LOGGER.debug("Received Describe event - sending Info")
            await self.write_event(self.wyoming_info.event())
            return True

        # Handle audio stream start
        elif AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.sample_rate = audio_start.rate
            self.sample_width = audio_start.width
            self.channels = audio_start.channels
            self.audio_buffer.clear()
            _LOGGER.debug(f"Audio stream started: {self.sample_rate}Hz, {self.sample_width}B, {self.channels}ch")
            return True

        # Handle audio data chunks
        elif AudioChunk.is_type(event.type):
            audio_chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(audio_chunk.audio)

            if not isinstance(CONFIG['max_audio_length'], float):
                raise ValueError("Invalid max_audio_length in configuration")

            # Limit buffer size to prevent too long recordings
            max_buffer_size = int(CONFIG['max_audio_length'] * self.sample_rate * self.sample_width)
            if len(self.audio_buffer) > max_buffer_size:
                _LOGGER.debug(f"Audio buffer reached limit ({CONFIG['max_audio_length']}s), processing...")
                await self._process_audio_buffer()

            return True

        # Handle audio stream end - process accumulated audio
        elif AudioStop.is_type(event.type):
            await self._process_audio_buffer()
            return True

        # Handle transcription requests
        elif Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            _LOGGER.debug(f"Transcription request: language={transcribe.language}")
            return True

        return True

    async def _process_audio_buffer(self):
        """Process the accumulated audio buffer with VAD and ASR"""
        if not self.audio_buffer:
            await self.write_event(Transcript("").event())
            return

        try:
            # Ensure models are loaded (lazy loading)
            if not self.vad_processor.is_loaded:
                _LOGGER.info("Loading VAD model on first use...")
                await self.vad_processor.load_model()

            if not self.asr_processor.is_loaded:
                _LOGGER.info("Loading ASR model on first use...")
                await self.asr_processor.load_model()
            # Convert buffer to numpy array
            audio_bytes = bytes(self.audio_buffer)

            # Convert based on sample width
            if self.sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            elif self.sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {self.sample_width}")

            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32)
            if self.sample_width == 2:
                audio_data /= 32768.0
            else:
                audio_data /= 2147483648.0

            # Handle stereo -> mono conversion
            if self.channels == 2:
                audio_data = audio_data[::2]  # Take every other sample

            # Resample if needed (basic resampling)
            if self.sample_rate != self.asr_processor.sample_rate:
                _LOGGER.debug(f"Resampling from {self.sample_rate} to {self.asr_processor.sample_rate} Hz")
                ratio = self.asr_processor.sample_rate / self.sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )

            # Check for speech using VAD
            if CONFIG['vad_enabled'] and self.vad_processor.is_loaded:
                if not self.vad_processor.has_speech(audio_data):
                    _LOGGER.debug("No speech detected by VAD - returning empty transcript")
                    await self.write_event(Transcript("").event())
                    return

            # Transcribe with selected ASR
            if self.asr_processor.is_loaded and len(audio_data) > 0:
                text = await self.asr_processor.transcribe(audio_data)
                await self.write_event(Transcript(text).event())
            else:
                await self.write_event(Transcript("ASR not available").event())

        except Exception as e:
            _LOGGER.error(f"Audio processing failed: {e}")
            await self.write_event(Transcript(f"Error: {str(e)}").event())
        finally:
            self.audio_buffer.clear()


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file"""
    global CONFIG

    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            CONFIG.update(file_config)
            _LOGGER.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            _LOGGER.warning(f"Failed to load config from {config_path}: {e}")
    else:
        _LOGGER.info("Using default configuration")

    return CONFIG


def create_wyoming_info() -> Info:
    """Create Wyoming service info for discovery"""
    if not isinstance(CONFIG['languages'], list):
        raise ValueError("Invalid languages in configuration")

    return Info(
        asr=[
            AsrProgram(
                name="parakeet-v3-vad",
                description="Parakeet v3 ASR with Silero VAD",
                attribution=Attribution(
                    name="NVIDIA NeMo",
                    url="https://github.com/NVIDIA/NeMo"
                ),
                installed=True,
                version="3.0",
                models=[
                    AsrModel(
                        name="parakeet-tdt-0.6b-v3-vad",
                        description="Parakeet TDT 0.6B v3 with VAD - 25 European languages",
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
                        ),
                        installed=True,
                        languages=CONFIG['languages'],
                        version="3.0"
                    )
                ]
            )
        ]
    )


async def main():
    """Main server function"""
    parser = argparse.ArgumentParser(description="Wyoming VAD ASR Server")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300", help="Server URI")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load configuration
    config_path = args.config or Path("/home/attila/voice-services/vad_asr_config.yaml")
    load_config(config_path)

    if not isinstance(CONFIG['languages'], list):
        raise ValueError("Invalid languages in configuration")

    _LOGGER.info("🚀 Starting Wyoming VAD ASR Server...")
    _LOGGER.info(f"🎤 VAD: {'enabled' if CONFIG['vad_enabled'] else 'disabled'}")
    _LOGGER.info(f"🌍 Languages: {', '.join(CONFIG['languages'])}")

    # Create Wyoming service info
    wyoming_info = create_wyoming_info()

    # Create and start server
    server = AsyncServer.from_uri(args.uri)

    _LOGGER.info(f"✅ Server ready on {args.uri}")
    _LOGGER.info("🔍 Home Assistant should now be able to discover this service")

    try:
        await server.run(
            lambda reader, writer: WyomingVADASRHandler(
                wyoming_info, reader=reader, writer=writer
            )
        )
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    except Exception as e:
        _LOGGER.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
