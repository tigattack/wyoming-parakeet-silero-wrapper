#!/usr/bin/env python3
"""
Wyoming Protocol Test Client
Tests Wyoming service discovery and protocol compliance

This tool validates that a Wyoming ASR server properly handles:
- Service discovery (Describe -> Info)
- Audio streaming (AudioStart -> AudioChunk -> AudioStop)
- Transcription requests
"""

import asyncio
import argparse
import logging
import wave
import struct
from pathlib import Path
from typing import Optional

# Wyoming protocol imports
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

_LOGGER = logging.getLogger(__name__)


class WyomingTestClient:
    """Test client for Wyoming ASR services"""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.client: Optional[AsyncClient] = None
        
    async def connect(self):
        """Connect to Wyoming server"""
        _LOGGER.info(f"Connecting to {self.uri}")
        try:
            self.client = AsyncClient.from_uri(self.uri)
            await self.client.connect()
            _LOGGER.info("✅ Connected successfully")
            return True
        except Exception as e:
            _LOGGER.error(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.client:
            await self.client.disconnect()
            _LOGGER.info("Disconnected")
    
    async def test_service_discovery(self) -> bool:
        """Test service discovery (Describe -> Info)"""
        if not self.client:
            return False
            
        _LOGGER.info("🔍 Testing service discovery...")
        
        try:
            # Send Describe event
            _LOGGER.debug("Sending Describe event")
            await self.client.write_event(Describe().event())
            
            # Wait for Info response
            event = await self.client.read_event()
            if Info.is_type(event.type):
                info = Info.from_event(event)
                _LOGGER.info("✅ Service discovery successful!")
                
                # Display service info
                print("\n" + "="*50)
                print("SERVICE INFORMATION")
                print("="*50)
                
                if info.asr:
                    for asr_program in info.asr:
                        print(f"ASR Program: {asr_program.name}")
                        print(f"Description: {asr_program.description}")
                        print(f"Version: {asr_program.version}")
                        print(f"Installed: {asr_program.installed}")
                        
                        if asr_program.models:
                            print("Models:")
                            for model in asr_program.models:
                                print(f"  - {model.name}: {model.description}")
                                print(f"    Languages: {', '.join(model.languages or [])}")
                                print(f"    Version: {model.version}")
                        
                        if asr_program.attribution:
                            print(f"Attribution: {asr_program.attribution.name}")
                            if asr_program.attribution.url:
                                print(f"URL: {asr_program.attribution.url}")
                        print()
                        
                print("="*50)
                return True
            else:
                _LOGGER.error(f"❌ Expected Info event, got: {event.type}")
                return False
                
        except Exception as e:
            _LOGGER.error(f"❌ Service discovery failed: {e}")
            return False
    
    async def test_audio_transcription(self, audio_file: Optional[Path] = None) -> bool:
        """Test audio transcription"""
        if not self.client:
            return False
            
        _LOGGER.info("🎤 Testing audio transcription...")
        
        try:
            # Generate test audio if no file provided
            if audio_file and audio_file.exists():
                audio_data = self._load_audio_file(audio_file)
            else:
                audio_data = self._generate_test_audio()
            
            # Send audio stream
            await self._send_audio_stream(audio_data)
            
            # Wait for transcript
            event = await self.client.read_event()
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                _LOGGER.info("✅ Transcription successful!")
                print(f"\n🗣️  Transcript: '{transcript.text}'")
                return True
            else:
                _LOGGER.error(f"❌ Expected Transcript event, got: {event.type}")
                return False
                
        except Exception as e:
            _LOGGER.error(f"❌ Audio transcription failed: {e}")
            return False
    
    async def _send_audio_stream(self, audio_data: bytes):
        """Send audio data as a stream"""
        sample_rate = 16000
        sample_width = 2  # 16-bit
        channels = 1  # mono
        
        # Send AudioStart
        _LOGGER.debug("Sending AudioStart")
        audio_start = AudioStart(
            rate=sample_rate,
            width=sample_width,
            channels=channels
        )
        await self.client.write_event(audio_start.event())
        
        # Send audio in chunks
        chunk_size = 1024  # bytes
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            audio_chunk = AudioChunk(
                rate=sample_rate,
                width=sample_width,
                channels=channels,
                audio=chunk
            )
            await self.client.write_event(audio_chunk.event())
            _LOGGER.debug(f"Sent audio chunk {i//chunk_size + 1}")
            
        # Send AudioStop
        _LOGGER.debug("Sending AudioStop")
        await self.client.write_event(AudioStop().event())
    
    def _load_audio_file(self, audio_file: Path) -> bytes:
        """Load audio data from WAV file"""
        _LOGGER.info(f"Loading audio from {audio_file}")
        with wave.open(str(audio_file), 'rb') as wav:
            return wav.readframes(-1)
    
    def _generate_test_audio(self) -> bytes:
        """Generate synthetic test audio (sine wave)"""
        _LOGGER.info("Generating synthetic test audio")
        
        import math
        sample_rate = 16000
        duration = 2.0  # seconds
        frequency = 440.0  # A4 note
        
        samples = []
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            # Generate sine wave with some variation to simulate speech-like patterns
            amplitude = 0.3 * (1 + 0.5 * math.sin(2 * math.pi * t * 5))  # Amplitude modulation
            sample = amplitude * math.sin(2 * math.pi * frequency * t)
            # Convert to 16-bit PCM
            samples.append(struct.pack('<h', int(sample * 32767)))
        
        return b''.join(samples)
    
    async def run_all_tests(self, audio_file: Optional[Path] = None) -> bool:
        """Run all tests"""
        print("🧪 Wyoming Protocol Test Suite")
        print("="*40)
        
        success = True
        
        # Test connection
        if not await self.connect():
            return False
        
        try:
            # Test 1: Service Discovery
            if not await self.test_service_discovery():
                success = False
            
            print()  # Spacing
            
            # Test 2: Audio Transcription
            if not await self.test_audio_transcription(audio_file):
                success = False
                
        finally:
            await self.disconnect()
        
        print("\n" + "="*40)
        if success:
            print("✅ All tests passed!")
            print("🎉 Wyoming server is Home Assistant compatible")
        else:
            print("❌ Some tests failed")
            print("🔧 Check server logs for details")
        print("="*40)
        
        return success


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Wyoming Protocol Test Client")
    parser.add_argument("--uri", default="tcp://localhost:10300", help="Wyoming server URI")
    parser.add_argument("--audio-file", type=Path, help="Audio file to test transcription (WAV format)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Run tests
    test_client = WyomingTestClient(args.uri)
    success = await test_client.run_all_tests(args.audio_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)