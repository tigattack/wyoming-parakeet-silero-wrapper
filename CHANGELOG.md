# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-12

### Added
- Initial release of Parakeet Wyoming Wrapper
- NVIDIA Parakeet v3 ASR integration with Wyoming Protocol
- Silero VAD for intelligent voice activity detection
- Support for 25 European languages including Hungarian
- Complete Home Assistant compatibility and auto-discovery
- Global model sharing for 35x faster inference (0.06s vs 20s)
- Comprehensive management scripts with service installation
- Systemd service for auto-start and auto-restart functionality
- Advanced log management with automatic rotation and cleanup
- Protocol compliance testing tools
- Production-ready configuration with GPU optimization
- Memory-efficient design using only 2.8GB VRAM (vs 6-8GB for Whisper)

### Features
- **High Performance**: 35x faster than Whisper after model loading
- **Low Resource Usage**: 60-70% less VRAM than Whisper
- **Smart Audio Processing**: VAD filtering reduces false transcriptions
- **Production Ready**: Systemd service, log rotation, auto-restart
- **Easy Management**: Single script for all operations
- **Multilingual**: Optimized for European languages
- **Drop-in Replacement**: Full Wyoming Protocol compliance

### Technical Details
- Wyoming Protocol 1.7.2+ compatibility
- Lazy loading architecture for optimal performance
- Global model instances to prevent reloading
- Configurable VAD thresholds and audio processing limits
- Automatic buffer management and timeout handling
- Comprehensive error handling and fallback mechanisms

### Documentation
- Complete setup and configuration guide
- Home Assistant integration instructions
- Performance tuning recommendations
- Troubleshooting and debugging information
- Contributing guidelines and project structure