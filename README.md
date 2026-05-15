# Audio Spectrogram

A real-time audio spectrogram visualizer that captures microphone input, converts it to mono, and displays a live spectrogram, waveform, and live FFT magnitude view using Matplotlib.

## Features

- **Real-time spectrogram** — Live FFT visualization of microphone input (64 time frames)
- **Live waveform** — Time-domain plot showing raw audio signal
- **Live FFT magnitude plot** — Raw and smoothed FFT magnitude curves with peak frequency marker
- **Mono conversion** — Automatically mixes stereo to mono
- **Configurable** — YAML-based configuration for all parameters
- **Auto-scaling** — Dynamic dB range adapts to audio levels
- **Window functions** — Supports Hann, Hamming, Blackman, and Bartlett windows
- **Cross-platform** — Works on Windows, macOS, and Linux

## Requirements

- Python 3.8+
- A working microphone

## Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/calderonf/audio-spectrogram.git
   cd audio-spectrogram
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.yaml` to customize the spectrogram behavior:

| Section | Parameter | Description |
|---------|-----------|-------------|
| audio | sample_rate | Sample rate in Hz (default: 44100) |
| audio | chunk_size | Buffer size in samples |
| audio | channels | Number of input channels (1 = mono) |
| audio | dtype | Audio data type (default: int16) |
| spectrogram | fft_window | FFT window size in samples |
| spectrogram | hop_length | Hop length for STFT overlap |
| spectrogram | window_function | Window type: hann, hamming, blackman, bartlett |
| spectrogram | frequency_limit | Upper frequency limit used by the spectrogram and FFT magnitude plot |
| plot | refresh_rate | Plot update frequency in Hz |
| plot | colormap | Matplotlib colormap name |
| plot | colorbar | Show/hide amplitude colorbar |
| plotfft | alpha | Exponential smoothing factor for the FFT magnitude view (recommended starting value: 0.1) |

The FFT magnitude figure uses `spectrogram.frequency_limit` to limit the displayed frequency range and `plotfft.alpha` to smooth the FFT curve over time.

## Usage

### Basic Usage

```bash
# Run with default configuration
python -m audio_spectrogram
```

Or run the main module directly:

```bash
cd audio_spectrogram
python -m audio_spectrogram.main
```

### Command-Line Options

```bash
python -m audio_spectrogram --help
```

Options:
- `-c, --config PATH` — Path to custom configuration file
- `-v, --verbose` — Enable verbose logging
- `--list-devices` — List available audio input devices

### Custom Configuration

Use your own YAML config file:

```bash
python -m audio_spectrogram --config my_config.yaml
```

## Examples

### Check Available Microphones

```bash
python -m audio_spectrogram --list-devices
```

This will show all available audio input devices with their details.

### Run with Verbose Output

```bash
python -m audio_spectrogram -v
```

Shows debug information including audio callback status and performance metrics.

## Troubleshooting

### "No audio input devices found"

- Make sure your microphone is connected and recognized by the system
- Check system permissions for microphone access
- On Windows: Go to Settings > Privacy > Microphone and enable access

### "Error: [Errno -9997] Invalid sample rate"

- The specified sample rate isn't supported by your device
- Edit `config.yaml` and use a standard rate (44100, 48000, etc.)

### Plot not updating

- Make sure you're running in an environment with display support
- On headless systems: use a virtual framebuffer (Xvfb) on Linux

## Project Structure

```
audio-spectrogram/
├── audio_spectrogram/
│   ├── __init__.py       # Main module & classes
│   └── main.py           # CLI entry point
├── config.yaml           # Default configuration
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # GPL license
```

## License

This project is licensed under the GNU General Public License v3.0.
See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
