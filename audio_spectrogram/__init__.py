#!/usr/bin/env python3
"""
Real-time Audio Spectrogram Visualizer

Captures audio from the default microphone, converts to mono, 
and displays a live spectrogram + waveform using Matplotlib.

Usage:
    python -m audio_spectrogram
    
Or simply:
    python -m audio_spectrogram
"""

import argparse
import copy
import logging
import sys
import gc
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd

# Use TkAgg backend for better macOS compatibility
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml


class SpectrogramConfig:
    """Configuration manager for the spectrogram application."""
    
    DEFAULT_CONFIG = {
        "audio": {
            "sample_rate": 44100,
            "chunk_size": 2048,
            "channels": 1,
            "dtype": "int16"
        },
        "spectrogram": {
            "fft_window": 2048,
            "hop_length": 512,
            "window_function": "hann",
            "frequency_limit": 22050
        },
        "plot": {
            "refresh_rate": 30,
            "colormap": "viridis",
            "colorbar": True,
            "title": "Real-time Audio Spectrogram & Waveform"
        },
        "plotfft": {
            "alpha": 0.1
        }
    }
    
    def __init__(self, config_path: str = None):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, path: str):
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge - update nested dicts
            for section, values in user_config.items():
                if section in self.config and isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
                    
        except Exception as e:
            logging.warning(f"Failed to load config from {path}: {e}")
    
    @property
    def sample_rate(self) -> int:
        return self.config["audio"]["sample_rate"]
    
    @property
    def chunk_size(self) -> int:
        return self.config["audio"]["chunk_size"]
    
    @property
    def channels(self) -> int:
        return self.config["audio"]["channels"]
    
    @property
    def dtype(self):
        return np.dtype(self.config["audio"]["dtype"])
    
    @property
    def fft_window(self) -> int:
        return self.config["spectrogram"]["fft_window"]
    
    @property
    def hop_length(self) -> int:
        return self.config["spectrogram"].get("hop_length", self.fft_window // 4)
    
    @property
    def window_function(self) -> str:
        return self.config["spectrogram"].get("window_function", "hann")
    
    @property
    def frequency_limit(self) -> int:
        return self.config["spectrogram"].get("frequency_limit", self.sample_rate // 2)
    
    @property
    def refresh_rate(self) -> float:
        return self.config["plot"].get("refresh_rate", 30)
    
    @property
    def colormap(self) -> str:
        return self.config["plot"].get("colormap", "viridis")
    
    @property
    def colorbar(self) -> bool:
        return self.config["plot"].get("colorbar", True)
    
    @property
    def title(self) -> str:
        return self.config["plot"].get("title", "Real-time Audio Spectrogram")

    @property
    def plotfft_alpha(self) -> float:
        alpha = self.config.get("plotfft", {}).get("alpha", 0.1)
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            logging.warning(f"Invalid plotfft.alpha '{alpha}', using 0.1")
            return 0.1
        if not 0 < alpha <= 1:
            logging.warning(f"plotfft.alpha must be in (0, 1], got {alpha}. Using 0.1")
            return 0.1
        return alpha


class AudioSpectrogram:
    """Real-time audio spectrogram + waveform visualization."""
    
    # Number of historical slices to keep for the spectrogram
    HISTORY_FRAMES = 64
    
    def __init__(self, config: SpectrogramConfig):
        self.config = config
        self.running = False
        
        # Pre-compute window function
        self.window = self._get_window_function()
        
        # Frequency array (only positive frequencies)
        self.freqs = np.fft.rfftfreq(config.fft_window, 1.0 / config.sample_rate)
        freq_mask = self.freqs <= config.frequency_limit
        self.freqs_limited = self.freqs[freq_mask]
        
        # Spectrogram history (rows = time, columns = frequency bins)
        self.spectrogram_history = np.zeros((self.HISTORY_FRAMES, len(self.freqs_limited)))
        
        # Waveform buffer - store more samples for visualization
        self.waveform_samples = 44100 // 2  # Show ~0.5 seconds of audio
        self.audio_buffer = deque(maxlen=self.waveform_samples)
        
        # Track min/max for auto-scaling
        self.db_min = -60
        self.db_max = 0

        # FFT plot state
        self.current_fft_magnitude = np.zeros(len(self.freqs_limited))
        self.smoothed_fft_magnitude = np.zeros(len(self.freqs_limited))
        self.peak_frequency_hz = None
        self.peak_magnitude = None
        
        # Matplotlib components
        self.fig = None
        self.fig_fft = None
        self.ax_spec = None
        self.ax_wave = None
        self.ax_fft = None
        self.image = None
        self.wave_line = None
        self.fft_line = None
        self.fft_smooth_line = None
        self.peak_marker = None
        self.animation = None
    
    def _get_window_function(self) -> np.ndarray:
        """Get the configured window function."""
        window_name = self.config.window_function.lower()
        
        if window_name == "hann":
            return np.hanning(self.config.fft_window)
        elif window_name == "hamming":
            return np.hamming(self.config.fft_window)
        elif window_name == "blackman":
            return np.blackman(self.config.fft_window)
        elif window_name == "bartlett":
            return np.bartlett(self.config.fft_window)
        else:
            logging.warning(f"Unknown window function '{window_name}', using Hann")
            return np.hanning(self.config.fft_window)
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio stream."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # Convert to mono (average channels if stereo)
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        
        # Add to waveform buffer
        for sample in mono:
            self.audio_buffer.append(sample)
        
        # Compute magnitude spectrum for this chunk
        windowed_data = mono * self.window
        
        # Zero-padding to match fft_window if needed (for final incomplete frame)
        if len(windowed_data) < self.config.fft_window:
            padded = np.zeros(self.config.fft_window)
            padded[:len(windowed_data)] = windowed_data
            windowed_data = padded
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_data)
        
        # Compute magnitude spectrum in dB scale
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Limit to frequency range and store
        freq_mask = self.freqs <= self.config.frequency_limit
        limited_magnitude = magnitude[freq_mask]
        self.current_fft_magnitude = limited_magnitude
        self.smoothed_fft_magnitude = (
            self.config.plotfft_alpha * limited_magnitude
            + (1 - self.config.plotfft_alpha) * self.smoothed_fft_magnitude
        )

        if len(limited_magnitude) > 0:
            peak_index = int(np.argmax(limited_magnitude))
            self.peak_frequency_hz = float(self.freqs_limited[peak_index])
            self.peak_magnitude = float(limited_magnitude[peak_index])
        else:
            self.peak_frequency_hz = None
            self.peak_magnitude = None
        
        # Shift history and add new frame at the end
        self.spectrogram_history = np.roll(self.spectrogram_history, -1, axis=0)
        self.spectrogram_history[-1] = magnitude_db[freq_mask]
        
        # Update dB range for auto-scaling (rolling window)
        recent_dbs = self.spectrogram_history[-16:]  # Last ~0.5 seconds
        self.db_min = np.percentile(recent_dbs, 5) - 10
        self.db_max = np.percentile(recent_dbs, 95) + 10
    
    def _init_plot(self):
        """Initialize the matplotlib plots."""
        # Create figure with two subplots (spectrogram on top, waveform below)
        self.fig, (self.ax_spec, self.ax_wave) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Audio Spectrogram & Waveform")
        self.fig.canvas.mpl_connect('close_event', self._on_figure_close)
        
        # ====== Spectrogram subplot ======
        extent = [0, self.config.frequency_limit / 1000, 0, self.HISTORY_FRAMES]
        
        self.image = self.ax_spec.imshow(
            self.spectrogram_history,
            aspect='auto',
            origin='lower',
            cmap=self.config.colormap,
            extent=extent,
            vmin=-60,
            vmax=0,
            interpolation='nearest'
        )
        
        if self.config.colorbar:
            cbar = plt.colorbar(self.image, ax=self.ax_spec)
            cbar.set_label('Amplitude (dB)')
        
        self.ax_spec.set_xlabel('Frequency (kHz)')
        self.ax_spec.set_ylabel('Time frame')
        self.ax_spec.set_title('Spectrogram')
        
        # ====== Waveform subplot ======
        # Initialize with zeros
        self.wave_line, = self.ax_wave.plot(
            np.linspace(0, len(self.audio_buffer) / self.config.sample_rate, len(self.audio_buffer)),
            list(self.audio_buffer),
            lw=0.5,
            color='blue'
        )
        
        self.ax_wave.set_xlim(0, self.waveform_samples / self.config.sample_rate)
        self.ax_wave.set_ylim(-32768, 32767)  # Full 16-bit range
        self.ax_wave.set_xlabel('Time (s)')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_title('Waveform')
        self.ax_wave.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Create a second figure for the FFT magnitude view.
        self.fig_fft, self.ax_fft = plt.subplots(figsize=(12, 4.5))
        self.fig_fft.canvas.manager.set_window_title("FFT Magnitude")
        self.fig_fft.canvas.mpl_connect('close_event', self._on_figure_close)

        self.fft_line, = self.ax_fft.plot(
            self.freqs_limited,
            self.current_fft_magnitude,
            color='tab:orange',
            linewidth=1.0,
            label='FFT magnitude'
        )
        self.fft_smooth_line, = self.ax_fft.plot(
            self.freqs_limited,
            self.smoothed_fft_magnitude,
            color='tab:green',
            linewidth=1.5,
            label=f'Smoothed magnitude (alpha={self.config.plotfft_alpha:.2f})'
        )
        self.peak_marker, = self.ax_fft.plot(
            [],
            [],
            marker='o',
            linestyle='None',
            color='tab:red',
            markersize=3,
            label='Peak frequency'
        )

        self.ax_fft.set_xlim(0, self.config.frequency_limit)
        self.ax_fft.set_ylim(0, 1)
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Magnitude')
        self.ax_fft.set_title('FFT Magnitude')
        self.ax_fft.grid(True, alpha=0.3)
        self.ax_fft.legend(loc='upper right')
        self.fig_fft.tight_layout()

    def _on_figure_close(self, event):
        """Stop the application if either figure is closed."""
        if self.running:
            self.stop()
            plt.close('all')
    
    def _update_plot(self, frame):
        """Update all plots - called by FuncAnimation."""
        if not self.running:
            return self.image, self.wave_line
        
        # ====== Update spectrogram ======
        self.image.set_array(self.spectrogram_history)
        self.image.set_clim(vmin=self.db_min, vmax=self.db_max)
        
        # ====== Update waveform ======
        audio_array = np.array(self.audio_buffer)
        time_axis = np.linspace(0, len(audio_array) / self.config.sample_rate, len(audio_array))
        self.wave_line.set_data(time_axis, audio_array)
        
        # Auto-scale y-axis based on current audio
        if len(audio_array) > 0:
            max_val = np.max(np.abs(audio_array))
            if max_val > 100:  # Only scale if there's actual signal
                self.ax_wave.set_ylim(-max_val * 1.2, max_val * 1.2)

        # ====== Update FFT magnitude figure ======
        self.fft_line.set_data(self.freqs_limited, self.current_fft_magnitude)
        self.fft_smooth_line.set_data(self.freqs_limited, self.smoothed_fft_magnitude)

        raw_fft_max = float(np.max(self.current_fft_magnitude)) if len(self.current_fft_magnitude) > 0 else 0.0
        smooth_fft_max = float(np.max(self.smoothed_fft_magnitude)) if len(self.smoothed_fft_magnitude) > 0 else 0.0
        fft_max = max(raw_fft_max, smooth_fft_max)
        self.ax_fft.set_ylim(0, max(fft_max * 1.1, 1.0))

        if self.peak_frequency_hz is not None and self.peak_magnitude is not None:
            self.peak_marker.set_data([self.peak_frequency_hz], [self.peak_magnitude])
            self.ax_fft.set_title(f'FFT Magnitude (Peak: {self.peak_frequency_hz:.1f} Hz)')
        else:
            self.peak_marker.set_data([], [])
            self.ax_fft.set_title('FFT Magnitude')

        if self.fig_fft is not None:
            self.fig_fft.canvas.draw_idle()
        
        return self.image, self.wave_line
    
    def start(self):
        """Start the visualization using FuncAnimation."""
        # Initialize plot
        self._init_plot()
        
        # Check for available input devices
        try:
            devices = sd.query_devices(kind='input')
            if not devices:
                logging.error("No audio input devices found!")
                print("Error: No microphone detected.")
                return
            
            device_name = devices.get('name', 'Unknown') if isinstance(devices, dict) else str(devices)
            print(f"Using audio device: {device_name}")
            
        except Exception as e:
            logging.error(f"Error querying devices: {e}")
            print(f"Error querying devices: {e}")
            return
        
        # Start the animation with audio stream
        self.running = True
        
        try:
            print("Streaming started. Close the window or press Ctrl+C to stop.")
            
            # Create animation and KEEP REFERENCE (prevents garbage collection)
            interval = 1000 / self.config.refresh_rate  # Convert Hz to ms interval
            
            self.animation = FuncAnimation(
                self.fig,
                self._update_plot,
                interval=interval,
                blit=False,  # Disable blit for better compatibility
                cache_frame_data=False
            )
            
            # Show the window - this blocks until closed
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                callback=self._audio_callback
            ):
                plt.show()
            
        except KeyboardInterrupt:
            print("\nStopping spectrogram...")
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            print(f"Error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the spectrogram and clean up."""
        self.running = False
        if self.animation is not None:
            self.animation.event_source.stop()


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for the spectrogram application."""
    parser = argparse.ArgumentParser(
        description='Real-time Audio Spectrogram Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m audio_spectrogram                      # Run with default settings
  python -m audio_spectrogram --config custom.yaml # Use custom config file
  python -m audio_spectrogram -v                   # Verbose output
  python -m audio_spectrogram --list-devices       # List audio devices
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio input devices and exit'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # List devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        print(sd.query_devices(kind='input'))
        return
    
    # Find config file (look in current dir, then package directory)
    config_path = Path(args.config)
    if not config_path.exists():
        # Try package default
        package_config = Path(__file__).parent.parent / "config.yaml"
        if package_config.exists():
            config_path = package_config
        else:
            logging.warning(f"Config file not found, using defaults")
            config_path = None
    
    # Create configuration and run
    config = SpectrogramConfig(str(config_path) if config_path else None)
    
    print("=" * 50)
    print("Audio Spectrogram Configuration:")
    print(f"  Sample Rate: {config.sample_rate} Hz")
    print(f"  Chunk Size: {config.chunk_size}")
    print(f"  FFT Window: {config.fft_window}")
    print(f"  Hop Length: {config.hop_length}")
    print(f"  Window Function: {config.window_function}")
    print("=" * 50)
    
    spectrogram = AudioSpectrogram(config)
    spectrogram.start()


if __name__ == "__main__":
    main()
