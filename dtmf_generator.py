"""
DTMF (Dual-Tone Multi-Frequency) Tone Generator
This module generates DTMF tones for phone keypad navigation
"""

import numpy as np
import pyaudio
import asyncio
from typing import Dict

# DTMF frequency pairs for each digit/symbol
DTMF_FREQUENCIES = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477),
}

class DTMFGenerator:
    def __init__(self, sample_rate=44100, duration=0.2, volume=0.3):
        """
        Initialize DTMF generator
        
        Args:
            sample_rate: Audio sample rate (Hz)
            duration: Duration of each tone (seconds)
            volume: Volume level (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.volume = volume
        self.p = pyaudio.PyAudio()
        
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
        )
    
    def generate_tone(self, digit: str) -> np.ndarray:
        """
        Generate DTMF tone for a specific digit
        
        Args:
            digit: The digit or symbol ('0'-'9', '*', '#')
            
        Returns:
            numpy array containing the audio samples
        """
        if digit not in DTMF_FREQUENCIES:
            raise ValueError(f"Invalid DTMF digit: {digit}")
        
        freq1, freq2 = DTMF_FREQUENCIES[digit]
        
        # Generate time array
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        
        # Generate the two sine waves and combine them
        wave1 = np.sin(2 * np.pi * freq1 * t)
        wave2 = np.sin(2 * np.pi * freq2 * t)
        
        # Combine and apply volume
        combined_wave = (wave1 + wave2) * self.volume / 2
        
        # Apply envelope to avoid clicks
        envelope_samples = int(0.01 * self.sample_rate)  # 10ms fade
        if envelope_samples > 0:
            fade_in = np.linspace(0, 1, envelope_samples)
            fade_out = np.linspace(1, 0, envelope_samples)
            
            combined_wave[:envelope_samples] *= fade_in
            combined_wave[-envelope_samples:] *= fade_out
        
        return combined_wave.astype(np.float32)
    
    async def play_tone(self, digit: str):
        """
        Asynchronously play a DTMF tone
        
        Args:
            digit: The digit or symbol to play
        """
        tone = self.generate_tone(digit)
        await asyncio.to_thread(self.stream.write, tone.tobytes())
    
    async def play_sequence(self, sequence: str, pause_between=0.1):
        """
        Play a sequence of DTMF tones
        
        Args:
            sequence: String of digits/symbols to play
            pause_between: Pause between tones (seconds)
        """
        for digit in sequence:
            if digit.isspace():
                await asyncio.sleep(pause_between * 3)  # Longer pause for spaces
            else:
                await self.play_tone(digit)
                if digit != sequence[-1]:  # Don't pause after last tone
                    await asyncio.sleep(pause_between)
    
    def close(self):
        """Close the audio stream and PyAudio"""
        if hasattr(self, 'stream'):
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()

# Example usage and testing
async def test_dtmf():
    """Test the DTMF generator"""
    dtmf = DTMFGenerator()
    
    print("Testing DTMF tones...")
    
    # Test individual digits
    test_digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '*', '#']
    for digit in test_digits:
        print(f"Playing {digit}")
        await dtmf.play_tone(digit)
        await asyncio.sleep(0.3)
    
    print("Playing sequence: 1-800-555-0199")
    await dtmf.play_sequence("18005550199", pause_between=0.15)
    
    dtmf.close()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_dtmf())
