import numpy as np
import sounddevice as sd

def main():
    # Musical scale notes with their frequencies (4th octave)
    notes = {
        'C': 261.63,
        'D': 293.66,
        'E': 329.63,
        'F': 349.23,
        'G': 392.00,
        'A': 440.00,
        'B': 493.88
    }
    
    sample_rate = 44100
    duration = 0.5  # Half second per note
    
    print("Playing C Major Scale: C, D, E, F, G, A, B")
    print("=" * 50)
    
    # Play each note in the scale
    for note_name, frequency in notes.items():
        mysound = sine_tone(frequency=frequency, duration=duration)
        
        print(f"Playing {note_name}: {frequency:.2f} Hz")
        
        sd.play(mysound, samplerate=sample_rate)
        sd.wait()
    
    print("=" * 50)
    print("Scale playback completed!")


def sine_tone(
        frequency: float=440,
        duration: float=1.0,
        amplitude: float=0.5,
        sample_rate: int=44100
    ) -> np.ndarray:

    n_samples = int(sample_rate * duration)  # Number of samples per second

    time_points = np.linspace(0, duration, n_samples, endpoint=False)  # Time points for the duration of the tone

    sine = np.sin(2 * np.pi * frequency * time_points)  # Generate the sine wave

    sine *= amplitude  # Scale the sine wave by the amplitude
    return sine.astype(np.float32)  # Ensure correct data type for audio playback


if __name__ == "__main__":
    main() 