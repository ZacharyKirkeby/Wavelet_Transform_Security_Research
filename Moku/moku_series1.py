import numpy as np
from moku.instruments import ArbitraryWaveformGenerator
import matplotlib.pyplot as plt

# Generate time vector
t = np.linspace(0, 1, 100000)

# Generate sine and cosine waves
sin_wave = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
cos_wave = np.cos(2 * np.pi * 10 * t)  # 10 Hz cosine wave

# Generate more complex waveform
complex_wave = np.sin(t) ** 2 * np.cos(t)

t = np.linspace(0, 10, 100000)

# Define parameters for the trigonometric components
freqs = [1, 3, 5, 7]  # Frequencies of the sine and cosine waves
phases = [0, np.pi/2, np.pi, 3*np.pi/2]  # Phases of the sine waves
amplitudes = [1, 0.5, 0.3, 0.2]  # Amplitudes of the sine and cosine waves

# Generate the noisy trigonometric time series
trig_series = np.zeros_like(t)
for freq, phase, amp in zip(freqs, phases, amplitudes):
    trig_series += amp * (np.sin(2 * np.pi * freq * t + phase) + np.cos(2 * np.pi * freq * t + phase))
    trig_series += np.random.normal(0, 0.1, size=len(t))  # Add random noise

# Plot the noisy trigonometric time series
plt.figure(figsize=(12, 4))
plt.plot(t, trig_series)
plt.title('Noisy Trigonometric Time Series')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Plot the generated waveforms
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, sin_wave)
plt.title('Sine Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, cos_wave)
plt.title('Cosine Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, complex_wave)
plt.title('Complex Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sin_wave)
plt.title('Sine Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, cos_wave)
plt.title('Cosine Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


# Connect to your Moku by its ip address ArbitraryWaveformGenerator('192.168.###.###')
# or by its serial ArbitraryWaveformGenerator(serial=123)
i = ArbitraryWaveformGenerator('[]', force_connect=True)

try:
    # Load and configure the waveform.
    i.generate_waveform(channel=1, sample_rate='Auto',
                        lut_data=list(sin_wave), frequency=10,
                        amplitude=1)
    i.generate_waveform(channel=2, sample_rate='Auto', lut_data=list(cos_wave),
                        frequency=10, amplitude=1)

    # Set channel 1 to pulse mode 
    # 2 dead cycles at 0Vpp
    i.pulse_modulate(channel=1, dead_cycles=2, dead_voltage=0)

    # Set Channel 2 to burst mode
    # Burst mode triggering from Input 1 at 0.1 V
    # 3 cycles of the waveform will be generated every time it is triggered
    i.burst_modulate(channel=2, trigger_source='Input1', trigger_mode='NCycle', burst_cycles=3, trigger_level=0.1)

except Exception as e:
    print(f'Exception occurred: {e}')

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
