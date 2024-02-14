from moku.instruments import ArbitraryWaveformGenerator
import numpy as np
import matplotlib.pyplot as plt

# Generate time vector
t = np.linspace(0, 2*np.pi, 1000)

# Connect to your Moku by its ip address ArbitraryWaveformGenerator('192.168.###.###')
# or by its serial ArbitraryWaveformGenerator(serial=123)
i = ArbitraryWaveformGenerator('[]', force_connect=True)

try:
    # Generate a sine wave signal
    i.generate_waveform(channel=1, sample_rate='Auto', lut_data=list(np.sin(t)), frequency=1/(2*np.pi), amplitude=1)

    # Set channel 1 to pulse mode 
    # 2 dead cycles at 0Vpp
    i.pulse_modulate(channel=1, dead_cycles=2, dead_voltage=0)

    # Add noise with different relative amplitudes
    noise_amplitudes = [0.01, 0.015, 0.02]  # 1%, 1.5%, and 2%
    plt.figure(figsize=(12, 18))

    for idx, noise_amplitude in enumerate(noise_amplitudes, start=1):
        noise = np.random.normal(0, noise_amplitude, size=len(t))
        noisy_signal = np.sin(t) + noise
        i.generate_waveform(channel=2, sample_rate='Auto', lut_data=list(noisy_signal), frequency=1/(2*np.pi), amplitude=1)

        plt.subplot(3, 1, idx)
        plt.plot(t, np.sin(t), label='Signal', linewidth=2)
        plt.plot(t, noisy_signal, label=f'Noisy Signal ({noise_amplitude*100}%)', alpha=0.7)
        plt.legend()
        plt.title(f'Noise Amplitude: {noise_amplitude*100}%')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f'Exception occurred: {e}')

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
