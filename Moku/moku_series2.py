from moku.instruments import ArbitraryWaveformGenerator
import numpy as np
import matplotlib.pyplot as plt

# Generate time vector
t = np.linspace(0, 2*np.pi, 1000)

# Connect to your Moku by its ip address ArbitraryWaveformGenerator('192.168.###.###')
# or by its serial ArbitraryWaveformGenerator(serial=123)
i = ArbitraryWaveformGenerator('[]', force_connect=True)

try:
    # Generate sine wave
    i.generate_waveform(channel=1, sample_rate='Auto', lut_data=list(np.sin(t)), frequency=1/(2*np.pi), amplitude=1)
    # Generate cosine wave
    i.generate_waveform(channel=2, sample_rate='Auto', lut_data=list(np.cos(t)), frequency=1/(2*np.pi), amplitude=1)

    # Set channel 1 to pulse mode 
    # 2 dead cycles at 0Vpp
    i.pulse_modulate(channel=1, dead_cycles=2, dead_voltage=0)

    # Set Channel 2 to burst mode
    # Burst mode triggering from Input 1 at 0.1 V
    # 3 cycles of the waveform will be generated every time it is triggered
    i.burst_modulate(channel=2, trigger_source='Input1', trigger_mode='NCycle', burst_cycles=3, trigger_level=0.1)

    # Plot the sine wave, cosine wave, and the sum of the two waves
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, np.sin(t), label='Sin Wave')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, np.cos(t), label='Cos Wave')
    plt.legend()

    plt.subplot(3, 1, 3)
    sum_wave = np.sin(t) + np.cos(t)
    plt.plot(t, sum_wave, label='Sum of Sin and Cos')
    plt.legend()

    plt.show()
    plt.figure(figsize=(12, 6))
    diff_wave = np.sin(t) - np.cos(t)
    plt.plot(t, diff_wave, label='Sin - Cos')
    plt.legend()

    plt.title('Subtraction of Cosine Wave from Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    plt.figure(figsize=(12, 6))
    prod_wave = np.sin(t) * np.cos(t)
    plt.plot(t, prod_wave, label='Sin * Cos')
    plt.legend()

    plt.title('Multiplication of Sine and Cosine Waves')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

except Exception as e:
    print(f'Exception occurred: {e}')

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
