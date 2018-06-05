from librosa import load, power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow
import os
import matplotlib.pyplot as plt
import numpy as np
from get_features import data_dir

# example of vocal fry spectrogram
y, sr = load(os.path.join(data_dir, 'TA_SF_43_Vox18-01.wav'), sr=None, offset=3.2, duration=2.4)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Vocal Fry Mel Spectrogram')

y, sr = load(os.path.join(data_dir, 'T_HA_20_LeadVoxHi-05.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.subplot(2, 1, 2)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('False Chord Mel Spectrogram')
plt.tight_layout()
plt.show()


y, sr = load(os.path.join(data_dir, 'DR_POM_28_LeadVox1-12.wav'), sr=None, offset=2.6, duration=3.2)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Vocal Fry Mel Spectrogram')

y, sr = load(os.path.join(data_dir, 'T_HA_21_LeadVoxLo-01.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.subplot(2, 1, 2)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('False Chord Mel Spectrogram')
plt.tight_layout()
plt.show()


y, sr = load(os.path.join(data_dir, 'CAT_B_40_LeadVox-06.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Vocal Fry Dominated Mel Spectrogram')

y, sr = load(os.path.join(data_dir, 'CAT_B_42_BackingVox1-06.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.subplot(2, 1, 2)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('False Chord Dominated Mel Spectrogram')
plt.tight_layout()
plt.show()


y, sr = load(os.path.join(data_dir, 'SOP_OIAHF_27_LeadVox1-04.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Vocal Fry Dominated Mel Spectrogram')

y, sr = load(os.path.join(data_dir, 'HG_LB_15_LeadVox1-01.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.subplot(2, 1, 2)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('False Chord Dominated Mel Spectrogram')
plt.tight_layout()
plt.show()

y, sr = load(os.path.join(data_dir, 'LL_WWIH_30_LeadVox2-17.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Low Pitch False Chord Dominated Mel Spectrogram')

y, sr = load(os.path.join(data_dir, 'LL_WWIH_30_LeadVox2-12.wav'), sr=None)
spec = melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels=256)
plt.subplot(2, 1, 2)
specshow(power_to_db(spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('High Pitch False Chord Dominated Mel Spectrogram')
plt.tight_layout()
plt.show()