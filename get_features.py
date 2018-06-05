import os
import pandas as pd
import numpy as np
from scipy import stats
import librosa

data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/EVA (MIR term project)/data'


def columns():
    feature_sizes = dict(mfcc=20, rmse=1, zcr=1, poly=2, spectral_centroid=1, spectral_bandwidth=1, spectral_contrast=7,
                         spectral_flatness=1, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i + 1)) for i in range(size))
            columns.extend(it)
    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid, fname):
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    filepath = os.path.join(data_dir, fname)
    y, sr = librosa.load(filepath)

    f = librosa.feature.zero_crossing_rate(y)
    feature_stats('zcr', f)

    stft = np.abs(librosa.stft(y))

    f = librosa.feature.rmse(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_flatness(S=stft)
    feature_stats('spectral_flatness', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)
    f = librosa.feature.poly_features(S=stft)
    feature_stats('poly', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel))
    feature_stats('mfcc', f)

    return features


if __name__ == '__main__':
    data = pd.read_csv('annotations.csv')
    tracks = data['File Name'].values
    features = pd.DataFrame(columns=columns(), dtype=np.float32)
    features.index.name = 'track_id'

    for id, name in enumerate(tracks):
        print("Running", name)
        features.loc[id] = compute_features(id, name)

    features.to_csv('features.csv')
    print(features.isnull().values.any())