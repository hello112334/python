import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import sklearn
from pydub import AudioSegment
import numpy as np

import cepstrum

input_file_path = 'C:\\Users\\TPI_CHEN\Documents\\TPI\\42_TK2_Dev\
\\06_AWS_WebSer\\NodeJS\\aws_webser\\Web_frontend\\src\\assets\\audio\\input\\'
# input_file_name = 'JLPT N1 Listening Sample Exam with Answers.mp3'
# input_file_name = "30歳まで童貞だと魔法使いになれるらしい チェリまほ テレ東 TSUTAYA【期間限定 1話無料公開中】『30歳まで童貞だと魔法使いになれるらしい』第1話.mp3"
# input_file_name = "newSong.mp3"
# input_file_name = "voice_jp_2.mp3"
input_file_name = "voice_jp_11.mp3"

output_file_path = 'C:\\Users\\TPI_CHEN\Documents\\TPI\\42_TK2_Dev\
\\06_AWS_WebSer\\NodeJS\\aws_webser\\Web_frontend\\src\\assets\\audio\\output\\'
output_file_name = 'voice_jp_25.mp3'

def main():

    dst = "tempConvert.wav"

    # file_name = input_file_path + input_file_name
    wav_file_name = input_file_path + dst
    # sound = AudioSegment.from_file( file_name , "mp3" )

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(output_file_path + output_file_name)
    sound.export(wav_file_name, format="wav")

    # file_name = input_file_path + input_file_name
    wav_file_name = input_file_path + dst

    # audio_data = file_name
    audio_data = wav_file_name

    x , sr = librosa.load(audio_data)

    try:
        # print(type(x), type(sr)) 

        ipd.Audio(audio_data)

        # plt.figure(figsize=(14, 5))
        # librosa.display.waveplot(x, sr=sr)
        
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        # spectral_centroids.shape(775,)

        # Computing the time variable for visualization
        plt.figure(figsize=(14, 4))
        frames = range(len(spectral_centroids))
        time_arr = librosa.frames_to_time(frames)

        #Plotting the Spectral Centroid along the waveform
        librosa.display.waveplot(x, sr=sr, alpha=0.4)
        plt.plot(time_arr, normalize(spectral_centroids), color='blue',alpha=0.5)
        # plt.plot(time_arr, normalize(spectral_centroids), color='gray', alpha=0.5)
        # plt.plot(time_arr, spectral_centroids, color='b')

        get_test = cepstrum.complex_cepstrum(normalize(spectral_centroids))
        # print(len(get_test[0]))

        # plt.figure(figsize=(14, 4))
        max_num = max(time_arr)
        time_range = max_num / len(get_test[0])
        time_arr2 = np.arange(0, max_num, time_range)
        plt.plot(time_arr2, get_test[0], color='red', alpha=0.8)
        plt.show()

        # plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')

    except Exception as err:
        print(err)


# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

"""
computes the spectral centroid from the (squared) magnitude spectrum

  Args:
    X: spectrogram (dimension FFTLength X Observations)
    f_s: sample rate of audio data

  Returns:
    vsc spectral centroid (in Hz)
"""
def FeatureSpectralCentroid(X, f_s):

    isSpectrum = X.ndim == 1

    # X = X**2 removed for consistency with book

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    vsc = np.dot(np.arange(0, X.shape[0]), X) / norm

    # convert from index to Hz
    vsc = vsc / (X.shape[0] - 1) * f_s / 2

    # if input is a spectrum, output scaler else if spectrogram, output 1d array
    vsc = np.squeeze(vsc) if isSpectrum else np.squeeze(vsc, axis=0)

    return vsc

# 本体
if __name__ == '__main__':

    print(" ------------ START ------------ ")

    # MAIN
    main()
    

    print(" ------------ END ------------ ")