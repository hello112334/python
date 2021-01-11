from mutagen.mp3 import MP3

from pydub import AudioSegment
from pydub.silence import split_on_silence

import librosa
import librosa.display

import ffmpeg
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
from scipy import fftpack

import matplotlib.pyplot as plt

import scipy.io.wavfile
# from itertools import izip_longest

input_file_path = 'C:\\Users\\TPI_CHEN\Documents\\TPI\\42_TK2_Dev\
\\06_AWS_WebSer\\NodeJS\\aws_webser\\Web_frontend\\src\\assets\\audio\\input\\'
# input_file_name = 'JLPT N1 Listening Sample Exam with Answers.mp3'
# input_file_name = "30歳まで童貞だと魔法使いになれるらしい チェリまほ テレ東 TSUTAYA【期間限定 1話無料公開中】『30歳まで童貞だと魔法使いになれるらしい』第1話.mp3"
# input_file_name = "newSong.mp3"
# input_file_name = "voice_jp_2.mp3"
input_file_name = "voice_jp_8.mp3"

output_file_path = 'C:\\Users\\TPI_CHEN\Documents\\TPI\\42_TK2_Dev\
\\06_AWS_WebSer\\NodeJS\\aws_webser\\Web_frontend\\src\\assets\\audio\\output\\'
output_file_name = 'newSong.mp3'


# startMin = 9
# startSec = 50

# endMin = 13
# endSec = 30

# Time to miliseconds
# startTime = startMin*60*1000+startSec*1000
# endTime = endMin*60*1000+endSec*1000

# 層チェック
cnum = 1

# 
spms = 1000

def voice_create():
    print(" -------- voice_create ")
    
    # 
    try:
        audio = MP3(input_file_path + input_file_name)
        print("total: {0} s.".format(audio.info.length))

        # Do something with the file
        newAudio = AudioSegment.from_mp3( input_file_path + input_file_name )
         
        # Works in milliseconds 
        t1 = 0 * 1000   
        t2 = 180 * 1000

        newAudio = newAudio[t1:t2]
        newAudio.export( output_file_path + output_file_name, format="mp3") #Exports to a wav file in the current path.

        # Saving
        # extract.export( file_name+'-extract.mp3', format="mp3")
    except IOError:
        print("File not accessible")
        print( input_file_path + input_file_name )

# 
def voice_split():
    print(" -------- split_voice ")
    print(input_file_name)

    # Opening file and extracting segment
    try:
        # f = open( input_file_path + file_name )
        audio = MP3(input_file_path + input_file_name)
        print("total: {0} s.".format(audio.info.length))

        # Do something with the file
        newAudio = AudioSegment.from_mp3( input_file_path + input_file_name )
        # extract = song[startTime:endTime]
        
        silence_check(newAudio)

        # t1 = 1 * 1000   #Works in milliseconds
        # t2 = 10 * 1000
        # newAudio = AudioSegment.from_wav("oldSong.wav")
        # newAudio = newAudio[t1:t2]
        # newAudio.export( output_file_path + output_file_name, format="mp3") #Exports to a wav file in the current path.

    except IOError:
        print("File not accessible")
        print( input_file_path + input_file_name )

# 
def silence_check(song):
    print(" -------- silence_check")

    global cnum, spms
    
    # Load your audio.
    # song = AudioSegment.from_mp3("your_audio.mp3")

    chunks = split_on_silence (
    # Use the loaded audio. 
    song, 

    # split on silences longer than 1000ms (1 sec)
    min_silence_len=spms,

    # anything under -16 dBFS is considered silence
    silence_thresh=-40, 

    # keep 200 ms of leading/trailing silence
    keep_silence=200
    )

    # now recombine the chunks so that the parts are at least 90 sec long
    # target_length = 10 * 1000
    # output_chunks = [chunks[0]]
    # for chunk in chunks[1:]:
    #     if len(output_chunks[-1]) < target_length:
    #         output_chunks[-1] += chunk
    #     else:
    #         # if the last output chunk is longer than the target length,
    #         # we can start a new one
    #         output_chunks.append(chunk)

    # now your have chunks that are bigger than 90 seconds (except, possibly the last one)

    # X (X * 1000) sec long
    target_length = 8 * 1000

    # 出力
    for i, chunk in enumerate(chunks):
        print("{0} disposing..{1}/{2}".format(spms, i, len(chunks)))
        output_chunks = [chunk]
        if len(output_chunks[-1]) > target_length:
            print("Over {0}s".format(target_length/1000))
            
            if spms > 100:
                spms = spms - 100
            elif spms < 20:
                break
            else:  
                spms = spms/2

            silence_check(chunk)
            spms = spms + 100
        else:  
            cnum = cnum + 1
            chunk.export("{0}voice_jp_{1}.mp3".format(output_file_path,cnum), format="mp3")


def fre_check():
    print(" -------- fre_check")

    file_name = input_file_path + input_file_name

    sound = AudioSegment.from_file( file_name , "mp3" )
    data = np.array(sound.get_array_of_samples())
    spec = np.fft.fft(data)   #2次元配列(実部，虚部)
    freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate) 
    spec = spec[:int(spec.shape[0]/2 + 1)]    #周波数がマイナスになるスペクトル要素の削除
    freq = freq[:int(freq.shape[0]/2 + 1)]    #周波数がマイナスになる周波数要素の削除
    max_spec=max(np.abs(spec))    #最大音圧を取得(音圧を正規化するために使用）
    plt.plot(freq, np.abs(spec)/max_spec)

    plt.grid()
    plt.xlim([0,4000])    #グラフに出力する周波数の範囲[Hz]
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Sound Pressure[-]")
    # plt.yscale("log")
    # plt.savefig(file_name + ".png")  #pngファイルで出力
    plt.show()

def f(x):
    pi2 = 2.*np.pi
    return 1.0*np.sin(0.1*pi2*x) + 1.0*np.cos(1.*pi2*x)


def fre_check_2():
    print(" -------- fre_check_2")

    file_name = input_file_path + input_file_name

    sound = AudioSegment.from_file( file_name , "mp3" )
    data = np.array(sound.get_array_of_samples())

    freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate) 

    #Periodic data with/without random noise
    # xdata = np.linspace(0, 100, num=1024)
    xdata = freq
    # np.random.seed(1234)
    ydata = f(xdata) #+ 10.*np.random.randn(xdata.size)

    time_step = xdata[1]-xdata[0]

    #FFT
    sample_freq = fftpack.fftfreq(ydata[:].size, d=time_step)
    y_fft = fftpack.fft(ydata[:])
    pidxs = np.where(sample_freq > 0)
    freqs, power = sample_freq[pidxs], np.abs(y_fft)[pidxs]
    freq = freqs[power.argmax()]

    #PLot
    plt.figure(figsize=(8,10))
    plt.subplot(211)
    plt.plot(xdata,ydata,'b-', linewidth=0.2)
    plt.xlabel('Time')
    plt.ylabel('Ydata')
    plt.grid(True)

    plt.subplot(212)
    #plt.semilogx(freqs, power,'b.-',lw=1)
    plt.loglog(freqs, power,'b.-',lw=1)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid(True)

    plt.show()


def fre_check_3():

    #音声ファイル読み込み
    # local_path = "F:/CHEN/01_devolment/04_Python/voice/file/"

    dst = "tempConvert.wav"

    file_name = input_file_path + input_file_name
    wav_file_name = input_file_path + dst
    # sound = AudioSegment.from_file( file_name , "mp3" )

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(file_name)
    sound.export(wav_file_name, format="wav")

    # analyze_filename = input_file_path + input_file_name
    rate, data = scipy.io.wavfile.read(wav_file_name)
         
    
    sound_sum = data.shape[0] # サンプル数
    sound_sec = float(format(sound.duration_seconds, '.2f'))
    dt    = float(format(sound_sec/len(data), '.8')) # サンプリング周期（ sec ）

    # 時間軸
    # time_arr = np.arange(0, sound_sum, dt)
    time_arr = np.arange(0, sound_sum, dt)

    #（振幅）の配列を作成
    # data = data / 32768

    ##### 周波数成分を表示する #####
    #縦軸：dataを高速フーリエ変換する（時間領域から周波数領域に変換する）
    # fft_data = np.abs(np.fft.fft(data))
    # fft_data = time_arr

    #横軸：周波数の取得　　#np.fft.fftfreq(データ点数, サンプリング周期)
    # freqList = np.fft.fftfreq(data.shape[0], d=1.0/rate)  

    #データプロット
    # plt.plot( freqList, fft_data)
    plt.plot( data, time_arr)
    
    # plt.xlim(0, 100000) #0～8000Hzまで表示
    plt.show()


# 本体
if __name__ == '__main__':

    print(" ------------ START ------------ ")

    # 自動分割
    # voice_split()

    # 単体
    # voice_create()

    # 周波数
    # fre_check()
    # fre_check_2()
    fre_check_3()

    print(" ------------ END ------------ ")