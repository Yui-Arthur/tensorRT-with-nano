import wave

audio = wave.open("data/audio/dog2.wav" , "r")
audio_data_size = audio.getsampwidth()
audio_size = audio.getnframes()
data = []
for _ in range(audio_size):
    int_16bit = int.from_bytes(audio.readframes(1) , byteorder ='little' , signed=True)
    max_16bit =  32768
    float_num = int_16bit / (max_16bit -1) if int_16bit > 0 else int_16bit / (max_16bit)
    data.append(float_num)
    # print(float_num)
print(len(data))
# print(audio.readframes(5000))