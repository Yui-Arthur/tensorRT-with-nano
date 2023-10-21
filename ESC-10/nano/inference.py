
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import time
import wave

def allocate_buffers(engine, batch_size):

    inputs = []
    outputs = []
    bindings = []
    # data_type = engine.get_binding_dtype(0)

    for binding in engine:
        # print(engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        dic = {
                "host_mem" : host_mem,
                "device_mem" : device_mem,
                "shape" : engine.get_binding_shape(binding),
                "dtype" : dtype
            }
        if engine.binding_is_input(binding):
            inputs.append(dic)
        else:
            outputs.append(dic)

    stream = cuda.Stream()
    return inputs , outputs , bindings , stream

def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(context, pics_1, inputs , outputs , bindings , stream, model_output_shape):

    start = time.perf_counter()
    load_images_to_buffer(pics_1, inputs[0]["host_mem"])

    [cuda.memcpy_htod_async(intput_dic['device_mem'], intput_dic['host_mem'], stream) for intput_dic in inputs]

    # Run inference.

    # context.profiler = trt.Profiler()
    context.execute(batch_size=1, bindings=bindings)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(output_dic["host_mem"], output_dic["device_mem"], stream) for output_dic in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.
    out = outputs[0]["host_mem"].reshape((outputs[0]['shape']))
    # out = h_output

    return out , time.perf_counter() - start

def softmax(x):
  # x -= np.max(x , axis=1 , keepdims = True)
  x = np.exp(x) / np.sum(np.exp(x))
  return x

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def preprocess_wav(wav_file):
    audio = wave.open(wav_file , "r")
    
    audio_size = audio.getnframes()
    data = []
    for _ in range(audio_size):
        int_16bit = int.from_bytes(audio.readframes(1) , byteorder ='little' , signed=True)
        max_16bit =  32768
        float_num = int_16bit / (max_16bit -1) if int_16bit > 0 else int_16bit / (max_16bit)
        data.append(float_num)
        # print(float_num)
    return np.array(data)

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--engine', type=str, help='engine path')
    parser.add_argument('--source', type=str, help='audio path')

    opt = parser.parse_args()
    return opt

def main(opt):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    input_audio_path = opt['source']
    model_path = opt['engine']
    

    class_dic = {
        0 : "chainsaw",
        1 : "clock_tick",
        2 : "crackling_fire",
        3 : "crying_baby",
        4 : "dog",
        5 : "helicopter",
        6 : "rain",
        7 : "rooster",
        8 : "sea_waves",
        9 : "sneezing"
    }

    input_data = preprocess_wav(input_audio_path)
    
    
    start_time = time.perf_counter()

    

    engine = load_engine(trt_runtime, model_path)
    inputs , outputs , bindings , stream = allocate_buffers(engine, 1)
    context = engine.create_execution_context()
    model_output_shape = outputs[0]['shape']
    
    out , infer_time = do_inference(context, input_data, inputs , outputs , bindings, stream, model_output_shape)
    
    # out = softmax(out.astype(np.float128))
    end_time = time.perf_counter()
    label = np.argmax(out)
    value = np.max(out)

    print(f"inference time : {(end_time - start_time)*1000} ms")
    print(class_dic[label] , value)

if __name__ == '__main__':
   opt = parse_opt()
   main(vars(opt))