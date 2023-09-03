import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import yaml
import time

def non_maximum_suppression_fast(boxes, overlapThresh=0.3):

    # If there is no bounding box, then return an empty list
    if len(boxes) == 0:
        return []
        
    # Initialize the list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    
    # Calculate the area of bounding boxes
    bound_area = (x2-x1+1) * (y2-y1+1)
    
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sort_index = np.argsort(y2)
    
    # Looping until nothing left in sort_index
    while sort_index.shape[0] > 0:
        # Get the last index of sort_index
        # i.e. the index of bounding box having the biggest y2
        last = sort_index.shape[0]-1
        i = sort_index[last]
        
        # Add the index to the pick list
        pick.append(i)
        
        # Compared to every bounding box in one sitting
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])        

        # Calculate the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlapping
        overlap = (w*h) / bound_area[sort_index[:last]]
        
        # Delete the bounding box with the ratio bigger than overlapThresh
        sort_index = np.delete(sort_index, 
                               np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes in pick list        
    # return boxes[pick]
    return pick

def load_engine(trt_runtime, plan_path):

    engine = trt_runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    return engine

def allocate_buffers(engine, batch_size, data_type):

   """
   This is the function to allocate buffers for input and output in the device
   Args:
      engine : The path to the TensorRT engine.
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32.

   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device.
      h_output_1: Output in the host.
      d_output_1: Output in the device.
      stream: CUDA stream.

   """

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
   # Allocate device memory for inputs and outputs.
   d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

   d_output = cuda.mem_alloc(h_output.nbytes)
   # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_1, d_input_1, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, model_output_shape):
    """
    This is the function to run the inference
    Args:
        engine : Path to the TensorRT engine
        pics_1 : Input images to the model.
        h_input_1: Input in the host
        d_input_1: Input in the device
        h_output_1: Output in the host
        d_output_1: Output in the device
        stream: CUDA stream
        batch_size : Batch size for execution time
        height: Height of the output image
        width: Width of the output image

    Output:
        The list of output images

    """
    start = time.perf_counter()
    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.

        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output.reshape((model_output_shape))
        # out = h_output

        return out , time.perf_counter() - start




def draw_detect(img , x , y , width , height , conf , class_id , label):
    # label = f'{CLASSES[class_id]} ({confidence:.2f})'
    # color = colors[class_id]
    
    print(x , y , width , height , conf , class_id)
    cv2.rectangle(img, (x, y), (x + width, y + height), (0,0,255), 2)

    cv2.putText(img, f"{label[class_id]} {conf:0.3}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

def show_detect(img , preds , iou_threshold , conf_threshold, class_label):
    boxes = []
    scores = []
    class_ids = []
    

    for pred_idx in range(preds.shape[2]):
        pred = preds[0,:,pred_idx]
        box = [pred[0] - 0.5*pred[2], pred[1] - 0.5*pred[3] , pred[2] , pred[3]]

        conf = pred[4:]
        label = np.argmax(conf)
        max_conf = np.max(conf)
        if max_conf < conf_threshold:
            continue
        boxes.append(box)
        
        scores.append(max_conf)
        class_ids.append(label)

    boxes = np.array(boxes)
    result_boxes = non_maximum_suppression_fast(boxes, overlapThresh=iou_threshold)
    

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        
        draw_detect(img, round(box[0]), round(box[1]),round(box[2]), round(box[3]),
            scores[index] , class_ids[index] , class_label)
    
    
    
    return
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=1, type=str, help='model path')
    parser.add_argument('--source', nargs=1 , type=str  ,help='inference target')
    parser.add_argument('--output-shape' , nargs='+' , type=int, help='model output shape')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--data', nargs=1 , type=str, help=' dataset.yaml path')

    opt = parser.parse_args()
    return opt

def main(opt):
    print(opt)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine_path = opt['weights'][0]
    WIDTH , HEIGHT = opt['imgsz']
    model_output_shape = opt['output_shape']
    engine = load_engine(trt_runtime, engine_path)
    source =  opt['source'][0]
    iou_threshold =  opt['iou_thres']
    conf_threshold = opt['conf_thres']
    yaml_path = opt['data'][0]

    with open(yaml_path, 'r') as stream:
        data = yaml.load(stream)
    
    label = data['names']
    print(label)

    if source.split('.')[-1] in ('jpg' , 'png' , 'jpeg'):
        image_inferences(source , WIDTH , HEIGHT , model_output_shape , engine , iou_threshold , conf_threshold , label)
    else:
        if len(source.split('.')) == 1: source = int(source)

        video_inferences(source , WIDTH , HEIGHT , model_output_shape , engine , iou_threshold , conf_threshold , label)



def video_inferences(video_path , WIDTH , HEIGHT , model_output_shape , engine , iou_threshold , conf_threshold , label):
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("VideoCapture Error")
        return
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        
        start_time = time.perf_counter()
        frame = cv2.resize(frame , (WIDTH , HEIGHT))            
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = np.array(im, dtype=np.float32, order='C')
        im = im.transpose((2, 0, 1))
        im = (2.0 / 255.0) * im - 1.0
        out , _ = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, model_output_shape)

        
        show_detect(frame , out , iou_threshold , conf_threshold , label)

        
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print((end_time - start_time))
        print(fps)
        cv2.putText(frame, f"fps : {int(fps)}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow("img" , frame)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()

def image_inferences(img_path , WIDTH , HEIGHT , model_output_shape , engine , iou_threshold , conf_threshold , label):
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
    img = cv2.imread(img_path)
    img = cv2.resize(img , (WIDTH , HEIGHT))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = np.array(im, dtype=np.float32, order='C')
    im = im.transpose((2, 0, 1))
    im = (2.0 / 255.0) * im - 1.0
    out , infer_time = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, model_output_shape)
    show_detect(img , out , iou_threshold , conf_threshold , label)
    print(f"success inference with {int(infer_time*1000)} ms")
    cv2.imshow("img" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__" :
    opt = parse_opt()
    main(vars(opt))