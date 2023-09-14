import numpy as np
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import yaml
import time
import onnxruntime as ort

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


def draw_detect(img , x1 , y1 , x2 , y2 , conf , class_id , label):
    # label = f'{CLASSES[class_id]} ({confidence:.2f})'
    # color = colors[class_id]
    
    # print(x1 , y1 , x2 , y2 , conf , class_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.putText(img, f"{label[class_id]} {conf:0.3}", (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

def show_detect(img , preds , iou_threshold , conf_threshold, class_label):
    boxes = []
    scores = []
    class_ids = []
    
    # print()
    max_conf = np.max(preds[0,4:,:] , axis=0)
    idx_list = np.where(max_conf > conf_threshold)[0]
    
    # for pred_idx in range(preds.shape[2]):
    for pred_idx in idx_list:

        pred = preds[0,:,pred_idx]
        conf = pred[4:]
        
        
        box = [pred[0] - 0.5*pred[2], pred[1] - 0.5*pred[3] , pred[0] + 0.5*pred[2] , pred[1] + 0.5*pred[3]]
        boxes.append(box)

        label = np.argmax(conf)
        
        scores.append(max_conf[pred_idx])
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
    # parser.add_argument('--output-shape' , nargs='+' , type=int, help='model output shape')
    parser.add_argument('--device', type=str , default='CPU', help='the running device')
    # parser.add_argument('--imgsz', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--data', nargs=1 , type=str, help=' dataset.yaml path')
    parser.add_argument('--show', action="store_true", help=' show detect result')

    opt = parser.parse_args()
    return opt

def main(opt):
    print(opt)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    onnx_path = opt['weights'][0]
    # WIDTH , HEIGHT = opt['imgsz']
    provider = opt['device'] + "ExecutionProvider"
    
    source =  opt['source'][0]
    iou_threshold =  opt['iou_thres']
    conf_threshold = opt['conf_thres']
    yaml_path = opt['data'][0]
    show = opt['show']

    # ort_sess = ort.InferenceSession(onnx_path)
    # outputs = ort_sess.run(None, {'input': x.numpy()})

    with open(yaml_path, 'r') as stream:
        data = yaml.load(stream)
    
    label = data['names']
    print(label)

    if source.split('.')[-1] in ('jpg' , 'png' , 'jpeg'):
        image_inferences(source , onnx_path , provider , iou_threshold , conf_threshold , label , show)
    else:
        if len(source.split('.')) == 1: source = int(source)

        video_inferences(source , onnx_path , provider , iou_threshold , conf_threshold , label , show)



def video_inferences(video_path , onnx_path , provider  , iou_threshold , conf_threshold , label , show):
    
    ort_sess = ort.InferenceSession(onnx_path , providers=[provider])
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    output_name = ort_sess.get_outputs()[0].name

    WIDTH = input_shape[2]
    HEIGHT = input_shape[3]

    total_infer_time = 0
    total_infer_count = 0
    warmup_count = 100

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
        im /=  255
        
        infer_start_time = time.perf_counter()
        output_data = ort_sess.run([output_name], {input_name: [im]})[0]
        infer_time = time.perf_counter() - infer_start_time

        show_detect(frame , output_data , iou_threshold , conf_threshold , label)

        
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        # print(f"do_inference : {infer_time}")
        # print(f"all_inference : {(end_time - start_time)}")
        # print(f"fps : {(fps)}")

        total_infer_count += 1
        if total_infer_count > warmup_count:
            total_infer_time += end_time - start_time
            print(f"avg FPS : {1/(total_infer_time / (total_infer_count-warmup_count))}")

        # print(fps)

        if show:
            cv2.putText(frame, f"fps : {int(fps)}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow("img" , frame)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print(f"avg FPS : {1/(total_infer_time / (total_infer_count-warmup_count))}")

def image_inferences(img_path , onnx_path , provider , iou_threshold , conf_threshold , label , show):
    ort_sess = ort.InferenceSession(onnx_path , providers=[provider])
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    output_name = ort_sess.get_outputs()[0].name

    WIDTH = input_shape[2]
    HEIGHT = input_shape[3]

    img = cv2.imread(img_path)
    img = cv2.resize(img , (WIDTH , HEIGHT))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = np.array(im, dtype=np.float32, order='C')
    im = im.transpose((2, 0, 1))
    im = (2.0 / 255.0) * im - 1.0
    
    
    output_data = ort_sess.run([output_name], {input_name: [im]})[0]

    show_detect(img , output_data , iou_threshold , conf_threshold , label)
    # print(f"success inference with {int(infer_time*1000)} ms")

    if show:
        cv2.imshow("img" , img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__" :
    opt = parse_opt()
    main(vars(opt))