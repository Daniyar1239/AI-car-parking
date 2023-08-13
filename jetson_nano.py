import numpy as np
import onnxruntime as ort
import cv2
import pandas as pd
import json
import requests
import time



def preprocess_image(patch, channels=3):
    
    patch = cv2.resize(patch, (96, 96), interpolation = cv2.INTER_AREA)
    patch_data = np.asarray(patch).astype(np.float32)
    patch_data = patch_data.transpose([2, 0, 1]) # transpose to CHW
    
    mean = np.array([0.33938432, 0.39506438, 0.3533833])
    std = np.array([0.12990639, 0.14369106, 0.13916087])

    for channel in range(patch_data.shape[0]):
        patch_data[channel, :, :] = (patch_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    
    return patch_data


def run_sample(df, frame, ort_sess):
    
    stack_tensor = np.zeros((df.shape[0], 3 , 96, 96))

    for i in range(df.shape[0]):
        
        x, y , w, h = df.loc[i, ['x', 'y', 'w', 'h']]
        patch = preprocess_image(frame[y: (y+h), x: (x+w)])
        patch_ext = np.expand_dims(patch , axis = 0)
        stack_tensor[i,:,:,:] = patch_ext 
        
    X = stack_tensor.astype('float32')
    test_pred_logits = ort_sess.run(None, {'image_inp': X})[0]
    # print(test_pred_logits)
    # test_pred_logits = torch.from_numpy(test_pred_logits[0])
    test_pred_labels = test_pred_logits.argmax(axis=1)
  
    return test_pred_labels


def convert_result_todict(array):
    result_dict = dict()

    for i in range(len(array)):
        slot = "busy" if array[i] == 1 else "free"
        result_dict["status_place_" + str(i+1)] = slot

    return result_dict

def max_frequent_values(arr):
    # Get the unique values and their counts for each column
    unique_vals, val_counts = np.apply_along_axis(np.unique, 0, arr, return_counts=True)
    
    # Get the indices of the maximum count for each column
    max_freq_indices = np.argmax(val_counts, axis=0)
    
    # Use the indices to get the maximum frequent value for each column
    max_freq_values = unique_vals[max_freq_indices, np.arange(arr.shape[1])]
    
    # Return the max_freq_values array
    return max_freq_values


if __name__ == '__main__':

    ort_sess = ort.InferenceSession('model.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=800, height=700, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # DATABASE_URL = "http://100.24.240.125:8000/posts"  # server url
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
    i = 0

    delay = 1

    total_result = []

    while True:
        
        ret, frame = cap.read()

        # if i == 11: break

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        df = pd.read_csv('box_coord.csv')

        result = run_sample(df, frame, ort_sess)
        total_result.append(result.tolist())

        if i % 7 == 0 and i > 0:
            new_total = np.array(total_result)
            x = max_frequent_values(new_total)
            result_dict = convert_result_todict(x)
            print(result_dict)
            # response = requests.post(url=DATABASE_URL, json=result_dict) #, headers=headers)
            # print(response.status_code)
            total_result = []

        # print(result_dict)
        print("-"*20)



        i += 1

        time.sleep(delay)




           
