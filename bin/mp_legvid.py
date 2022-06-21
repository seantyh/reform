import numpy as np
import cv2
import mediapipe as mp
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from collections import namedtuple
from google.protobuf.json_format import MessageToDict

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mask_people(image, mask):
    condition = np.stack((mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (192,192,192)
    annotated_image = np.where(~condition, image, bg_image)
    return annotated_image

def protobuf_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [protobuf_to_dict(x) for x in obj]
    else:
        return MessageToDict(obj)

def write_to_files(results_list, out_path_templ, video_info):
    timestamp = datetime.now().strftime("v%y%m%d.%H%M%S")
    fps = video_info["fps"]
    every_n_frame = video_info["every_n_frame"]
    video_path = str(Path(video_info["video_path"]).absolute())    
    
    metainfo = {
        "video_path": video_path,
        "timestamp": timestamp,
        "fps": fps,
        "nframes": video_info["n_frames"],
        "every_n_frame": every_n_frame,
        "sample_rate": 1/(every_n_frame/fps),
        "size": (video_info["width"], video_info["height"])        
    }

    for mptype_x in ("faces", "meshes", "poses"):
        with open(out_path_templ.format(mptype=mptype_x), 
                    "w", encoding="UTF-8") as fout:
            ## dict union is only added after Python 3.9
            mpdict_list = [x[mptype_x] for x in results_list]
            out_dict = {**metainfo, mptype_x: mpdict_list}
            json.dump(out_dict, fout, ensure_ascii=False)

def main(video_path, out_dir=None, debug=False):   
    if out_dir:
        out_path_templ = str(Path(out_dir) / Path(video_path).stem) + ".{mptype}.mp.json"
    else:
        out_path_templ = video_path.replace(".mp4", ".{mptype}.mp.json")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    pbar = tqdm(total=n_frames)
    static_image_mode = False
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_mesh.FaceMesh(static_image_mode=static_image_mode, 
                                max_num_faces=3, refine_landmarks=True,
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=static_image_mode, 
                        model_complexity=1, 
                        enable_segmentation=True,
                        min_detection_confidence=0.5)    

    results_list = []

    while cap.isOpened():    
            
        success, image = cap.read()            
        if not success:            
            # If loading a video, use 'break' instead of 'continue'.
            break
        pbar.update(1)        
        
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

        sample_rate = 5  # in Hz
        every_n_frame = round(fps/sample_rate)        
        if frame_idx % every_n_frame > 0:   
            continue    
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        
        if static_image_mode:        
            ## make an image copy for visualization
            raw_image = image.copy()
        
        ## This is always true for multi-pose detection
        image.flags.writeable = True
        
        ## mediapipe processing
        face_results = face_detection.process(image)
        mesh_results = face_mesh.process(image)
        
        pose_results_list = []
        if face_results.detections:
            n_face = len(face_results.detections)
        else:
            n_face = 0
            
        for _ in range(n_face):
            pose_results = pose.process(image)
            pose_results_list.append(pose_results)
            if pose_results.segmentation_mask is not None:
                image = mask_people(image, pose_results.segmentation_mask)
            
        # Draw the face detection annotations on the image.
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame_info = {
            "frame": frame_idx, 
            "offset": frame_idx/fps 
        }                

        faces_dict = protobuf_to_dict(face_results.detections)
        mesh_dict = protobuf_to_dict(mesh_results.multi_face_landmarks)
        poses_dict_list = [protobuf_to_dict(x.pose_landmarks)
                           for x in pose_results_list]
                
        results_list.append({
            "faces": {**frame_info, "results": faces_dict},
            "meshes": {**frame_info, "results": mesh_dict},
            "poses": {**frame_info, "results": poses_dict_list}})

        if debug and len(results_list) > 10:
            break
        
    cap.release()
    face_mesh.close()
    face_detection.close()
    pose.close()
    pbar.close()

    write_to_files(results_list, out_path_templ, dict(
        video_path = video_path, 
        width=width, height=height, n_frames=n_frames,
        fps=fps, every_n_frame=every_n_frame        
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to mediapipe")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
