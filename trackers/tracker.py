from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
# sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_center_of_bbox, get_bbox_width 
import cv2
import numpy as np
import pandas as pd

# ,frame_rate=self.fps, track_thresh=0.25, match_thresh=0.8, track_buffer=50
class Tracker:
    def __init__(self, model_path, video_path):

        self.model = YOLO(model_path)
        # self.pitch_detection_model = YOLO(pitch_detection_model_path)

         # Get FPS from video
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # self.frame_interval = max(int(self.fps / 15), 1)

        self.tracker = sv.ByteTrack()
        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])


        # interpolating missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        

    def detect_frames(self, frames):

        batch_size = 20
        detections = []
        # for i in range(0, len(frames), batch_size):  
        #     detections.append(self.model.predict(frames[i], conf=0.05))
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks =  pickle.load(f)
            return tracks    

        detections = self.detect_frames(frames)

        tracks={
            'players':[],
            'referees':[],
            'ball':[],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Converting the detections to supervision supported format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Converting Goalkeeper to Player
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) 
            # detection_with_tracks = self.tracker.update_with_detections(
            #                             detection_supervision,
                                        
                                        
            #                             track_buffer=int(self.fps * 2)
                                        
            #                         )
 
            

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox =frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}   

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center,y2), axes = (int(width), int(0.35*width)), 
                    angle =0.0, startAngle=-45, endAngle=235, 
                    color= color, thickness = 2, lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rectangle = x_center - rectangle_width//2
        x2_rectangle = x_center + rectangle_width//2
        y1_rectangle = y2 - rectangle_height//2 + 15
        y2_rectangle = y2 + rectangle_height//2 + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rectangle), int(y1_rectangle)),
                           (int(x2_rectangle), int(y2_rectangle)),
                           color, cv2.FILLED)
            
            x1_text = x1_rectangle + 12
            if track_id > 99 : 
                x1_text -=10

            cv2.putText(frame, f'{track_id}',
                        (int(x1_text), int(y2_rectangle - 2)) ,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),2)
               
            
        
        return frame


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    

    def interpolate_ball_positions(self, ball_positions):
        # Extract ball bounding boxes, replacing missing detections with NaNs
        ball_positions_list = [x.get(1, {}).get('bbox', [np.nan] * 4) for x in ball_positions]

        # Convert to DataFrame
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Mark original detections to avoid overwriting them
        detected_mask = df_ball_positions.notna().all(axis=1)

        # Interpolate only missing values
        df_ball_positions.interpolate(method='linear', limit_direction='both', inplace=True)

        # Apply rolling average smoothing to only the interpolated values
        smoothed_df = df_ball_positions.rolling(window=3, min_periods=1).mean()

        # Restore original detections (overwrite interpolated values with actual detections)
        df_ball_positions.loc[detected_mask] = smoothed_df.loc[detected_mask]

        # Convert back to the original format
        ball_positions = [{1: {'bbox': x.tolist()}} for x in df_ball_positions.to_numpy()]

        return ball_positions


    
    def draw_ball_control(self, frame, frame_num, team_ball_control):

        overlay = frame.copy()
        cv2.rectangle(overlay, (650, 0), (1300, 70), (0,0,0), -1)
        alpha_value = 0.4
        cv2.addWeighted(overlay, alpha_value, frame, 1 - alpha_value, 0, frame)

        ball_control_till_current_frame = team_ball_control[:frame_num+1]

        team_A_frames = ball_control_till_current_frame[ball_control_till_current_frame == 1].shape[0]
        team_B_frames = ball_control_till_current_frame[ball_control_till_current_frame == 2].shape[0]
        team_A = team_A_frames/(team_A_frames + team_B_frames) * 100
        team_B = team_B_frames/(team_A_frames + team_B_frames) * 100

        cv2.putText(frame, f'Team A: {team_A:.2f}%', (675, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 4)
        cv2.putText(frame, f'Team B: {team_B:.2f}%', (1000, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 4)

        return frame
    

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            #Drawing players

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

                

            #Drawing referees

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255))    

            #Drawing ball

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))
            
            #now adding the team ball control percentage display
            frame = self.draw_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames