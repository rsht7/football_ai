from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import math

def main():
    print('Hello World')

    video_path = 'input_videos/08fd33_4.mp4'

    # Reading video
    video_frames = read_video(video_path)

    # Initializing the tracker
    tracker = Tracker('models/bestx.pt', video_path)
    tracks = tracker.get_object_tracks(video_frames,
                              read_from_stub=True,
                              stub_path='stubs/track_stubs.pkl')
    
    #Save cropped img of player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     #saving cropped img
    #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
    #     break
    
    

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #Assigning player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])


    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assigning ball to player
    team_ball_control = []
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        # if 1 not in tracks['ball'][frame_num]:
        #     continue  # Skip if the ball is not detected in this frame
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player!= -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # for frame_num, player_track in enumerate(tracks['players']):
    #     ball_data = tracks['ball'][frame_num]
    
    #     # Check if ball data exists and is valid
    #     if not ball_data or 'bbox' not in ball_data[1]:
    #         continue  # Skip this frame if ball data is missing
    
    #     ball_bbox = ball_data[1]['bbox']

    #     # Check if bbox contains NaN values
    #     if any(math.isnan(coord) for coord in ball_bbox):
    #         continue  # Skip this frame

    #     assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

    #     if assigned_player != -1:
    #         tracks['players'][frame_num][assigned_player]['has_ball'] = True


    #Drawing o/p with circles and obj_tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)


    # Saving video
    save_video(output_video_frames, 'output_videos/output_videov5x.avi')



if __name__ == '__main__':
    main()    
    