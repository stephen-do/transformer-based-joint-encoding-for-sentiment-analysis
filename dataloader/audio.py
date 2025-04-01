import os
import argparse
import cv2
import time
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract(dataset):
    dataset = dataset.upper()
    input_directory_path = f'dataset/{dataset}/Raw'
    output_directory_path = f'dataset/{dataset}/wav'
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    for folder in tqdm(os.listdir(input_directory_path)):

        input_folder_path = os.path.join(input_directory_path, folder)
        output_folder_path = os.path.join(output_directory_path, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        try:
            for file_name in os.listdir(input_folder_path):
                if file_name.split(".")[-1] != "mp4" or file_name.split(".")[1] != "mp4":
                    continue
                input_file_path = os.path.join(input_folder_path, file_name)
                output_file_path = os.path.join(output_folder_path, file_name)
                if "-edited.mp4" in output_file_path:
                    output_file_path = output_file_path.replace("-edited.mp4", ".mp4")
                output_file_path = output_file_path.replace(".mp4", ".wav")
                # Skip if the video file is already edited
                if os.path.exists(input_file_path.replace(".mp4", "-edited.mp4")):
                    continue
                # Skip if the audio file already exists
                if os.path.exists(output_file_path):
                    continue
                try:
                    # Load the video file
                    video = VideoFileClip(input_file_path)

                    # Extract the audio from the video
                    audio = video.audio

                    # Set the desired sampling rate
                    desired_sampling_rate = 16000  # Replace this value with your desired sampling rate

                    # Resample the audio to the desired sampling rate
                    # resampled_audio = audio.set_fps(desired_sampling_rate)

                    # Save the extracted and resampled audio to a WAV file
                    audio.write_audiofile(output_file_path, codec='pcm_s16le', logger=None, fps=desired_sampling_rate)
                except Exception as e:
                    # one edited video lost audio, so we extract audio from the original video instead
                    # there are also 6 videos that are corrupted, so we skip them
                    print(input_file_path, e)
                    if "-edited.mp4" in input_file_path:
                        input_file_path = input_file_path.replace("-edited.mp4", ".mp4")
                        video = VideoFileClip(input_file_path)
                        audio = video.audio
                        desired_sampling_rate = 16000
                        resampled_audio = audio.set_fps(desired_sampling_rate)
                        resampled_audio.write_audiofile(output_file_path, codec='pcm_s16le', verbose=False, logger=None)
        except:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CMU-MOSI', help='dataset name')
    args = parser.parse_args()
    extract(args.dataset)
