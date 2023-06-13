import pandas as pd
from pytube import YouTube
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context


# Function to download a YouTube video using the video ID
def download_video_by_id(video_id, output_path, video_label):
    try:
        # Construct the YouTube video URL using the video ID
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Create a YouTube object with the video URL
        youtube = YouTube(video_url)
        # Get the first stream (highest resolution) with the "video" and "progressive" filters
        stream = youtube.streams.filter(progressive=True, file_extension='mp4').first()

        # Download the video to the specified output path

        stream.download(output_path)
        print("Download completed successfully!")
    except Exception as e:
        print("An error occurred while downloading the video:", str(e))


def Download(video_id, video_label):
    link = f"https://www.youtube.com/watch?v={video_id}"
    print(link)
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()

    except:
        print("An error has occurred")
    print("Download is completed successfully")


data = pd.read_csv(r'k400test.csv')
df = pd.DataFrame(data)
videos = df['youtube_id']
labels = df['label']


i = 0
parent_directory = "kinetic_dataset"
for video_id in videos:
    directory_path = os.path.join(parent_directory, labels[i])

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(i)
    download_video_by_id(video_id, directory_path, labels[i])
    i += 1
