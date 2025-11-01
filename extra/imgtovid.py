# Install MoviePy if not already installed
# pip install moviepy

from moviepy.editor import ImageClip

# Parameters
image_path = r"C:\Users\ASUS PC\Desktop\baby.jpg"
output_video = r"C:\Users\ASUS PC\Desktop\baby.mp4"

duration = 0.5  # seconds

# Create an ImageClip from the JPEG
clip = ImageClip(image_path, duration=duration)

# Set FPS to get 12 frames in 0.5 seconds
clip = clip.set_fps(12 / duration)  # 12 frames / 0.5s = 24 FPS

# Write the video
clip.write_videofile(output_video, codec="libx264")

print(f"Video saved as {output_video}")
