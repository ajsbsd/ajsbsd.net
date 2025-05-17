ffmpeg \
-i background.jpg \
-framerate 20 \
-i gif_frames/frame-%03d.png \
-c:v libx264 \
-pix_fmt yuv420p \
-crf 23 \
-preset fast \
output_with_background.mp4
