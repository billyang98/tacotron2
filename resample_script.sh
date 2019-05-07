for i in {1..9}
do
  ffmpeg -i sound-0$1-0$i.wav -ar 22050 resampled_audio/sound-0$1-0$i.wav
  ffmpeg -i resampled_audio/sound-0$1-0$i.wav -map 0 -map_metadata -1  -c  copy stripped_audio/sound-0$1-0$i.wav
done
