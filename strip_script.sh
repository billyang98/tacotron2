for j in {3..15}
do
  mkdir david_full_audio_dir/audio_$j/orig
  mv david_full_audio_dir/audio_$j/*.wav david_full_audio_dir/audio_$j/orig/
  for i in {1..9}
  do
    ffmpeg -i david_full_audio_dir/audio_$j/orig/sound-0$i.wav -map 0 -map_metadata -1  -c  copy david_full_audio_dir/audio_$j/sound-0$i.wav
  done
  for i in {10..680}
  do
    ffmpeg -i david_full_audio_dir/audio_$j/orig/sound-$i.wav -map 0 -map_metadata -1  -c  copy david_full_audio_dir/audio_$j/sound-$i.wav
  done
done
