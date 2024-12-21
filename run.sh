#!/bin/bash
folder_list=("bmx-bumps")
stroke_num=(16)
for video_file in "${folder_list[@]}"; do
    video_file="${video_file}.mp4"
    echo "$video_file"
    cd InSPyReNet/
    python run/Inference.py --config configs/InSPyReNet_SwinB.yaml --source /target_videos/$video_file --type map --gpu --verbose
    cd ..
    for num_strokes in "${stroke_num[@]}"; do
        echo "$num_strokes"
        python run_object_sketching.py --target_file "$video_file" --num_iter 2001 --num_strokes $num_strokes --frame_cut 220
    done
done
