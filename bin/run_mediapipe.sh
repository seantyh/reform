#! /bin/bash

if [ $# -eq 2 ]; then
    JOBIDX=$1
    JOBS=$2
else
    JOBIDX=0
    JOBS=1
fi;

shopt -s globstar
MP_OUT_DIR="../data/mp"

for video_path in ~/legvid/legislative/**/*.mp4; do
    VIDEO_ID=${video_path##*/}
    VIDEO_ID=${VIDEO_ID%.mp4}
    VIDEO_NUM=${VIDEO_ID#*-}
    
    if [ $(($VIDEO_NUM % $JOBS)) -ne $JOBIDX ]; then
        continue
    fi
    
    if ls $MP_OUT_DIR/*${VIDEO_ID}* 1> /dev/null 2>&1; then
        echo "${VIDEO_ID} exists"
    else
        echo "Processing ${VIDEO_ID}"
        python mp_legvid.py ${video_path} --out_dir=${MP_OUT_DIR}
    fi
done
