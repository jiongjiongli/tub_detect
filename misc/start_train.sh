
cd /project/train/src_repo/tub_detect
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/tub_detect

echo 'Reset env...'
rm -rf /project/train/models/*
mkdir -p /project/train/models
rm -rf /project/train/tensorboard/*
mkdir -p /project/train/tensorboard
rm -rf /project/train/log/*
mkdir -p /project/train/log
rm -rf /project/train/result-graphs/*
mkdir -p /project/train/result-graphs

echo 'Start data_analyzer...'
python misc/data_analyzer.py

echo 'Start data_config...'
python tub_det/data_config.py

echo 'Start cvmart_train...'
python tub_det/cvmart_train.py

echo 'Completed!'
