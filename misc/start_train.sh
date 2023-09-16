
cd /project/train/src_repo/tub_detect
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/tub_detect

echo 'Reset env...'
rm -rf /project/train/models
rm -rf /project/train/tensorboard
rm -f /project/train/log/log.txt

echo 'Start data_analyzer...'
python misc/data_analyzer.py

echo 'Start data_config...'
python tub_det/data_config.py

echo 'Start cvmart_train...'
python tub_det/cvmart_train.py

echo 'Completed!'
