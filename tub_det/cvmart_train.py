import logging
from pathlib import Path
import shutil

import distutils.version
from torch.utils.tensorboard import SummaryWriter
from ultralytics.yolo.utils import USER_CONFIG_DIR, LOGGER as logger
from ultralytics.yolo.utils.callbacks.tensorboard import callbacks as tb_callbacks
from ultralytics import YOLO


class TensorboardLogger:
    def __init__(self):
        self.writer = None

    def _log_scalars(self, scalars, step=0):
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    def on_pretrain_routine_start(self, trainer):
            tensorboard_log_dir_path = Path('/project/train/tensorboard')
            tensorboard_log_dir_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_log_dir_path.as_posix())

    def on_batch_end(self, trainer):
        self._log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)


    def on_fit_epoch_end(self, trainer):
        self._log_scalars(trainer.metrics, trainer.epoch + 1)

        # Copy images such as PR curve.
        model_save_dir_path = Path('/project/train/models')
        result_graphs_dir_path = Path('/project/train/result-graphs')
        file_paths = model_save_dir_path.rglob('*.*')

        for file_path in file_paths:
            if file_path.suffix in ['.png', '.jpg']:
                dest_file_path = result_graphs_dir_path / file_path.relative_to(model_save_dir_path)
                dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file_path, dest_file_path)


def main():
    repo_dir_path = Path('/project/train/src_repo')
    # model_file_path = repo_dir_path / 'yolov8n.pt'
    data_root_path = Path(r'/home/data')
    dataset_config_file_path = data_root_path / 'custom_dataset.yaml'
    model_save_dir_path = Path('/project/train/models')
    model_file_path = model_save_dir_path  / 'train/weights' / 'last.pt'
    result_graphs_dir_path = Path('/project/train/result-graphs')
    font_file_names = ['Arial.ttf']
    log_file_path = Path('/project/train/log/log.txt')

    file_handler = logging.FileHandler(log_file_path.as_posix(), mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    tb_logger = TensorboardLogger()
    tb_callbacks['on_pretrain_routine_start'] = tb_logger.on_pretrain_routine_start
    tb_callbacks['on_fit_epoch_end'] = tb_logger.on_fit_epoch_end
    tb_callbacks['on_batch_end'] = tb_logger.on_batch_end

    result_graphs_dir_path.mkdir(parents=True, exist_ok=True)

    for font_file_name in font_file_names:
        font_file_path = repo_dir_path / font_file_name
        dest_file_path = USER_CONFIG_DIR / font_file_name
        shutil.copyfile(font_file_path, dest_file_path)

    model = YOLO(model_file_path.as_posix())

    model.train(
        data=dataset_config_file_path.as_posix(),
        batch=16,
        seed=7,
        resume=True,
        epochs=150,
        project=model_save_dir_path.as_posix())


if __name__ == '__main__':
    main()
