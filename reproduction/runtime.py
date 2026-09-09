import json
from pathlib import Path
from transformers import TrainerCallback
from reproduction.prepare import write_json


class ArtifactCallback(TrainerCallback):
    def __init__(self, directory):
        self.directory = Path(directory)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': state.global_step, 'epoch': state.epoch, **(logs or {})}) + '\n')


def training_record(trainer, artifact_dir, final_path, parent):
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(Path(artifact_dir) / 'trainer_state.json'))
        write_json(Path(artifact_dir) / 'training.json', {
            'parent_model': parent, 'final_checkpoint': str(Path(final_path).resolve()),
            'final_step': trainer.state.global_step,
            'selected_step': int(Path(trainer.state.best_model_checkpoint).name.split('-')[-1]) if trainer.state.best_model_checkpoint else trainer.state.global_step,
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_metric': trainer.state.best_metric,
            'selection': 'SFT: minimum validation loss; RL: maximum validation ranking reward',
            'training_args': trainer.args.to_dict()})
