'''
Summary:
  Exports everything the in-browser explainer (web/) needs from the trained
  dogs-vs-cats model:

    1. A multi-output "activation" sub-model (Chapter 2) that returns three
       early-layer feature maps — edges, textures, patterns — in one forward
       pass. Saved as .h5 here; converted to TF.js by the companion shell step.
    2. Python-side predictions on the src/test/ images (Chapter 3/5 ground
       truth) so the browser TF.js model can be verified against the real model.
    3. The real epoch-by-epoch training history (Chapter 4), parsed from the
       TensorBoard event files in logs/fit/.

  Run inside the .venv-tfjs environment (TF 2.15 / Keras 2.15), which matches
  the stack the model was trained on.

Outputs:
  model/activation_model.h5         (intermediate -> converted to docs/model-activations/)
  docs/data/predictions.json
  docs/data/training_history.json

The TF.js model conversions are run separately (see README), e.g.:
  tensorflowjs_converter --input_format=keras model/cats_and_dogs_mobilenet.h5 docs/model
  tensorflowjs_converter --input_format=keras model/activation_model.h5 docs/model-activations
'''

import os
import glob
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
WEB_DATA_DIR = os.path.join(PROJECT_DIR, 'docs', 'data')
TEST_DIR = os.path.join(PROJECT_DIR, 'src', 'test')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs', 'fit')

MODEL_PATH = os.path.join(MODEL_DIR, 'cats_and_dogs_mobilenet.h5')
ACTIVATION_MODEL_PATH = os.path.join(MODEL_DIR, 'activation_model.h5')

IMG_SIZE = 150

# Early -> deep: edges, textures, larger patterns. Keeping the deepest at
# block_5 bounds the activation sub-model's weight size for the browser.
ACTIVATION_LAYERS = [
    ('Conv1_relu', 'edges'),
    ('block_2_expand_relu', 'textures'),
    ('block_5_expand_relu', 'patterns'),
]


def build_activation_model(full_model):
    '''Multi-output sub-model: image -> [edges, textures, patterns] feature maps.'''
    base = full_model.get_layer(index=0)  # the nested MobileNetV2 Functional model
    outputs = [base.get_layer(name).output for name, _ in ACTIVATION_LAYERS]
    submodel = Model(inputs=base.input, outputs=outputs, name='activations')
    submodel.save(ACTIVATION_MODEL_PATH)
    print(f'Saved activation sub-model -> {ACTIVATION_MODEL_PATH}')
    for (name, label), out in zip(ACTIVATION_LAYERS, submodel.outputs):
        print(f'  {label:10s} {name:22s} shape={tuple(out.shape)}')
    return submodel


def load_image(path):
    '''Load + resize + rescale exactly as training/serving did: [0,1], 150x150.'''
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    arr = img.numpy() / 255.0
    return arr.astype('float32')


def export_predictions(full_model):
    '''Run the real model on the test images so the browser can be verified.'''
    paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))
    preds = []
    for p in paths:
        arr = load_image(p)
        score = float(full_model.predict(arr[None, ...], verbose=0)[0][0])
        is_dog = score >= 0.5
        preds.append({
            'file': os.path.basename(p),
            'score': round(score, 6),          # raw sigmoid: 1.0 = dog, 0.0 = cat
            'label': 'Dog' if is_dog else 'Cat',
            'confidence': round(score if is_dog else 1 - score, 4),
        })
        print(f'  {os.path.basename(p):12s} score={score:.4f} -> {preds[-1]["label"]}')
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    out = os.path.join(WEB_DATA_DIR, 'predictions.json')
    with open(out, 'w') as f:
        json.dump({'img_size': IMG_SIZE, 'convention': '1.0=Dog, 0.0=Cat',
                   'predictions': preds}, f, indent=2)
    print(f'Saved predictions -> {out}')


def _scalar_from_event(value):
    '''Keras TensorBoard writes epoch scalars as rank-0 tensors in TF2.

    The log also carries non-scalar summaries (run config, strings); accept
    only size-1 numeric tensors and skip everything else.'''
    if value.HasField('simple_value'):
        return float(value.simple_value)
    if value.HasField('tensor'):
        try:
            arr = tf.make_ndarray(value.tensor)
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                return float(arr.reshape(-1)[0])
        except Exception:
            return None
    return None


def _read_run(run_dir):
    '''Return {tag: {step: value}} for a single run dir (train/ + validation/).'''
    series = {}
    for sub in ('train', 'validation'):
        for ev in glob.glob(os.path.join(run_dir, sub, 'events.out.tfevents.*')):
            for e in tf.compat.v1.train.summary_iterator(ev):
                for v in e.summary.value:
                    val = _scalar_from_event(v)
                    if val is None:
                        continue
                    tag = ('val_' + v.tag) if sub == 'validation' else v.tag
                    series.setdefault(tag, {})[e.step] = val
    return series


def _ordered(series, tag):
    '''Values for a tag ordered by step.'''
    d = series.get(tag, {})
    return [d[s] for s in sorted(d)]


def export_training_history():
    '''Parse TB logs into a per-epoch history for the Chapter 4 narrative.

    Phase 1 = the non-finetune run; Phase 2 = the latest *_finetune run.'''
    runs = sorted(glob.glob(os.path.join(LOGS_DIR, '*')))
    phase1 = next((r for r in runs if not r.endswith('_finetune')), None)
    finetunes = [r for r in runs if r.endswith('_finetune')]
    phase2 = finetunes[-1] if finetunes else None

    def phase_payload(run_dir):
        if not run_dir:
            return None
        s = _read_run(run_dir)
        # Keras logs tags as 'epoch_accuracy' / 'epoch_loss'.
        return {
            'run': os.path.basename(run_dir),
            'accuracy': _ordered(s, 'epoch_accuracy'),
            'val_accuracy': _ordered(s, 'val_epoch_accuracy'),
            'loss': _ordered(s, 'epoch_loss'),
            'val_loss': _ordered(s, 'val_epoch_loss'),
        }

    history = {'phase1': phase_payload(phase1), 'phase2': phase_payload(phase2)}
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    out = os.path.join(WEB_DATA_DIR, 'training_history.json')
    with open(out, 'w') as f:
        json.dump(history, f, indent=2)
    for name, p in history.items():
        if p:
            print(f'  {name}: {len(p["val_accuracy"])} epochs, '
                  f'best val_acc={max(p["val_accuracy"]) if p["val_accuracy"] else float("nan"):.4f}')
    print(f'Saved training history -> {out}')


if __name__ == '__main__':
    print('Loading full model...')
    model = load_model(MODEL_PATH)
    print('\n== Activation sub-model ==')
    build_activation_model(model)
    print('\n== Predictions on test images ==')
    export_predictions(model)
    print('\n== Training history ==')
    export_training_history()
    print('\nDone.')
