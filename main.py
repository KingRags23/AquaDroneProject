
import numpy as np

from google.colab import files

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    'Aerials/train',
    ['trees']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'Aerials/validate'
    ['trees']
)

spec = model_spec.get('efficientdet_lite1')
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='deforest_model.tflite')
model.evaluate_tflite('deforest_model.tflite', val_data)

files.download('deforest_model.tflite')

populator_dst = metadata.MetadataPopulator.with_model_file('deforest_model_edgetpu.tflite')

with open('deforest_model.tflite', 'rb') as f:
  populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()
updated_model_buf = populator_dst.get_model_buffer()

files.download('deforest_model_edgetpu.tflite')
