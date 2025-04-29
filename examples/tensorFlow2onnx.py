import tf2onnx
import tensorflow as tf

print("Podaj sciezke do modelu AI: ", end="")
fileName = input()

kerasModel = tf.keras.models.load_model(fileName)
kerasModel.output_names = ['outputLayer']

spec = (tf.TensorSpec(kerasModel.inputs[0].shape, tf.float32, name="input"),)

onnxModel = tf2onnx.convert.from_keras(kerasModel, output_path="model.onnx", input_signature=spec)
