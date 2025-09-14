import numpy as np
import tflite_runtime.interpreter as tflite

# --- Cargar el modelo TFLite ---
interpreter = tflite.Interpreter(model_path="and_model.tflite")
interpreter.allocate_tensors()

# --- Obtener detalles de entrada y salida ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Datos de entrada (las combinaciones de la compuerta AND) ---
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

print("\nResultados de la compuerta AND con TFLite:\n")

# --- Ejecutar inferencia ---
for entrada in X:
    # Ajustar forma de entrada según lo que espera el modelo
    entrada = np.expand_dims(entrada, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], entrada)
    interpreter.invoke()
    salida = interpreter.get_tensor(output_details[0]['index'])

    # Redondeamos a 0 o 1
    prediccion = int(np.round(salida[0][0]))
    print(f"{entrada[0]} -> {prediccion}  (valor real: {salida[0][0]:.4f})")