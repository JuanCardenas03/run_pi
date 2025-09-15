import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time

# --- Configuración de pines GPIO ---
PIN_A = 17   # switch A
PIN_B = 27   # switch B
PIN_LED = 22 # salida LED

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_A, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIN_B, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIN_LED, GPIO.OUT)

# --- Cargar modelo TFLite ---
interpreter = tflite.Interpreter(model_path="and_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Presiona Ctrl+C para salir.\n")

try:
    while True:
        # Leer entradas de los switches
        a = GPIO.input(PIN_A)
        b = GPIO.input(PIN_B)

        # Preparar entrada para el modelo
        entrada = np.array([[a, b]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], entrada)
        interpreter.invoke()
        salida = interpreter.get_tensor(output_details[0]['index'])

        prediccion = int(np.round(salida[0][0]))
        print(f"Entradas: {a}, {b} -> Predicción: {prediccion} (valor real: {salida[0][0]:.4f})")

        # Encender/apagar LED según la predicción
        GPIO.output(PIN_LED, GPIO.HIGH if prediccion == 1 else GPIO.LOW)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nSaliendo...")
finally:
    GPIO.cleanup()