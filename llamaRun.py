import numpy as np
from huggingface_hub import InferenceClient
import base64
import cv2
import requests

# encode the camera image
def encode_image_camera(image):
     _, buffer = cv2.imencode('.jpg', image)  
     return base64.b64encode(buffer).decode('utf-8')

# Settings of the user token
client = InferenceClient(api_key="seu_token_hf_aqui")

# Settings of the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

print("Aperte 'espaço' para capturar um frame")

while True:
	ret, image = cap.read()
	cv2.imshow('Leitura', image)
	cv2.waitKey(1)
	base64_image = encode_image_camera(image)

	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": f"detect the objects on the image and return them in a list separated by ', '. (Example: 'object1, object2, object3, object4). (Example: 'object1, object2, object3). (Example: 'object1, object2, object3, object4, object5). Be strict to the format of the output, and return nothing else"
				},
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{base64_image}"
					}
				}
			]
		}
	]

	completion = client.chat.completions.create(
		model="meta-llama/Llama-3.2-11B-Vision-Instruct",
		#model="Qwen/Qwen2-VL-7B-Instruct",
		messages=messages,
		max_tokens=500
	)

	print(f"{completion.choices[0].message.content} \n\n")



	cv2.waitKey(0)