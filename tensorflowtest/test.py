from tensorflow.keras.applications import imagenet_utils

image = 'plane.jpg'

img = load_img(image, target_size=(224, 224))
img_array = img_to_array(img)
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

mobile = tf.keras.applications.mobilenet.MobileNet()
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)

print(results[0][0][1], results[0][0][2] * 100)

# test
