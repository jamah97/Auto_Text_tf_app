import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

	



def main():
# step 1 allow the end-user to upload dataset of choice
	st.subheader("Auto Text Classifier App")
	st.write("Application created by Hassan Jama")
	st.write("About Section: In this application, you will be able to upload any CSV file that you may want to use to perform text classification. After uploading your CSV file, you'll have the opportunity to preview the data. Furthermore, there will also be a list of pre-train models that you will be able to select from. All of the models are provided through the TensorFlow hub. This is also known as transfer learning. 3 of the pre-train models are type Neural-Net Language Model (NNLM) with dimensions of 20, 50, and 128. The other pre-train model is wiki-words-250- with- normalization, which is token-based text embedding trained on Wikipedia corpus.")
	st.write("For better performance, please provide a CSV file that is under 3MB and in the English language.")
	data = st.file_uploader("Upload Dataset",type=["csv"], )
	if data is not None:
		df = pd.read_csv(data)
		if st.checkbox("Preview Dataset"):
			try:
				st.dataframe(df.head(int(st.text_input("Select Number of Rows To View"))))
			except ValueError:
				pass
		algorithm = ["nnlm-en-dim20", "nnlm-en-dim50", "nnlm-en-dim128", "Wiki-words-250-with-normalization/2"]
		choice = st.selectbox("Select Algorithm",algorithm)
		testsize = st.slider('Testing size: Select percent of data that will be used as testing data.', 0, 50)


		if choice == "nnlm-en-dim20":
			iv = st.selectbox("Select predictor variable", df.columns.to_list())
			dv = st.selectbox("Select independent variable", df.columns.to_list())
			iv1 = df[iv]
			dv1 = df[dv].values
			dv2 = pd.DataFrame(data=dv1)
			iv2 = pd.DataFrame(data=iv1)

			loss_function = ["binary_crossentropy", "categorical_crossentropy"]
			loss = st.selectbox("Select loss function",loss_function)

			if st.button("Model Performance"):
				train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
				endcoder = preprocessing.LabelEncoder()
				train_y = endcoder.fit_transform(train_y)
				test_y = endcoder.fit_transform(test_y)
				num_c = np.max(train_y) + 1
				st.write("number of classes", num_c)
				train_y = utils.to_categorical(train_y, num_c)
				test_y = utils.to_categorical(test_y, num_c)
				#train_x = tf.convert_to_tensor(train_x)
				#test_x = tf.convert_to_tensor(test_x)
				hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20], input_shape=[], dtype=tf.string)
				model = tf.keras.models.Sequential([ hub_layer,
													tf.keras.layers.Dense(128, activation='relu'),
													tf.keras.layers.Dropout(.2),
													tf.keras.layers.Dense(num_c, activation='sigmoid')])

				model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

				history = model.fit(train_x,train_y, epochs=50,batch_size=32,validation_data=(test_x, test_y),verbose=0)

				score = model.evaluate(test_x, test_y, verbose = 0)
				st.write('Test loss:', score[0])
				st.write('Test accuracy:', score[1])




# step 2 allow end-user to select pre-train text algorithm
		if choice == "nnlm-en-dim50":
			iv = st.selectbox("Select predictor variable", df.columns.to_list())
			dv = st.selectbox("Select independent variable", df.columns.to_list())
			iv1 = df[iv]
			dv1 = df[dv].values
			dv2 = pd.DataFrame(data=dv1)
			iv2 = pd.DataFrame(data=iv1)

			loss_function = ["binary_crossentropy", "categorical_crossentropy"]
			loss = st.selectbox("Select loss function",loss_function)

			if st.button("Model Performance"):
				train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
				endcoder = preprocessing.LabelEncoder()
				train_y = endcoder.fit_transform(train_y)
				test_y = endcoder.fit_transform(test_y)
				num_c = np.max(train_y) + 1
				st.write("number of classes", num_c)
				train_y = utils.to_categorical(train_y, num_c)
				test_y = utils.to_categorical(test_y, num_c)
				#train_x = tf.convert_to_tensor(train_x)
				#test_x = tf.convert_to_tensor(test_x)
				hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1", output_shape=[50], input_shape=[], dtype=tf.string)
				model = tf.keras.models.Sequential([ hub_layer,
													tf.keras.layers.Dense(128, activation='relu'),
													tf.keras.layers.Dropout(.2),
													tf.keras.layers.Dense(num_c, activation='sigmoid')])


				model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

				history = model.fit(train_x,train_y, epochs=50,batch_size=32,validation_data=(test_x, test_y),verbose=0)

				score = model.evaluate(test_x, test_y, verbose = 0)
				st.write('Test loss:', score[0])
				st.write('Test accuracy:', score[1])

		if choice == "nnlm-en-dim128":
			iv = st.selectbox("Select predictor variable", df.columns.to_list())
			dv = st.selectbox("Select independent variable", df.columns.to_list())
			iv1 = df[iv]
			dv1 = df[dv].values
			dv2 = pd.DataFrame(data=dv1)
			iv2 = pd.DataFrame(data=iv1)

			loss_function = ["binary_crossentropy", "categorical_crossentropy"]
			loss = st.selectbox("Select loss function",loss_function)

			if st.button("Model Performance"):
				train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
				endcoder = preprocessing.LabelEncoder()
				train_y = endcoder.fit_transform(train_y)
				test_y = endcoder.fit_transform(test_y)
				num_c = np.max(train_y) + 1
				st.write("number of classes", num_c)
				train_y = utils.to_categorical(train_y, num_c)
				test_y = utils.to_categorical(test_y, num_c)
				#train_x = tf.convert_to_tensor(train_x)
				#test_x = tf.convert_to_tensor(test_x)
				hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1", output_shape=[128], input_shape=[], dtype=tf.string)
				model = tf.keras.models.Sequential([ hub_layer,
													tf.keras.layers.Dense(128, activation='relu'),
													tf.keras.layers.Dropout(.2),
													tf.keras.layers.Dense(num_c, activation='sigmoid')])



				model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

				history = model.fit(train_x,train_y, epochs=50, batch_size=32,validation_data=(test_x, test_y),verbose=0)

				score = model.evaluate(test_x, test_y, verbose = 0)
				st.write('Test loss:', score[0])
				st.write('Test accuracy:', score[1])



		if choice == "Wiki-words-250-with-normalization/2":
			iv = st.selectbox("Select predictor variable", df.columns.to_list())
			dv = st.selectbox("Select independent variable", df.columns.to_list())
			iv1 = df[iv]
			dv1 = df[dv].values
			dv2 = pd.DataFrame(data=dv1)
			iv2 = pd.DataFrame(data=iv1)

			loss_function = ["binary_crossentropy", "categorical_crossentropy"]
			loss = st.selectbox("Select loss function",loss_function)

			if st.button("Model Performance"):
				train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
				endcoder = preprocessing.LabelEncoder()
				train_y = endcoder.fit_transform(train_y)
				test_y = endcoder.fit_transform(test_y)
				num_c = np.max(train_y) + 1
				st.write("number of classes", num_c)
				train_y = utils.to_categorical(train_y, num_c)
				test_y = utils.to_categorical(test_y, num_c)
				#train_x = tf.convert_to_tensor(train_x)
				#test_x = tf.convert_to_tensor(test_x)
				hub_layer = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-250-with-normalization/2", input_shape=[], dtype=tf.string)
				model = tf.keras.models.Sequential([ hub_layer,
													tf.keras.layers.Dense(128, activation='relu'),
													tf.keras.layers.Dropout(.2),
													tf.keras.layers.Dense(num_c, activation='sigmoid')])



				model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

				history = model.fit(train_x,train_y, epochs=50, batch_size=32,validation_data=(test_x, test_y),verbose=0)

				score = model.evaluate(test_x, test_y, verbose = 0)
				st.write('Test loss:', score[0])
				st.write('Test accuracy:', score[1])



if __name__ == '__main__':
	main()
