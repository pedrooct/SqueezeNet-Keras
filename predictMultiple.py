import cv2
import numpy as np
import os,glob
import matplotlib.pyplot as plt
import argparse
import utils
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


def predict(model, model_parms,img_test_batch):

    prob = model.predict(x=img_test_batch, batch_size=1, verbose=1, steps=None)
    prediction = np.argmax(prob, axis=1)[0]
    return model_parms['classes'][prediction]

if __name__ == "__main__":
    # Parse arguements
    parser = argparse.ArgumentParser(description="SqueezeNet Prediction.")
    parser.add_argument("--test-folder", type=str, default='./dataset/test/*',
                        dest='test_folder', help="The full path for the test folder")
    parser.add_argument("--mean-image", type=str, default='./images/mean_image.jpg',
                        dest='mean_image', help="The full path for mean image of training dataset.")
    parser.add_argument("--saved-model", type=str, default='./model/squeezenet_model.h5',
                        dest='saved_model', help="The trained squeezenet keras model (.h5)")
    parser.add_argument("--model-parms", type=str, default='./model/model_parms.json',
                        dest='model_parms', help="The dictionary of model params (classes and image dimensions)")
    args = parser.parse_args()

    # Load trained model
    model = load_model(args.saved_model)
    # Load model parms
    model_parms = utils.load_model_parms(args.model_parms)
    # loop here !
    for file in glob.glob(args.test_folder):
    # Mean image
        img_mean_array = img_to_array(load_img(args.mean_image,target_size=(model_parms['height'], model_parms['width'])))
        # Test image
        img = cv2.imread(file)
        for r in range(0,img.shape[0],64):
            for c in range(0,img.shape[1],64):
                img64=img1[r:r+64, c:c+64,:]
                img32 = cv2.resize(img64,(32,32))
                img_test_array = img_to_array(load_img(img32,target_size=(model_parms['height'], model_parms['width'])))
                img_test_array -= img_mean_array
                img_test_batch = np.expand_dims(img_test_array, axis=0)
                # Predict the class of the image
                predicted_class = predict(model,model_parms,img_test_batch)
                print("Class:"+predicted_class)
    ##END loop
        # Display the image and predicted class
        #img_test = load_img(args.test_image)
        #fig = plt.figure()
        #plt.imshow(img_test)
        #plt.title(predicted_class)
        #plt.axis('off')
        #plt.show()
