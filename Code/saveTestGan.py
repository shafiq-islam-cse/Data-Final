#FOr single image
# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn

# load model
model = load_model(r'D:\Atik\pythonScripts\WCNN\vibImages\ballFig\generator_model_0.h5')

#To create same image, suppy same vector each time
# all 0s
#vector = asarray([[0. for _ in range(100)]])  #Vector of all zeros

#To create random images each time...
vector = []
for i in range(0,200):
    #Vector of random numbers (creates a column, need to reshape)
    a = randn(100)  
    a = a.reshape(1, 100)
    vector.append(a)
    
# generate image
for i in range(0,len(vector)):
    X = model.predict(vector[i])
    # plot the result
    pyplot.imshow(X[0, :, :, 0], cmap='gray')
    pyplot.axis('off')
    pyplot.savefig(
        "D:/Atik/pythonScripts/WCNN/vibImages/ballFig/0/Fake/fakeFig_%d.png" % (i+1),
         bbox_inches='tight', pad_inches=0, dpi = (96*20/289))
    pyplot.close()