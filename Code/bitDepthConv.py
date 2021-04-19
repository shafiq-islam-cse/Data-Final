# coverting bit depth from 32 to 24
from PIL import Image

fault = "Outer0.021"

dir = "D:/Atik/pythonScripts/Data Final/Gan/Bearing/Fake2/"

for i in range(1,1001):
    im = Image.open(dir+"{}/Fake{}.png".format(fault,i)).convert('RGB')
    im.save(dir+"{}/Fake{}.png".format(fault,i))