# coverting bit depth from 32 to 24
from PIL import Image

fault = "twist"
type = "t1t2t3"
dir = "D:/Atik/pythonScripts/Data Final/Gan/Blade/Fake"

for i in range(1,3001):
    im = Image.open(dir+"/{}/{}/{}Fake{}.png".format(fault,type,type,i)).convert('RGB')
    im.save(dir+"/{}/{}/{}Fake{}.png".format(fault,type,type,i))