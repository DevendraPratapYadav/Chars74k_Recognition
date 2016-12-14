"""

Devendra Pratap Yadav - 2014CSB1010
Mayank Kumar - 2014CSB1022
CSL461 - Digital Image Analysis - Project

Neural Network using Scikit-learn
"""


import os, struct
from array import array as pyarray

from numpy import *
from pylab import *
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.externals import joblib
from sklearn.metrics import classification_report,confusion_matrix

from csv import reader



def load_all(filename):
	
	images=[]; labels=[];
	with open(filename) as afile:
		r = reader(afile)
		for line in r:
			labels.append(int (line[0]));
			images.append( map(float,line[1:]) );
	
	return images,labels;


def load_mnist(dataset="training", digits=np.arange(10), path="."):
   

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = 't10k-images-idx3-ubyte' #os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = 't10k-labels-idx1-ubyte' #os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])#.reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
	

#np.set_printoptions(threshold='nan')

# **************************************************************************
# use this Flag to toggle the dataset used for training

useHOG=True;	

# **************************************************************************

filename="";
if (useHOG):
	filename="features-handHOG.txt";
else:
	filename="features-handPIX.txt";
	
images,labels=load_all(filename);

images=array(images);
labels=array(labels);

print images.shape, " , ", labels.shape;

N=len(images);

x = np.arange(1,len(images));
np.random.shuffle(x);
shuf=x[:];

#print x

comb=zip(x,images);
comb=sorted(comb);

images=[x[1] for x in comb];

comb=zip(shuf,labels);
comb=sorted(comb);

labels=[x[1] for x in comb];

comb=zip(labels,images);
#print comb;


spl= int(0.8*N); #split location for train-test

trI=array(images[:spl]); trL=array(labels[:spl]);
teI=array(images[spl:]); teL=array(labels[spl:]);


trI=trI.astype('float'); teI=teI.astype('float');

if (useHOG==False):
	trI/=255.0;	
	teI/=255.0;

#print trI[0];

#print trI, trL;

#print teI, teL;

nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
	dropout_rate=0.2,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=20,
	#learning_rule='adam',
    n_iter=100,
	verbose=True)


nn.fit(trI,trL);	

res=nn.score(teI,teL);
print res


yres = nn.predict(teI);

  
print("\tReport:")
print(classification_report(teL,yres))
print '\nConfusion matrix:\n',confusion_matrix(teL, yres)


#print yres,teL

print "\n\nAccuracy : ", res,"\n";


"""
74 % accuracy


nn = Classifier(
    layers=[
        Layer("Rectifier", units=700),
		Layer("Rectifier", units=300),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.1,
	#learning_momentum=0.7,
	#batch_size=10,
	#learning_rule='adam',
    n_iter=20,
	verbose=True)
	
"""























"""

	
images, labels = load_mnist('training')#,digits=[3,8,5,6,9])
#imshow(images.mean(axis=0), cmap=cm.gray)
#show()

xd=images[:5000]; yd=labels[:5000];
#xt=images[6000:7000]; yt=labels[6000:7000];

#xt=images[10000:12000]; yt=labels[10000:12000];

xt,yt = load_mnist('testing')

#yd=yd.reshape(1,len(yd))
print xd.shape,", ",yd.shape
xd=xd.astype('float'); xt=xt.astype('float');
xd/=255; xt/=255;

#print xd[0]

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
		#Layer("Rectifier", units=200),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.05,
	dropout_rate=0.3,
	valid_size=0.1,
	learning_momentum=0.7,
	#batch_size=10,
	#learning_rule='adam',
    n_iter=10,
	verbose=True)

nn.fit(xd,yd)	

res=nn.score(xt,yt)
print res


yres = nn.predict(xt)

  
print("\tReport:")
print(classification_report(yt,yres))
print '\nConfusion matrix:\n',confusion_matrix(yt, yres)

#joblib.dump(nn, 'nn300-200_lr005_lmom07_drop03_bs10.pkl') 
#clf = joblib.load('filename.pkl') 

"""	





"""

87% for 36 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=400),
		Layer("Rectifier", units=200),
		Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.1,
	learning_momentum=0.4,
	batch_size=5,
	#learning_rule='adam',
    n_iter=50,
	verbose=True)

"""



"""
75 % for 62 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
	dropout_rate=0.4,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=10,
	#learning_rule='adam',
    n_iter=30,
	verbose=True)
	
	
"""	

"""
75 % for 62 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=500),
		Layer("Rectifier", units=200),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=10,
	#learning_rule='adam',
    n_iter=50,
	verbose=True)
"""



"""
77 % for 62 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=20,
	#learning_rule='adam',
    n_iter=50,
	verbose=True)
"""


"""
78 % for 62 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
	dropout_rate=0.3,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=20,
	#learning_rule='adam',
    n_iter=100,
	verbose=True)

"""


"""
79 % for 62 classes


nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
	dropout_rate=0.3,
	valid_size=0.15,
	learning_momentum=0.3,
	batch_size=15,
	#learning_rule='adam',
    n_iter=100,
	verbose=True)

	"""
	
	
	

#*******************************************************************************************
# HOG
	
"""
	82 % for 62 classes
	
nn = Classifier(
    layers=[
        Layer("Rectifier", units=600),
		Layer("Rectifier", units=300),
		#Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.01,
	dropout_rate=0.3,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=10,
	#learning_rule='adam',
    n_iter=100,
	verbose=True)
	
	
	
"""
	
	
	
"""
	84 % for 62 classes
	
	
	
	
nn = Classifier(
    layers=[
        Layer("Rectifier", units=800),
		Layer("Rectifier", units=400),
		#Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.015,
	dropout_rate=0.35,
	valid_size=0.15,
	learning_momentum=0.4,
	batch_size=20,
	#learning_rule='adam',
    n_iter=100,
	verbose=True)
	
	"""
	
	
	
	
	
	
	
	
	
	
	