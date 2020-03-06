import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import glob
import imageio

images_train = []
names = []

for i in range(1,101):
    names.append('train/' + str(i) + '.jpg')

for filename in names:
    im = np.float_(plt.imread(filename).flatten())
    images_train.append(im)

files_test = glob.glob('test/*.jpg')
images_test = []
for filename in files_test:
    im = np.float_(plt.imread(filename).flatten())
    images_test.append(im)
    
label_man = np.arange(2,100,2)
label_man = np.append(label_man, [59, 67])
label = np.zeros(100)
label[label_man - 1] = 1

x_train, x_test, y_train, y_test = train_test_split(images_train, label, train_size = 0.7)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
data_val = scaler.transform(images_test)

c = np.logspace(-2,2,100)
score = []
for i in c:
    svc = SVC(C=i, kernel = 'linear')
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    score.append(f1_score(y_test, y_pred))

score = np.array(score)
ii = np.argmax(score)
c_max = c[ii]

svc = SVC(C = c_max, kernel = 'linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(data_val)

out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, y_pred):
    print(f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))
out.close()

#for names, count in zip(files_test, y_pred):
#    print(names, count)