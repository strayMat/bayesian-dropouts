import matplotlib.pyplot as plt

import os
from PIL import Image
"""Creation of the notmnist dataset of size 10*nb_img_per_class using the .jpeg version of notmnist available here:
http://yaroslavvb.com/upload/notMNIST/
"""

path = 'data/notMNIST_large/'

data = [[]]*10
y_train = [[]]*10

# test of PIL.Image
tpic = Image.open(path +os.listdir(path )[0] + '/' + os.listdir(path + os.listdir(path)[0])[0])
tpic = np.array(tpic.getdata()).reshape(tpic.size[0]* tpic.size[1])
plt.imshow(tpic.reshape(28,28), cmap = "gray")
plt.show()
tpic.shape

#dictionnary to swap between numerical label and letter
letter2label = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], np.arange(10)))
label2letter = dict(zip(np.arange(10), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']))

# parameter for the size of the constructed dataset
nb_img_per_class = int(1e4)

# loop over the images
for d, i in zip(os.listdir(path), np.arange(10)):
    print('converting class', d)
    tmp_class = np.zeros((nb_img_per_class, 28 * 28))
    for f, j in zip(os.listdir(path + d)[:nb_img_per_class], np.arange(nb_img_per_class)):
        if j % 1000 == 0: print(j)
        local_path = path + d + '/'+ f
        try:
            pic = Image.open(local_path)
            pic = np.array(pic.getdata()).reshape((-1, pic.size[0]* pic.size[1]))
        except:
            # if we raise an exception (bad file name or anything stupid just take the precedent image )
            pic = tmp_class[j-1,:]
        tmp_class[j, :] = pic
    data[i] = tmp_class
    # adding label
    y[i] = np.array([letter2label[d]]*nb_img_per_class)

x_train = np.concatenate([c for c in data])
y_train = np.concatenate([c for c in y])

# checking
ix = 90001
print(label2letter[y_train[ix]])
plt.imshow(x_train[ix].reshape(28,28), cmap = "gray")
plt.show()

# shufle the dataset and conserving the correspondance label/image
shuffle = np.arange(len(y_train))
np.random.shuffle(shuffle)
xx_train = x_train[shuffle]
yy_train = y_train[shuffle]

## saving the dataset: uncomment to trigger
#train_data = dict({'x_train':xx_train, 'y_train': yy_train})
#np.save("notmnist_train", train_data)