import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB6

# import pathlib
# root_path = "/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/Segmented_train_120epochs"
# data_dir = pathlib.Path(root_path)


"""**Splitting the data into training and validation**"""

BATCH_SIZE = 64
IMG_HEIGHT, IMG_WIDTH = (256, 256)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size =BATCH_SIZE 
)


"""**Training the model**"""

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = img_augmentation(inputs)  
model = EfficientNetB2(include_top=False, input_tensor=x, weights="imagenet")

# Freeze the pretrained weights
model.trainable = False

# take a tensor and compute the average value of all values across the entire matrix for each of the input channels.
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
# 
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(8, activation="softmax", name="pred")(x)

# Compile
model = tf.keras.Model(inputs, outputs, name="EfficientNet")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#20 epochs

epochs = 20
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

model.save('/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/rms_Processed_Segmented_B2_efficientnet_model.h5')
print("EfficientNet Model saved")

#40 epochs

model.trainable = True
epochs=20
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

model.save('/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/rms.h5')
print("EfficientNet Model saved")

#45 epochs

epochs=5
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

model.save('/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/Segmented1_efficientnet_45_epochs.h5')
print("EfficientNet Model saved")

#50 epochs

epochs=5
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

model.save('/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/Segmented_efficientnet_50_epochs.h5')
print("EfficientNet Model saved")

"""**Validation Testing**"""

from keras.models import load_model
model=load_model('/content/gdrive/MyDrive/ISIC_DATASETS/ISIC_Datasets/Classification/Segmented_efficientnet_50_epochs.h5')

y_val = np.zeros(5066, dtype=object)
x_val = np.zeros([5066, 256, 256, 3])
i=0
j=0

for x,y in val_ds:
  for k in x:
    x_val[j, :,:] = k
    j+=1
  for p in y:
    y_val[i] = p
    i+=1

y_label = np.empty(5066, dtype=object)
for i in range(len(y_val)):
  y_label[i] = np.array(y_val[i])

classnames = train_ds.class_names
classnames

pred=model.predict(x_val,BATCH_SIZE =64, verbose=1)

y_true=[]
for i in range(len(y_label)):
  y_true.append(classnames[y_label[i]])

output_class=[]
for i in range(len(pred)):
  output_class.append(classnames[np.argmax(pred[i])])

"""**Displaying few of the predicted samples**"""

for i in range(50):
  print("\nImage : ", i+1)
  cv2_imshow(x_val[i])
  print("The actual class is", y_true[i])
  print("The predicted class is", output_class[i])

Crct_Counter={'AK':0,'MEL':0,'BCC':0,'NV':0,'BKL':0,'DF':0,'VASC':0,'SCC':0}
Total_Counter={'AK':0,'MEL':0,'BCC':0,'NV':0,'BKL':0,'DF':0,'VASC':0,'SCC':0}
for i in range(len(pred)):
  Total_Counter[y_true[i]]+=1
  if(y_true[i]==output_class[i]):
    Crct_Counter[y_true[i]]+=1

"""**Accuracies for each class**"""

print("Class_name \t\t Crct_pred \t\t  tot_images \t\t Accuracy")
print("AK\t\t\t",Crct_Counter['AK'],"\t\t\t ",Total_Counter['AK'],"\t\t\t",round(Crct_Counter['AK']/Total_Counter['AK'],2))
print("MEL\t\t\t",Crct_Counter['MEL'],"\t\t\t ",Total_Counter['MEL'],"\t\t\t",round(Crct_Counter['MEL']/Total_Counter['MEL'],2))
print("BCC\t\t\t",Crct_Counter['BCC'],"\t\t\t ",Total_Counter['BCC'],"\t\t\t",round(Crct_Counter['BCC']/Total_Counter['BCC'],2))
print("NV\t\t\t",Crct_Counter['NV'],"\t\t\t ",Total_Counter['NV'],"\t\t\t",round(Crct_Counter['NV']/Total_Counter['NV'],2))
print("BKL\t\t\t",Crct_Counter['BKL'],"\t\t\t ",Total_Counter['BKL'],"\t\t\t",round(Crct_Counter['BKL']/Total_Counter['BKL'],2))
print("DF\t\t\t",Crct_Counter['DF'],"\t\t\t ",Total_Counter['DF'],"\t\t\t",round(Crct_Counter['DF']/Total_Counter['DF'],2))
print("VASC\t\t\t",Crct_Counter['VASC'],"\t\t\t ",Total_Counter['VASC'],"\t\t\t",round(Crct_Counter['VASC']/Total_Counter['VASC'],2))
print("SCC\t\t\t",Crct_Counter['SCC'],"\t\t\t ",Total_Counter['SCC'],"\t\t\t",round(Crct_Counter['SCC']/Total_Counter['SCC'],2))

"""**Displaying the classification report**"""

from sklearn.metrics import classification_report
target_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
print(classification_report(y_true, output_class, target_names=target_names))

"""**Displaying the Confusion Matrix**"""

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, output_class, labels=target_names)