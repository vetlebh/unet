import os
import numpy as np

from skimage.transform import resize
from sklearn.model_selection import train_test_split



def data_gen(training_path, img_name, gt_name, im_height=256, im_width=256, test_size=0.1, val_size=0.1, random_state=1):

    walk = next(os.walk(training_path))[1]

    X = np.zeros((len(walk)*15, im_height, im_width, 1))
    y = np.zeros((len(walk)*15, im_height, im_width, 1))

    img_nr = 0
    sum_slices = 0
    patients_not_found = 0
    for ids in walk:

        try:
            img = np.load(os.path.join(training_path, ids, img_name))
            gt = np.load(os.path.join(training_path, ids, gt_name))
            slices = img.shape[2]

            for slice_nr in range(slices):

                img_slice, gt_slice = img[:, :, slice_nr], gt[:, :, slice_nr]
                img_resized = resize(img_slice, (im_height, im_width, 1), mode = 'edge', preserve_range = True, anti_aliasing=True)
                gt_resized = resize(gt_slice, (im_height, im_width, 1), mode = 'edge', preserve_range = True, anti_aliasing=True)

                # We are only interested in the classes 'heart' and 'background' for this experiment
                gt_resized = (gt_resized > 0.5).astype(np.uint8)

                X[sum_slices, :, :, :] = img_resized/255.0
                y[sum_slices, :, :, :] = gt_resized

                sum_slices +=1

        except:
            print(f'{ids} not found')
            patients_not_found += 1
            continue

        if(img_nr%10 == 0):
            print(f'{img_nr} images and {sum_slices} slices loaded to array')
        img_nr += 1
    print(f'Image load complete. {img_nr} images and {sum_slices} slices loaded successfully. ')

    X, y = X[:sum_slices, :, :, :], y[:sum_slices, :, :, :]
    print(X.shape, y.shape)
    print(np.unique(y))

    # Split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Split train data into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    print(f'Training size: {X_train.shape[0]}, Validation size: {X_valid.shape[0]}, Test size: {X_test.shape[0]}')

    return X_train, X_valid, X_test, y_train, y_valid, y_test
