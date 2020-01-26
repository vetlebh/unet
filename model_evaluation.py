#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[3]:


import pickle


# In[ ]:


import numpy as np
import maplotlib.pyplot as plt


# ### Load model and create predictions

# In[401]:


model = load_model('results/model_5_3.h5')
model.load_weights('results/weights_5_3.h5')


# In[495]:


preds = model.predict(X, verbose=1)


# In[404]:


# Create ground truth masks
y_mask = y.copy()
y_mask = np.ma.masked_where(y_mask < 0.5, y_mask)

# Create prediction masks
preds_t_mask = preds_t.copy()
preds_t_mask = np.ma.masked_where(preds_t_mask < 0.5, preds_t_mask)


# ### Vizualisation

# In[411]:


def plot_sample_two_models(X, y, y_mask, preds, preds2, preds_mask, preds_mask2, savename='temp.pdf', ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(3, 2, figsize=(10, 16))
    
    # Turn off tick labels
    ax[0,0].set_yticklabels([])
    ax[0,0].set_xticklabels([])
    ax[0,1].set_yticklabels([])
    ax[0,1].set_xticklabels([])
    ax[1,0].set_yticklabels([])
    ax[1,0].set_xticklabels([])
    ax[1,1].set_yticklabels([])
    ax[1,1].set_xticklabels([])
    ax[2,0].set_yticklabels([])
    ax[2,0].set_xticklabels([])
    ax[2,1].set_yticklabels([])
    ax[2,1].set_xticklabels([])
    
    ax[0,0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0,0].imshow(y_mask[ix, ..., 0], cmap='prism', interpolation='none', alpha=0.6)
    if has_mask:
        ax[0,0].contour(y[ix].squeeze(), colors='y', levels=[0.5])
    ax[0,0].set_title('Ground truth MRI')

    ax[0,1].imshow(y[ix].squeeze())
    ax[0,1].set_title('Ground truth mask')
    
    ax[1,0].imshow(X[ix, ..., 0], cmap='gray')
    ax[1,0].imshow(preds_mask[ix, ..., 0], cmap='prism', interpolation='none', alpha=0.6)
    if has_mask:
        ax[1,0].contour(preds[ix].squeeze(), colors='y', levels=[0.5])
    ax[1,0].set_title('5c Predicted MRI')
    
    ax[1,1].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[1,1].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    ax[1,1].set_title('5c Predicted mask');
    
    ax[2,0].imshow(X[ix, ..., 0], cmap='gray')
    ax[2,0].imshow(preds_mask2[ix, ..., 0], cmap='prism', interpolation='none', alpha=0.6)
    if has_mask:
        ax[2,0].contour(preds2[ix].squeeze(), colors='y', levels=[0.5])
    ax[2,0].set_title('5a Predicted MRI')
    
    ax[2,1].imshow(preds2[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2,1].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    ax[2,1].set_title('5a Predicted mask');
    
    plt.savefig(savename)


# In[237]:


def plot_sample_one_model(X, y, y_mask, preds, preds_mask, savename='temp.pdf', ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3, figsize=(13, 13))
    
    # Turn off tick labels
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])

    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].imshow(preds_mask[ix, ..., 0], cmap='prism', interpolation='none', alpha=0.6)
    if has_mask:
        ax[0].contour(preds[ix].squeeze(), colors='y', levels=[0.5])
    ax[0].set_title('Predicted MRI')
    
    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Ground truth mask')
    
    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    ax[2].set_title('Predicted mask');
    
    plt.savefig('results/'+savename)


# ### Plot Training History

# In[125]:


# Import training history
hist = []
for runs in range (1, 6):
    with open('results/trainHistoryDict_' + str(runs) + '.dms', 'rb') as f:
        hist.append(pickle.load(f))


# In[126]:


hist1 = hist[0]
hist2 = hist[1]
hist3 = hist[2]
hist4 = hist[3]
hist5 = hist[4]


# In[128]:


# Plot training history for 
plt.figure(figsize=(8,8))
plt.title('Learning curve for 5 runs on 60 epochs')
plt.plot(np.log(hist1['val_loss']), label='Run 1')
plt.plot(np.argmin(hist1['val_loss']), np.min(np.log(hist1['val_loss'])), marker="x", color="r")
plt.plot(np.log(hist2['val_loss']), label='Run 2')
plt.plot(np.argmin(hist2['val_loss']), np.min(np.log(hist2['val_loss'])), marker="x", color="r")
plt.plot(np.log(hist3['val_loss']), label='Run 3')
plt.plot(np.argmin(hist3['val_loss']), np.min(np.log(hist3['val_loss'])), marker="x", color="r")
plt.plot(np.log(hist4['val_loss']), label='Run 4')
plt.plot(np.argmin(hist4['val_loss']), np.min(np.log(hist4['val_loss'])), marker="x", color="r")
plt.plot(np.log(hist5['val_loss']), label='Run 5')
plt.plot(np.argmin(hist5['val_loss']), np.min(np.log(hist5['val_loss'])), marker="x", color="r", label="Best Model")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend();
plt.savefig('learningcurve.pdf')


# In[25]:


hist_2 = []
with open('results/trainHistoryDict_5.dms', 'rb') as f:
    hist_2.append(pickle.load(f))
with open('results/trainHistoryDict_5_2.dms', 'rb') as f:
    hist_2.append(pickle.load(f))
with open('results/trainHistoryDict_5_3.dms', 'rb') as f:
    hist_2.append(pickle.load(f))


# In[27]:


hist5a = hist_2[0]
hist5b = hist_2[1]
hist5c = hist_2[2]


# In[32]:


plt.figure(figsize=(14,8))
plt.title('Learning curve for 3 consecutive trainings on 60 epochs each')
plt.plot(np.log(hist5a['val_loss']), label='Run 5a')
plt.plot(np.arange(60, 120),np.log(hist5b['val_loss']), label='Run 5b')
plt.plot(np.arange(120, 180),np.log(hist5c['val_loss']), label='Run 5c')
plt.plot(np.argmin(hist5c['val_loss'])+120, np.min(np.log(hist5c['val_loss'])), marker="x", color="r", label="Best Model")
plt.xlabel("Epochs")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend();
plt.savefig('learningcurve5.pdf')


# ### Metrics

# In[180]:


met = []
for runs in range (1, 6):    
    with open('results/metricsDict_'+str(runs)+'.dms', 'rb') as f:
        met.append(pickle.load(f))


# In[487]:


with open('results/metricsDict_5_3.dms', 'rb') as f:
    met5c = pickle.load(f)


# In[488]:


met5c


# ### IoU Histogram

# In[469]:


def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculate intersection over union for the given batch."""
    # Cast both arrays to boolean arrays according to a cut-off of 0.5
    y_pred = (y_pred > 0.5)
    y_true = (y_true > 0.5)

    # Mark each pixel as True if both predicts True (TP)
    intersection_mask = tf.math.logical_and(y_pred, y_true)

    # Mark each pixel as True if either true positive, false negative, or false
    # positive (TP + FP + FN)
    union_mask = tf.math.logical_or(y_pred, y_true)

    # Calculate sum of true positives
    intersection = tf.math.count_nonzero(intersection_mask, axis=(1, 2))

    # Calculate sum of true positives fales negatives/positives
    union = tf.math.count_nonzero(union_mask, axis=(1, 2))

    # Set union to 1 where there are no true positives in order to prevent
    # division by zero
    dividing_union = tf.where(union == 0, tf.ones_like(union), union)

    # Calculate intersection over union
    ious = intersection / dividing_union

    # Set IoU to 1 where there are no true positives
    ious = tf.where(union == 0, tf.ones_like(ious), ious)

    # Return the mean IoU across the batch
    return tf.reduce_mean(ious)


# In[479]:


split = zip(X_test, y_test)
iou_array = []

for pair in split:

    model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics + [
                "binary_accuracy",
                "FalseNegatives",
                "FalsePositives",
                "Precision",
                "Recall",
                iou
            ],)

    evaluation = model.evaluate(np.reshape(pair[0], (1, 256, 256, 1)), np.reshape(pair[1], (1, 256, 256, 1)))
    metrics = {
        name: value
        for name, value
        in zip(model.metrics_names, evaluation)
        }

    iou_array.append(metrics['iou'])
    
    


# In[494]:



plt.figure(figsize=(16,8))
plt.hist(iou_array, bins=100)
plt.title('Histogram of IoU on predictions by model 5c on 95 images')
plt.xlabel("IoU")
plt.ylabel("Count")
plt.savefig('histo.pdf')


# In[492]:


np.mean(iou_array)
np.shape(iou_array)

