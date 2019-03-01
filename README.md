# Brightside

Python library for plotting common plots and formatting text and tables in Jupyter Notebook.

## Examples

Loading the libraries:


```bash
%matplotlib inline

import brightside as bs
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
```

### Loss and Accuracy plots

Loading the log data into numpy arrays.

```python
epochs, train_loss, train_acc, val_loss, val_acc = np.loadtxt('training.log', delimiter=' ', 
                                                              usecols=(1, 3, 5, 7, 9), unpack=True)
labels = ['Train', 'Validation']
```

Plotting the loss

```python
fig, ax = bs.plot_loss([train_loss,val_loss], labels, epochs=epochs, figsize=(10,3))
plt.savefig('loss_plot.png', format='png', dpi=300)
```
<center><img src="examples/loss_plot.png"></img></center>

Plotting the accuracy

```python
fig, ax = bs.plot_accuracy([train_acc,val_acc], labels, epochs=epochs, figsize=(5,5))
plt.savefig('acc_plot.png', format='png', dpi=300)
```
<center><img src="examples/acc_plot.png" height="450"></img></center>

### Confusion matrix plot

Loading predictions and groundtruth into numpy arrays.

```python
predictions, groundtruth = np.loadtxt('predictions.csv', delimiter=' ', unpack=True)
labels = np.genfromtxt('labels.txt',dtype='str')
```

Plotting the confusion matrix

```python
fig, ax = bs.show_confusion_matrix(predictions, groundtruth, labels,
                                   figsize=(8,8), annot=False,
                                   linewidths=0.005, linecolor='gray',
                                   square=True, show_xticks=True,
                                   cbar_kws={"shrink": 0.75})
plt.savefig('cm.png', format='png', dpi=300)
```

<center><img src="examples/cm.png" height="550"></img></center>

### Printing a table on Jupyter

Making an array of random numbers.

```python
np.random.seed(42)
rand_numbers = np.random.uniform(0,1,100)
```

Showing the table 

```python
bs.print_table('Statistics', [('Min:', rand_numbers.min()),
('Max:', rand_numbers.max()),
('Mean:', rand_numbers.mean()),
('Median:', np.median(rand_numbers)),
('StdDev:', rand_numbers.std()),
('Mode:', stats.mode(rand_numbers).mode[0]),
('Distinct:', np.size(np.unique(rand_numbers)))])
```

<b>Statistics</b>
<table>
  <tr>
    <td><b>Min:</b></td>
    <td>0.005522117123602399</td>    
  </tr>
  <tr>
    <td><b>Max:</b></td>
    <td>0.9868869366005173</td>    
  </tr>
  <tr>
    <td><b>Mean:</b></td>
    <td>0.47018074337820936</td>    
  </tr>
  <tr>
    <td><b>Median:</b></td>
    <td>0.4641424546894926</td>    
  </tr>
  <tr>
    <td><b>StdDev:</b></td>
    <td>0.29599822663249037</td>    
  </tr>
  <tr>
    <td><b>Mode:</b></td>
    <td>0.005522117123602399</td>    
  </tr>
  <tr>
    <td><b>Distinct:</b></td>
    <td>100</td>    
  </tr>
</table>