# GLANCE
### General Anomaly Classification Engine

## Project
In repetitive manufacturing there are a lot of machines and equipment that runs in cycles. The standard approach to keep them working well is regular inspections but they can be slow and expensive. A better approach is to apply sensors and do continuous monitoring. However, anomalies can be subtle and hard to detect and it is very often impractical to collect labelled anomaly data. Imagine the cost of breaking a $100K machine in every way that it can break just to get anomaly data.

Research has shown that variational autoencoders (VAE), a type of neural network, can work for detecting skin cancer anomalies from 2D images. I have adapted this idea to work on 1D time-series sensor data. The idea is to learn a lower dimensional probabilistic representation for "normal" operation using the VAE and use the output error to detect outliers (anomalies). This works but it is not sufficient for high accuracy. As a final step the representation space is clustered and a likelihood is calculated for each new sample (using GMMs). New cycles are streamed to the detector and anomalies are flagged below a threshold. If the anomaly rate grows too high it is then relatively straightforward to flag the equipment for maintenance. This complete pipeline achieves over 80% accuracy on the test dataset.

# How to Use
The functionality of the trained model can be demonstrated live with the included flask app. Depending on current hosting this may be accessible at glance.mattmacdonald.me. 
The notebooks demonstrate the training and assessment of the model, particularly:
* VAE_training_hydraulic_faults.ipynb
* VAE_error_classification.ipynb
* VAE_clustering_classification.ipynb
