### AeroNet RUL

https://aeronetrul-raquel.streamlit.app/



This is a turbo fan engines Remaining Useful Life (RUL) predictor for prognostics and health management (PHM) or engines using NASA’s C-MAPSS (Commercial Modular Aero-Propulsion System Simulation). 




The proposed architecture utilizes a hybrid Convolutional Autoencoder (CAE) and Bidirectional Long Short-Term Memory (Bi-LSTM) framework enhanced by a temporal attention mechanism for Remaining Useful Life (RUL) estimation. While the model adopts a CAE-inspired structure, it maintains a constant temporal resolution through Conv1D layers with 'same' padding, rather than employing dimensionality reduction via MaxPooling and UpSampling. Sequential dependencies are captured through a stacked Bi-LSTM backbone, while a Softmax-driven attention layer dynamically weights critical sensor states. The final transition from sequential features to regression is facilitated by a GlobalAveragePooling1D layer followed by a high-capacity Dense hidden layer with Dropout regularization, ensuring robust feature extraction and Keras 3 compatibility for stable performance across diverse turbofan degradation profiles





References:
Elsherif, S.M., Hafiz, B., Makhlouf, M.A. et al. A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life. Sci Rep 15, 26251 (2025). 
https://doi.org/10.1038/s41598-025-09155-z

Data: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
