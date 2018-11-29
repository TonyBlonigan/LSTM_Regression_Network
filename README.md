# LSTM_Regression_Network
Class for creating LSTM Regression Networks with simple interface for hyper parameter setting

## Usage
```
# build model
rnn = RnnNetwork(rnn_units=[200, 100], dense_units=[50, 20, 10], n_out=1, 
                 lr=0.000005, decay=0.0005,
                 l2_rnn_kernel=0.00, l2_rnn_activity=0.001, l2_fc=0.0, 
                 clipnorm=0.05, epochs=100,
                 batch_size=5, stateful=True, shuffle=False, verbose=0)

rnn.build_model(x_train=x_train)

# train model
rnn.fit_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

# plot the training mse path
plt.plot(np.sqrt(rnn.loss), label='train')
plt.plot(np.sqrt(rnn.val_loss), label='test')
plt.legend()
plt.title('MSE by Epoch')
plt.show()

# plot diagnostics
y = y_scaler.inverse_transform(y_val.ravel())
y_hat = rnn.model.predict(x_val, batch_size=rnn.batch_size).ravel()
y_hat = y_scaler.inverse_transform(y_hat)

print('MSE: %f' % np.sqrt(mean_squared_error(y, y_hat)))
print('r2: %f' % r2_score(y, y_hat))

# residuals plot
plt.plot(y, y - y_hat, 'o')
plt.title('Residuals Plot')
plt.xlabel('y')
plt.ylabel('residuals')
plt.show()

# plot actual v predicted values over time
plt.plot(y, label='y')
plt.plot(y_hat, label='y_hat')
plt.legend()
plt.title('Predictions at Each Time Step in Validation Data Set')
plt.show()
```
