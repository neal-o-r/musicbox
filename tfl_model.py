import numpy as np
import tflearn

songs = np.load('mid.npy')

X = np.zeros((100000, 20, 156))
Y = np.zeros((100000, 156))
c = 0
for s in songs:
	if s.shape[0] < 120 or c == 100000:
		continue

	for i in range(100):	
		X[c, :, :] = s[i:20+i]
		Y[c] = s[i]	
		c += 1
	

del songs

g = tflearn.input_data(shape=[None, 20, 156])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, 156, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary={i:i for i in range(156)},
                              seq_maxlen=20,
                              clip_gradients=5.0)

seed = list(X[np.random.randint(X.shape[0])].argmax(1))
m.fit(X, Y, validation_set=0.1, batch_size=128,
	n_epoch=10)
print("-- TESTING...")
print("-- Test with temperature of 1.2 --")	
seq = m.generate(50, temperature=1.2, seq_seed=seed)

out = np.zeros((len(seq), 156))
for i, s in enumerate(seq):
	out[i, s] = 1

np.save('out.npy', out)

