import torch
import sys
import struct
import numpy as np
print('processing', sys.argv[1])
map = torch.load(sys.argv[1])
with open('.'.join(sys.argv[1].split('.')[:-1])+'.bin', 'wb') as f:
	f.write(struct.pack(str(map.shape[0]*map.shape[0])+'d', *list(map.T.reshape(map.shape[0]*map.shape[0]))))
with open('.'.join(sys.argv[1].split('.')[:-1])+'.bin', 'rb') as f:
	buffer = f.read()
	map2 = np.array(struct.unpack(str(map.shape[0]*map.shape[0])+'d', buffer))
	print(np.max(np.absolute(map.T.reshape(map.shape[0]*map.shape[0])-map2)))
