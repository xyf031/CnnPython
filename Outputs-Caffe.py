

#################################################### dir(caffe)
AdaDeltaSolver
AdaGradSolver
AdamSolver
Classifier
Detector
Layer
NesterovSolver
Net
NetSpec
RMSPropSolver
SGDSolver
TEST
TRAIN
__builtins__
__doc__
__file__
__name__
__package__
__path__
__version__
_caffe
classifier
detector
get_solver
io
layer_type_list
layers
net_spec
params
proto
pycaffe
set_device
set_mode_cpu
set_mode_gpu
set_random_seed
to_proto



#################################################### dir(solver = caffe.SGDSolver)
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__hash__
__init__
__instance_size__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__
restore
snapshot
iter
solve
step
net
test_nets


#################################################### dir(solver.net)
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__hash__
__init__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__

_backward
_batch
_blob_loss_weights
_blob_names
_blobs
_bottom_ids
_forward
_inputs
_layer_names
_outputs
_set_input_arrays
_top_ids

blob_loss_weights
bottom_names
top_names
inputs
outputs

copy_from
save
reshape
set_input_arrays
share_with

backward
forward
forward_all
forward_backward_all

params
blobs
layers



#################################################### pl(solver.net._blob_names)
# solver.net._blob_names[i]
data
label
conv1
pool1
conv2
pool2
ip1
ip2
loss



#################################################### pl(solver.net._layer_names)
# solver.net._layer_names[i]
mnist
conv1L
pool1L
conv2L
pool2L
ip1L
relu1L
ip2L
lossL



#################################################### dir(solver.net.blobs["conv1"])
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__hash__
__init__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__

data        # numpy.ndarray, .data.shape = (64, 1, 28, 28)
diff        # Same as .data
reshape     # function

shape       # caffe._caffe.IntVec object, [64, 1, 28, 28]
count       # = 50176 = 64*1*28*28
num         # = 64
channels    # = 1
height      # = 28
width       # = 28



#################################################### dir(solver.net.layers[0])
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__hash__
__init__
__instance_size__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__

type        # str("Data")
reshape     # function
setup       # function
blobs       # <caffe._caffe.BlobVec object at 0x34fad70> No idea what this is.
# >>> pl(trainLayers[0].blobs)
# >>> pl(trainLayers[1].blobs)
# <caffe._caffe.Blob object at 0x3a45398>
# <caffe._caffe.Blob object at 0x3a452a8>
# >>> pl(trainLayers[2].blobs)
# >>> pl(trainLayers[3].blobs)
# <caffe._caffe.Blob object at 0x3a452a8>
# <caffe._caffe.Blob object at 0x3a45398>
# >>> pl(trainLayers[4].blobs)
# >>> pl(trainLayers[5].blobs)
# <caffe._caffe.Blob object at 0x3a45398>
# <caffe._caffe.Blob object at 0x3a452a8>
# >>> pl(trainLayers[6].blobs)
# >>> pl(trainLayers[7].blobs)
# <caffe._caffe.Blob object at 0x3a452a8>
# <caffe._caffe.Blob object at 0x3a45398>
# >>> pl(trainLayers[8].blobs)



#################################################### dir(solver.net.params)
_OrderedDict__map
_OrderedDict__marker
_OrderedDict__root
_OrderedDict__update
__class__
__cmp__
__contains__
__delattr__
__delitem__
__dict__
__doc__
__eq__
__format__
__ge__
__getattribute__
__getitem__
__gt__
__hash__
__init__
__iter__
__le__
__len__
__lt__
__module__
__ne__
__new__
__reduce__
__reduce_ex__
__repr__
__reversed__
__setattr__
__setitem__
__sizeof__
__str__
__subclasshook__
__weakref__

clear
copy
fromkeys
get
has_key
items
iteritems
iterkeys
itervalues
keys
pop
popitem
setdefault
update
values
viewitems
viewkeys
viewvalues



###########################   Test   #########################
#################################################### dir(solver.test_nets[0])
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__hash__
__init__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__

_backward
_batch
_blob_loss_weights
_blobs
_bottom_ids
_forward
_inputs
_outputs
_set_input_arrays
_top_ids
_blob_names
_layer_names

blobs
layers
blob_loss_weights
bottom_names
top_names
clear_param_diffs
copy_from

backward
forward
forward_all
forward_backward_all

inputs
load_hdf5
outputs
params
reshape
save
save_hdf5
set_input_arrays
share_with



#################################################### pl(testNameBlobs)
data
label
label_mnist_1_split_0
label_mnist_1_split_1
conv1
pool1
conv2
pool2
ip1
ip2
ip2_ip2L_0_split_0
ip2_ip2L_0_split_1
accuracy
loss



#################################################### pl(testNameLayers)
mnist
label_mnist_1_split
conv1L
pool1L
conv2L
pool2L
ip1L
relu1L
ip2L
ip2_ip2L_0_split
accuracyL
lossL



#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(solver.net.blobs["conv1L"])




#################################################### dir(numpy.ndarray)
__abs__
__add__
__and__
__array__
__array_finalize__
__array_interface__
__array_prepare__
__array_priority__
__array_struct__
__array_wrap__
__class__
__contains__
__copy__
__deepcopy__
__delattr__
__delitem__
__delslice__
__div__
__divmod__
__doc__
__eq__
__float__
__floordiv__
__format__
__ge__
__getattribute__
__getitem__
__getslice__
__gt__
__hash__
__hex__
__iadd__
__iand__
__idiv__
__ifloordiv__
__ilshift__
__imod__
__imul__
__index__
__init__
__int__
__invert__
__ior__
__ipow__
__irshift__
__isub__
__iter__
__itruediv__
__ixor__
__le__
__len__
__long__
__lshift__
__lt__
__mod__
__mul__
__ne__
__neg__
__new__
__nonzero__
__oct__
__or__
__pos__
__pow__
__radd__
__rand__
__rdiv__
__rdivmod__
__reduce__
__reduce_ex__
__repr__
__rfloordiv__
__rlshift__
__rmod__
__rmul__
__ror__
__rpow__
__rrshift__
__rshift__
__rsub__
__rtruediv__
__rxor__
__setattr__
__setitem__
__setslice__
__setstate__
__sizeof__
__str__
__sub__
__subclasshook__
__truediv__
__xor__

all
any
argmax
argmin
argpartition
argsort
astype
base
byteswap
choose
clip
compress
conj
conjugate
copy
ctypes
cumprod
cumsum
data
diagonal
dot
dtype
dump
dumps
fill
flags
flat
flatten
getfield
imag
item
itemset
itemsize
max
mean
min
nbytes
ndim
newbyteorder
nonzero
partition
prod
ptp
put
ravel
real
repeat
reshape
resize
round
searchsorted
setfield
setflags
shape
size
sort
squeeze
std
strides
sum
swapaxes
take
tobytes
tofile
tolist
tostring
trace
transpose
var
view
T



























params = [(paramName, paramBlob[0].data.shape, paramBlob[1].data.shape) for paramName, paramBlob in solver.net.params.items()]

ax.imshow(solver.net.blobs['dataA'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print solver.net.blobs['label'].data[:8]

ax.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print solver.test_nets[0].blobs['label'].data[:8]

solver.net.forward()
solver.test_nets[]
solver.step(1)
ax.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')


%%time
niter = 200
test_interval = 25
test_batch = 8
train_loss = zeros(niter)  # zeros(200)
test_acc = zeros(int(np.ceil(niter / test_interval)))  # zeros(200/25 = 8)
output = zeros((niter, test_batch, 10))  # zeros((200, test_batch, all_class))

for i in range(niter):
	solver.step(1)
	train_loss[i] = solver.net.blobs['loss'].data
	solver.test_nets[0].forward(start='conv1')  # start the forward pass at conv1 to avoid loading new data
	output[i] = solver.test_nets[0].blobs['score'].data[:test_batch]

	if i % test_interval == 0:
        print 'Iteration', i, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
        test_acc[i // test_interval] = correct / 1e4



_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)  # arange()?
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# ax2.set_ylabel('test accuracy')
# ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))




# We'll plot time on the x axis and each possible label on the y, with lightness indicating confidence.
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')  # blobs[].data.shape = (100, 1, 28, 28)
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    # xlabel('iteration')
    # ylabel('label')

for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')  # softmax
    # xlabel('iteration')
    # ylabel('label')



Logging before InitGoogleLogging() is written to STDERR

/home/x/Documents/caffe-master/python/caffe/pycaffe.py:13: 
RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Net<float> > already registered; second conversion method ignored.
  from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \

/home/x/Documents/caffe-master/python/caffe/pycaffe.py:13: 
RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Blob<float> > already registered; second conversion method ignored.
  from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \

/home/x/Documents/caffe-master/python/caffe/pycaffe.py:13: 
RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Solver<float> > already registered; second conversion method ignored.
  from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \









