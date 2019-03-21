import os
import tensorflow as tf
import numpy as np
import DEN
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = "4，5"  # 这是为了控制使用哪块GPU自己加的

np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 4300, "Epoch to train")   # 与代码运行输出日志一致，决定训练时最大的迭代次数
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_integer("dims", [784, 312, 128, 10], "Dimensions about layers including output")
# [784,312]是task1的shape；[312,128]是task2的shape；
flags.DEFINE_integer("n_classes", 10, 'The number of classes at each task')
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
#模型扩展过程中，每次增加的单元数
flags.DEFINE_integer("ex_k", 10, "The number of units increased in the expansion processing")
flags.DEFINE_float('loss_thr', 0.01, "Threshold of dynamic expansion")
flags.DEFINE_float('spl_thr', 0.05, "Threshold of split and duplication")
FLAGS = flags.FLAGS
# 0 数据准备
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trainX = mnist.train.images  # 是[55000,784]的张量
valX = mnist.validation.images
testX = mnist.test.images

# 将0-783这784个数据打乱顺序存储到列表task_permutation中，相当于task_permutation是一个10行，784列的列表
task_permutation = []
for task in range(10):
	task_permutation.append(np.random.permutation(784))

# 将训练测试和验证集的图像784维顺序都打乱我不太能理解，这样的话，那个图片的显示不就变了吗？
# 答：这是一种MT（多任务）模式，分类问题是唯一的，但是多个任务是相互独立的。
# 将维度顺序打乱了只是另外一种形式的表现形式。不同的打乱顺序代表不同的任务。
trainXs, valXs, testXs = [], [], []
for task in range(10):
	trainXs.append(trainX[:, task_permutation[task]])
	valXs.append(valX[:, task_permutation[task]])
	testXs.append(testX[:, task_permutation[task]])

# 1 2 3 前传和反传（即模型准备），因为终身学习的原因，模型迭代也包含在了这里
model = DEN.DEN(FLAGS)
params = dict()  # 为参数创建一个空字典
avg_perf = []   # 创建一个空列表

# 终身学习中的任务控制
# 以每个任务中的类别数作为总共要训练的任务数
# 为什么待分类的数据是10类，这里就设10个任务，还有就是这些数据只是打乱了784维图片的维度排列，为什么就算是不同的任务了呢？
# 答：我认为这里设置几个任务跟类别数是无关的，也可以设置5个。
# 784维图片重新排列，是数字的不同表现形式，相当于另一门语言那个数字的写法，对于模型来说就是不同的任务了。
for t in range(FLAGS.n_classes):
	data = (trainXs[t], mnist.train.labels, valXs[t], mnist.validation.labels, testXs[t], mnist.test.labels)
	model.sess = tf.Session()
	print("\n\n\tTASK %d TRAINING\n"%(t+1))

	model.T = model.T+1
	model.task_indices.append(t+1)
	model.load_params(params, time = 1)
	perf, sparsity, expansion = model.add_task(t+1, data)  # add_task是DEN模型调用的核心
	# writer = tf.summary.FileWriter('./improved_graph2', model.sess.graph)  # 这是为了可视化计算图自己加的

	params = model.get_params()  # 每一轮训练完的参数会存储起来
	print("查看参数存储")
	print(params)
	model.destroy_graph()
	model.sess.close()

	model.sess= tf.Session()
	print('\n OVERALL EVALUATION')
	model.load_params(params)
	temp_perfs = []
	for j in range(t+1):
		temp_perf = model.predict_perform(j+1, testXs[j], mnist.test.labels)
		temp_perfs.append(temp_perf)
	avg_perf.append( sum(temp_perfs) / float(t+1) )
	print("   [*] avg_perf: %.4f"%avg_perf[t])
	model.destroy_graph()
	model.sess.close()

