�!	,D����B@,D����B@!,D����B@	�c��@�?�c��@�?!�c��@�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6,D����B@�ʆ5�E<@1o��ܚ4@A�4�8EG�?I6�!�Q@Y)�� ��?*	�����X�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�
F%u�?!�9����K@)��Q���?1$�_�h�E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map��x�&1�?!��  >@)t�����?1�~���8@:Preprocessing2F
Iterator::Model�/L�
F�?!�Є�#@)	�^)˰?1�_�@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenateޓ��ZӬ?!]�� �@)�Q��?1�P٥�@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat �o_Ω?!�Թh�@)HP�s�?1�K���@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate�#�����?!AoPɆ�	@)Zd;�O��?1��Q �@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!{����@)HP�sג?1wA��u@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch��y�):�?!d�:�� @)��y�):�?1d�:�� @:Preprocessing2U
Iterator::Model::ParallelMapV2�Q���?!-��� @)�Q���?1-��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�MbX9�?!�7��NN@)�
F%u�?1{�����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory�&1�|?!��yΐ��?)y�&1�|?1��yΐ��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP�s�r?!wA��u�?)HP�s�r?1wA��u�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor�����g?!`�]=��?)�����g?1`�]=��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensorǺ���V?!(N.�@A�?)Ǻ���V?1(N.�@A�?:Preprocessing2�
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSliceǺ���V?!(N.�@A�?)Ǻ���V?1(N.�@A�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�13.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�c��@�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ʆ5�E<@�ʆ5�E<@!�ʆ5�E<@      ��!       "	o��ܚ4@o��ܚ4@!o��ܚ4@*      ��!       2	�4�8EG�?�4�8EG�?!�4�8EG�?:	6�!�Q@6�!�Q@!6�!�Q@B      ��!       J	)�� ��?)�� ��?!)�� ��?R      ��!       Z	)�� ��?)�� ��?!)�� ��?JGPUY�c��@�?b �"-
IteratorGetNext/_3_Send���1f��?!���1f��?"-
IteratorGetNext/_1_Send�t~Ym��?!J��i��?"R
6autoencoder_12/sequential_18/dense_20/Tensordot/MatMulMatMul�9��e��?!����A�?"g
Kgradient_tape/autoencoder_12/sequential_19/dense_21/Tensordot/MatMul/MatMulMatMul[�a�b�?!�5����?"g
Kgradient_tape/autoencoder_12/sequential_18/dense_20/Tensordot/MatMul/MatMulMatMulg�����?!��H����?"i
Mgradient_tape/autoencoder_12/sequential_19/dense_21/Tensordot/MatMul/MatMul_1MatMul�Jj�?!���8d�?"R
6autoencoder_12/sequential_19/dense_21/Tensordot/MatMulMatMul�(a��?!� _�7�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�����?!�sϟ0�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��
ў?!%�Z ('�?"
CastCast�ܤ�"��?!�$)���?Q      Y@Y�Q�/�~@@aW?��P@q�w匄�.@y��
Ѿ?"�
both�Your program is POTENTIALLY input-bound because 75.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�13.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�15.3838% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 