	ط���S5@ط���S5@!ط���S5@	f�˹@f�˹@!f�˹@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ط���S5@�"�J -@1whX��@A��v�$$�?I"���@Y�#��t��?*	    �~@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�sF���?!��2
#^N@)�ڊ�e��?1$�k�G@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map�q����?!�\*j��:@)㥛� ��?1�����-@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�(��?!��z���'@)�\m����?1����&@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate����镢?!�Ԧ6��@)�-����?1��t��@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate��ׁsF�?!��.@)�� �rh�?1�o�W��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj�t��?!�鑦�@)2U0*��?1��Er��	@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch2�%䃎?!W����Y@)2�%䃎?1W����Y@:Preprocessing2U
Iterator::Model::ParallelMapV2���_vO�?!��zI0@)���_vO�?1��zI0@:Preprocessing2F
Iterator::Model�X�� �?!:'�B[�@)V-��?1���;��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�/�$�?!h�J���P@)�j+��݃?1p7�İ��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����w?! �cG��?)�����w?1 �cG��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP�s�r?!�N��M�?)HP�s�r?1�N��M�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor����Mb`?!t��-&�?)����Mb`?1t��-&�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensora2U0*�S?!$<>fa�?)a2U0*�S?1$<>fa�?:Preprocessing2�
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlice-C��6J?!��~�W��?)-C��6J?1��~�W��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 68.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�16.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9f�˹@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�"�J -@�"�J -@!�"�J -@      ��!       "	whX��@whX��@!whX��@*      ��!       2	��v�$$�?��v�$$�?!��v�$$�?:	"���@"���@!"���@B      ��!       J	�#��t��?�#��t��?!�#��t��?R      ��!       Z	�#��t��?�#��t��?!�#��t��?JGPUYf�˹@b 