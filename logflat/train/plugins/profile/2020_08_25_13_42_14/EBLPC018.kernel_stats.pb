
À
W_Z18sgemm_largek_lds64ILb1ELb0ELi5ELi5ELi4ELi4ELi4ELi34EEvPfPKfS2_iiiiiiS2_S2_ffiiPiS3_*28È@ÈHÈXbKgradient_tape/autoencoder_13/sequential_21/dense_23/Tensordot/MatMul/MatMulh
«
W_Z18sgemm_largek_lds64ILb0ELb0ELi5ELi5ELi4ELi4ELi4ELi32EEvPfPKfS2_iiiiiiS2_S2_ffiiPiS3_*28Ø@ØHØXb6autoencoder_13/sequential_20/dense_22/Tensordot/MatMulh
i
sgemm_32x32x32_NN_vec*28ð@ðHðXb6autoencoder_13/sequential_21/dense_23/Tensordot/MatMulh
~
sgemm_32x32x32_TN_vec*28@HXbKgradient_tape/autoencoder_13/sequential_20/dense_22/Tensordot/MatMul/MatMulh
~
sgemm_32x32x32_TN_vec*28ð@ðHðbMgradient_tape/autoencoder_13/sequential_21/dense_23/Tensordot/MatMul/MatMul_1h

ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKdLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbCast_1h
î
­_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_28scalar_squared_difference_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28è@èHèb$mean_squared_error/SquaredDifferenceh
ç
¦_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ø@ØHØb$gradient_tape/mean_squared_error/subh

ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKdLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ø@ØHØbCasth

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 Ð@ ÐH Ðb$Adam/Adam/update_2/ResourceApplyAdamh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Ð@ÐHÐb"Adam/Adam/update/ResourceApplyAdamh
æ
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ð@ÐHÐb&gradient_tape/mean_squared_error/mul_1h
z
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28 @ H b-autoencoder_13/sequential_21/dense_23/BiasAddh
Ú
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28è@èHèb$gradient_tape/mean_squared_error/Mulh
à
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_18scalar_quotient_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28è@èHèb(gradient_tape/mean_squared_error/truedivh
²
O_ZN10tensorflow7functor24ColumnReduceSimpleKernelIPKfPfN3cub3SumEEEvT_T0_iiiT1_*28@HbGgradient_tape/autoencoder_13/sequential_21/dense_23/BiasAdd/BiasAddGradh
½
_ZN3cub27DeviceSegmentedReduceKernelINS_18DeviceReducePolicyIfiN10tensorflow7functor3SumIfEEE9Policy600EPfNS2_23TransformOutputIteratorIffNS3_9DividesByIffEExEENS_22TransformInputIteratorIiNS3_9RowOffsetENS_21CountingInputIteratorIixEExEEiS5_fEEvT0_T1_T2_SK_iT4_T5_*28@Hbmean_squared_error/Meanh
À
÷_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy2EEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28@Hb,gradient_tape/mean_squared_error/BroadcastToh
Ä
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIxxEEKNS4_INS5_IKxLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAdam/addh
Æ
÷_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy2EEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28X@XHXb5gradient_tape/mean_squared_error/weighted_loss/Tile_1h
ì
¤_ZN10tensorflow67_GLOBAL__N__43_dynamic_stitch_op_gpu_cu_compute_70_cpp1_ii_86968eec19DynamicStitchKernelIiEEviiNS_20GpuDeviceArrayStructIiLi8EEENS2_IPKT_Li8EEEPS4_*288@8H8b.gradient_tape/mean_squared_error/DynamicStitchh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*288@8H8b$Adam/Adam/update_3/ResourceApplyAdamh
Ì
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28(@(H(bGgradient_tape/autoencoder_13/sequential_20/dense_22/BiasAdd/BiasAddGradh
Î
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*28(@(H(bGgradient_tape/autoencoder_13/sequential_20/dense_22/BiasAdd/BiasAddGradh
÷
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(b?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanh
Á
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(bAdam/Powh
À
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(bMulh
¦
U_Z11scal_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28 @ H Xb6autoencoder_13/sequential_20/dense_22/Tensordot/MatMulh
»
U_Z11scal_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28 @ H XbKgradient_tape/autoencoder_13/sequential_21/dense_23/Tensordot/MatMul/MatMulh
w
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28 @ H b-autoencoder_13/sequential_20/dense_22/BiasAddh

:_ZN10tensorflow14GatherOpKernelIiiLb1EEEvPKT_PKT0_PS1_xxxx*28 @ H b:autoencoder_13/sequential_20/dense_22/Tensordot/GatherV2_1h

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 @ H b$Adam/Adam/update_1/ResourceApplyAdamh
±
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28 @ H b$mean_squared_error/weighted_loss/Sumh
Â
u_ZN10tensorflow7functor17BlockReduceKernelIPiS2_Li256ENS0_4ProdIiEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28 @ H b4autoencoder_13/sequential_20/dense_22/Tensordot/Prodh
±
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H bAssignAddVariableOph

:_ZN10tensorflow14GatherOpKernelIiiLb1EEEvPKT_PKT0_PS1_xxxx*28@Hb8autoencoder_13/sequential_21/dense_23/Tensordot/GatherV2h

:_ZN10tensorflow14GatherOpKernelIiiLb1EEEvPKT_PKT0_PS1_xxxx*28@Hb:autoencoder_13/sequential_21/dense_23/Tensordot/GatherV2_1h
Â
u_ZN10tensorflow7functor17BlockReduceKernelIPiS2_Li256ENS0_4ProdIiEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28@Hb4autoencoder_13/sequential_21/dense_23/Tensordot/Prodh
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb
LogicalAndh
Â
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb
div_no_nanh
Þ
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb&mean_squared_error/weighted_loss/valueh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb
Adam/Pow_1h

ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbCast_2h
°
ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb2mean_squared_error/weighted_loss/num_elements/Casth

ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKxLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAdam/Cast_1h
³
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAssignAddVariableOp_1h
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAdam/Adam/AssignAddVariableOph
³
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAssignAddVariableOp_2h

:_ZN10tensorflow14GatherOpKernelIiiLb1EEEvPKT_PKT0_PS1_xxxx*28@Hb8autoencoder_13/sequential_20/dense_22/Tensordot/GatherV2h
£
ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb%gradient_tape/mean_squared_error/Casth