Ѕљ	
тх
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ир
ё
sae-hidden-0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ѕђ*$
shared_namesae-hidden-0/kernel
}
'sae-hidden-0/kernel/Read/ReadVariableOpReadVariableOpsae-hidden-0/kernel* 
_output_shapes
:
ѕђ*
dtype0
ё
sae-hidden-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*$
shared_namesae-hidden-1/kernel
}
'sae-hidden-1/kernel/Read/ReadVariableOpReadVariableOpsae-hidden-1/kernel* 
_output_shapes
:
ђђ*
dtype0
Ѓ
sae-hidden-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*$
shared_namesae-hidden-2/kernel
|
'sae-hidden-2/kernel/Read/ReadVariableOpReadVariableOpsae-hidden-2/kernel*
_output_shapes
:	ђ@*
dtype0
Ј
classifier-hidden0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ**
shared_nameclassifier-hidden0/kernel
ѕ
-classifier-hidden0/kernel/Read/ReadVariableOpReadVariableOpclassifier-hidden0/kernel*
_output_shapes
:	@ђ*
dtype0
љ
classifier-hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ**
shared_nameclassifier-hidden1/kernel
Ѕ
-classifier-hidden1/kernel/Read/ReadVariableOpReadVariableOpclassifier-hidden1/kernel* 
_output_shapes
:
ђђ*
dtype0
Ѓ
activation-0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђv*$
shared_nameactivation-0/kernel
|
'activation-0/kernel/Read/ReadVariableOpReadVariableOpactivation-0/kernel*
_output_shapes
:	ђv*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
њ
Adam/sae-hidden-0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ѕђ*+
shared_nameAdam/sae-hidden-0/kernel/m
І
.Adam/sae-hidden-0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-0/kernel/m* 
_output_shapes
:
ѕђ*
dtype0
њ
Adam/sae-hidden-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*+
shared_nameAdam/sae-hidden-1/kernel/m
І
.Adam/sae-hidden-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-1/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Љ
Adam/sae-hidden-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*+
shared_nameAdam/sae-hidden-2/kernel/m
і
.Adam/sae-hidden-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-2/kernel/m*
_output_shapes
:	ђ@*
dtype0
Ю
 Adam/classifier-hidden0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*1
shared_name" Adam/classifier-hidden0/kernel/m
ќ
4Adam/classifier-hidden0/kernel/m/Read/ReadVariableOpReadVariableOp Adam/classifier-hidden0/kernel/m*
_output_shapes
:	@ђ*
dtype0
ъ
 Adam/classifier-hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*1
shared_name" Adam/classifier-hidden1/kernel/m
Ќ
4Adam/classifier-hidden1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/classifier-hidden1/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Љ
Adam/activation-0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђv*+
shared_nameAdam/activation-0/kernel/m
і
.Adam/activation-0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/activation-0/kernel/m*
_output_shapes
:	ђv*
dtype0
њ
Adam/sae-hidden-0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ѕђ*+
shared_nameAdam/sae-hidden-0/kernel/v
І
.Adam/sae-hidden-0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-0/kernel/v* 
_output_shapes
:
ѕђ*
dtype0
њ
Adam/sae-hidden-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*+
shared_nameAdam/sae-hidden-1/kernel/v
І
.Adam/sae-hidden-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-1/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Љ
Adam/sae-hidden-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*+
shared_nameAdam/sae-hidden-2/kernel/v
і
.Adam/sae-hidden-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-2/kernel/v*
_output_shapes
:	ђ@*
dtype0
Ю
 Adam/classifier-hidden0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*1
shared_name" Adam/classifier-hidden0/kernel/v
ќ
4Adam/classifier-hidden0/kernel/v/Read/ReadVariableOpReadVariableOp Adam/classifier-hidden0/kernel/v*
_output_shapes
:	@ђ*
dtype0
ъ
 Adam/classifier-hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*1
shared_name" Adam/classifier-hidden1/kernel/v
Ќ
4Adam/classifier-hidden1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/classifier-hidden1/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Љ
Adam/activation-0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђv*+
shared_nameAdam/activation-0/kernel/v
і
.Adam/activation-0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/activation-0/kernel/v*
_output_shapes
:	ђv*
dtype0

NoOpNoOp
љ5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╦4
value┴4BЙ4 Bи4
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
^

#kernel
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
^

,kernel
-	variables
.regularization_losses
/trainable_variables
0	keras_api
R
1	variables
2regularization_losses
3trainable_variables
4	keras_api
^

5kernel
6	variables
7regularization_losses
8trainable_variables
9	keras_api
┤
:iter

;beta_1

<beta_2
	=decay
>learning_ratem|m}m~#m,mђ5mЂvѓvЃvё#vЁ,vє5vЄ
*
0
1
2
#3
,4
55
 
*
0
1
2
#3
,4
55
Г
?layer_regularization_losses
@metrics
	variables
regularization_losses
Alayer_metrics
trainable_variables

Blayers
Cnon_trainable_variables
 
_]
VARIABLE_VALUEsae-hidden-0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Г
Dlayer_regularization_losses
Emetrics
	variables
regularization_losses
Flayer_metrics
trainable_variables

Glayers
Hnon_trainable_variables
_]
VARIABLE_VALUEsae-hidden-1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Г
Ilayer_regularization_losses
Jmetrics
	variables
regularization_losses
Klayer_metrics
trainable_variables

Llayers
Mnon_trainable_variables
_]
VARIABLE_VALUEsae-hidden-2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Г
Nlayer_regularization_losses
Ometrics
	variables
regularization_losses
Player_metrics
trainable_variables

Qlayers
Rnon_trainable_variables
 
 
 
Г
Slayer_regularization_losses
Tmetrics
	variables
 regularization_losses
Ulayer_metrics
!trainable_variables

Vlayers
Wnon_trainable_variables
ec
VARIABLE_VALUEclassifier-hidden0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

#0
 

#0
Г
Xlayer_regularization_losses
Ymetrics
$	variables
%regularization_losses
Zlayer_metrics
&trainable_variables

[layers
\non_trainable_variables
 
 
 
Г
]layer_regularization_losses
^metrics
(	variables
)regularization_losses
_layer_metrics
*trainable_variables

`layers
anon_trainable_variables
ec
VARIABLE_VALUEclassifier-hidden1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

,0
 

,0
Г
blayer_regularization_losses
cmetrics
-	variables
.regularization_losses
dlayer_metrics
/trainable_variables

elayers
fnon_trainable_variables
 
 
 
Г
glayer_regularization_losses
hmetrics
1	variables
2regularization_losses
ilayer_metrics
3trainable_variables

jlayers
knon_trainable_variables
_]
VARIABLE_VALUEactivation-0/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE

50
 

50
Г
llayer_regularization_losses
mmetrics
6	variables
7regularization_losses
nlayer_metrics
8trainable_variables

olayers
pnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1
 
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	stotal
	tcount
u	variables
v	keras_api
D
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

u	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

z	variables
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE Adam/classifier-hidden0/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE Adam/classifier-hidden1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/activation-0/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/sae-hidden-2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE Adam/classifier-hidden0/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE Adam/classifier-hidden1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/activation-0/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Є
"serving_default_sae-hidden-0_inputPlaceholder*(
_output_shapes
:         ѕ*
dtype0*
shape:         ѕ
М
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sae-hidden-0_inputsae-hidden-0/kernelsae-hidden-1/kernelsae-hidden-2/kernelclassifier-hidden0/kernelclassifier-hidden1/kernelactivation-0/kernel*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_236366
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Е
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'sae-hidden-0/kernel/Read/ReadVariableOp'sae-hidden-1/kernel/Read/ReadVariableOp'sae-hidden-2/kernel/Read/ReadVariableOp-classifier-hidden0/kernel/Read/ReadVariableOp-classifier-hidden1/kernel/Read/ReadVariableOp'activation-0/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/sae-hidden-0/kernel/m/Read/ReadVariableOp.Adam/sae-hidden-1/kernel/m/Read/ReadVariableOp.Adam/sae-hidden-2/kernel/m/Read/ReadVariableOp4Adam/classifier-hidden0/kernel/m/Read/ReadVariableOp4Adam/classifier-hidden1/kernel/m/Read/ReadVariableOp.Adam/activation-0/kernel/m/Read/ReadVariableOp.Adam/sae-hidden-0/kernel/v/Read/ReadVariableOp.Adam/sae-hidden-1/kernel/v/Read/ReadVariableOp.Adam/sae-hidden-2/kernel/v/Read/ReadVariableOp4Adam/classifier-hidden0/kernel/v/Read/ReadVariableOp4Adam/classifier-hidden1/kernel/v/Read/ReadVariableOp.Adam/activation-0/kernel/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_236710
ѕ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesae-hidden-0/kernelsae-hidden-1/kernelsae-hidden-2/kernelclassifier-hidden0/kernelclassifier-hidden1/kernelactivation-0/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/sae-hidden-0/kernel/mAdam/sae-hidden-1/kernel/mAdam/sae-hidden-2/kernel/m Adam/classifier-hidden0/kernel/m Adam/classifier-hidden1/kernel/mAdam/activation-0/kernel/mAdam/sae-hidden-0/kernel/vAdam/sae-hidden-1/kernel/vAdam/sae-hidden-2/kernel/v Adam/classifier-hidden0/kernel/v Adam/classifier-hidden1/kernel/vAdam/activation-0/kernel/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_236801┬П
ўw
»
"__inference__traced_restore_236801
file_prefix8
$assignvariableop_sae_hidden_0_kernel:
ѕђ:
&assignvariableop_1_sae_hidden_1_kernel:
ђђ9
&assignvariableop_2_sae_hidden_2_kernel:	ђ@?
,assignvariableop_3_classifier_hidden0_kernel:	@ђ@
,assignvariableop_4_classifier_hidden1_kernel:
ђђ9
&assignvariableop_5_activation_0_kernel:	ђv&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: B
.assignvariableop_15_adam_sae_hidden_0_kernel_m:
ѕђB
.assignvariableop_16_adam_sae_hidden_1_kernel_m:
ђђA
.assignvariableop_17_adam_sae_hidden_2_kernel_m:	ђ@G
4assignvariableop_18_adam_classifier_hidden0_kernel_m:	@ђH
4assignvariableop_19_adam_classifier_hidden1_kernel_m:
ђђA
.assignvariableop_20_adam_activation_0_kernel_m:	ђvB
.assignvariableop_21_adam_sae_hidden_0_kernel_v:
ѕђB
.assignvariableop_22_adam_sae_hidden_1_kernel_v:
ђђA
.assignvariableop_23_adam_sae_hidden_2_kernel_v:	ђ@G
4assignvariableop_24_adam_classifier_hidden0_kernel_v:	@ђH
4assignvariableop_25_adam_classifier_hidden1_kernel_v:
ђђA
.assignvariableop_26_adam_activation_0_kernel_v:	ђv
identity_28ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ц
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*░
valueдBБB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesИ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ё
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityБ
AssignVariableOpAssignVariableOp$assignvariableop_sae_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOp&assignvariableop_1_sae_hidden_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ф
AssignVariableOp_2AssignVariableOp&assignvariableop_2_sae_hidden_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3▒
AssignVariableOp_3AssignVariableOp,assignvariableop_3_classifier_hidden0_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_classifier_hidden1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ф
AssignVariableOp_5AssignVariableOp&assignvariableop_5_activation_0_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6А
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Б
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Б
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9б
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11А
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12А
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Б
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Б
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Х
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_sae_hidden_0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Х
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_sae_hidden_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Х
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_sae_hidden_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╝
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_classifier_hidden0_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╝
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_classifier_hidden1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Х
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_activation_0_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Х
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_sae_hidden_0_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Х
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_sae_hidden_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Х
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_sae_hidden_2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_classifier_hidden0_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╝
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_classifier_hidden1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Х
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_activation_0_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp░
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27Б
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї
Ѕ
3__inference_classifier-hidden1_layer_call_fn_236564

inputs
unknown:
ђђ
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_2360832
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_236092

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
І3
р
!__inference__wrapped_model_236006
sae_hidden_0_inputL
8sequential_1_sae_hidden_0_matmul_readvariableop_resource:
ѕђL
8sequential_1_sae_hidden_1_matmul_readvariableop_resource:
ђђK
8sequential_1_sae_hidden_2_matmul_readvariableop_resource:	ђ@Q
>sequential_1_classifier_hidden0_matmul_readvariableop_resource:	@ђR
>sequential_1_classifier_hidden1_matmul_readvariableop_resource:
ђђK
8sequential_1_activation_0_matmul_readvariableop_resource:	ђv
identityѕб/sequential_1/activation-0/MatMul/ReadVariableOpб5sequential_1/classifier-hidden0/MatMul/ReadVariableOpб5sequential_1/classifier-hidden1/MatMul/ReadVariableOpб/sequential_1/sae-hidden-0/MatMul/ReadVariableOpб/sequential_1/sae-hidden-1/MatMul/ReadVariableOpб/sequential_1/sae-hidden-2/MatMul/ReadVariableOpП
/sequential_1/sae-hidden-0/MatMul/ReadVariableOpReadVariableOp8sequential_1_sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
ѕђ*
dtype021
/sequential_1/sae-hidden-0/MatMul/ReadVariableOp╬
 sequential_1/sae-hidden-0/MatMulMatMulsae_hidden_0_input7sequential_1/sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2"
 sequential_1/sae-hidden-0/MatMulД
sequential_1/sae-hidden-0/ReluRelu*sequential_1/sae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2 
sequential_1/sae-hidden-0/ReluП
/sequential_1/sae-hidden-1/MatMul/ReadVariableOpReadVariableOp8sequential_1_sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype021
/sequential_1/sae-hidden-1/MatMul/ReadVariableOpУ
 sequential_1/sae-hidden-1/MatMulMatMul,sequential_1/sae-hidden-0/Relu:activations:07sequential_1/sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2"
 sequential_1/sae-hidden-1/MatMulД
sequential_1/sae-hidden-1/ReluRelu*sequential_1/sae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2 
sequential_1/sae-hidden-1/Relu▄
/sequential_1/sae-hidden-2/MatMul/ReadVariableOpReadVariableOp8sequential_1_sae_hidden_2_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype021
/sequential_1/sae-hidden-2/MatMul/ReadVariableOpу
 sequential_1/sae-hidden-2/MatMulMatMul,sequential_1/sae-hidden-1/Relu:activations:07sequential_1/sae-hidden-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2"
 sequential_1/sae-hidden-2/MatMulд
sequential_1/sae-hidden-2/ReluRelu*sequential_1/sae-hidden-2/MatMul:product:0*
T0*'
_output_shapes
:         @2 
sequential_1/sae-hidden-2/Relu«
sequential_1/dropout_3/IdentityIdentity,sequential_1/sae-hidden-2/Relu:activations:0*
T0*'
_output_shapes
:         @2!
sequential_1/dropout_3/IdentityЬ
5sequential_1/classifier-hidden0/MatMul/ReadVariableOpReadVariableOp>sequential_1_classifier_hidden0_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype027
5sequential_1/classifier-hidden0/MatMul/ReadVariableOpШ
&sequential_1/classifier-hidden0/MatMulMatMul(sequential_1/dropout_3/Identity:output:0=sequential_1/classifier-hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&sequential_1/classifier-hidden0/MatMul╣
$sequential_1/classifier-hidden0/ReluRelu0sequential_1/classifier-hidden0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2&
$sequential_1/classifier-hidden0/Reluх
sequential_1/dropout_4/IdentityIdentity2sequential_1/classifier-hidden0/Relu:activations:0*
T0*(
_output_shapes
:         ђ2!
sequential_1/dropout_4/Identity№
5sequential_1/classifier-hidden1/MatMul/ReadVariableOpReadVariableOp>sequential_1_classifier_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype027
5sequential_1/classifier-hidden1/MatMul/ReadVariableOpШ
&sequential_1/classifier-hidden1/MatMulMatMul(sequential_1/dropout_4/Identity:output:0=sequential_1/classifier-hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&sequential_1/classifier-hidden1/MatMul╣
$sequential_1/classifier-hidden1/ReluRelu0sequential_1/classifier-hidden1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2&
$sequential_1/classifier-hidden1/Reluх
sequential_1/dropout_5/IdentityIdentity2sequential_1/classifier-hidden1/Relu:activations:0*
T0*(
_output_shapes
:         ђ2!
sequential_1/dropout_5/Identity▄
/sequential_1/activation-0/MatMul/ReadVariableOpReadVariableOp8sequential_1_activation_0_matmul_readvariableop_resource*
_output_shapes
:	ђv*
dtype021
/sequential_1/activation-0/MatMul/ReadVariableOpс
 sequential_1/activation-0/MatMulMatMul(sequential_1/dropout_5/Identity:output:07sequential_1/activation-0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v2"
 sequential_1/activation-0/MatMul»
!sequential_1/activation-0/SigmoidSigmoid*sequential_1/activation-0/MatMul:product:0*
T0*'
_output_shapes
:         v2#
!sequential_1/activation-0/Sigmoid▒
IdentityIdentity%sequential_1/activation-0/Sigmoid:y:00^sequential_1/activation-0/MatMul/ReadVariableOp6^sequential_1/classifier-hidden0/MatMul/ReadVariableOp6^sequential_1/classifier-hidden1/MatMul/ReadVariableOp0^sequential_1/sae-hidden-0/MatMul/ReadVariableOp0^sequential_1/sae-hidden-1/MatMul/ReadVariableOp0^sequential_1/sae-hidden-2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2b
/sequential_1/activation-0/MatMul/ReadVariableOp/sequential_1/activation-0/MatMul/ReadVariableOp2n
5sequential_1/classifier-hidden0/MatMul/ReadVariableOp5sequential_1/classifier-hidden0/MatMul/ReadVariableOp2n
5sequential_1/classifier-hidden1/MatMul/ReadVariableOp5sequential_1/classifier-hidden1/MatMul/ReadVariableOp2b
/sequential_1/sae-hidden-0/MatMul/ReadVariableOp/sequential_1/sae-hidden-0/MatMul/ReadVariableOp2b
/sequential_1/sae-hidden-1/MatMul/ReadVariableOp/sequential_1/sae-hidden-1/MatMul/ReadVariableOp2b
/sequential_1/sae-hidden-2/MatMul/ReadVariableOp/sequential_1/sae-hidden-2/MatMul/ReadVariableOp:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
Ш
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_236553

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
╣
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_236083

inputs2
matmul_readvariableop_resource:
ђђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
љ
│
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_236474

inputs2
matmul_readvariableop_resource:
ѕђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ѕђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ѕ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
ё
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_236557

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ї
▓
H__inference_activation-0_layer_call_and_return_conditional_losses_236606

inputs1
matmul_readvariableop_resource:	ђv
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђv*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:         v2	
Sigmoidw
IdentityIdentitySigmoid:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
■
ѓ
-__inference_sae-hidden-2_layer_call_fn_236496

inputs
unknown:	ђ@
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_2360452
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬
F
*__inference_dropout_4_layer_call_fn_236543

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2360732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
╣
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_236572

inputs2
matmul_readvariableop_resource:
ђђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
і
ѕ
3__inference_classifier-hidden0_layer_call_fn_236530

inputs
unknown:	@ђ
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_2360642
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         @: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┬
F
*__inference_dropout_5_layer_call_fn_236577

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2360922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ђ
a
E__inference_dropout_3_layer_call_and_return_conditional_losses_236523

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ё
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_236142

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_236587

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ё
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_236591

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_236073

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
љ
│
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_236033

inputs2
matmul_readvariableop_resource:
ђђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
є)
Я
H__inference_sequential_1_layer_call_and_return_conditional_losses_236431

inputs?
+sae_hidden_0_matmul_readvariableop_resource:
ѕђ?
+sae_hidden_1_matmul_readvariableop_resource:
ђђ>
+sae_hidden_2_matmul_readvariableop_resource:	ђ@D
1classifier_hidden0_matmul_readvariableop_resource:	@ђE
1classifier_hidden1_matmul_readvariableop_resource:
ђђ>
+activation_0_matmul_readvariableop_resource:	ђv
identityѕб"activation-0/MatMul/ReadVariableOpб(classifier-hidden0/MatMul/ReadVariableOpб(classifier-hidden1/MatMul/ReadVariableOpб"sae-hidden-0/MatMul/ReadVariableOpб"sae-hidden-1/MatMul/ReadVariableOpб"sae-hidden-2/MatMul/ReadVariableOpХ
"sae-hidden-0/MatMul/ReadVariableOpReadVariableOp+sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
ѕђ*
dtype02$
"sae-hidden-0/MatMul/ReadVariableOpЏ
sae-hidden-0/MatMulMatMulinputs*sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-0/MatMulђ
sae-hidden-0/ReluRelusae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-0/ReluХ
"sae-hidden-1/MatMul/ReadVariableOpReadVariableOp+sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02$
"sae-hidden-1/MatMul/ReadVariableOp┤
sae-hidden-1/MatMulMatMulsae-hidden-0/Relu:activations:0*sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-1/MatMulђ
sae-hidden-1/ReluRelusae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-1/Reluх
"sae-hidden-2/MatMul/ReadVariableOpReadVariableOp+sae_hidden_2_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02$
"sae-hidden-2/MatMul/ReadVariableOp│
sae-hidden-2/MatMulMatMulsae-hidden-1/Relu:activations:0*sae-hidden-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
sae-hidden-2/MatMul
sae-hidden-2/ReluRelusae-hidden-2/MatMul:product:0*
T0*'
_output_shapes
:         @2
sae-hidden-2/ReluЄ
dropout_3/IdentityIdentitysae-hidden-2/Relu:activations:0*
T0*'
_output_shapes
:         @2
dropout_3/IdentityК
(classifier-hidden0/MatMul/ReadVariableOpReadVariableOp1classifier_hidden0_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02*
(classifier-hidden0/MatMul/ReadVariableOp┬
classifier-hidden0/MatMulMatMuldropout_3/Identity:output:00classifier-hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden0/MatMulњ
classifier-hidden0/ReluRelu#classifier-hidden0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden0/Reluј
dropout_4/IdentityIdentity%classifier-hidden0/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_4/Identity╚
(classifier-hidden1/MatMul/ReadVariableOpReadVariableOp1classifier_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02*
(classifier-hidden1/MatMul/ReadVariableOp┬
classifier-hidden1/MatMulMatMuldropout_4/Identity:output:00classifier-hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden1/MatMulњ
classifier-hidden1/ReluRelu#classifier-hidden1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden1/Reluј
dropout_5/IdentityIdentity%classifier-hidden1/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_5/Identityх
"activation-0/MatMul/ReadVariableOpReadVariableOp+activation_0_matmul_readvariableop_resource*
_output_shapes
:	ђv*
dtype02$
"activation-0/MatMul/ReadVariableOp»
activation-0/MatMulMatMuldropout_5/Identity:output:0*activation-0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v2
activation-0/MatMulѕ
activation-0/SigmoidSigmoidactivation-0/MatMul:product:0*
T0*'
_output_shapes
:         v2
activation-0/Sigmoidо
IdentityIdentityactivation-0/Sigmoid:y:0#^activation-0/MatMul/ReadVariableOp)^classifier-hidden0/MatMul/ReadVariableOp)^classifier-hidden1/MatMul/ReadVariableOp#^sae-hidden-0/MatMul/ReadVariableOp#^sae-hidden-1/MatMul/ReadVariableOp#^sae-hidden-2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2H
"activation-0/MatMul/ReadVariableOp"activation-0/MatMul/ReadVariableOp2T
(classifier-hidden0/MatMul/ReadVariableOp(classifier-hidden0/MatMul/ReadVariableOp2T
(classifier-hidden1/MatMul/ReadVariableOp(classifier-hidden1/MatMul/ReadVariableOp2H
"sae-hidden-0/MatMul/ReadVariableOp"sae-hidden-0/MatMul/ReadVariableOp2H
"sae-hidden-1/MatMul/ReadVariableOp"sae-hidden-1/MatMul/ReadVariableOp2H
"sae-hidden-2/MatMul/ReadVariableOp"sae-hidden-2/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
Ђ
Ѓ
-__inference_sae-hidden-1_layer_call_fn_236481

inputs
unknown:
ђђ
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_2360332
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
њ
И
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_236064

inputs1
matmul_readvariableop_resource:	@ђ
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         @: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
г@
ѕ
__inference__traced_save_236710
file_prefix2
.savev2_sae_hidden_0_kernel_read_readvariableop2
.savev2_sae_hidden_1_kernel_read_readvariableop2
.savev2_sae_hidden_2_kernel_read_readvariableop8
4savev2_classifier_hidden0_kernel_read_readvariableop8
4savev2_classifier_hidden1_kernel_read_readvariableop2
.savev2_activation_0_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_sae_hidden_0_kernel_m_read_readvariableop9
5savev2_adam_sae_hidden_1_kernel_m_read_readvariableop9
5savev2_adam_sae_hidden_2_kernel_m_read_readvariableop?
;savev2_adam_classifier_hidden0_kernel_m_read_readvariableop?
;savev2_adam_classifier_hidden1_kernel_m_read_readvariableop9
5savev2_adam_activation_0_kernel_m_read_readvariableop9
5savev2_adam_sae_hidden_0_kernel_v_read_readvariableop9
5savev2_adam_sae_hidden_1_kernel_v_read_readvariableop9
5savev2_adam_sae_hidden_2_kernel_v_read_readvariableop?
;savev2_adam_classifier_hidden0_kernel_v_read_readvariableop?
;savev2_adam_classifier_hidden1_kernel_v_read_readvariableop9
5savev2_adam_activation_0_kernel_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*░
valueдBБB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names└
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЁ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_sae_hidden_0_kernel_read_readvariableop.savev2_sae_hidden_1_kernel_read_readvariableop.savev2_sae_hidden_2_kernel_read_readvariableop4savev2_classifier_hidden0_kernel_read_readvariableop4savev2_classifier_hidden1_kernel_read_readvariableop.savev2_activation_0_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_sae_hidden_0_kernel_m_read_readvariableop5savev2_adam_sae_hidden_1_kernel_m_read_readvariableop5savev2_adam_sae_hidden_2_kernel_m_read_readvariableop;savev2_adam_classifier_hidden0_kernel_m_read_readvariableop;savev2_adam_classifier_hidden1_kernel_m_read_readvariableop5savev2_adam_activation_0_kernel_m_read_readvariableop5savev2_adam_sae_hidden_0_kernel_v_read_readvariableop5savev2_adam_sae_hidden_1_kernel_v_read_readvariableop5savev2_adam_sae_hidden_2_kernel_v_read_readvariableop;savev2_adam_classifier_hidden0_kernel_v_read_readvariableop;savev2_adam_classifier_hidden1_kernel_v_read_readvariableop5savev2_adam_activation_0_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Щ
_input_shapesУ
т: :
ѕђ:
ђђ:	ђ@:	@ђ:
ђђ:	ђv: : : : : : : : : :
ѕђ:
ђђ:	ђ@:	@ђ:
ђђ:	ђv:
ѕђ:
ђђ:	ђ@:	@ђ:
ђђ:	ђv: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ѕђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђ@:%!

_output_shapes
:	@ђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђv:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ѕђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђ@:%!

_output_shapes
:	@ђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђv:&"
 
_output_shapes
:
ѕђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђ@:%!

_output_shapes
:	@ђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђv:

_output_shapes
: 
ё
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_236165

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
І
▓
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_236045

inputs1
matmul_readvariableop_resource:	ђ@
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         @2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ђ
Ѓ
-__inference_sae-hidden-0_layer_call_fn_236466

inputs
unknown:
ѕђ
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_2360212
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ѕ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
љ
│
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_236021

inputs2
matmul_readvariableop_resource:
ѕђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ѕђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ѕ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
Ы
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_236054

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Р
Џ
-__inference_sequential_1_layer_call_fn_236400

inputs
unknown:
ѕђ
	unknown_0:
ђђ
	unknown_1:	ђ@
	unknown_2:	@ђ
	unknown_3:
ђђ
	unknown_4:	ђv
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2362592
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
┬
F
*__inference_dropout_5_layer_call_fn_236582

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2361422
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З*
▄
H__inference_sequential_1_layer_call_and_return_conditional_losses_236259

inputs'
sae_hidden_0_236237:
ѕђ'
sae_hidden_1_236240:
ђђ&
sae_hidden_2_236243:	ђ@,
classifier_hidden0_236247:	@ђ-
classifier_hidden1_236251:
ђђ&
activation_0_236255:	ђv
identityѕб$activation-0/StatefulPartitionedCallб*classifier-hidden0/StatefulPartitionedCallб*classifier-hidden1/StatefulPartitionedCallб$sae-hidden-0/StatefulPartitionedCallб$sae-hidden-1/StatefulPartitionedCallб$sae-hidden-2/StatefulPartitionedCallњ
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallinputssae_hidden_0_236237*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_2360212&
$sae-hidden-0/StatefulPartitionedCall╣
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_236240*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_2360332&
$sae-hidden-1/StatefulPartitionedCallИ
$sae-hidden-2/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-1/StatefulPartitionedCall:output:0sae_hidden_2_236243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_2360452&
$sae-hidden-2/StatefulPartitionedCall■
dropout_3/PartitionedCallPartitionedCall-sae-hidden-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2361882
dropout_3/PartitionedCallк
*classifier-hidden0/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0classifier_hidden0_236247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_2360642,
*classifier-hidden0/StatefulPartitionedCallЁ
dropout_4/PartitionedCallPartitionedCall3classifier-hidden0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2361652
dropout_4/PartitionedCallк
*classifier-hidden1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0classifier_hidden1_236251*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_2360832,
*classifier-hidden1/StatefulPartitionedCallЁ
dropout_5/PartitionedCallPartitionedCall3classifier-hidden1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2361422
dropout_5/PartitionedCallГ
$activation-0/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0activation_0_236255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation-0_layer_call_and_return_conditional_losses_2361022&
$activation-0/StatefulPartitionedCallэ
IdentityIdentity-activation-0/StatefulPartitionedCall:output:0%^activation-0/StatefulPartitionedCall+^classifier-hidden0/StatefulPartitionedCall+^classifier-hidden1/StatefulPartitionedCall%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall%^sae-hidden-2/StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2L
$activation-0/StatefulPartitionedCall$activation-0/StatefulPartitionedCall2X
*classifier-hidden0/StatefulPartitionedCall*classifier-hidden0/StatefulPartitionedCall2X
*classifier-hidden1/StatefulPartitionedCall*classifier-hidden1/StatefulPartitionedCall2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall2L
$sae-hidden-2/StatefulPartitionedCall$sae-hidden-2/StatefulPartitionedCall:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
о
ъ
$__inference_signature_wrapper_236366
sae_hidden_0_input
unknown:
ѕђ
	unknown_0:
ђђ
	unknown_1:	ђ@
	unknown_2:	@ђ
	unknown_3:
ђђ
	unknown_4:	ђv
identityѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_2360062
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
є	
Д
-__inference_sequential_1_layer_call_fn_236122
sae_hidden_0_input
unknown:
ѕђ
	unknown_0:
ђђ
	unknown_1:	ђ@
	unknown_2:	@ђ
	unknown_3:
ђђ
	unknown_4:	ђv
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2361072
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
ў+
У
H__inference_sequential_1_layer_call_and_return_conditional_losses_236316
sae_hidden_0_input'
sae_hidden_0_236294:
ѕђ'
sae_hidden_1_236297:
ђђ&
sae_hidden_2_236300:	ђ@,
classifier_hidden0_236304:	@ђ-
classifier_hidden1_236308:
ђђ&
activation_0_236312:	ђv
identityѕб$activation-0/StatefulPartitionedCallб*classifier-hidden0/StatefulPartitionedCallб*classifier-hidden1/StatefulPartitionedCallб$sae-hidden-0/StatefulPartitionedCallб$sae-hidden-1/StatefulPartitionedCallб$sae-hidden-2/StatefulPartitionedCallъ
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputsae_hidden_0_236294*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_2360212&
$sae-hidden-0/StatefulPartitionedCall╣
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_236297*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_2360332&
$sae-hidden-1/StatefulPartitionedCallИ
$sae-hidden-2/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-1/StatefulPartitionedCall:output:0sae_hidden_2_236300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_2360452&
$sae-hidden-2/StatefulPartitionedCall■
dropout_3/PartitionedCallPartitionedCall-sae-hidden-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360542
dropout_3/PartitionedCallк
*classifier-hidden0/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0classifier_hidden0_236304*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_2360642,
*classifier-hidden0/StatefulPartitionedCallЁ
dropout_4/PartitionedCallPartitionedCall3classifier-hidden0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2360732
dropout_4/PartitionedCallк
*classifier-hidden1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0classifier_hidden1_236308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_2360832,
*classifier-hidden1/StatefulPartitionedCallЁ
dropout_5/PartitionedCallPartitionedCall3classifier-hidden1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2360922
dropout_5/PartitionedCallГ
$activation-0/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0activation_0_236312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation-0_layer_call_and_return_conditional_losses_2361022&
$activation-0/StatefulPartitionedCallэ
IdentityIdentity-activation-0/StatefulPartitionedCall:output:0%^activation-0/StatefulPartitionedCall+^classifier-hidden0/StatefulPartitionedCall+^classifier-hidden1/StatefulPartitionedCall%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall%^sae-hidden-2/StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2L
$activation-0/StatefulPartitionedCall$activation-0/StatefulPartitionedCall2X
*classifier-hidden0/StatefulPartitionedCall*classifier-hidden0/StatefulPartitionedCall2X
*classifier-hidden1/StatefulPartitionedCall*classifier-hidden1/StatefulPartitionedCall2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall2L
$sae-hidden-2/StatefulPartitionedCall$sae-hidden-2/StatefulPartitionedCall:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
Ї
▓
H__inference_activation-0_layer_call_and_return_conditional_losses_236102

inputs1
matmul_readvariableop_resource:	ђv
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђv*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:         v2	
Sigmoidw
IdentityIdentitySigmoid:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Р
Џ
-__inference_sequential_1_layer_call_fn_236383

inputs
unknown:
ѕђ
	unknown_0:
ђђ
	unknown_1:	ђ@
	unknown_2:	@ђ
	unknown_3:
ђђ
	unknown_4:	ђv
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2361072
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
ў+
У
H__inference_sequential_1_layer_call_and_return_conditional_losses_236341
sae_hidden_0_input'
sae_hidden_0_236319:
ѕђ'
sae_hidden_1_236322:
ђђ&
sae_hidden_2_236325:	ђ@,
classifier_hidden0_236329:	@ђ-
classifier_hidden1_236333:
ђђ&
activation_0_236337:	ђv
identityѕб$activation-0/StatefulPartitionedCallб*classifier-hidden0/StatefulPartitionedCallб*classifier-hidden1/StatefulPartitionedCallб$sae-hidden-0/StatefulPartitionedCallб$sae-hidden-1/StatefulPartitionedCallб$sae-hidden-2/StatefulPartitionedCallъ
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputsae_hidden_0_236319*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_2360212&
$sae-hidden-0/StatefulPartitionedCall╣
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_236322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_2360332&
$sae-hidden-1/StatefulPartitionedCallИ
$sae-hidden-2/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-1/StatefulPartitionedCall:output:0sae_hidden_2_236325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_2360452&
$sae-hidden-2/StatefulPartitionedCall■
dropout_3/PartitionedCallPartitionedCall-sae-hidden-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2361882
dropout_3/PartitionedCallк
*classifier-hidden0/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0classifier_hidden0_236329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_2360642,
*classifier-hidden0/StatefulPartitionedCallЁ
dropout_4/PartitionedCallPartitionedCall3classifier-hidden0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2361652
dropout_4/PartitionedCallк
*classifier-hidden1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0classifier_hidden1_236333*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_2360832,
*classifier-hidden1/StatefulPartitionedCallЁ
dropout_5/PartitionedCallPartitionedCall3classifier-hidden1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2361422
dropout_5/PartitionedCallГ
$activation-0/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0activation_0_236337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation-0_layer_call_and_return_conditional_losses_2361022&
$activation-0/StatefulPartitionedCallэ
IdentityIdentity-activation-0/StatefulPartitionedCall:output:0%^activation-0/StatefulPartitionedCall+^classifier-hidden0/StatefulPartitionedCall+^classifier-hidden1/StatefulPartitionedCall%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall%^sae-hidden-2/StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2L
$activation-0/StatefulPartitionedCall$activation-0/StatefulPartitionedCall2X
*classifier-hidden0/StatefulPartitionedCall*classifier-hidden0/StatefulPartitionedCall2X
*classifier-hidden1/StatefulPartitionedCall*classifier-hidden1/StatefulPartitionedCall2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall2L
$sae-hidden-2/StatefulPartitionedCall$sae-hidden-2/StatefulPartitionedCall:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
њ
И
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_236538

inputs1
matmul_readvariableop_resource:	@ђ
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         @: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Й
F
*__inference_dropout_3_layer_call_fn_236509

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360542
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┬
F
*__inference_dropout_4_layer_call_fn_236548

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2361652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
■
ѓ
-__inference_activation-0_layer_call_fn_236598

inputs
unknown:	ђv
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation-0_layer_call_and_return_conditional_losses_2361022
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З*
▄
H__inference_sequential_1_layer_call_and_return_conditional_losses_236107

inputs'
sae_hidden_0_236022:
ѕђ'
sae_hidden_1_236034:
ђђ&
sae_hidden_2_236046:	ђ@,
classifier_hidden0_236065:	@ђ-
classifier_hidden1_236084:
ђђ&
activation_0_236103:	ђv
identityѕб$activation-0/StatefulPartitionedCallб*classifier-hidden0/StatefulPartitionedCallб*classifier-hidden1/StatefulPartitionedCallб$sae-hidden-0/StatefulPartitionedCallб$sae-hidden-1/StatefulPartitionedCallб$sae-hidden-2/StatefulPartitionedCallњ
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallinputssae_hidden_0_236022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_2360212&
$sae-hidden-0/StatefulPartitionedCall╣
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_236034*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_2360332&
$sae-hidden-1/StatefulPartitionedCallИ
$sae-hidden-2/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-1/StatefulPartitionedCall:output:0sae_hidden_2_236046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_2360452&
$sae-hidden-2/StatefulPartitionedCall■
dropout_3/PartitionedCallPartitionedCall-sae-hidden-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360542
dropout_3/PartitionedCallк
*classifier-hidden0/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0classifier_hidden0_236065*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_2360642,
*classifier-hidden0/StatefulPartitionedCallЁ
dropout_4/PartitionedCallPartitionedCall3classifier-hidden0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2360732
dropout_4/PartitionedCallк
*classifier-hidden1/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0classifier_hidden1_236084*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_2360832,
*classifier-hidden1/StatefulPartitionedCallЁ
dropout_5/PartitionedCallPartitionedCall3classifier-hidden1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2360922
dropout_5/PartitionedCallГ
$activation-0/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0activation_0_236103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation-0_layer_call_and_return_conditional_losses_2361022&
$activation-0/StatefulPartitionedCallэ
IdentityIdentity-activation-0/StatefulPartitionedCall:output:0%^activation-0/StatefulPartitionedCall+^classifier-hidden0/StatefulPartitionedCall+^classifier-hidden1/StatefulPartitionedCall%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall%^sae-hidden-2/StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2L
$activation-0/StatefulPartitionedCall$activation-0/StatefulPartitionedCall2X
*classifier-hidden0/StatefulPartitionedCall*classifier-hidden0/StatefulPartitionedCall2X
*classifier-hidden1/StatefulPartitionedCall*classifier-hidden1/StatefulPartitionedCall2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall2L
$sae-hidden-2/StatefulPartitionedCall$sae-hidden-2/StatefulPartitionedCall:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs
І
▓
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_236504

inputs1
matmul_readvariableop_resource:	ђ@
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         @2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ы
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_236519

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Й
F
*__inference_dropout_3_layer_call_fn_236514

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2361882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
љ
│
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_236489

inputs2
matmul_readvariableop_resource:
ђђ
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         ђ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
є	
Д
-__inference_sequential_1_layer_call_fn_236291
sae_hidden_0_input
unknown:
ѕђ
	unknown_0:
ђђ
	unknown_1:	ђ@
	unknown_2:	@ђ
	unknown_3:
ђђ
	unknown_4:	ђv
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         v*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2362592
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ѕ
,
_user_specified_namesae-hidden-0_input
Ђ
a
E__inference_dropout_3_layer_call_and_return_conditional_losses_236188

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ы%
Я
H__inference_sequential_1_layer_call_and_return_conditional_losses_236459

inputs?
+sae_hidden_0_matmul_readvariableop_resource:
ѕђ?
+sae_hidden_1_matmul_readvariableop_resource:
ђђ>
+sae_hidden_2_matmul_readvariableop_resource:	ђ@D
1classifier_hidden0_matmul_readvariableop_resource:	@ђE
1classifier_hidden1_matmul_readvariableop_resource:
ђђ>
+activation_0_matmul_readvariableop_resource:	ђv
identityѕб"activation-0/MatMul/ReadVariableOpб(classifier-hidden0/MatMul/ReadVariableOpб(classifier-hidden1/MatMul/ReadVariableOpб"sae-hidden-0/MatMul/ReadVariableOpб"sae-hidden-1/MatMul/ReadVariableOpб"sae-hidden-2/MatMul/ReadVariableOpХ
"sae-hidden-0/MatMul/ReadVariableOpReadVariableOp+sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
ѕђ*
dtype02$
"sae-hidden-0/MatMul/ReadVariableOpЏ
sae-hidden-0/MatMulMatMulinputs*sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-0/MatMulђ
sae-hidden-0/ReluRelusae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-0/ReluХ
"sae-hidden-1/MatMul/ReadVariableOpReadVariableOp+sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02$
"sae-hidden-1/MatMul/ReadVariableOp┤
sae-hidden-1/MatMulMatMulsae-hidden-0/Relu:activations:0*sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-1/MatMulђ
sae-hidden-1/ReluRelusae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
sae-hidden-1/Reluх
"sae-hidden-2/MatMul/ReadVariableOpReadVariableOp+sae_hidden_2_matmul_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02$
"sae-hidden-2/MatMul/ReadVariableOp│
sae-hidden-2/MatMulMatMulsae-hidden-1/Relu:activations:0*sae-hidden-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
sae-hidden-2/MatMul
sae-hidden-2/ReluRelusae-hidden-2/MatMul:product:0*
T0*'
_output_shapes
:         @2
sae-hidden-2/ReluК
(classifier-hidden0/MatMul/ReadVariableOpReadVariableOp1classifier_hidden0_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02*
(classifier-hidden0/MatMul/ReadVariableOpк
classifier-hidden0/MatMulMatMulsae-hidden-2/Relu:activations:00classifier-hidden0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden0/MatMulњ
classifier-hidden0/ReluRelu#classifier-hidden0/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden0/Relu╚
(classifier-hidden1/MatMul/ReadVariableOpReadVariableOp1classifier_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02*
(classifier-hidden1/MatMul/ReadVariableOp╠
classifier-hidden1/MatMulMatMul%classifier-hidden0/Relu:activations:00classifier-hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden1/MatMulњ
classifier-hidden1/ReluRelu#classifier-hidden1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2
classifier-hidden1/Reluх
"activation-0/MatMul/ReadVariableOpReadVariableOp+activation_0_matmul_readvariableop_resource*
_output_shapes
:	ђv*
dtype02$
"activation-0/MatMul/ReadVariableOp╣
activation-0/MatMulMatMul%classifier-hidden1/Relu:activations:0*activation-0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v2
activation-0/MatMulѕ
activation-0/SigmoidSigmoidactivation-0/MatMul:product:0*
T0*'
_output_shapes
:         v2
activation-0/Sigmoidо
IdentityIdentityactivation-0/Sigmoid:y:0#^activation-0/MatMul/ReadVariableOp)^classifier-hidden0/MatMul/ReadVariableOp)^classifier-hidden1/MatMul/ReadVariableOp#^sae-hidden-0/MatMul/ReadVariableOp#^sae-hidden-1/MatMul/ReadVariableOp#^sae-hidden-2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         v2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ѕ: : : : : : 2H
"activation-0/MatMul/ReadVariableOp"activation-0/MatMul/ReadVariableOp2T
(classifier-hidden0/MatMul/ReadVariableOp(classifier-hidden0/MatMul/ReadVariableOp2T
(classifier-hidden1/MatMul/ReadVariableOp(classifier-hidden1/MatMul/ReadVariableOp2H
"sae-hidden-0/MatMul/ReadVariableOp"sae-hidden-0/MatMul/ReadVariableOp2H
"sae-hidden-1/MatMul/ReadVariableOp"sae-hidden-1/MatMul/ReadVariableOp2H
"sae-hidden-2/MatMul/ReadVariableOp"sae-hidden-2/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ѕ
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_default▓
R
sae-hidden-0_input<
$serving_default_sae-hidden-0_input:0         ѕ@
activation-00
StatefulPartitionedCall:0         vtensorflow/serving/predict:Жъ
ћF
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
ѕ__call__
Ѕ_default_save_signature
+і&call_and_return_all_conditional_losses"хB
_tf_keras_sequentialќB{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sae-hidden-0_input"}}, {"class_name": "Dense", "config": {"name": "sae-hidden-0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "sae-hidden-1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "sae-hidden-2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "classifier-hidden0", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "classifier-hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "activation-0", "trainable": true, "dtype": "float32", "units": 118, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 520}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 520]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 520]}, "float32", "sae-hidden-0_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sae-hidden-0_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "sae-hidden-0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "sae-hidden-1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "sae-hidden-2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "classifier-hidden0", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "classifier-hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "activation-0", "trainable": true, "dtype": "float32", "units": 118, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 24}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╚	

kernel
	variables
regularization_losses
trainable_variables
	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"Ф
_tf_keras_layerЉ{"name": "sae-hidden-0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "sae-hidden-0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 520]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 520}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 520]}}
М

kernel
	variables
regularization_losses
trainable_variables
	keras_api
Ї__call__
+ј&call_and_return_all_conditional_losses"Х
_tf_keras_layerю{"name": "sae-hidden-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "sae-hidden-1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
м

kernel
	variables
regularization_losses
trainable_variables
	keras_api
Ј__call__
+љ&call_and_return_all_conditional_losses"х
_tf_keras_layerЏ{"name": "sae-hidden-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "sae-hidden-2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 
	variables
 regularization_losses
!trainable_variables
"	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 10}
Я

#kernel
$	variables
%regularization_losses
&trainable_variables
'	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses"├
_tf_keras_layerЕ{"name": "classifier-hidden0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "classifier-hidden0", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 
(	variables
)regularization_losses
*trainable_variables
+	keras_api
Ћ__call__
+ќ&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 14}
Р

,kernel
-	variables
.regularization_losses
/trainable_variables
0	keras_api
Ќ__call__
+ў&call_and_return_all_conditional_losses"┼
_tf_keras_layerФ{"name": "classifier-hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "classifier-hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 
1	variables
2regularization_losses
3trainable_variables
4	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 18}
┘

5kernel
6	variables
7regularization_losses
8trainable_variables
9	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses"╝
_tf_keras_layerб{"name": "activation-0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "activation-0", "trainable": true, "dtype": "float32", "units": 118, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
К
:iter

;beta_1

<beta_2
	=decay
>learning_ratem|m}m~#m,mђ5mЂvѓvЃvё#vЁ,vє5vЄ"
	optimizer
J
0
1
2
#3
,4
55"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
#3
,4
55"
trackable_list_wrapper
╬
?layer_regularization_losses
@metrics
	variables
regularization_losses
Alayer_metrics
trainable_variables

Blayers
Cnon_trainable_variables
ѕ__call__
Ѕ_default_save_signature
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
-
Юserving_default"
signature_map
':%
ѕђ2sae-hidden-0/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
░
Dlayer_regularization_losses
Emetrics
	variables
regularization_losses
Flayer_metrics
trainable_variables

Glayers
Hnon_trainable_variables
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
':%
ђђ2sae-hidden-1/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
░
Ilayer_regularization_losses
Jmetrics
	variables
regularization_losses
Klayer_metrics
trainable_variables

Llayers
Mnon_trainable_variables
Ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
&:$	ђ@2sae-hidden-2/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
░
Nlayer_regularization_losses
Ometrics
	variables
regularization_losses
Player_metrics
trainable_variables

Qlayers
Rnon_trainable_variables
Ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Slayer_regularization_losses
Tmetrics
	variables
 regularization_losses
Ulayer_metrics
!trainable_variables

Vlayers
Wnon_trainable_variables
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
,:*	@ђ2classifier-hidden0/kernel
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
░
Xlayer_regularization_losses
Ymetrics
$	variables
%regularization_losses
Zlayer_metrics
&trainable_variables

[layers
\non_trainable_variables
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
]layer_regularization_losses
^metrics
(	variables
)regularization_losses
_layer_metrics
*trainable_variables

`layers
anon_trainable_variables
Ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
-:+
ђђ2classifier-hidden1/kernel
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
░
blayer_regularization_losses
cmetrics
-	variables
.regularization_losses
dlayer_metrics
/trainable_variables

elayers
fnon_trainable_variables
Ќ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
glayer_regularization_losses
hmetrics
1	variables
2regularization_losses
ilayer_metrics
3trainable_variables

jlayers
knon_trainable_variables
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
&:$	ђv2activation-0/kernel
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
░
llayer_regularization_losses
mmetrics
6	variables
7regularization_losses
nlayer_metrics
8trainable_variables

olayers
pnon_trainable_variables
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
	stotal
	tcount
u	variables
v	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 30}
Ќ
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api"л
_tf_keras_metricх{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 24}
:  (2total
:  (2count
.
s0
t1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
,:*
ѕђ2Adam/sae-hidden-0/kernel/m
,:*
ђђ2Adam/sae-hidden-1/kernel/m
+:)	ђ@2Adam/sae-hidden-2/kernel/m
1:/	@ђ2 Adam/classifier-hidden0/kernel/m
2:0
ђђ2 Adam/classifier-hidden1/kernel/m
+:)	ђv2Adam/activation-0/kernel/m
,:*
ѕђ2Adam/sae-hidden-0/kernel/v
,:*
ђђ2Adam/sae-hidden-1/kernel/v
+:)	ђ@2Adam/sae-hidden-2/kernel/v
1:/	@ђ2 Adam/classifier-hidden0/kernel/v
2:0
ђђ2 Adam/classifier-hidden1/kernel/v
+:)	ђv2Adam/activation-0/kernel/v
ѓ2 
-__inference_sequential_1_layer_call_fn_236122
-__inference_sequential_1_layer_call_fn_236383
-__inference_sequential_1_layer_call_fn_236400
-__inference_sequential_1_layer_call_fn_236291└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
!__inference__wrapped_model_236006┬
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *2б/
-і*
sae-hidden-0_input         ѕ
Ь2в
H__inference_sequential_1_layer_call_and_return_conditional_losses_236431
H__inference_sequential_1_layer_call_and_return_conditional_losses_236459
H__inference_sequential_1_layer_call_and_return_conditional_losses_236316
H__inference_sequential_1_layer_call_and_return_conditional_losses_236341└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
О2н
-__inference_sae-hidden-0_layer_call_fn_236466б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_236474б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_sae-hidden-1_layer_call_fn_236481б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_236489б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_sae-hidden-2_layer_call_fn_236496б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_236504б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њ2Ј
*__inference_dropout_3_layer_call_fn_236509
*__inference_dropout_3_layer_call_fn_236514┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╚2┼
E__inference_dropout_3_layer_call_and_return_conditional_losses_236519
E__inference_dropout_3_layer_call_and_return_conditional_losses_236523┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
П2┌
3__inference_classifier-hidden0_layer_call_fn_236530б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Э2ш
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_236538б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њ2Ј
*__inference_dropout_4_layer_call_fn_236543
*__inference_dropout_4_layer_call_fn_236548┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╚2┼
E__inference_dropout_4_layer_call_and_return_conditional_losses_236553
E__inference_dropout_4_layer_call_and_return_conditional_losses_236557┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
П2┌
3__inference_classifier-hidden1_layer_call_fn_236564б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Э2ш
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_236572б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њ2Ј
*__inference_dropout_5_layer_call_fn_236577
*__inference_dropout_5_layer_call_fn_236582┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╚2┼
E__inference_dropout_5_layer_call_and_return_conditional_losses_236587
E__inference_dropout_5_layer_call_and_return_conditional_losses_236591┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
О2н
-__inference_activation-0_layer_call_fn_236598б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation-0_layer_call_and_return_conditional_losses_236606б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
оBМ
$__inference_signature_wrapper_236366sae-hidden-0_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Е
!__inference__wrapped_model_236006Ѓ#,5<б9
2б/
-і*
sae-hidden-0_input         ѕ
ф ";ф8
6
activation-0&і#
activation-0         vе
H__inference_activation-0_layer_call_and_return_conditional_losses_236606\50б-
&б#
!і
inputs         ђ
ф "%б"
і
0         v
џ ђ
-__inference_activation-0_layer_call_fn_236598O50б-
&б#
!і
inputs         ђ
ф "і         v«
N__inference_classifier-hidden0_layer_call_and_return_conditional_losses_236538\#/б,
%б"
 і
inputs         @
ф "&б#
і
0         ђ
џ є
3__inference_classifier-hidden0_layer_call_fn_236530O#/б,
%б"
 і
inputs         @
ф "і         ђ»
N__inference_classifier-hidden1_layer_call_and_return_conditional_losses_236572],0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ Є
3__inference_classifier-hidden1_layer_call_fn_236564P,0б-
&б#
!і
inputs         ђ
ф "і         ђЦ
E__inference_dropout_3_layer_call_and_return_conditional_losses_236519\3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ Ц
E__inference_dropout_3_layer_call_and_return_conditional_losses_236523\3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ }
*__inference_dropout_3_layer_call_fn_236509O3б0
)б&
 і
inputs         @
p 
ф "і         @}
*__inference_dropout_3_layer_call_fn_236514O3б0
)б&
 і
inputs         @
p
ф "і         @Д
E__inference_dropout_4_layer_call_and_return_conditional_losses_236553^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ Д
E__inference_dropout_4_layer_call_and_return_conditional_losses_236557^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ 
*__inference_dropout_4_layer_call_fn_236543Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ
*__inference_dropout_4_layer_call_fn_236548Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђД
E__inference_dropout_5_layer_call_and_return_conditional_losses_236587^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ Д
E__inference_dropout_5_layer_call_and_return_conditional_losses_236591^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ 
*__inference_dropout_5_layer_call_fn_236577Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ
*__inference_dropout_5_layer_call_fn_236582Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђЕ
H__inference_sae-hidden-0_layer_call_and_return_conditional_losses_236474]0б-
&б#
!і
inputs         ѕ
ф "&б#
і
0         ђ
џ Ђ
-__inference_sae-hidden-0_layer_call_fn_236466P0б-
&б#
!і
inputs         ѕ
ф "і         ђЕ
H__inference_sae-hidden-1_layer_call_and_return_conditional_losses_236489]0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ Ђ
-__inference_sae-hidden-1_layer_call_fn_236481P0б-
&б#
!і
inputs         ђ
ф "і         ђе
H__inference_sae-hidden-2_layer_call_and_return_conditional_losses_236504\0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         @
џ ђ
-__inference_sae-hidden-2_layer_call_fn_236496O0б-
&б#
!і
inputs         ђ
ф "і         @┴
H__inference_sequential_1_layer_call_and_return_conditional_losses_236316u#,5DбA
:б7
-і*
sae-hidden-0_input         ѕ
p 

 
ф "%б"
і
0         v
џ ┴
H__inference_sequential_1_layer_call_and_return_conditional_losses_236341u#,5DбA
:б7
-і*
sae-hidden-0_input         ѕ
p

 
ф "%б"
і
0         v
џ х
H__inference_sequential_1_layer_call_and_return_conditional_losses_236431i#,58б5
.б+
!і
inputs         ѕ
p 

 
ф "%б"
і
0         v
џ х
H__inference_sequential_1_layer_call_and_return_conditional_losses_236459i#,58б5
.б+
!і
inputs         ѕ
p

 
ф "%б"
і
0         v
џ Ў
-__inference_sequential_1_layer_call_fn_236122h#,5DбA
:б7
-і*
sae-hidden-0_input         ѕ
p 

 
ф "і         vЎ
-__inference_sequential_1_layer_call_fn_236291h#,5DбA
:б7
-і*
sae-hidden-0_input         ѕ
p

 
ф "і         vЇ
-__inference_sequential_1_layer_call_fn_236383\#,58б5
.б+
!і
inputs         ѕ
p 

 
ф "і         vЇ
-__inference_sequential_1_layer_call_fn_236400\#,58б5
.б+
!і
inputs         ѕ
p

 
ф "і         v┬
$__inference_signature_wrapper_236366Ў#,5RбO
б 
HфE
C
sae-hidden-0_input-і*
sae-hidden-0_input         ѕ";ф8
6
activation-0&і#
activation-0         v