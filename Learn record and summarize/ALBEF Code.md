# configs

- config_bert.json

  它是一个 BERT 模型的配置文件，其作用是定义Bert模型的超参数和架构

- Grounding.yaml

  它是一个 YAML 格式的配置文件，主要用于配置模型训练和测试的各种参数

  指定训练集和测试集的数据文件路径、 BERT 模型的配置文件路径、图片的分辨率、训练时的批量大小、优化器和学习率调度器的类型和参数设置等

- NLVR_pretrain.yaml

  它是一个 YAML 格式的配置文件，用于配置模型的预训练任务 NLVR（Natural Language Visual Reasoning）

  指定用于预训练的数据集文件列表等等

- NLVR.yaml

  这里是一个自然语言推理数据集NLVR2的训练配置文件

  "NLVR_pretrain.yaml" 是用于预训练模型的，而"NLVR.yaml" 是用于针对 NLVR 数据集 fine-tune 预训练好的模型。具体来说，"NLVR_pretrain.yaml" 首先从多个数据集（如 COCO、VG、CC3M、SBU）中收集大量的图片和相应的文字描述，用于预训练一个图像文字匹配模型。而 "NLVR.yaml" 中则只包含 NLVR 数据集的训练、验证和测试数据文件路径、优化器、学习率调度器以及其他一些超参数，它用于训练和评估预训练好的图像文字匹配模型在 NLVR 数据集上的表现。因此，"NLVR.yaml" 是 fine-tuning 阶段的配置文件，而"NLVR_pretrain.yaml" 是预训练阶段的配置文件

- Pretrain.yaml

  ALBEF 模型预训练配置文件

  它定义了数据输入、Bert模型的配置以及优化器的超参数等

- Retrieval_coco.yaml

  它是一个用于图像检索任务的配置文件，检索的数据集是coco

- Retrieval_flickr.yaml

  它是一个用于图像检索任务的配置文件，检索的数据集是flickr

- VE.yaml

  它是用于训练图像和自然语言联合理解任务的视觉表征 (VE) 模型的配置文件

- VQA.yaml

  用于指定图像和文本之间的视觉问答（Visual Question Answering，VQA）任务中的训练、验证和测试数据的位置和特征提取器的超参数等相关参数

  

# dataset

- init.py

  定义了几个函数来创建不同的PyTorch数据集，以及处理它们的Dataloader。这些数据集包括预训练(pretrain)、图像推理(re)、视觉问答(vqa)、自然语言推理(nlvr)、视觉推理(ve)和图像标注(grounding)

  不同数据集的创建方法略有不同，但它们都采用了一些常见的数据增强技术，例如随机裁剪(RandomResizedCrop)、随机翻转(RandomHorizontalFlip)和随机增强(RandomAugment)。

  此外，该代码还定义了一个自定义的collate函数(vqa_collate_fn)，以处理VQA数据集的批处理和权重。该函数的作用是遍历batch中的每个数据点，并将图像、问题和权重分别添加到其相应的列表中。回答列表是通过将每个数据点的可能答案列表加入到一个大列表中得到的

  最后，该代码定义了两个函数(create_sampler和create_loader)，用于创建并行数据加载器，以便在多个GPU上训练模型。这些函数都涉及到使用PyTorch内置的DistributedSampler和DataLoader类来处理数据并进行并行训练。

- caption_dataset.py

  re_train_dataset类用于重新训练图像检索模型。初始化函数加载一个或多个注释文件（ann_file），并将所有注释存储在self.ann列表中。除此之外，还初始化了一些其他变量，如变换（transform）、图像根路径（image_root）和最大单词数（max_words）等。__len__方法返回self.ann的长度，即数据集中注释的数量。__getitem__方法使用给定的索引（index）获取对应的注释，加载对应的图像文件并应用变换，最后返回图像、标题和图像ID。

  re_eval_dataset类用于重新评估图像检索模型。它初始化方式与re_train_dataset类相似，但是只加载一个注释文件（ann_file）。__len__方法返回数据集中图像的数量。__getitem__方法使用给定的索引（index）获取对应的注释，加载对应的图像文件并应用变换，最后返回图像和图像ID。

  pretrain_dataset类用于预训练图像检索模型。初始化函数与re_train_dataset类类似，但是不需要图像根路径（image_root）。__len__方法返回self.ann的长度，即数据集中注释的数量。__getitem__方法使用给定的索引（index）获取对应的注释，加载对应的图像文件并应用变换，最后返回图像和标题。如果注释中的标题是列表，则从列表中随机选择一个标题。

- grounding_dataset.py

  这段代码定义了一个名为grounding_dataset的PyTorch数据集类。这个数据集类的主要作用是为图像文本相关性任务提供数据，包括训练和测试模式。通过加载注释文件和图像文件，并应用变换和标题转换，将数据准备好以供模型使用。

  初始化函数接收一个或多个注释文件路径（ann_file）、变换（transform）、图像根路径（image_root）、最大单词数（max_words）和模式（mode），默认为train模式。在初始化过程中，首先加载所有注释并将其存储在self.ann列表中。然后，根据模式（mode），初始化不同的变量，如果是train模式，则初始化img_ids字典，用于存储图像ID和对应的索引。最后，将初始化的变量存储在实例变量中。

  __len__方法返回self.ann的长度，即数据集中注释的数量。__getitem__方法使用给定的索引（index）获取对应的注释，加载对应的图像文件并应用变换，最后返回图像、标题和图像ID（仅在训练模式下）或引用ID（在测试模式下）。

  其中，注释中的图像路径（ann['image']）是相对于图像根路径（image_root）的相对路径，因此需要使用os.path.join将它们组合在一起。pre_caption函数用于将标题转换为整数列表。如果是训练模式，将返回图像、标题和对应的图像ID，否则将返回图像、标题和对应的引用ID。 

- nlvr_dataset.py

  这段代码定义了一个名为nlvr_dataset的自定义数据集类，用于加载NLVR（Natural Language for Visual Reasoning）数据集中的图像和相应的自然语言描述，并将其用于视觉推理任务。该类的__getitem__方法返回的是一个包含图像、自然语言描述和标签的元组，其中图像和自然语言描述用于输入模型，标签用于训练和评估模型。

- ve_dataset.py

  自定义数据集类ve_dataset，用于加载视觉推理数据集中的数据，包括图片、描述句子和标签

- vqa_dataset.py

  该代码实现了一个 VQA 数据集的 PyTorch Dataset 类，其中包括以下方法：

  1. `__init__`: 构造函数，用于初始化数据集参数和读取注释文件。
  2. `__len__`: 返回数据集的大小，即注释文件中的样本数。
  3. `__getitem__`: 根据索引返回数据集中的样本，包括图像、问题、答案及答案权重（训练集中）或问题 ID（测试集中）。

- randaugment.py

  这段代码定义了一系列图像变换操作(函数),并封装成 RandomAugment 类,用于随机组合这些操作进行数据增广。

  定义了Identity,AutoContrast,Equalize,Rotate,Solarize,Color,Contrast,Brightness,Sharpness,ShearX,TranslateX,TranslateY,Posterize,ShearY 等 14 个图像变换操作函数。func_dict 字典存储了这 14 个函数的映射关系。arg_dict 字典存储了这 14 个函数的参数生成函数的映射关系。这些参数生成函数用于根据增强级别 level 生成相应函数的参数

  RandomAugment 类初始化时可选择 augs 参数指定要使用的图像变换操作函数的名称列表。否则默认使用所有的 14 个操作函数。RandomAugment.call 方法中使用 np.random.choice 随机选择 N 个操作函数,并使用arg_dict 生成这 N 个函数的参数。然后逐个调用这 N 个函数对输入图像 img 进行变换,实现随机数据增广的效果。如果传入 isPIL=True,则会先将 PIL.Image 的图像数据转换成 NumPy 数组进行处理。

- utils.py

  该代码实现了 VQA 和 Grouding 任务过程中的一些常用功能和指标计算。

  该代码实现了 question 和 caption 的预处理,包括小写化、去除标点符号、使用空格替换“-”和“/”等操作。对 VQA 的结果进行评估,计算准确率等指标。收集所有进程/设备的结果,并保存到文件中。这部分实现了对分布式训练的支持。实现了 Grouding 结果的评估,计算 IoU 大于 0.5 的准确率等指标。IoU 计算函数的实现。



# models

- model_nlvr.py

- model_pretrain_nlvr.py

- model_pretrain.py

  ALBEF 模型预训练

  定义视觉编码器，文本编码器，动量模型，前向过程

- model_retrieval.py

- model_ve.py

- model_vqa.py

  针对 VQA 问题的 ALBEF 模型预训练

- tokenization_bert.py

- vit.py

  这段代码实现了Vision Transformer模型,主要包含以下部分:

  1. PatchEmbed:实现patch embedding,即将image分割成patch,并embedding为patch embedding。
  2. Positional Embedding:为序列元素加入绝对位置编码,以编码序列中元素的相对位置信息。
  3. Attention:实现attention机制,包括多头注意力计算以及前向传播。
  4. Mlp:实现MLP中两个全连接层的前向传播。
  5. Block:实现Transformers的Encoder Block,包含LayerNorm、Attention、MLP三个子层。
  6. VisionTransformer:组合以上模块,实现整体模型的前向传播。
  7. interpolate_pos_embed: 用于在finetune时interpolate position embedding,使之匹配当前的图片大小与patch数。
  8. 其他:还包含残差连接、层归一化等技术。

- xbert.py

  实现 BERT 模型



# optim

- adafactor.py

  Adafactor：它是一种自适应学习率优化器，可以在保持速度和性能的同时，提高模型的泛化能力。Adafactor优化器的特点是自适应地调整学习率和梯度修正。

- adahessian.py

  Adahessian：它是一种二阶优化器，可以在保持速度和性能的同时，提高模型的泛化能力。Adahessian优化器的特点是利用二阶信息来更新模型参数，从而提高模型的收敛速度和泛化能力。

- adamp.py

  AdamP：它是Adam优化器的一个改进版本，主要是在Adam优化器的基础上加入了权重衰减和自适应梯度修正。AdamP优化器的目标是在保持Adam优化器的速度和性能的同时，提高模型的泛化能力。

- adamw.py

  AdamW：它是Adam优化器的一个变种，主要是在Adam优化器的基础上加入了权重衰减。AdamW优化器的目标是在Adam优化器的基础上，解决过拟合问题。

- lookahead.py

  Lookahead：它是一种优化器的wrapper，可以在保持速度和性能的同时，提高模型的泛化能力。Lookahead优化器的特点是在原有优化器的基础上，加入了一种lookahead机制，使得模型能够更好地探索参数空间。

- nadam.py

  Nadam：它是一种Adam优化器的变种，主要是在Adam优化器的基础上加入了Nesterov动量。Nadam优化器的目标是在Adam优化器的基础上，进一步提高模型的收敛速度和泛化能力。

- novograd.py

  NovoGrad：它是一种自适应梯度优化器，可以在保持速度和性能的同时，提高模型的泛化能力。NovoGrad优化器的特点是自适应地调整梯度修正，并且采用了类似于Adam优化器的动量。

- nvnovograd.py

  NvNovoGrad：它是一种NovoGrad优化器的变种，主要是在NovoGrad优化器的基础上加入了Nesterov动量。NvNovoGrad优化器的目标是在NovoGrad优化器的基础上，进一步提高模型的收敛速度和泛化能力。

- optim_factory.py

- radam.py

  RAdam：它是一种自适应学习率优化器，可以在保持速度和性能的同时，提高模型的泛化能力。RAdam优化器的特点是在原有优化器的基础上，加入了一种自适应学习率机制，使得模型能够更好地探索参数空间。

- rmsprop_tf.py

  RMSpropTF：它是一种优化器的wrapper，可以在保持速度和性能的同时，提高模型的泛化能力。RMSpropTF优化器的特点是在原有优化器的基础上，加入了一种自适应学习率机制，使得模型能够更好地探索参数空间。

- sgdp.py

  SGDP：它是一种带有权重衰减的SGD优化器，可以在保持速度和性能的同时，提高模型的泛化能力。SGDP优化器的特点是在原有SGD优化器的基础上，加入了一种权重衰减机制，使得模型能够更好地控制过拟合。



# refTools

一个参考指标



# scheduler

- cosine_lr.py

  定义了一个余弦退火学习率调度器类 CosineLRScheduler，可以用于深度学习模型的训练中，动态调整学习率以提高模型的收敛速度和泛化能力。

  在构造函数中，调度器会对部分参数进行检查和计算，如判断周期长度、学习率最小值是否合法，计算预热阶段的学习率值和倍增的周期数等。

  该调度器的核心函数是 _get_lr(self, t)，它根据当前的周期长度 t，计算出对应的学习率值。如果当前周期处于预热阶段，学习率会逐步从预热初始值增加到目标值；否则，会根据周期长度计算出当前所在的周期，进而计算出学习率的衰减系数、最小值和最大值，并根据余弦函数计算出当前的学习率值。

  除此之外，该调度器还实现了其他函数，如 get_epoch_values 和 get_update_values，用于在训练过程中获取当前 epoch 或更新次数对应的学习率值，以及 get_cycle_length，用于获取周期长度。

- plateau_lr.py

  实现了一个基于验证集损失的学习率调度器类 PlateauLRScheduler，可以用于深度学习模型的训练中，在验证集损失停滞时动态调整学习率以提高模型的收敛速度和泛化能力。并且该调度器支持学习率预热和学习率扰动。

  该调度器类继承了 PyTorch 自带的学习率调度器类 torch.optim.lr_scheduler.ReduceLROnPlateau，并在其基础上实现了学习率预热和学习率扰动的功能。该调度器类的 step 方法会根据当前的训练周期和验证集损失更新学习率，如果当前周期小于等于预热周期，则会根据预热步长和预热初始值更新学习率；否则，会使用 torch.optim.lr_scheduler.ReduceLROnPlateau 的 step 方法更新学习率，并根据学习率扰动参数在指定周期范围内对学习率进行扰动。扰动的方式根据 noise_type 参数指定，可以是正态分布噪声或均匀分布噪声。

- scheduler_factory.py

  该代码定义了一个名为“create_scheduler”的函数，该函数接受两个参数：args和optimizer。args是一个命名空间对象，其中包含有关训练过程的各种参数。optimizer是一个PyTorch优化器对象，用于更新模型参数。

  函数首先从args参数中提取一些值，并将它们分配给变量。然后，根据args参数中给定的调度器类型，函数创建一个相应的学习率调度器对象。四种支持的调度器类型分别是：cosine、tanh、step和plateau。

  如果调度器类型是cosine或tanh，则函数使用相应的类来创建一个新的调度器对象。这些调度器使用余弦退火和tanh函数来调整学习率。如果调度器类型是step，则函数使用StepLRScheduler类创建一个新的调度器对象。这个调度器使用分段常数函数来调整学习率。如果调度器类型是plateau，则函数使用PlateauLRScheduler类创建一个新的调度器对象。这个调度器使用一个类似于指数移动平均的方法来调整学习率。

  最后，函数返回一个包含新创建的学习率调度器对象和num_epochs（根据所选调度器类型和args参数计算的训练周期数）的元组。

- scheduler.py

  这是一个PyTorch的参数调度器基类，可以用来调度任何优化器参数组。这个基类的调度器是为了减少内置的PyTorch调度器中“last_epoch”和-1值特殊行为的混淆而设计的。所有的epoch和update计数都必须在训练代码中跟踪，并在相应的step或step_update调用中显式传递给调度器。

  这个基类的设计是为了让子类尽可能地保持无状态，以便简化其实现。子类应该实现get_epoch_values和get_update_values来提供适当的参数组值。然后，调用step或step_update来更新每个参数组的当前值。在这些函数中，子类可以使用update_groups函数来将更新后的值应用到每个参数组中。如果需要，子类也可以重写_add_noise函数来添加噪声。

- step_lr.py

  该代码实现了一个基本的 step LR 调度程序，包括渐进、噪声等功能。具体来说，这个调度程序会在训练过程中按照指定的步骤对学习率进行调整，以达到更好的训练效果。

- tanh_lr.py

  这段代码实现了一个基于双曲正切函数（Tanh）的学习率调度器，该调度器可以进行学习率的warmup、周期性衰减、噪声等操作

  

# vqaTools

- vqa.py

  这段代码定义了一个VQA类，用于处理和访问VQA数据集。VQA数据集包含问题和答案对，每个问题有多个答案。该类提供了以下主要功能：

  1. 初始化函数：从VQA注释文件和问题文件中加载数据集并创建索引
  2. 创建索引函数：创建数据集的索引
  3. 获取符合给定过滤条件的问题ID的函数
  4. 获取符合给定过滤条件的图像ID的函数
  5. 加载指定问题ID的问题和答案函数
  6. 显示指定注释函数
  7. 加载结果文件并创建结果对象

  该类的实现方式是，通过在创建索引时将注释数据和问题数据组合在一起，使其易于查询和访问。此外，还提供了函数来获取符合给定过滤条件的问题和图像ID。最后，还提供了函数来加载结果文件并创建结果对象。

- vqaEval.py

  该代码实现VQA(视觉问题回答)的评估器。具体做法如下:

  1. 实例化VQAEval类,传入vqa(视觉问题回答真实数据)和vqaRes(视觉问题回答模型生成的结果)。同时指定除法精度n。

  2. 定义 contractions( contractions = {"aint": "ain't", ...) ,用于contracted form的映射。

  3. 定义 punct(用于标点符号的列表),articles(所),manualMap(数字与文字的映射)等辅助变量。

  4. 定义 processPunctuation 和 processDigitArticle 方法,用于处理标点符号和数字与文字的对应,以标准化答案。

  5. evaluate方法,用于评估准确率。首先获取需要评估的问题标识quesIds。然后获取每个问题的真实答案gts和结果答案res。

  6. 计算准确率: accQA:所有问题的准确率平均值。accQuesType:按问题类型划分的准确率。accAnsType:按答案类型划分的准确率。

  7. 根据准确率结果,设置accuracy,evalQA,evalQuesType和evalAnsType属性。

  8. 在计算过程中使用updateProgress方法展示进度。

  9. processPunctuation和processDigitArticle方法用于数据标准化,以减少答案中的标点符号和数字与文字的对应影响对准确率判断的影响。

     

# Main Function

- Grounding.py

  这段代码主要完成如下几个任务:

  1. 解析命令行参数,对训练进行配置。参数包括模型配置文件路径、 checkpoint 重载文件路径、输出目录、gradcam 模式、block序号(用于选择需要 visualisation 的block)、text encoder 类型、是否仅评估模型、使用的设备、随机种子、分布式训练相关参数等。
  2. 初始化分布式训练环境(如果使用)并设置种子用于 reproducibility。
  3. 创建训练和验证数据集,如果使用分布式训练也会创建采样器。
  4. 创建ALBEF 模型。如果提供 checkpoint 会加载 checkpoint 进行模型重载。并将模型移到指定设备。如果使用分布式训练会进行包装。
  5. 设置优化器、学习率调度器。根据配置创建。
  6. 训练循环。每个epoch会进行训练、评估两步。训练使用批次循环并更新参数。评估会产生gradcam结果。并计算各种metric 进行评估。在验证指标上有提升会保存模型。
  7. 如果仅评估 True 会直接进入评估阶段,不进行训练。
  8. 计算并打印整体训练时间。
  9. 相关结果保存到 output_dir 中,包括损失 logs 、模型 checkpoint 、gradcam结果等。

  总的来说,这段代码实现了RefCOCO数据集上的ALBEF模型的训练与评估,同时可选择产生gradcam可视化结果用于理解。

- NLVR.py

  NLVR(Natural Language Visual Reasoning)模型的训练程序,实现非回复性阅读理解。

  1. 解析命令行参数,包括配置文件、检测点、输出目录、设备、种子等参数。这些参数用于控制训练过程和结果保存。
  2. 从配置文件中加载NLVR模型配置参数。配置参数控制训练详细设置,如epochs、warmup_steps、optimizer类型等。
  3. 在训练设备(通常cuda device)上设置随机种子,用于保证训练结果可 reproducability。
  4. 创建NLVR数据集及其DataLoader。如果使用分布式训练,需要为每个task创建Sampler。
  5. 加载 Bert 文本编码器及其词典,用于处理输入文本。
  6. 构建ALBEF模型。ALBEF是NLVR模型的具体实现。如果提供了检测点,从检测点恢复模型参数。
  7. 根据配置,构建优化器和学习率调节器。优化器用于更新模型参数,学习率调节器控制学习率随epoch变化。
  8. 根据配置设置训练参数,如epochs、warmup_steps等。这些参数控制整体训练过程。
  9. 开始训练循环,每轮运行训练、验证和测试阶段。在每轮训练结束后,保存当前最佳模型和学习率。
  10. 训练结束后,记录和打印训练总时间。如果是主任务,在日志文件中记录最佳模型epoch。
  11. 将最终配置、最佳模型等参数保存至指定输出目录。

  综上,这个程序实现了NLVR模型的训练、优化和评估。它支持分布式训练,并在训练过程中对学习率进行调节。最终,它保存了最佳NLVR模型及训练过程中产生的信息。

- Pretrain_nlvr.py

  这个代码实现了在distributed环境下的预训练模型训练。主要步骤如下:

  1. 加载配置文件和设置训练参数,如 epochs, warmup_epochs等。检查分布式训练设置。
  2. 加载数据集和创建数据加载器。如果使用分布式训练,会给每个gpu分配一个dataloader。
  3. 加载文本编码器bert tokenizer。
  4. 创建模型ALBEF。如果使用检查点重载模型,会对模型进行适应,以适应增加的visual encoder块。
  5. 创建优化器和学习率调度器。
  6. 如果使用检查点重载模型,会加载检查点并对模型进行适应。
  7. 如果使用分布式训练, would wrap the model with DistributedDataParallel.
  8. 训练循环。每轮会减小学习率。每轮训练后 would save检查点。
  9. 收集训练日志与统计信息,保存至json日志文件和yaml配置文件。
  10. 训练完成后计算总耗时并打印。

  主要步骤涵盖了分布式环境下的模型预训练,包括dataloader分布、学习率调整、检查点保存等。关键步骤中有对模型进行适应扩展的逻辑,以适应增加的visual encoder块。总体实现了分布式环境下常见的预训练循环。

- Predict.py

  这段代码实现了一个图像 captioning 增强型预测器,具体做了以下事情:

  设置了数据预处理方法,包括图像Resize、ToTensor、Normalize normalization变换。还设置了BertTokenizer用于处理文本输入。

  构建了VL_Transformer_ITM模型,它由VisionTransformer作为视觉编码器,和BertModel作为语言编码器组成。它还包含一个itm_head头用于画生成分类预测。

  从检测点加载了模型参数,使模型进入评估模式。

  选择了模型内的第8层crossattention模块,启用了其save_attention机制来保存注意力权重,用于后续进行Grad-CAM可视化。

  定义了predict方法用于进行预测、损失计算和Grad-CAM生成。输入包含图像和图像描述文字。输出包含生成的Grad-CAM可视化结果图像。

  实现了pre_caption方法用于文本输入的预处理,包括去标点、转换为小写、去划线、最大词数截断等。

  实现了getAttMap方法用于将Grad-CAM权重图和图像融合成型,可通过参数控制模糊处理和覆盖等效果。

  最后,定义了Predictor类,它用于设置模型、预处理和predict方法,形成了一个完整的预测器。

  总体来说,这个代码实现了一个能够根据用户输入图像描述文字生成相应Grad-CAM heatmap 的图像 captioning 增强型预测器。它采用了BERT和ViT作为主要模型组件,并对Grad-CAM过程进行定制化,以提供更直观的结果可视化。

- Pretrain.py

  这个代码主要实现ALBERT模型的预训练。具体的操作包括:

  1. 解析命令行参数,获取训练配置等信息。
  2. 初始化 PyTorch 的分布式训练模式。设置随机种子确保结果 reproducible。
  3. 创建预训练数据集和DataLoader。使用分布式训练的Sampler进行采样。
  4. 加载预训练模型ALBERT。如果提供了checkpoint参数,则恢复之前训练好的模型。且可能根据resume参数恢复优化器和学习率调节器的状态。
  5. 创建优化器和学习率调度器,配置模型对应的优化参数。
  6. 循环进行训练。每个epoch:
     - 根据epoch和warmup steps调整学习率。
     - 调用train()方法进行一轮训练。train()方法定义了训练的过程,包括构建训练数据批次、前馈计算损失并进行反向传播、更新参数等。
     - 记录当前 epoch 的训练统计信息。如果当前进程是主进程,则将统计信息保存至日志文件。
     - 使用dist.barrier()进行过程同步,确保所有模型参数更新完成。
  7. 计算并报告整体训练时间。
  8. 保存当前模型的状态至checkpoint文件。

  总体来说,这个脚本 demontrates PyTorch 如何进行分布式训练的 ALBERT 预训练。主要实现了如下功能:

  1. 配置模型、优化器、学习率调度器以及训练参数。
  2. 构建数据集并 DataLoader 进行数据采样。
  3. 根据 checkpoint 恢复模型状态,或者进行新的训练。
  4. 循环进行训练并统计信息。
  5. 保存中间结果和训练结果。

- Retrieval.py

  训练 loop 有以下步骤:

  1. 将模型置于训练模式。
  2. 初始化一些Metric Logger对象来追踪训练损失、学习率等指标。
  3. 为每个epoch打印一个header。
  4. 按指定的频率(每50步)打印训练进度。
  5. 从数据 loader 获得一个batch的图像和对应的text。
  6. 将图像和text输入张量送到device(GPU或CPU)。输入text还需要过tokenizer切词。
  7. 根据epoch决定是否使用warm up学习率并调整alpha参数。
  8. 根据image与text计算两种损失(失焦损失和区域损失),合并为总损失。
  9. 进行梯度反传和优化器梯度下降步骤。
  10. 更新Metric Logger中的相关指标(损失、学习率)。如果在warm up期间并在specified step内,则调整学习率。
  11. 在每个epoch结束时,同步所有测量指标,并打印平均值。
  12. 返回一个词典,其中包含各种Metric Logger测量的平均值。

  评估模型的方法做以下事情:

  1. 将模型置于评估模式。
  2. 初始化一个新的Metric Logger来跟踪评估过程。
  3. 打印一个header。
  4. 计算图像和文本特征Embeddings。这需要对所有的图像和文本序列进行一次通过模型获取特征。这需要分批处理。
  5. 计算图像与文本 Embeddings 之间的相似度矩阵。这需要使用所有进程共享GPU或CPU进行并行计算。结果将合并在一起。
  6. 根据相似度矩阵的topk结果,为每个图像batch选择k个最相似的文本,并构建所需的输入以用于score预测。
  7. 使用Iterative Modification Tower(IMT)头计算图像与选择的k个文本之间的相互信息 score 。
  8. 构造相反方向的相似度矩阵和score预测。
  9. 如果使用分布式训练,则所有的操作使用barrier()同步,并使用all_reduce()操作来并行合并所有的处理器上的计算结果。
  10. 计算并打印总的评估时间。
  11. 返回两种方向(图像到文本和文本到图像)的相似度矩阵。

  可以分析如下几点:

  1. 这段代码是用来评估一个模型的检索性能的。它会计算Image -> Text和Text -> Image方向的R@1、R@5、R@10准确率和平均准确率。
  2. 它首先获得验证集和测试集的检索分数 scores_i2t 和 scores_t2i 。然后计算相应的R@1、R@5、R@10准确率,并取得平均准确率。
  3. 它会保存最高准确率的模型和对应的最佳epoch。可以通过args.evaluate 这个参数控制是否仅进行评估,还是进行训练和评估。
  4. 它使用BertTokenizer来对文本进行tokenize,使用ALBEF模型来进行检索。ALBEF模型需要一个text_encoder参数来指定使用的BERT预训练模型。
  5. 它使用AdamW优化器进行训练,使用LinearWarmupCosineAnnealingLR学习率调整器进行学习率warmup和 anneal。训练相关超参数可以通过yaml配置文件指定。
  6. 该脚本使用torch.distributed进行分布训练。节点总数和本地rank可以通过命令行参数指定。
  7. 训练过程中会不断记录训练准确率、验证准确率和测试准确率,以及学习率等信息,用于tensorboard可视化和分析。
  8. 评估阶段会在验证集和测试集上分别计算Image -> Text和Text -> Image的R@1、R@5、R@10准确率,并取得平均准确率,以评估模型的检索能力。

  综上,这段代码主要用于训练和评估一个基于BERT的检索模型,其能检索图像和文本。

- utils.py

  这片段代码主要关于下列几个内容:

  1. SmoothedValue类用于实现滑动平均值,可以配置滑动窗口大小,可以同步平均值计算的多个GPU/进程之间。利用deque实现滑动窗口,使用torch.tensor同步平均值计算。
  2. MetricLogger类用于在训练过程中收集和计算多个指标的滑动平均值和全局平均值。利用SmoothedValue类,可以实现多个GPU/进程同步计算这些指标。可通过[]方式访问这些指标的属性,也支持__getattr__方式实现动态添加新指标。支持str()打印所有指标的当前值和不同统计量。
  3. compute_acc用于计算准确率,compute_n_params用于计算模型参数数量。
  4. setup_for_distributed用于控制是否显示命令行打印信息,用于在多GPU/多进程训练时只在master进程打印信息。通过修改__builtin__.print实现。
  5. is_dist_avail_and_initialized、get_world_size、get_rank等用于检查分布式训练环境是否初始化成功,获取当前进程在所有进程中的rank等。
  6. init_distributed_mode用于初始化分布式训练环境。支持SLURM和RANK/WORLD_SIZE环境变量两种方式获取进程信息,设置设备、后端、世界大小等信息,并调用torch.distributed.init_process_group初始化分布式环境。也会调用setup_for_distributed控制打印。
  7. save_on_master用于只在master进程上保存模型和optimizer参数,避免在多个设备上重复保存。

  总体来说,此代码主要提供了一些工具类用于在多GPU/多进程训练时收集指标,控制打印,同步计算等,也提供了分布式环境的初始化设置。

- VE.py

  主要的逻辑如下:

  1. 处理命令行参数和配置文件参数,确定训练环境(分布式训练或者单机训练)
  2. 设定随机数种子和设备(cuda)
  3. 载入训练数据集和验证集
  4. 加载BertTokenizer用于 tokenizer input text
  5. 构建模型ALBEF,如果提供了checkpoint则从checkpoint恢复模型参数
  6. 实例化优化器和学习率调度器
  7. 训练循环,每 epoch:
     - 如果是训练阶段,针对训练集训练模型,计算训练损失和准确率
     - 针对验证集和测试集计算准确率作为val_stats和test_stats
     - 如果val_stats中的acc较好则保存最佳 checkpoint
     - 根据学习率调度器更新学习率
  8. 总训练时间和最佳epoch会记录在log.txt中

  总体来说,这段代码实现了ALBEF模型的训练,在每次训练epoch后对验证集和测试集进行评估,并保存最优的模型参数。

- VQA.py

  这段代码实现了一个BERT-based的模型在VQA(Visual Question Answering)datasets上的训练和评估:

  1. 读入模型配置和命令行参数,并默认补全一些必要参数,确定训练设备等。
  2. 确定模型输入(图像、问题、答案)需要用的tokenizer,并赋值给tokenizer变量。
  3. 根据配置创建模型(ALBEF),并将模型移到训练设备上。
  4. 根据配置创建优化器(Optimizer)和学习率调度器(LR Scheduler),并将optimizer赋值给optimizer变量,scheduler赋值给scheduler变量。
  5. 如果提供了 checkpoint参数,则从checkpoint中恢复模型的权重,并对图像分辨率改变做positional embedding的插值。
  6. 将模型分为子处理器(DistributedDataParallel)以进行分布式训练。
  7. 根据Epoch循环进行训练。在每个Epoch开始前,调整学习率;在每个Batch后,进行优化步骤。在每个Epoch结束后,保存Checkpoint和tensorboard log。
  8. 如果指定evaluate参数,则进入评估阶段,使用最新Checkpoint进行测试集上的评分,并将结果保存为JSON文件。
  9. 训练结束后,计算和打印总训练时间。

  相关细节:

  1. 使用torch.utils.data.DataLoader加载数据集,并用vqa_collate_fn进行批处理。
  2. 在训练过程中,使用MetricLogger进行 Metric Monitoring 和 Smoothing。
  3. 在分布式训练中,使用torch.distributed进行barrier同步和sampler.set_epoch()设置Epoch。
  4. 结果保存为JSON文件。
  5. 使用ruamel_yaml进行YAML配置文件读写。

  所以总体来说,这段代码实现了一个BERT-based的VQA模型,支持分布式训练,具有恢复Checkpoint、学习率 Warmup 等Feature,可用来进行VQA任务上的训练和评估。



