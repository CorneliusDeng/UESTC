# Multimodal Machine Learning

## 概念 Concept

- **模态**

  每一种信息的来源或者形式，都可以称为一种模态，例如：触觉，听觉，视觉，嗅觉；信息的媒介，有语音、视频、文字；传感器，如雷达、红外、加速度计

  Our experience of the world is multimodal - we see objects, hear sounds, feel texture, smell odors, and taste flavors. Modality refers to the way in which something happens or is experienced and a research problem is characterized as multimodal when it includes multiple such modalities.

- **多模态机器学习MultiModal Machine Learning**

  Multimodal machine learning aims to build models that can process and relate information from multiple modalities.
  
  旨在通过机器学习的方法实现处理和理解多源模态信息的能力

## 分类 Taxonomy

- Representation

  A first fundamental challenge is learning how to represent and summarize multimodal data in a way that exploits the complementarity and redundancy of multiple modalities. The heterogeneity of multimodal data makes it challenging to construct such representations. For example, language is often symbolic while audio and visual modalities will be represented as signals.
  第一个基本挑战是学习如何以一种利用多种模式的互补性和冗余性的方式表示和总结多模态数据。多模态数据的异构性使得构建这样的表示具有挑战性。例如，语言通常是象征性的，而音频和视觉形式将表示为信号。

- Translation

  A second challenge addresses how to translate (map) data from one modality to another. Not only is the data heterogeneous, but the relationship between modalities is often open-ended or subjective. For example, there exist a number of correct ways to describe an image and and one perfect translation may not exist.
  第二个挑战是如何将数据从一种模式转换(映射)到另一种模式。不仅数据是异质的，而且模式之间的关系通常是开放的或主观的。例如，有许多正确的方法来描述一幅图像，而一个完美的翻译可能不存在。

- Alignment

  A third challenge is to identify the direct relations between (sub)elements from two or more different modalities. For example, we may want to align the steps in a recipe to a video showing the dish being made. To tackle this challenge we need to measure similarity between different modalities and deal with possible long-range dependencies and ambiguities.
  第三个挑战是确定来自两个或两个以上不同模式的(子)元素之间的直接关系。例如，我们可能想要将菜谱中的步骤与演示菜肴制作过程的视频对齐。为了应对这一挑战，我们需要衡量不同模式之间的相似性，并处理可能的长期依赖性和模糊性。

- Fusion

  A fourth challenge is to join information from two or more modalities to perform a prediction. For example, for audio-visual speech recognition, the visual description of the lip motion is fused with the speech signal to predict spoken words. The information coming from different modalities may have varying predictive power and noise topology, with possibly missing data in at least one of the modalities.
  第四个挑战是连接来自两个或多个模式的信息来执行预测。例如，在视听语音识别中，将对嘴唇运动的视觉描述与语音信号融合在一起，以预测口语。来自不同模式的信息可能具有不同的预测能力和噪声拓扑结构，其中至少有一种模式可能缺失数据。

- Co-learning
  A fifth challenge is to transfer knowledge between modalities, their representation, and their predictive models. This is exemplified by algorithms of co-training, conceptual grounding, and zero shot learning. Co-learning explores how knowledgelearning from one modality can help a computational model trained on a different modality. This challengeis particularly relevant when one of the modalities has limited resources (e.g., annotated data).
  第五个挑战是在模式、它们的表示和它们的预测模型之间转移知识。联合训练、概念基础和零样本学习的算法就是例证。联合学习探索了从一种模式中学习的知识如何帮助在另一种模式下训练的计算模型。当其中一种模式的资源有限时(例如，注释数据)，这一挑战尤其相关。

## 应用 Application

- One of the earliest examples of multimodal research is audio-visual speech recognition (AVSR). 视听语音识别
- A second important category of multimodal applicationscomes from the field of multimedia content indexing and retrieval. While earlier approaches for indexing and searching these multimedia videos were keyword-based , new research problems emerged when trying to search the visual and multimodal content directly.
- A third category of applications was established in the early 2000s around the emerging field of multimodal interaction with the goal of understanding human multimodal behaviors during social interactions.
- Most recently, a new category of multimodal applications emerged with an emphasis on language and vision: media description. One of the most representative applications is image captioning where the task is to generate a text description of the input image. The main challenges media description and generation is evaluation: how to evaluate the quality of the predicted descriptions and media.

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/A%20Summary%20of%20Applications%20Enabled%20by%20Multimodal%20Machine%20Learning.png)



# **多模态表示 Multimodal Representation**

## Basic Knowledge

- **A multimodal representation is a representation of data using information from multiple entities.**
  使用多个实体信息的数据表示

- difficulties

  - how to combine the data from heterogeneous sources

  - how to deal with differentlevels of noise

  - how to handle missing data

- good representations

  - smoothness，temporal and spatial coherence，sparsity，natural clustering amongst others
  - similarity in the representation space should reflect the similarity of the corresponding concepts, the representation should be easy to obtain even in the absence of some modalities, and finally, it should be possible to fill-in missing modalities given the observed ones

- Two categories of Multimodal Representation

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/Structure%20of%20joint%20and%20coordinated%20representations..png)

- A Summary of Multimodal Representation Techniques

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/A%20Summary%20of%20Multimodal%20Representation%20Techniques.png)

## Joint Representation

- Joint representations combine the unimodal signals into the same representation space. Joint representations are projected to the same space using all of the modalities as input. 将多个模态的信息一起映射到一个统一的多模态向量空间
  - Joint representations project multimodal data into a common space and are best suited for situations when all of the modalities are present during inference.
  - They have been extensively used for AVSR, affect, and multimodal gesture recognition.
- Joint representations are mostly (but not exclusively) used in tasks where multimodal data is present both during training and inference steps.
- Neural networks have become a very popular method for unimodal data representation
  - To construct a multimodal representation using neural networks each modality starts with several individual neural layers followed by a hidden layer that projects the modalities into a joint space.
  - The joint multimodal representation is then be passed through multiple hidden layers itself or used directly for prediction.
  - The major advantage of neural network based joint representations comes from their ability to pre-train from unlabled data when labeled data is not enough for supervised learning.
  - One of the disadvantages comes from the model not being able to handle missing data naturally although there are ways to alleviate this issue.
- Probabilistic graphical models can be used to construct representations through the use of latent random variables.
  - One such way to represent data is through deep Boltzmann machines (DBM), that stack restricted Boltzmann machines (RBM) as building blocks.
  - One of the big advantages of using multimodal DBMs for learning multimodal representations is their generative nature, which allows for an easy way to deal with missing data even if a whole modality is missing, the model has a natural way to cope.
  - The major disadvantageof DBMs is the difficulty of training them high computational cost, and the need to use approximate variational training methods.
- Sequential Representation, model that represent varying length sequences such as sentences,videos, or audio streams.
  - Recurrent neural networks(RNNs), and their variants such as long-short term memory(LSTMs) networks, have recently gained popularity due to their success in sequence modeling across various tasks.

## Coordinated Representation

- Coordinated representations process unimodal signals separately, but enforce certain similarity constraints on them to bring them to what we term a coordinated space. Coordinated representations exist in their own space, but are coordinated through a similarity (e.g., Euclidean distance) or structure constraint (e.g., partial order). 将多模态中的每个模态分别映射到各自的表示空间，但映射后的向量之间满足一定的相关性约束
  - Coordinated representations, project each modality into a separate but coordinated space, making them suitable for applications where only one modality is present at test time.
  - such as: multimodal retrieval and translation, grounding, and zero shot learning .
- Similarity models minimize the distance between modalities in the coordinated space.
- More recently, neural networks have become a popular way to construct coordinated representations, due to their ability to learn representations.
  - Their advantage lies in the fact that they can jointly learn coordinated representationsin an end-to-end manner.
  - An example is DeViSE—a deep visual-semantic embedding. DeViSE uses a similar inner product and ranking loss function to WSABIE but uses more complex image and word embeddings.
- Structured coordinated spaces are commonly used in cross-modal hashing-compression of high dimensional data into compact binary codes with similar binary codes for similar objects.
  - Hashing enforces certain constraints on the resulting multimodal space: 
    - it has to be an N-dimensional Hamming space—a binary representation with controllablenumber of bits; 
    - the same object from different modalities has to have a similar hash code; 
    - the space has to be similarity-preserving.



# **多模态转化 Multimodal Translation**

## Basic Knowledge

- **Given an entity in one modality the task is to generate the same entity in a different modality.** 
  由一种模态生成同义的另一模态

  - For example given an image we might want to generate a sentence describing it or given a textual description generate an image matching it.

- Two categories of Multimodal Translation

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/Example-based%20and%20generative%20multimodal%20translation.png)

- Taxonomy of Multimodal Translation Research

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/Taxonomy%20of%20Multimodal%20Translation%20Research.png)

- Model Evaluation and Discussion

  - A major challenge facing multimodal translation methods is that they are very difficult to evaluate.
  - Often the ideal way to evaluate a subjective task is through human judgment.
  - While human studies are a gold standard for evaluation,a number of automatic alternatives have been proposed for the task of media description: BLEU, ROUGE, Meteor, and CIDEr. 

## Example-Based

- Example-based models use a dictionary when translating between the modalities.  It retrieves the best translation from a dictionary.
- Example-based algorithms are restricted by their training data dictionary
- We identify two types of Example-based algorithms: retrieval based, and combination based.
  - Retrieval-based models directly use the retrieved translation without modifying it.
    - Retrieval-based models are arguably the simplest form of multimodal translation. They rely on finding the closest sample in the dictionary and using that as the translated result.
    - Retrieval approaches in semantic space tend to perform better than their unimodal counterparts as they are retrieving examples in a more meaningful space that reflects both modalities and that is often optimized for retrieval.
  - Combination-based models rely on more complex rules to create translations based on a number of retrieved instances.
    - Combination-based models take the retrieval based approaches one step further. Instead of just retrieving examples from the dictionary, they combine them in a meaningful way to construct a better translation.
    - Combination basedmedia description approaches are motivated by the fact that sentence descriptions of images share a common and simple structure that could be exploited.
- A big problem facing example-based approaches for translation is that the model is the entire dictionary-making the model large and inference slow. 
- Another issue facing example-based translation is that it is unrealistic to expect that a single comprehensive and accurate translation relevant to the source example will always exist in the dictionary—unless the task is simple or the dictionary is very large.

## Generative Approaches

- Generative models construct a model that is able to produce a translation. It trains a translation model on the dictionary and then uses that model for translation.
- We identify three broad categories of generative models: grammar-based, encoder-decoder, and continuous generation models.
  - Grammar based models simplify the task by restricting the target domain by using a grammar.
    - Grammar-based models rely on a pre-defined grammar for generating a particular modality.
    - They start by detecting high level concepts from the source modality. These detections are then incorporated together with a generation procedure based ona pre-defined grammar to result in a target modality.
    - Eg. Barbuet al proposed a video description model that generatessentences of the form: who did what to whom and where and how they did it.
    - Some grammar-based approaches rely on graphical models to generate the target modality.
    - An advantage of grammar-based methods is that they are more likely to generate syntactically (in case of language)or logically correct target instances as they use predefined templates and restricted grammars. However, this limits them to producing formulaic rather than creative translations. Furthermore, grammar-based methods rely on complex pipelines for concept detection, with each concept requiring a separate model and a separate training dataset.
  - Encoder-decoder models first encode the source modality to a latent representation which is then used by a decoder to generate the target modality.
    - Encoder-decoder models based on end-to-end trained, neural networks are popular.
    - The main idea behind the model is to first encode a source modality into a vectorial representation and then to use a decoder module to generate the target modality.
    - The first step of the encoder-decoder model is to encode the source object, this is done in modality specific way. Popular models to encode acoustic signals include RNNs and DBNs
    - Decoding is most often performed by an RNNor an LSTMusing the encoded representation as the initial hidden state.
  - Continuous generation models generate the target modality continuously based on a stream of source modality inputs and are most suited for translating between temporal sequences
    - Continuous generation models are intended for sequence translation and produce outputs at every timestep in an online manner.



# **多模态对齐 Multimodal Alignment**

## Basic Knowledge

- **We define multimodal alignment as finding relationships and correspondences between sub-components of instances from two or more modalities.**
  对来自多个实例的不同模态信息的子元素寻找对应关系

- Two categories of Multimodal Alignment—implicit and explicit.

- Summary of  Taxonomy for the Multimodal Alignment Challenge

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/Summary%20of%20Our%20Taxonomy%20for%20MA.png)

- Multimodal alignment faces a number of difficulties: 

  - there are few datasets with explicitly annotated alignments.
  -  it is difficult to design similarity metrics between modalities.
  - there may exist multiple possible alignments and not all elements in one modality have correspondences in another.

## Explicit Alignment

- In explicit alignment, we are explicitly interested in aligning sub-components between modalities.
- We categorize papers as performing explicit alignment if their main modeling objective is alignment between subcomponents of instances from two or more modalities.
- A very important part of explicit alignment is the similarity metric.
- We identify two types of algorithms that tackle explicit alignment—unsupervised and (weakly)supervised.
  - Unsupervised multimodal alignment tackles modality alignment without requiring any direct alignment labels between instances from the different modalities. 
    - Dynamic time warping (DTW) is a dynamic programming approach that has been extensively used to align multi-view time series.
    - Various graphical models have also been popular form ultimodal sequence alignment in an unsupervised manner.
  - Supervised alignment methods rely on labeled aligned instances. 
    - Deep learning based approaches are becoming popular for explicit alignment (specifically for measuring similarity) due to very recent availability of aligned datasets in the languageand vision communities.

## Implicit Alignment

- Implicit alignment is used as an intermediate(often latent) step for another task. This allows for better performance in a number of tasks including speech recognition, machine translation, media description,and visual question-answering.
- Such models do not explicitly align data and do not rely on supervised alignment examples, but learn how to latently align the data during model training.
- We identify two types of implicit alignment models: earlier work based on graphical models, and more modern neural network methods.
  - Graphical models have seen some early work used to better align words between languages for machine translation and alignment of speech phonemes with their transcriptions. However, they require manual construction of a mapping between the modalities.
  - Neural networks Translation is an example of a modeling task that can often be improved if alignment is performed as a latent intermediate step.



# **多模态融合 Multimodal Fusion**

## Basic Knowledge

- **Multimodal fusion is the concept of integrating information from multiple modalities with the goal of predicting an outcome measure: a class through classification, or a continuous value through regression.** 
  联合多个模态的信息，进行目标预测、分类、回归

- We classify multimodal fusion into two main categories: model-agnostic approaches and model based approaches.

- A Summary of Taxonomy of Multimodal Fusion Approaches

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/A%20Summary%20of%20Our%20Taxonomy%20of%20Multimodal%20Fusion%20Approaches.png)
  
- multimodal fusion faces the following challenges: 

  - signals might not be temporally aligned (possibly dense continuous signal and a sparseevent)
  - it is difficult to build models that exploit supplementary and not only complementary information 
  - each modality might exhibit different types and different levels of noise at different points in time
  

## Model-Agnostic Approaches 

- Model-Agnostic Approaches are not directly dependent on a specific machine learning method.
- An advantageof model agnostic approaches is that they can be implemented using almost any unimodal classifiers or regressors.
- Model-Agnostic Approaches can be split into early (i.e., feature-based), late(i.e., decision-based) and hybrid fusion.
  - Early fusion integrates features immediately after they are extracted(often by simply concatenating their representations).
    - Early fusion could be seen as an early attempt by multimodal researchers to perform multimodal representation learning—as it can learn to exploit the correlation and interactions between low level features of each modality.
    - It alsoonly requires the training of a single model, making the training pipeline easier compared to late and hybrid fusion.
  - Late fusion performs integration after each ofthe modalities has made a decision(e.g., classification orregression).
    - Late fusion uses unimodal decision values and fuses them using a fusion mechanism such as averaging, voting schemes, weighting based on channel noise and signal variance, or a learned model.
    - It allows for the use of different models for each modality as different predictors can model each individual modality better, allowing for more flexibility.
    - It makes it easier to make predictions when one or more ofthe modalities is missing and even allows for training when no parallel data is available.
    - However, late fusion ignores the low level interaction between the modalities.
  - Hybrid fusion combines outputs from early fusion and individual unimodal predictors.
    - Hybrid fusion attempts to exploit the advantages of both of the above described methods in a common framework.
    - It has been used successfully for multimodal speaker identification and multimedia event detection.

## Model Based Approaches

- Model Based approaches that explicitly address fusion in their construction—such as kernel-based approaches, graphical models, and neural networks.
- There are three categories of approaches that are designed to perform multimodal fusion: kernel based methods, graphical models, and neural networks.
  - Multiple kernel learning (MKL) methods are an extension to kernel support vector machines (SVM) that allow for the use of different kernels for different modalities/views of the data.
    - MKL approaches have been an especially popular method for fusing visual descriptors for object detection and only recently have been overtaken by deep learning methods for the task.
    - Besides flexibility in kernel selection, an advantage of MKL is the fact that the loss function is convex, allowing for model training using standard optimization packages and global optimum solutions.
    - MKL can be used to both perform regression and classification.
    - One of the main disadvantages of MKL is the reliance on training data (support vectors) during test time, leading to slow inference and a large memory footprint.
  - Majority of graphical models can be classified into two main categories: generative modeling joint probability; or discriminative—modeling conditional probability
    - The benefit of graphical models is their ability to easily exploit spatial and temporal structure of the data, making them especially popular for temporal modeling tasks.
  - Neural networks have been used extensively for the task of multimodal fusion
    - Both shallow and deep neural models have been explored for multimodal fusion.
    - Neural networks have also been used for fusing temporal multimodal information through the use of RNNs and LSTMs.
    - A big advantage of deep neural network approaches in data fusion is their capacity to learn from large amount of data. Second, recent neural architectures allow for end-toend training of both the multimodal representation component and the fusion component. Finally, they show good performance when compared to non neural network based system and are able to learn complex decision boundaries that other approaches struggle with.
    - The major disadvantage of neural network approaches is their lack of interpretability.



# **多模态协同学习 Multimodal Co-learning**

## Basic Knowledge

- **Co-learning aiding the modeling of a (resource poor) modality by exploiting knowledge from another (resource rich) modality.**
  使用一个资源丰富的模态信息来辅助另一个资源相对贫瘠的模态进行学习，协同学习是与需要解决的任务无关的，因此它可以用于辅助多模态映射、融合及对齐等问题的研究

- Multimodal co-learning allows for one modality to influence the training of another, exploiting the complementary information across modalities. It is important to note that co-learning is task independent and could be used to create better fusion, translation, and alignment models.

- We identify three types of co-learning approaches based on their training resources: parallel, non-parallel, and hybrid.

- Types of data parallelism used in co-learning

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/MultiModal%20Machine%20Learning/Types%20of%20data%20parallelism%20used%20in%20co-learning.png)

## Parallel Data

- Papallel data approaches require training datasets where the observations from one modality are directly linked to the observations from other modalities. 
  Modalities are from the same dataset and there is a direct correspondence between instances. 
- Co-training is the process of creating more labeled training samples when we have few labeled samples in a multimodal problem.
- The basic algorithm builds weak classifiers in each modality to bootstrap each other with labels for the unlabeled data.
- Transfer learning（迁移学习） is another way to exploit co-learning with parallel data.

## Non-Parallel Data

- Non-parallel data approaches do not require direct links between observations from different modalities. These approaches usually achieve co-learning by using overlap in terms of categories.
  Modalities are from different datasets and do not have overlapping instances, but overlap in general categories or concepts.
- Non-parallel co-learning approaches can help when learning representations, allow for better semantic concept understanding and even perform unseen object recognition.
- Transfer learning is also possible on non-parallel data and allows to learn better representations through transferring information from a representation built using a data rich or clean modality to a data scarce or noisy modality.
- Conceptual grounding（概念基础） refers to learning semantic meanings or concepts not purely based on language but also on additional modalities. 
  - While the majority of concept learning approaches are purely language-based, representations of meaning in humans are not merely a product of our linguistic exposure, but are also grounded through our sensorimotor experience and perceptual system. 
  - However, one has to be careful as grounding does not always lead to betterperformance, and only makes sense when grounding has relevance for the task—such as groundingusing images for visually-related concepts.
- Zero shot learning (ZSL 零样本学习) refers to recognizing a concept without having explicitly seen any examples of it.
  - There are two main types of ZSL—unimodal and multimodal.
  - The unimodal ZSL looks at component parts or attributes of the object, such as phonemes to recognize an unheard word or visual attributes such as color, size, and shape to predict an unseen visual class.
  - The multimodal ZSL recognizes the objects in the primary modality through the help of the secondary one—in which the object has been seen.

## Hybrid Data

- In the hybrid data setting the modalities are bridged through a shared modality or a dataset.
  The instances or concepts are bridged by a third modality or a dataset.
- In the hybrid data setting two non-parallel modalities arebridged by a shared modality or a dataset.
- Themost notable example is the Bridge Correlational Neural Network, which uses a pivot modality to learn coordinated multimodal representations in presence of non-parallel data.
- For example, for multilingual image captioning, the image modality would be paired with at least one caption in any language.

