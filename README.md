# [NLP Repository](https://ryuseonghan.github.io/NLP-Repository)

A curated list of resources about natural language processing (NLP).

Maintained by [Seonghan Ryu](https://github.com/ryuseonghan).

Inspired by the [awsome lists](https://github.com/sindresorhus/awesome).

## Sharing

[![Facebook](https://github.com/ryuseonghan/NLP-Repository/blob/master/img/fb.png?raw=true)](https://www.facebook.com/sharer/sharer.php?u=https://ryuseonghan.github.io/NLP-Repository)
[![Twitter](https://github.com/ryuseonghan/NLP-Repository/blob/master/img/tt.png?raw=true)](http://twitter.com/home?status=https://ryuseonghan.github.io/NLP-Repository)

## Table of Contents

- [Video Lectures](#video-lectures)
- [Presentations](#presentations)
- [Articles](#articles)
- [Books](#books)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Research Papers](#research-papers)
- [References](#references)

## Video Lectures

- [Deep Learning for NLP (Stanford)](http://cs224d.stanford.edu/), Richard Socher, 2016
- [Deep NLP (Oxford)](https://github.com/oxford-cs-deepnlp-2017/lectures), Phil Blunsom, 2017 
- [Deep Learning for NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2015-10_Lecture), Nils Reimers, 2015

## Presentations

- [Neural Machine Translation](https://www.youtube.com/watch?v=z4CNmiLF-YU), Kyunghyun Cho, 2016
- [Deep Natural Language Understanding](https://www.youtube.com/watch?v=K_zKimkoVk8), Kyunghyun Cho, 2016

## Articles

- [Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/), Suriyadeepan Ramamoorthy, 2016
- [Multi-Task Learning in Tensorflow](http://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html), Jonathan Godwin, 2016
- [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/), Denny Britz (WildML), 2016
- [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/), Beomsu Kim, 2016
- [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) [[Part 1]](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) [[Part 2]](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/) [[Part 3]](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) [[Part 4]](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/), Denny Britz (WildML), 205
- [Introduction to Neural Machine Translation with GPUs](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/) [[Part 1]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/) [[Part 2]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/) [[Part 3]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/), Kyunghyun Cho, 2015
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/), Denny Britz (WildML), 2015
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), Denny Britz (WildML), 2015
- [Recurrent Neural Networks with Word Embeddings](http://www.deeplearning.net/tutorial/rnnslu.html),  Grégoire Mesnil (deeplearning.net), 2013
- [LSTM Networks for Sentiment Analysis](http://www.deeplearning.net/tutorial/lstm.html), Pierre Luc Carrier (deeplearning.net), 2012

## Books

- [Natural Language Understanding with Distributed Representation](https://arxiv.org/abs/1511.07916), Kyunghyun Cho, Lecture Note, 2016

## Datasets

- Question Answering
	- [The (20) QA bAbI tasks (Facebook)](https://research.fb.com/projects/babi/)
	- [The Children's Book test (Facebook)](https://research.fb.com/projects/babi/)
	- [The SimpleQuestions dataset (Facebook)](https://research.fb.com/projects/babi/)
	- [WebQuestions](http://www-nlp.stanford.edu/software/sempre/)
	- [DeepMind Q&A Dataset](http://cs.nyu.edu/~kcho/DMQA/)
	- [SQuAD (Stanford)](https://rajpurkar.github.io/SQuAD-explorer/)
	- [Question-Answer Dataset (CMU)](http://www.cs.cmu.edu/~ark/QA-data/)
	- [NewsQA dataset (Maluuba)](https://datasets.maluuba.com/NewsQA)
	- [Question Pairs (Quora)](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- Dialog
	- [The (6) Dialog bAbI tasks (Facebook)](https://research.fb.com/projects/babi/)
	- [The Movie Dialog dataset (Facebook)](https://research.fb.com/projects/babi/)
	- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
	- [Frames Dataset (Maluuba)](https://datasets.maluuba.com/Frames)
- Language Model
	- [The WikiText Language Modeling Dataset](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
	- [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
	- [Common Crawl](http://commoncrawl.org/the-data/)
- Sentiment Analysis
	- [Large Movie Review Dataset (IMDB)](http://ai.stanford.edu/~amaas/data/sentiment/)

## Libraries

- [TensorFlow](https://www.tensorflow.org/), Apache License 2.0
- [Theano](http://www.deeplearning.net/software/theano/)
- [Keras](https://keras.io/), MIT license
- [The Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) (previously known as CNTK)
- [RNNLM Toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
- [word2vec](https://code.google.com/p/word2vec/), Apache License 2.0
- [GloVe](https://github.com/stanfordnlp/GloVe), Apache License 2.0
- [gensimn](https://github.com/RaRe-Technologies/gensim), LGPL v2.1

## Research Papers
 
- Word Embedding
	- [Distributed Representations of Sentences and Document](https://arxiv.org/abs/1405.4053) (Paragraph Vector), Quoc V. Le et al., ICML, 2014
	- [Glove: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington et al., EMNLP, 2014
	- [Learning word embeddings efficiently with noise-contrastive estimation](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf), Andriy Mnih et al., NIPS, 2013
	- [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/pdf/1310.4546.pdf) (Negative Sampling), Tomas Mikolov et al., NIPS, 2013
	- [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (CBOW and Skip-gram), Tomas Mikolov et al., ICLR, 2013
- Task-oriented Dialog System
	- [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562v2.pdf), Tsung-Hsien Wen et al., arXiv, 2016
	- [Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning](https://arxiv.org/abs/1606.02560), Tiancheng Zhao et al., SIGDIAL, 2016
	- [Stochastic Language Generation in Dialogue using Recurrent Neural Networks with Convolutional Sentence Reranking](http://www.sigdial.org/workshops/conference16/proceedings/pdf/SIGDIAL39.pdf), Tsung-Hsien Wen et al., SIGDIAL, 2015
	- [Word-Based Dialog State Tracking with Recurrent Neural Networks](http://www.sigdial.org/workshops/sigdial2014/proceedings/pdf/W14-4340.pdf), Matthew Henderson et al., SIGDIAL, 2014
	- [Investigation of Recurrent-Neural-Network Architectures and Learning Methods for Spoken Language Understanding](https://www.microsoft.com/en-us/research/publication/investigation-of-recurrent-neural-network-architectures-and-learning-methods-for-spoken-language-understanding/), Grégoire Mesnil et al., Interspeech, 2013 [[Code]](https://github.com/mesnilgr/is13)
- Question Answering
	- [Question Answering over Freebase with Multi-Column Convolutional Neural Networks](http://www.anthology.aclweb.org/P/P15/P15-1026.pdf), Li Dong et al., ACL-IJCNLP, 2015
	- [Large-scale Simple Question Answering with Memory Networks](https://arxiv.org/abs/1506.02075), Antoine Bordes et al., arXiv, 2015
	- [End-to-end Memory Networks](https://arxiv.org/abs/1503.08895) (MemN2N), Sainbayar Sukhbaatar et al., NIPS, 2015 [[Code]](https://github.com/facebook/MemNN)
	- [Question Answering with Subgraph Embeddings](https://arxiv.org/pdf/1406.3676v3.pdf), Antoine Bordes et al., EMNLP, 2014
	- [Memory Networks](https://arxiv.org/abs/1410.3916) (MemNN), Jason Weston et al., arXiv, 2014
	- [Semantic Parsing for Single-Relation Question Answering](https://aclweb.org/anthology/P/P14/P14-2105.pdf), Wen-tau Yih et al., ACL, 2014 [[Poster]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ACL-14-SRQA-Poster.pdf)
	- [Reasoning With Neural Tensor Networks for Knowledge Base Completion](http://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf), Richard Socher et al., NIPS, 2013.
	- [Semantic Parsing on Freebase from Question-Answer Pairs](http://cs.stanford.edu/~pliang/papers/freebase-emnlp2013.pdf) (SEMPRE), Jonathan Berant, EMNLP, 2013 [[Code]](http://nlp.stanford.edu/software/sempre/)
- Chatbot
	- [A Persona-Based Neural Conversation Model](https://www.microsoft.com/en-us/research/publication/persona-based-neural-conversation-model/), Jiwei Li et al., ACL, 2016
	- [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808), Iulian V. Serban et al., AAAI, 2016
	- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869), Oriol Vinyals et al., ICML, 2015
	- [Neural Responding Machine for Short-Text Conversation](https://arxiv.org/abs/1503.02364), Lifeng Shang et al., ACL, 2015
- Language Model
	- [Linguistic Regularities in Continuous Space Word Representations](http://research.microsoft.com/pubs/189726/rvecs.pdf), Tomas Mikolov et al., NAACL-HLT, 2013
	- [Recurrent Neural Network based Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf), Tomas Mikolov et al., Interspeech, 2010
	- [A Scalable Hierarchical Distributed Language Model](https://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model) (Hierarchical Softmax), Andriy Mnih, NIPS, 2008
- Image Captioning
	- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), Kelvin Xu, ICML, 2015
	- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555), Oriol Vinyals et al., CVPR, 2015
	- [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf), Andrej Karpathy et al., CVPR, 2015 [[Code]](https://github.com/karpathy/neuraltalk) [[Demo]](http://cs.stanford.edu/people/karpathy/deepimagesent/rankingdemo/)
- Sentiment Analysis
	- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), Yoon Kim, EMNLP, 2014
	- [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf), Richard Socher et al., EMNLP, 2013
- Machine Translation
	- [Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558), Melvin Johnson et al., arXiv, 2016
	- [A Character-level Decoder without Explicit Segmentation for Neural Machine Translation](https://www.aclweb.org/anthology/P/P16/P16-1160.pdf), Junyoung Chung et al., ACL, 2016
	- [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473), Dzmitry Bahdanau et al., ICLR, 2015
	- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), Ilya Sutskever et al., NIPS, 2014
	- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Kyunghyun Cho et al., EMNLP, 2014
- Entity Linking
	- [Modeling Mention, Context and Entity with Neural Networks for Entity Disambiguation](http://ir.hit.edu.cn/~dytang/paper/ijcai2015/ijcai15-yaming.pdf), Yaming Sun et al., IJCAI, 2015
	- [Local and Global Algorithms for Disambiguation to Wikipedia](http://web.eecs.umich.edu/~mrander/pubs/RatinovDoRo.pdf), Lev Ratinov et al. ACL, 2011 
- Information Retrieval
	- [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](http://dl.acm.org/citation.cfm?id=2505665), Po-Sen Huang et al., CIKM, 2013
- Syntactic Parsing
	- [Parsing Natural Scenes and Natural Language with Recursive Neural Networks](http://ai.stanford.edu/~ang/papers/icml11-ParsingWithRecursiveNeuralNetworks.pdf), Richard Socher et al., ICML, 2011