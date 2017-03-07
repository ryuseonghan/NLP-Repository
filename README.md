# [NLP Repository](https://ryuseonghan.github.io/NLP-Repository)

A curated list of resources about natural language processing (NLP).

Maintained by [Seonghan Ryu](https://github.com/ryuseonghan).

Inspired by the [awsome lists](https://github.com/sindresorhus/awesome).

## Sharing

[![Facebook](https://github.com/ryuseonghan/NLP-Repository/blob/master/img/fb.png?raw=true)](https://www.facebook.com/sharer/sharer.php?u=https://ryuseonghan.github.io/NLP-Repository)
[![Twitter](https://github.com/ryuseonghan/NLP-Repository/blob/master/img/tt.png?raw=true)](http://twitter.com/home?status=https://ryuseonghan.github.io/NLP-Repository)

## Table of Contents

- [NLP in General](#nlp-in-general)
- [Distributed Representation](#distributed-representation)
- [Task-oriented Dialog System](#task-oriented-dialog-system)
- [Question Answering](#question-answering)
- [Machine Translation](#machine-translation)
- [Chatbot](#chatbot)
- [Sentiment Analysis](#sentiment-analysis)
- [Language Modeling](#language-modeling)
- [Uncategorized](#uncategorized)

## NLP in General

Video Lectures

- [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/), Richard Socher, Stanford University, 2016
- [Deep Natural Language Processing](https://github.com/oxford-cs-deepnlp-2017/lectures), Phil Blunsom, Oxford University, 2017 
- [Deep Learning for NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2015-10_Lecture), Nils Reimers, 2015
- [Natural Language Processing](https://www.youtube.com/playlist?list=PL6397E4B26D00A269), Dan Jurafsky & chris Manning, Coursera, 2012 

Books

- [Natural Language Understanding with Distributed Representation](https://arxiv.org/abs/1511.07916), Kyunghyun Cho, 2016
- [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), Yoav Goldberg, 2015

Related Conferences & Workshops

- Annual Meeting of the Association for Computational Linguistics (ACL)
- Conference on Empirical Methods in Natural Language Processing (EMNLP)
- International Conference on Computational Linguistics (COLING)
- Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)
- European Chapter of the Association for Computational Linguistics (EACL)
- Interspeech
- IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
- IEEE Workshop on Spoken Language Technology (SLT)
- Annual SIGdial Meeting on Discourse and Dialogue (SIGdial)
- International Conference on Language Resources and Evaluation (LREC)
- International Workshop on Spoken Dialog Systems (IWSDS)
- International Joint Conference on Natural Language Processing (IJCNLP)
- AAAI Conference on Artificial Intelligence (AAAI)
- Annual Conference on Neural Information Processing Systems (NIPS)

Shared Tasks

- The SIGNLL Conference on Computational Natural Language Learning (CoNLL)
- Semantic Evaluation (SemEval)
- Text REtrieval Conference (TREC)
- Conference and Labs of the Evaluation Forum (CLEF)
- NII Testbeds and Community for Information access Research (NTCIR)
- Dialog State Tracking Challenge (DSTC)

Reference

- [Awesome-Korean-NLP](https://github.com/datanada/Awesome-Korean-NLP)

## Distributed Representation

Reviews & Tutorials

- [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/), Beomsu Kim, Personal Blog, 2016

Research Papers

- [Distributed Representations of Sentences and Document](https://arxiv.org/abs/1405.4053) (Paragraph Vector), Quoc V. Le et al., ICML, 2014
- [Glove: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington et al., EMNLP, 2014 [[GloVe]](https://github.com/stanfordnlp/GloVe)
- [Learning word embeddings efficiently with noise-contrastive estimation](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf), Andriy Mnih et al., NIPS, 2013
- [Linguistic Regularities in Continuous Space Word Representations](http://research.microsoft.com/pubs/189726/rvecs.pdf), Tomas Mikolov et al., NAACL-HLT, 2013
- [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/pdf/1310.4546.pdf) (Negative Sampling), Tomas Mikolov et al., NIPS, 2013
- [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (word2vec, CBOW, and Skip-gram), Tomas Mikolov et al., ICLR, 2013 [[Code]](https://code.google.com/p/word2vec/)
- [Software Framework for Topic Modelling with Large Corpora](https://radimrehurek.com/gensim/lrec2010_final.pdf) (gensim), Radim Řehůřek et al., LREC, 2010 [[Code]](https://github.com/RaRe-Technologies/gensim)

## Task-oriented Dialog System

Reviews & Tutorials

- [파이썬으로 챗봇 만들기](https://www.slideshare.net/KimSungdong1/20170227-72644192), Kim Sungdong, 2017

Research Papers

- [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562v2.pdf), Tsung-Hsien Wen et al., arXiv, 2016
- [Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning](https://arxiv.org/abs/1606.02560), Tiancheng Zhao et al., SIGDIAL, 2016
- [Stochastic Language Generation in Dialogue using Recurrent Neural Networks with Convolutional Sentence Reranking](http://www.sigdial.org/workshops/conference16/proceedings/pdf/SIGDIAL39.pdf), Tsung-Hsien Wen et al., SIGDIAL, 2015
- [Word-Based Dialog State Tracking with Recurrent Neural Networks](http://www.sigdial.org/workshops/sigdial2014/proceedings/pdf/W14-4340.pdf), Matthew Henderson et al., SIGDIAL, 2014
- [Investigation of Recurrent-Neural-Network Architectures and Learning Methods for Spoken Language Understanding](https://www.microsoft.com/en-us/research/publication/investigation-of-recurrent-neural-network-architectures-and-learning-methods-for-spoken-language-understanding/), Grégoire Mesnil et al., Interspeech, 2013 [[Code]](https://github.com/mesnilgr/is13) [[Tutorial]](http://www.deeplearning.net/tutorial/rnnslu.html) #NER #ATIS

Datasets

- [Airline Travel Information Systems (ATIS)](https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0) #NER
- [Facebook Dialog bAbI tasks](https://research.fb.com/projects/babi/)
- [Facebook Movie Dialog dataset](https://research.fb.com/projects/babi/)
- [Maluuba Frames Dataset](https://datasets.maluuba.com/Frames)

## Question Answering

Research Papers

- [Question Answering over Freebase with Multi-Column Convolutional Neural Networks](http://www.anthology.aclweb.org/P/P15/P15-1026.pdf), Li Dong et al., ACL-IJCNLP, 2015
- [Large-scale Simple Question Answering with Memory Networks](https://arxiv.org/abs/1506.02075), Antoine Bordes et al., arXiv, 2015
- [End-to-end Memory Networks](https://arxiv.org/abs/1503.08895) (MemN2N), Sainbayar Sukhbaatar et al., NIPS, 2015 [[Code]](https://github.com/facebook/MemNN)
- [Question Answering with Subgraph Embeddings](https://arxiv.org/pdf/1406.3676v3.pdf), Antoine Bordes et al., EMNLP, 2014
- [Memory Networks](https://arxiv.org/abs/1410.3916) (MemNN), Jason Weston et al., arXiv, 2014
- [Semantic Parsing for Single-Relation Question Answering](https://aclweb.org/anthology/P/P14/P14-2105.pdf), Wen-tau Yih et al., ACL, 2014 [[Poster]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ACL-14-SRQA-Poster.pdf)
- [Reasoning With Neural Tensor Networks for Knowledge Base Completion](http://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf), Richard Socher et al., NIPS, 2013.
- [Semantic Parsing on Freebase from Question-Answer Pairs](http://cs.stanford.edu/~pliang/papers/freebase-emnlp2013.pdf) (SEMPRE), Jonathan Berant, EMNLP, 2013 [[Code]](http://nlp.stanford.edu/software/sempre/)

Datasets

- [Facebook QA bAbI tasks](https://research.fb.com/projects/babi/)
- [Facebook Children's Book test](https://research.fb.com/projects/babi/)
- [Facebook SimpleQuestions dataset](https://research.fb.com/projects/babi/)
- [WebQuestions](http://www-nlp.stanford.edu/software/sempre/)
- [DeepMind Q&A Dataset](http://cs.nyu.edu/~kcho/DMQA/)
- [The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)
- [CMU Question-Answer Dataset](http://www.cs.cmu.edu/~ark/QA-data/)
- [Maluuba NewsQA dataset](https://datasets.maluuba.com/NewsQA)
- [Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

## Machine Translation

Reviews & Tutorials

- [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/), Denny Britz, WildML, 2016
- [Deep Natural Language Understanding](http://videolectures.net/deeplearning2016_cho_language_understanding/), Kyunghyun Cho, Deep Learning Summer School, 2016
- [Neural Machine Translation](https://www.youtube.com/watch?v=z4CNmiLF-YU), Kyunghyun Cho, Microsoft Research Invited Talk, 2016
- [Introduction to Neural Machine Translation with GPUs](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/) [[Part 1]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/) [[Part 2]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/) [[Part 3]](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/), Kyunghyun Cho, NVIDIA Blog, 2015

Research Papers

- [Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558), Melvin Johnson et al., arXiv, 2016
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), Yonghui Wu et al., arXiv, 2016 
- [A Character-level Decoder without Explicit Segmentation for Neural Machine Translation](https://www.aclweb.org/anthology/P/P16/P16-1160.pdf), Junyoung Chung et al., ACL, 2016
- [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473), Dzmitry Bahdanau et al., ICLR, 2015 #Attention
- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (seq2seq), Ilya Sutskever et al., NIPS, 2014 #Google #seq2seq #LSTM
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Kyunghyun Cho et al., EMNLP, 2014

## Chatbot

Reviews & Tutorials

- [Developing Korean Chatbot 101](https://www.youtube.com/watch?v=i0sQB1DRh84&index=3&list=PLlMkM4tgfjnLHjEoaRKLdbpSIDJhiLtZE), Jaemin Cho, TensorFlow Korea, 2017 [[Slide]](https://www.slideshare.net/JaeminCho6/developing-korean-chatbot-101-71013451)
- [Building AI Chat bot using Python 3 & TensorFlow](https://speakerdeck.com/inureyes/building-ai-chat-bot-using-python-3-and-tensorflow), Jeongkyu Shin, PyCon, 2016
- [Deep Learning for Chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/), Denny Britz, WildML, 2016
- [Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/), Suriyadeepan Ramamoorthy, Personal Blog, 2016

Research Papers

- [A Persona-Based Neural Conversation Model](https://www.microsoft.com/en-us/research/publication/persona-based-neural-conversation-model/), Jiwei Li et al., ACL, 2016
- [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808), Iulian V. Serban et al., AAAI, 2016
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869), Oriol Vinyals et al., ICML, 2015 #Google #LSTM #seq2seq
- [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](https://arxiv.org/abs/1506.06714), Alessandro Sordoni et al., NAACL-HLT, 2015
- [Neural Responding Machine for Short-Text Conversation](https://arxiv.org/abs/1503.02364), Lifeng Shang et al., ACL, 2015

Datasets

- [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

## Sentiment Analysis

Reviews & Tutorials

- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/), Denny Britz, WildML, 2015 [[Code]](https://github.com/dennybritz/cnn-text-classification-tf)
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), Denny Britz, WildML, 2015
- [LSTM Networks for Sentiment Analysis](http://www.deeplearning.net/tutorial/lstm.html), Pierre Luc Carrier, deeplearning.net, 2012

Research Papers

- [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188v1), Nal Kalchbrenner et al., ACL, 2014 
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), Yoon Kim, EMNLP, 2014
- [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf), Richard Socher et al., EMNLP, 2013

Datasets

- [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [Large Movie Review Dataset (IMDB)](http://ai.stanford.edu/~amaas/data/sentiment/)

## Language Modeling

Reviews & Tutorials

- [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) [[Part 1]](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) [[Part 2]](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/) [[Part 3]](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) [[Part 4]](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/), Denny Britz, WildML, 2015 [[Code]](https://github.com/dennybritz/rnn-tutorial-gru-lstm)

Research Papers

- [Recurrent Neural Network based Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf), Tomas Mikolov et al., Interspeech, 2010 [[Code]](http://www.fit.vutbr.cz/~imikolov/rnnlm/)
- [A Scalable Hierarchical Distributed Language Model](https://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model) (Hierarchical Softmax), Andriy Mnih, NIPS, 2008
- [A Neural Probabilistic Language Model](https://papers.nips.cc/paper/1839-a-neural-probabilistic-language-model.pdf), Yoshua Bengio et al., NIPS, 2001

Datasets

- [The WikiText Language Modeling Dataset](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
- [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
- [Common Crawl](http://commoncrawl.org/the-data/)

## Uncategorized

Research Papers

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), Kelvin Xu, ICML, 2015
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555), Oriol Vinyals et al., CVPR, 2015
- [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf), Andrej Karpathy et al., CVPR, 2015 [[Code]](https://github.com/karpathy/neuraltalk) [[Demo]](http://cs.stanford.edu/people/karpathy/deepimagesent/rankingdemo/)
- [Modeling Mention, Context and Entity with Neural Networks for Entity Disambiguation](http://ir.hit.edu.cn/~dytang/paper/ijcai2015/ijcai15-yaming.pdf), Yaming Sun et al., IJCAI, 2015
- [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](http://dl.acm.org/citation.cfm?id=2505665), Po-Sen Huang et al., CIKM, 2013
- [Parsing Natural Scenes and Natural Language with Recursive Neural Networks](http://ai.stanford.edu/~ang/papers/icml11-ParsingWithRecursiveNeuralNetworks.pdf), Richard Socher et al., ICML, 2011
- [Local and Global Algorithms for Disambiguation to Wikipedia](http://web.eecs.umich.edu/~mrander/pubs/RatinovDoRo.pdf), Lev Ratinov et al. ACL, 2011 