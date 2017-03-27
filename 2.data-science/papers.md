# Deep Learning Papers Reading Roadmap

---------------------------------------

# 1 Deep Learning History and Basics

## 1.1 ImageNet Evolution（Deep Learning broke out from here）

**[1]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[2]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[3]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:

**[4]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

## 1.2 Speech Recognition Evolution

**[1]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(Breakthrough in speech recognition)**:star::star::star::star:

**[2]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN)**:star::star::star:

**[3]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:

**[4]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](http://arxiv.org/pdf/1507.06947) **(Google Speech Recognition System)** :star::star::star:

**[5]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(Baidu Speech Recognition System)** :star::star::star::star:

**[6]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](https://arxiv.org/pdf/1610.05256v1) **(State-of-the-art in speech recognition, Microsoft)** :star::star::star::star:


# 2 Deep Learning Method

## 2.1 Model

**[1]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** :star::star::star:

**[2]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) :star::star::star:

**[3]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)** :star::star::star::star:

**[4]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)** :star::star::star::star:

**[5]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**  :star::star::star:

**[6]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

**[7]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)** :star::star::star:

**[8]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)** :star::star::star:

## 2.2 Optimization

**[1]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** :star::star:

**[2]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)** :star::star::star:

**[3]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[4]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[5]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[1]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[2]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[3]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[4]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[5]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[6]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[7]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[1]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[2]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[3]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[4]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[5]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:

**[5]** CY Wu, A Ahmed, A Beutel, AJ Smola, H Jing. "**Recurrent Recommender Networks**." WSDM (2017). [[pdf]](http://alexbeutel.com/papers/rrn_wsdm2017.pdf))

## 2.5 Deep Reinforcement Learning

**[1]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[2]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) **(Milestone)** :star::star::star::star::star:

**[3]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)**  :star::star::star::star:

**[4]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** :star::star::star::star::star:

**[5]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** :star::star::star::star:

**[6]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[7]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[8]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)** :star::star::star::star::star:

## 2.6 Deep Transfer Learning / Lifelong Learning / especially for RL

**[1]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(A Tutorial)** :star::star::star:

**[2]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(A brief discussion about lifelong learning)**  :star::star::star:

**[3]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(Godfather's Work)** :star::star::star::star:

**[4]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)** :star::star::star:

**[5]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** :star::star::star:

**[6]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(Outstanding Work, A novel idea)** :star::star::star::star::star:


## 2.7 One Shot Deep Learning

**[1]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[2]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[3]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)** :star::star::star::star:

**[4]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[5]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI tasks)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail cloze style questions)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(state-of-the-art in text classification)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(slightly worse than state-of-the-art, but a lot faster)** :star::star::star:


## 3.2 Image Caption
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## 3.3 Machine Translation


**[1]** Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](http://arxiv.org/pdf/1410.8206) :star::star::star::star:


**[2]** Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf):star::star::star:

**[3]** Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](http://arxiv.org/pdf/1508.04025) :star::star::star::star:

**[4]** Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf):star::star:

**[5]** Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf):star::star::star::star::star:

**[6]** Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) **(Milestone)** :star::star::star::star:


## 3.4 Art

**[1]** Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**
:star::star::star::star:

**[2]** Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](http://arxiv.org/pdf/1508.06576) **(Outstanding Work, most successful method currently)** :star::star::star::star::star:

**[3]** Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](https://arxiv.org/pdf/1609.03552) **(iGAN)** :star::star::star::star:

**[4]** Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](http://arxiv.org/pdf/1603.01768) **(Neural Doodle)** :star::star::star::star:

**[5]** Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](http://arxiv.org/pdf/1603.08511) :star::star::star::star:

**[6]** Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](https://arxiv.org/pdf/1603.08155.pdf) :star::star::star::star:

**[7]** Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf) :star::star::star::star:

**[8]** Gatys, Leon and Ecker, et al."**Controlling Perceptual Factors in Neural Style Transfer**." arXiv preprint arXiv:1611.07865 (2016). [[pdf]](https://arxiv.org/pdf/1611.07865.pdf) **(control style transfer over spatial location,colour information and across spatial scale)**:star::star::star::star:

**[9]** Ulyanov, Dmitry and Lebedev, Vadim, et al. "**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**." arXiv preprint arXiv:1603.03417(2016). [[pdf]](http://arxiv.org/abs/1603.03417) **(texture generation and style transfer)** :star::star::star::star:


## 3.5 Object Segmentation

**[1]** J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf) :star::star::star::star::star:

**[2]** L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) :star::star::star::star::star:

**[3]** Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf) :star::star::star::star:

**[4]** Dai, J., He, K., Sun, J. "**Instance-aware semantic segmentation via multi-task network cascades**." in CVPR. 2016 [[pdf]](https://arxiv.org/pdf/1512.04412v1.pdf) :star::star::star:

**[5]** Dai, J., He, K., Sun, J. "**Instance-sensitive Fully Convolutional Networks**." arXiv preprint arXiv:1603.08678 (2016). [[pdf]](https://arxiv.org/pdf/1603.08678v1.pdf) :star::star::star:


# 4 Generative Adversary Network

## 4.1 GAN Theory

**[1]** **GAN**: Goodfellow, Ian, et al. "**Generative adversarial nets.**" Advances in Neural Information Processing Systems. 2014.

**[2]** **LAPGAN**: Denton, Emily L., Soumith Chintala, and Rob Fergus. "**Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks.**" Advances in neural information processing systems. 2015.


**[3]** **DCGAN**: Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks.**" arXiv preprint arXiv:1511.06434 (2015).


**[4]** **Improved GAN**: Salimans, Tim, et al. "**Improved techniques for training gans.**" arXiv preprint arXiv:1606.03498 (2016).


**[5]** **InfoGAN**: Chen, Xi, et al. "**Infogan: Interpretable representation learning by information maximizing generative adversarial nets.**" arXiv preprint arXiv:1606.03657(2016).


**[6]** **EnergyGAN**: Zhao, Junbo, Michael Mathieu, and Yann LeCun. "**Energy-based Generative Adversarial Network.**" arXiv preprint arXiv:1609.03126 (2016).


**[7]** Creswell, Antonia, and Anil A. Bharath. "**Task Specific Adversarial Cost Function.**" arXiv preprint arXiv:1609.08661 (2016).


**[8]** **f-GAN**: Nowozin, Sebastian, Botond Cseke, and Ryota Tomioka. "**f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization.**" arXiv preprint arXiv:1606.00709 (2016).


**[9]** **Unrolled Generative Adversarial Networks**, ICLR 2017 Open Review


**[10]** **Improving Generative Adversarial Networks with Denoising Feature Matching**, ICLR 2017 Open Review


**[11]** **Mode Regularized Generative Adversarial Networks**, ICLR 2017 Open Review


**[12]** **b-GAN**: **Unified Framework of Generative Adversarial Networks**, ICLR 2017 Open Review


**[13]** Mohamed, Shakir, and Balaji Lakshminarayanan. "**Learning in Implicit Generative Models.**" arXiv preprint arXiv:1610.03483 (2016).


## 4.2 GAN in Semi-Supervised

**[1]** Springenberg, Jost Tobias. "**Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks.**" arXiv preprint arXiv:1511.06390 (2015).

**[2]** Odena, Augustus. "**Semi-Supervised Learning with Generative Adversarial Networks.**" arXiv preprint arXiv:1606.01583 (2016).

## 4.3 Muti-GAN

**[1]** CoupledGAN: Liu, Ming-Yu, and Oncel Tuzel. "**Coupled Generative Adversarial Networks.**" arXiv preprint arXiv:1606.07536 (2016).

**[2]** Wang, Xiaolong, and Abhinav Gupta. "**Generative Image Modeling using Style and Structure Adversarial Networks.**" arXiv preprint arXiv:1603.05631(2016).
Generative Adversarial Parallelization, ICLR 2017 Open Review

**[3]** **LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation**, ICLR 2017 Open Review

## 4.4 GAN with other Generative Model

**[1]** Dosovitskiy, Alexey, and Thomas Brox. "**Generating images with perceptual similarity metrics based on deep networks.**" arXiv preprint arXiv:1602.02644(2016).

**[2]** Larsen, Anders Boesen Lindbo, Søren Kaae Sønderby, and Ole Winther. "**Autoencoding beyond pixels using a learned similarity metric.**" arXiv preprint arXiv:1512.09300 (2015).

**[3]** Theis, Lucas, and Matthias Bethge. "**Generative image modeling using spatial lstms.**" Advances in Neural Information Processing Systems. 2015.

## 4.5 GAN with RNN

**[1]** Im, Daniel Jiwoong, et al. "**Generating images with recurrent adversarial networks.**" arXiv preprint arXiv:1602.05110 (2016).

**[2]** Kwak, Hanock, and Byoung-Tak Zhang. "**Generating Images Part by Part with Composite Generative Adversarial Networks.**" arXiv preprint arXiv:1607.05387 (2016).

**[3]** Yu, Lantao, et al. "**SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient.**" arXiv preprint arXiv:1609.05473 (2016).

## 4.6 GAN in Application

**[1]** Zhu, Jun-Yan, et al. "**Generative visual manipulation on the natural image manifold.**" European Conference on Computer Vision. Springer International Publishing, 2016.

**[2]** Creswell, Antonia, and Anil Anthony Bharath. "**Adversarial Training For Sketch Retrieval.**" European Conference on Computer Vision. Springer International Publishing, 2016.

**[3]** Reed, Scott, et al. "**Generative adversarial text to image synthesis.**" arXiv preprint arXiv:1605.05396 (2016).

**[4]** Ravanbakhsh, Siamak, et al. "**Enabling Dark Energy Science with Deep Generative Models of Galaxy Images.**" arXiv preprint arXiv:1609.05796(2016).

**[5]** Abadi, Martín, and David G. Andersen. "**Learning to Protect Communications with Adversarial Neural Cryptography.**" arXiv preprint arXiv:1610.06918(2016).

**[6]** Odena, Augustus, Christopher Olah, and Jonathon Shlens. "**Conditional Image Synthesis With Auxiliary Classifier GANs.**" arXiv preprint arXiv:1610.09585 (2016).

**[7]** Ledig, Christian, et al. "**Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.**" arXiv preprint arXiv:1609.04802 (2016).

**[8]** Nguyen, Anh, et al. "**Synthesizing the preferred inputs for neurons in neural networks via deep generator networks.**" arXiv preprint arXiv:1605.09304(2016).


# 5 Other Topic

**[1]** Thorsten Joachims, Adith Swaminathan, Tobias Schnabel. "**Unbiased Learning-to-Rank with Biased Feedback**" arXiv:1608.04468.[[pdf]](https://arxiv.org/abs/1608.04468)

**[2]** Han Cai, Kan Ren, Weinan Zhang, Kleanthis Malialis, Jun Wang, Yong Yu, Defeng Guo. "**Real-Time Bidding by Reinforcement Learning in Display Advertising**" arXiv:1701.02490.[[pdf]](https://arxiv.org/abs/1701.02490)

**[3]** Eugene Kharitonov, Alexey Drutsa, Pavel Serdyukov. "**Learning Sensitive Combinations of A/B Test Metrics**" [[pdf]](http://dl.acm.org/citation.cfm?id=3018708)

**[4]** Michael Bendersky, Xuanhui Wang, Donald Metzler, Marc Najork. "**Learning from User Interactions in Personal Search via Attribute Parameterization**" [[pdf]](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45660.pdf)
