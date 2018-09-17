# paper

## Recent Trends in Deep Learning BasedNatural Language Processing: [English](https://arxiv.org/pdf/1708.02709.pdf), [Korean](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)

What is different from Machine learning on NLP to Deep Learning on NLP

    * shallow models(e.g. SVM and logistic regression) -> dense vector representations
    * hand-crafted features ->  multi-level automatic feature


1. Distributed Represntation
    * Word embedding :  
        Measuring similarity between vectors is possible, using measures such as cosine similarity. 
        Word Embedding are often used as the first data processing layer in a deep learning model.
    * Word2vec(Apply the CBOW or Skip-gram)
        
        * Frameworks providing embedding tools and methods
        
          [S-Space(Java)](https://github.com/fozziethebeat/S-Space), 
          [Semanticvectors(Java)](https://github.com/semanticvectors/), 
          [Gensim(Python)](https://radimrehurek.com/gensim/), 
          [Pydsm(Python)](https://github.com/jimmycallin/pydsm/), 
          [Dissect(Python)](http://clic.cimec.unitn.it/composes/toolkit/), 
          [Fasttext(Python)](https://fasttext.cc/)
          
    * Character Embeddings
         
        For tasks such as POS-tagging and NER, intra-word morphological and shape information can also be very useful. A common phenomenon for languages with large vocabularies is the unknown word issue or out-of-vocabulary word(OOV)
issue.

2. Convolution Neural Networks


   A look-up table was used to transform each word into a vector of user-defined dimensions. Thus, an input sequence {s1, s2, ...sn} of n words was transformed into a series of vectors {ws1, ws2, ...wsn} by applying the look-up table to each of its words
 
   * Basic CNN
   
       [for korea](http://docs.likejazz.com/cnn-text-classification-tf/), [source](https://github.com/likejazz/cnn-text-classification-tf), korea paper ([SNU : Large-Scale Text Classification Methodology
with Convolutional Neural Network](https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2015W_JoHY.pdf))
       [Yoon Kim's site](http://www.people.fas.harvard.edu/~yoonkim/)
       
      
      * Sentence Modeling
      
         Sequential convolutions help in improved mining of the sentence to grasp a truly abstract representations comprising rich semantic information. 

      * window approach
         
         many NLP tasks, such as NER, POS tagging, and SRL, require word-based predictions
         To adapt CNNsfor such tasks, a window approach is used, which assumes that the tag of a word primarily depends on its neighboring words.
         The ultimate goal of word-level classification is generally to assign a sequence of labels to the entire sentence. 


   * Applications
   
      This simple network, however, had many shortcomings with the CNN’s inability to model long distance dependencies standing as the main issue
      
      [Kalchbrenner et al. [44]](http://www.aclweb.org/anthology/P14-1062), who published a prominent paper where they proposed a dynamic convolutional neural network (DCNN) for semantic modeling of sentences.
      [source example: Theano](https://github.com/FredericGodin/DynamicCNN), [Review of a paper](https://github.com/YBIGTA/DeepNLP-Study/wiki/Review-of-a-paper-:-A-Convolutional-Neural-Network-for-Modeling-Sentences)
      
      Tasks involving sentiment analysis also require effective extraction of aspects along with their sentiment polarities. [Ruder et al.](http://www.aclweb.org/anthology/S16-1053) applied a CNN where in the input they concatenated an aspect vector with the word embeddings to get competitive results. [source example](https://github.com/hurshprasad/ABSA)
      
      [Denil et al.](https://arxiv.org/pdf/1406.3830.pdf) applied DCNN to map meanings of words that constitute a sentence to that of documents for summarization.
      
      In the domain of QA, [Yih et al.]((http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)) proposed to measure the semantic similarity between a question and entries in a knowledge base (KB) to determine what supporting fact in the KB to look for when answering a question
      
      [Dong et al.](http://www.aclweb.org/anthology/P15-1026) introduced a multi-column CNN (MCCNN) to analyze and understand questions from multiple aspects and create their representations.
      
      [Severyn and Moschitti](https://arxiv.org/pdf/1604.01178.pdf) also used CNN network to model optimal representations of question and answer sentences.
      
      [Chen et al.](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf) proposed a modified pooling strategy: dynamic multi-pooling CNN (DMCNN) To overcome this loss of information for multiple-event modeling because traditional max-pooling strategies perform this in a translation invariant form
      
      Speech recognition also requires such invariance and, thus, [Abdel-Hamid et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf?ranMID=24542&ranEAID=je6NUbpObpQ&ranSiteID=je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw&epi=je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw&irgwc=1&OCID=AID681541_aff_7593_1243925&tduid=(ir_d32d3c05N7d3d1b2f5d8a9996762b938b)(7593)(1243925)(je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw)()&irclickid=d32d3c05N7d3d1b2f5d8a9996762b938b) used a hybrid CNN-HMM model which provided invariance to frequency shifts along the frequency axis.
      
      [Palaz et al.](https://ronan.collobert.com/pub/matos/2015_cnnspeech_interspeech.pdf) performed extensive analysis of CNN-based speech recognition systems when given raw speech as input.
      
       [Tu et al.](http://www.aclweb.org/anthology/P15-2088) addressed this task by considering both the semantic similarity of the translation pair and their respective contexts.
       
       Overall, CNNs are extremely effective in mining semantic clues in contextual windows. However, they are very data heavy
models. 


3. Recurrent Neural Networks
   
   The term “recurrent” applies as they perform the same task over each instance of the sequence such that the output is dependent on the previous computations and results. Given that an RNN performs sequential processing by modeling units in sequence, it has the ability to capture the inherent sequential nature present in language, where units are characters, words or even sentences. RNNs are tailor-made for modeling such context dependencies in language and similar sequence modeling tasks, which resulted to be a strong motivation for researchers to use RNNs over CNNs in these areas. Another factor aiding RNN’s suitability for [sequence modeling tasks lies in its ability to model variable length of text,including very long sentences, paragraphs and even documents](http://aclweb.org/anthology/D15-1167)
   
NLP tasks 
   
   **language modeling:** 
   
   [Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
   
   [Generating Text with Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
   
   **machine translation:**
   
   [A Recursive Recurrent Neural Network for Statistical Machine Translation](http://www.aclweb.org/anthology/P14-1140.pdf)
   
   [Joint Language and Translation Modeling with Recurrent Neural Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EMNLP2013RNNMT.pdf)
   
   [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
         
   **speech recognition:** 
   
   [THE USE OF RECURRENT NEURAL NETWORKS IN CONTINUOUS SPEECH RECOGNITION](http://www.cstr.ed.ac.uk/downloads/publications/1996/rnn4csr96.pdf)
   
   [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf) 
   
   [Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1701.02720.pdf)
   
   [LONG SHORT-TERM MEMORY BASED RECURRENT NEURAL NETWORK ARCHITECTURES FOR LARGE VOCABULARY SPEECH RECOGNITION](https://arxiv.org/pdf/1402.1128.pdf)
   
   **image captioning:** 
   
   [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
               
   
   Recently, several works provided contrasting evidence on the superiority of CNNs over RNNs. Even in RNN-suited tasks like language modeling, [CNNs achieved competitive performance over RNNs](https://arxiv.org/pdf/1612.08083.pdf)
   
   [Yin et al.](https://arxiv.org/pdf/1702.01923.pdf) provided interesting insights on the comparative performance between RNNs and CNNs. After testing on multiple NLP tasks that included sentiment classification, QA, and POS tagging, they concluded that there is no clear winner
   
   * RNN Model
   
      * Simple RNN
         
         In the context of NLP, RNNs are primarily based on [Elman network](http://psych.colorado.edu/~kimlab/Elman1990.pdf) and they are originally threelayer networks.
         
         In the figure, xt is taken as the input to the network at time step t and st represents the hidden state at the same time step. Calculation of st is based as per the equation: st = f(Uxt + Wst−1) 
         
      * Long Short-Term Memory(LSTM)
      
         LSTM [1](http://www.bioinf.jku.at/publications/older/2604.pdf), [2](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf) has additional “forget” gates over the simple RNN. Its unique mechanism enables it to overcome both the vanishing and exploding gradient problem. Consisting of three gates: input, forget and output gates
         
      * Gated Recuurent Units(GRU)
      
         Another gated RNN variant called [GRU](https://arxiv.org/pdf/1406.1078.pdf) of lesser complexity was invented with empirically similar performances to LSTM in most tasks. GRU comprises of two gates, reset gate and update gate, and handles the flow of information like an LSTM sans a memory unit


   * Applications
   
      * RNN for word-level classification
         
         **bidirectional LSTM for NER([Lample et al](https://arxiv.org/pdf/1603.01360.pdf))**: The network captured arbitrarily long context information around the target word (curbing the limitation of a fixed window size) resulting in two fixed-size vector, on top of which another fully-connected layer was built. They used a CRF layer at last for the final entity tagging.
         
         **Generating Sequences With RNNs([Graves](https://arxiv.org/pdf/1308.0850.pdf))**: the effectiveness of RNNs in modeling complex sequences with long range context structure. also proposed deep RNNs where multiple layers of hidden states were used to enhance the modeling
         
         **prediction of a word on the words ahead using by replacing a feed-forward neural network with an RNN([Sundermeyer et al](https://www.lsv.uni-saarland.de/fileadmin/teaching/seminars/ASR-2015/DL-Seminar/From_Feedforward_to_Recurrent_LSTM_Neural_Networks_for_Language_Modeling.pdf)**: In their work, they proposed a typical hierarchy in neural network architectures where feed-forward neural networks gave considerable improvement over traditional count-based language models, which in turn were superseded by RNNs and later by LSTMs. 
         
      * RNN for sentence-level classification
      
         **predicting sentiment polarity ([Wang et al.](http://www.aclweb.org/anthology/P15-1130))**: This simple strategy proved competitive to the more complex DCNN structure
         
         **semantic matching between texts([Lowe et al](https://arxiv.org/pdf/1506.08909.pdf)**: o match a message with candidate responses with Dual-LSTM, which encodes both as fixed-size vectors and then measure their inner product as the basis to rank candidate responses
         
      * RNN for generating language: 
      
         

      
      
      
      
      
       

      
      
