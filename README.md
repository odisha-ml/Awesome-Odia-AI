# Awesome-Odia-AI
Curated list of all AI related resources in Odia Language.

Table of Contents
=================
* [NLP](#nlp)
  * [Translation](#translation)
  * [Transliteration](#transliteration)
  * [Language Understanding](#language-understanding)
  * [Language Generation](#language-generation)
  * [Text Classification](#text-classification)
  * [Text Dataset](#text-dataset)
    * [Parallel Translation Corpus](#parallel-translation-corpus)
    * [Monolingual Corpus](#monolingual-corpus)
    * [Lexical Resources](#lexical-resources)
    * [POS Tagged Corpus](#pos-tagged-corpus)
    * [Dialect Detection Corpus](#dialect-detection-corpus)
  * [Models](#models)
      * [Language Model](#language-model)
      * [Word Embedding](#word-embedding)
      * [Morphanalyzers](#morphanalyzers)
  * [NLP Libraries / Tools](#nlp-libraries--tools)
  * [Other NLP Resources](#other-nlp-resources)
* [Audio](#audio)
  * [Speech Recognition](#speech-recognition)
  * [Text-to-Speech](#text-to-speech)
  * [Speech Dataset](#speech-dataset)
* [Computer Vision](#computer-vision)
  * [OCR](#ocr)
* [Events](#events)
* [Community](#community)

## NLP

### Translation

- Sua: Machine Translation from English to Odia language [Repo](https://github.com/soumendrak/MTEnglish2Odia)
- IndicTrans: [Paper](https://arxiv.org/abs/2104.05596) [Web](https://ai4bharat.iitm.ac.in/indictrans/)
- IndicTrans2: [Paper](https://arxiv.org/abs/2305.16307) [Web](https://ai4bharat.iitm.ac.in/indic-trans2/) [Code](https://github.com/AI4Bharat/IndicTrans2)
  
### Transliteration

- IndicXlit: [Paper](https://arxiv.org/abs/2205.03018)[Web](https://ai4bharat.iitm.ac.in/areas/transliteration/") [Code](https://github.com/AI4Bharat/IndicTrans2) [Demo](https://xlit.ai4bharat.org/) [PyPi](https://pypi.org/project/ai4bharat-transliteration)

### Language Understanding
### Language Generation
### Text Dataset

#### Parallel Translation Corpus
* <a href="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3211">OdiEnCorp 2.0</a> : This dataset contains 97K English-Odia parallel sentences and serving in <a href="http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2020/index.html"> WAT2020</a> for Odia-English machine translation task. <a href="https://www.aclweb.org/anthology/2020.wildre-1.3.pdf">Paper</a> 
* <a href="http://opus.nlpl.eu/">OPUS Corpus</a> : It contains parallel sentences of other languages with Odia. The collection of data are domain-specific and noisy.  
* <a href="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879">OdiEnCorp 1.0</a> : This dataset contains 30K English-Odia parallel sentences. <a href="https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47">Paper</a> 
* <a href="https://github.com/anoopkunchukuttan/indowordnet_parallel">IndoWordnet Parallel Corpus</a> : Parallel corpora mined from IndoWordNet gloss and/or examples for Indian-Indian language corpora (6.3 million segments, 18 languages including Odia). <a href="https://github.com/anoopkunchukuttan/indowordnet_parallel/blob/master/iwn_parallel_2020.pdf">Paper</a>
* <a href="http://data.statmt.org/pmindia/">PMIndia</a> : Parallel corpus for En-Indian languages mined from Mann ki Baat speeches of the PM of India. It contains 38K English-Odia parallel sentences.<a href="https://arxiv.org/abs/2001.09907">Paper</a> 
* <a href="http://preon.iiit.ac.in/~jerin/bhasha/">CVIT PIB</a> : Parallel corpus for En-Indian languages mined from press information bureau website of India. It contains 60K English-Odia parallel sentences.
* <a href="https://ai4bharat.iitm.ac.in//samanantar/">Samanantar</a> is the largest publicly available parallel corpora collection for Indic languages. The corpus has 49.6M sentence pairs between English to Indian Languages.
* <a href="https://ai4bharat.iitm.ac.in/bpcc/">BPCC</a> is a comprehensive and publicly available parallel corpus containing a mix of Human labelled data and automatically mined data; totaling to approximately 230 million bitext pairs.



#### Monolingual Corpus
* <a href="https://www.lancaster.ac.uk/fass/projects/corpus/emille/">EMILLE Corpus</a> : It contains fourteen monolingual corpora for Indian languages including Odia.<a href="https://www.lancaster.ac.uk/fass/projects/corpus/emille/MANUAL.htm">Manual</a> 
* <a href="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879">OdiEnCorp 1.0</a> : This dataset contains 221K Odia sentences.<a href="https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47">Paper</a> 
* <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus">AI4Bharat-IndicNLP Corpus</a> : The text corpus not available now (will be available later). It used 3.5M Odia sentences to build the embedding. Vocabulary frequency files are available.<a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf">Paper</a>
* <a href="https://oscar-corpus.com/">OSCAR Corpus</a> : It contains around 300K Odia sentences.

#### Lexical Resources
* <a href="http://www.cfilt.iitb.ac.in/indowordnet/">IndoWordNet</a> : Wordnet for Indian languages including Odia.


#### POS Tagged corpus
* <a href="http://sanskrit.jnu.ac.in/ilci/index.jsp/">Indian Language Corpora Initiative</a> : It contains parallel annotated corpora in 12 Indian languages including Odia (tourism and health domain). 
*  <a href="https://github.com/UniversalDependencies/UD_Odia-ODTB/tree/dev">Odia Treebank</a> : The treebank contains approx. 1082 tokens (100 sentences) in Odia.
<a href="https://lnkd.in/evgspdqm">Paper</a>


#### Dialect Detection corpus
* <a href="https://github.com/shantipriyap/Odia-Santali-Dialect-Detection-Dataset/">Odia-Santali Dialect Detection Corpus</a> : This corpus contains text data of Odia and Santali written in Odia script. 

### Models

#### Language Model
* <a href="https://github.com/goru001/nlp-for-odia">Language Model</a> : Pretrained Odia Language Model. 
* <a href="https://colab.research.google.com/gist/satyapb2002/aeb7bf9a686a9c7294ec5725ff53fa49/odiabert_languagemodel.ipynb#scrollTo=xy_H5EjNTdRE">BertOdia</a> : Bert-based Odia Language Model.

#### Word Embedding
* <a href="https://fasttext.cc/docs/en/crawl-vectors.html">FastText (CommonCrawl + Wikipedia)</a> : Pretrained Word vector (CommonCrawl + Wikipedia). Trained on Common Crawl and Wikipedia using fastText. Select the language "oriya" from the model list.
* <a href="https://fasttext.cc/docs/en/pretrained-vectors.html">FastText (Wikipedia)</a> : Pretrained Word vector (Wikipedia). Trained on Wikipedia using fastText. Select the language "oriya" from the model list.
* <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus">AI4Bharat IndicNLP Project</a> : Pretrained Word embeddings for 10 Indian languages including Odia. <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf">Paper</a>

#### Morphanalyzers
* <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus">IndicNLP Morphanalyzers</a> : Unsupervised morphanalyzers for 10 Indian languages including Odia learnt using morfessor.


### Text Classification
* <a href="https://www.kaggle.com/disisbig/odia-news-dataset">Odia News Article Classification</a> : This dataset contains approxmately 19,000 news article headlines collected from Odia news websites. The labeled dataset is splitted into training and testset suitable for supervised text classification. 
* <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus">AI4Bharat IndicNLP News Articles</a> : This datasets comprising news articles and their categories for 9 languages including Odia. For Odia language, it has 4 classes (business, crime, entertainment, sports) and each class contains 7.5K news articles. The dataset is balanced across classes. <a href="https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf">Paper</a>

### NLP Libraries / Tools 
* <a href="https://github.com/anoopkunchukuttan/indic_nlp_library">Indic NLP Library</a> : It is a python based NLP library for Indian language text processing including Odia.
* <a href="https://indic-ocr.github.io/">Indic-OCR</a> : OCR tools for Indic scripts including Odia. Also, supports Ol Chiki (Santali).
* <a href="https://github.com/shantipriyap/odia_nlp">Odia Romanization Script</a> : The perl script "odiaroman" maps the Devnagri (Odia) to Latin.

### Other NLP Resources
* <a href="http://tdil-dc.in/index.php?lang=en">TDIL</a> : It contains language application, resources, and tools for Indian languages including Odia. It contains many language applications, resources, and tools for Odia such as Odia terminology application, Odia language search engine, wordnet, English-Odia parallel text corpus, English-Odia machine-assisted translation, text-to-speech software, and many more.  

## Audio
### Speech Recognition
### Text-to-Speech
- Indic-TTS [Paper](https://arxiv.org/abs/2211.09536) [Code](https://github.com/AI4Bharat/Indic-TTS) [Try It Live](https://models.ai4bharat.org/#/tts)] [Video](https://youtu.be/I3eo8IUAP7s)


### Speech Dataset
* <a href="https://www.iitm.ac.in/donlab/tts/index.php">IIT Madras IndicTTS</a> : The Indic TTS project develops the text-to-speech (TTS) synthesis system for Indian languages including Odia. The database contains spoken sentences/utterances recorded by both Male and Female native speakers.
* <a href="http://www.ldcil.org/resourcesSpeechCorpOriya.aspx">LDC-IL</a> :  It includes Odia annotated speech corpora which has voices of 450 different native speakers.
## Computer Vision
### OCR

## Events
- Global Conference: [2023 Pt 2](https://www.youtube.com/live/KZB9bfKkLgM?si=3i9eY22xT-1yZTD8) [2023 Pt 1](https://www.youtube.com/live/GPkWL-9akQc?si=uh0Ay0SKEVlRnX3U) |[2022](https://www.youtube.com/live/MPrU-3s8ccw?si=gxbOFyfI3j3g8UsH) |[2021](https://www.youtube.com/live/iX59_YJzINs?si=TiZmMMeB6Hy28JcZ) |[2020](https://www.youtube.com/live/PF5DScCr5SI?si=znfuwHbrIgHSzgnO)||
- Summer School: [2022](https://youtube.com/playlist?list=PLQCNXbSwgbGwMW4rGHr_LIfSCMh-7lgbR&si=f_b94K73yVAKST1E) ||

## Community
- [Odias in AI/ML](https://www.odishaai.org/)
