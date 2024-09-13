# Awesome-Odia-AI
[![Awesome](_static/awesome.webp)](https://github.com/sindresorhus/awesome) 
[![Check Markdown links](https://github.com/odisha-ml/Awesome-Odia-AI/actions/workflows/deadlink_checker.yml/badge.svg)](https://github.com/odisha-ml/Awesome-Odia-AI/actions/workflows/deadlink_checker.yml)

Curated list of all AI related resources in Odia Language.

<details>
  <summary>
    Table of Contents
  </summary>
=================
* [NLP](#nlp)
  * [Translation](#translation)
  * [Transliteration](#transliteration)
  * [Language Understanding](#language-understanding)
   * [Language Model](#language-model)
   * [Word Embedding](#word-embedding)
   * [Morphanalyzers](#morphanalyzers)
  * [Language Generation](#language-generation)
  * [Text Classification](#text-classification)
  * [Text Dataset](#text-dataset)
    * [Parallel Translation Corpus](#parallel-translation-corpus)
    * [Monolingual Corpus](#monolingual-corpus)
    * [Lexical Resources](#lexical-resources)
    * [POS Tagged Corpus](#pos-tagged-corpus)
    * [Dialect Detection Corpus](#dialect-detection-corpus)
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

</details>

## NLP
### Translation

- Sua: Machine Translation from English to Odia language [[dataset](https://github.com/soumendrak/MTEnglish2Odia)[code](https://github.com/OdiaNLP/NMT)]
- IndicTrans: [[paper](https://arxiv.org/abs/2104.05596)] [[web](https://ai4bharat.iitm.ac.in/indictrans/)]
- IndicTrans2: [[paper](https://arxiv.org/abs/2305.16307)][[web](https://ai4bharat.iitm.ac.in/indic-trans2/)][[code](https://github.com/AI4Bharat/IndicTrans2)]
  
### Transliteration

- IndicXlit: [[paper](https://arxiv.org/abs/2205.03018)][[web](https://ai4bharat.iitm.ac.in/areas/transliteration/")][[code](https://github.com/AI4Bharat/IndicTrans2) ][[Demo](https://xlit.ai4bharat.org/)][[PyPi](https://pypi.org/project/ai4bharat-transliteration)]
- [open-source unicode converter](https://github.com/OdiaWikimedia/Converter) to transliterate between various languages to Odia language [Demo](https://or.wikipedia.org/s/1hv1).

### Language Understanding
#### Datasets
- IndicCorp: Large sentence-level monolingual corpora for 11 Indian languages and Indian English containing 8.5 billions words (250 million sentences) from multiple news domain sources. [[paper]()][[code]()][[web](https://ai4bharat.iitm.ac.in/indiccorp)]  
- Naamapadam: Training and evaluation datasets for named entity recognition in multiple Indian language. [[paper](https://arxiv.org/abs/2212.10168)][[huggingface](https://huggingface.co/datasets/ai4bharat/naamapadam)][[web](https://ai4bharat.iitm.ac.in/naamapadam/)]  
- IndicCorp v2: he largest collection of texts for Indic languages consisting of 20.9 billion tokens of which 14.4B tokens correspond to 23 Indic languages and 6.5B tokens of Indian English content curated from Indian websites. [[paper](https://arxiv.org/abs/2212.05409)][[code](https://github.com/AI4Bharat/IndicBERT/tree/main?tab=readme-ov-file#indiccorp-v2)]  

  
#### Language Model
- [Language Model](https://github.com/goru001/nlp-for-odia) : Pretrained Odia Language Model. 
- [BertOdia](https://colab.research.google.com/gist/satyapb2002/aeb7bf9a686a9c7294ec5725ff53fa49/odiabert_languagemodel.ipynb#scrollTo=xy_H5EjNTdRE) : Bert-based Odia Language Model.
- IndicBERT: Multilingual, compact ALBERT language model trained on IndicCorp covering 11 major Indian and English. Small model (18 million parameters) that is competitive with large LMs for Indian language tasks. [[paper](https://aclanthology.org/2020.findings-emnlp.445/)][[code](https://github.com/AI4Bharat/Indic-BERT-v1)][[web](https://ai4bharat.iitm.ac.in/language-understanding)]
- IndicNER: Named Entity Recognizer models for multiple Indian languages. The models are trained on the Naampadam NER dataset mined from Samanantar parallel corpora. [[paper](https://arxiv.org/abs/2212.10168)][[huggingface](https://huggingface.co/ai4bharat/IndicNER)][[web](https://ai4bharat.iitm.ac.in/language-understanding)]
- IndicBERTv2: Language model trained on IndicCorp v2 with competitive performance on IndicXTREME [[paper](https://arxiv.org/abs/2212.05409)][[code](https://github.com/AI4Bharat/IndicBERT)][[web](https://ai4bharat.iitm.ac.in/language-understanding)]

#### Word Embedding
- [FastText (CommonCrawl + Wikipedia)](https://fasttext.cc/docs/en/crawl-vectors.html) : Pretrained Word vector (CommonCrawl + Wikipedia). Trained on Common Crawl and Wikipedia using fastText. Select the language "oriya" from the model list.
- [FastText (Wikipedia)](https://fasttext.cc/docs/en/pretrained-vectors.html) : Pretrained Word vector (Wikipedia). Trained on Wikipedia using fastText. Select the language "oriya" from the model list.
- IndicFT: Word embeddings for 11 Indian languages trained on IndicCorp. The embeddings are based on the fastText model and are well suited for the morphologically rich nature of Indic languages. [[paper](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf)][[code]()][[web](https://ai4bharat.iitm.ac.in/indicft)]

  
#### Morphanalyzers
* [IndicNLP Morphanalyzers](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : Unsupervised morphanalyzers for 10 Indian languages including Odia learnt using morfessor.

### Language Generation

### Text Dataset

#### Parallel Translation Corpus
* [OdiEnCorp 2.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3211) : This dataset contains 97K English-Odia parallel sentences and serving in [WAT2020](http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2020/index.html) for Odia-English machine translation task. [Paper](https://www.aclweb.org/anthology/2020.wildre-1.3.pdf) 
* [OPUS Corpus](http://opus.nlpl.eu/) : It contains parallel sentences of other languages with Odia. The collection of data are domain-specific and noisy.  
* [OdiEnCorp 1.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879) : This dataset contains 30K English-Odia parallel sentences. [Paper](https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47) 
* [IndoWordnet Parallel Corpus](https://github.com/anoopkunchukuttan/indowordnet_parallel) : Parallel corpora mined from IndoWordNet gloss and/or examples for Indian-Indian language corpora (6.3 million segments, 18 languages including Odia). [Paper](https://github.com/anoopkunchukuttan/indowordnet_parallel/blob/master/iwn_parallel_2020.pdf)
* [PMIndia](http://data.statmt.org/pmindia/) : Parallel corpus for En-Indian languages mined from Mann ki Baat speeches of the PM of India. It contains 38K English-Odia parallel sentences.[Paper](https://arxiv.org/abs/2001.09907) 
* [CVIT PIB](http://preon.iiit.ac.in/~jerin/bhasha/) : Parallel corpus for En-Indian languages mined from press information bureau website of India. It contains 60K English-Odia parallel sentences.
* [Samanantar](https://ai4bharat.iitm.ac.in//samanantar/) is the largest publicly available parallel corpora collection for Indic languages. The corpus has 49.6M sentence pairs between English to Indian Languages.
* [BPCC](https://ai4bharat.iitm.ac.in/bpcc/) is a comprehensive and publicly available parallel corpus containing a mix of Human labelled data and automatically mined data; totaling to approximately 230 million bitext pairs.

#### Monolingual Corpus
* [EMILLE Corpus](https://www.lancaster.ac.uk/fass/projects/corpus/emille/) : It contains fourteen monolingual corpora for Indian languages including Odia.[Manual](https://www.lancaster.ac.uk/fass/projects/corpus/emille/MANUAL.htm) 
* [OdiEnCorp 1.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879) : This dataset contains 221K Odia sentences.[Paper](https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47) 
* [AI4Bharat-IndicNLP Corpus](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : The text corpus not available now (will be available later). It used 3.5M Odia sentences to build the embedding. Vocabulary frequency files are available.[Paper](https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf)
* [OSCAR Corpus](https://oscar-corpus.com/) : It contains around 300K Odia sentences.

#### Lexical Resources
* [IndoWordNet](http://www.cfilt.iitb.ac.in/indowordnet/) : Wordnet for Indian languages including Odia.


#### POS Tagged corpus
* [Indian Language Corpora Initiative](http://sanskrit.jnu.ac.in/ilci/index.jsp/) : It contains parallel annotated corpora in 12 Indian languages including Odia (tourism and health domain). 
*  [Odia Treebank](https://github.com/UniversalDependencies/UD_Odia-ODTB/tree/dev) : The treebank contains approx. 1082 tokens (100 sentences) in Odia.
[Paper](https://lnkd.in/evgspdqm)

#### Dialect Detection corpus
* [Odia-Santali Dialect Detection Corpus](https://github.com/shantipriyap/Odia-Santali-Dialect-Detection-Dataset/) : This corpus contains text data of Odia and Santali written in Odia script. 

### Text Classification
* [Odia News Article Classification](https://www.kaggle.com/disisbig/odia-news-dataset) : This dataset contains approxmately 19,000 news article headlines collected from Odia news websites. The labeled dataset is splitted into training and testset suitable for supervised text classification. 
* [AI4Bharat IndicNLP News Articles](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : This datasets comprising news articles and their categories for 9 languages including Odia. For Odia language, it has 4 classes (business, crime, entertainment, sports) and each class contains 7.5K news articles. The dataset is balanced across classes. [Paper](https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf)

### NLP Libraries / Tools 
* [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) : It is a python based NLP library for Indian language text processing including Odia.
* [Odia Romanization Script](https://github.com/shantipriyap/odia_nlp) : The perl script "odiaroman" maps the Devnagri (Odia) to Latin.

### Other NLP Resources
* [TDIL](http://tdil-dc.in/index.php?lang=en) : It contains language application, resources, and tools for Indian languages including Odia. It contains many language applications, resources, and tools for Odia such as Odia terminology application, Odia language search engine, wordnet, English-Odia parallel text corpus, English-Odia machine-assisted translation, text-to-speech software, and many more.  

## Audio

### Speech Recognition

### Text-to-Speech
- Indic-TTS [[Paper](https://arxiv.org/abs/2211.09536)][[Code](https://github.com/AI4Bharat/Indic-TTS)] [[Try It Live](https://models.ai4bharat.org/#/tts)]][[Video](https://youtu.be/I3eo8IUAP7s)]


### Speech Dataset
* [IIT Madras IndicTTS](https://www.iitm.ac.in/donlab/tts/index.php) : The Indic TTS project develops the text-to-speech (TTS) synthesis system for Indian languages including Odia. The database contains spoken sentences/utterances recorded by both Male and Female native speakers.
* [LDC-IL](http://www.ldcil.org/resourcesSpeechCorpOriya.aspx) :  It includes Odia annotated speech corpora which has voices of 450 different native speakers.
  
## Computer Vision

### OCR
* [Indic-OCR](https://indic-ocr.github.io/) : OCR tools for Indic scripts including Odia. Also, supports Ol Chiki (Santali).

## Events
- Global Conference: [2023 Pt 2](https://www.youtube.com/live/KZB9bfKkLgM?si=3i9eY22xT-1yZTD8) [2023 Pt 1](https://www.youtube.com/live/GPkWL-9akQc?si=uh0Ay0SKEVlRnX3U) |[2022](https://www.youtube.com/live/MPrU-3s8ccw?si=gxbOFyfI3j3g8UsH) |[2021](https://www.youtube.com/live/iX59_YJzINs?si=TiZmMMeB6Hy28JcZ) |[2020](https://www.youtube.com/live/PF5DScCr5SI?si=znfuwHbrIgHSzgnO)||
- Summer School: [2022](https://youtube.com/playlist?list=PLQCNXbSwgbGwMW4rGHr_LIfSCMh-7lgbR&si=f_b94K73yVAKST1E) ||

## Community
- [Odias in AI/ML](https://www.odishaai.org/)
