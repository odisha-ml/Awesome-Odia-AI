# Awesome-Odia-AI
[![Awesome](_static/awesome.webp)](https://github.com/sindresorhus/awesome) 
[![Check Markdown links](https://github.com/odisha-ml/Awesome-Odia-AI/actions/workflows/deadlink_checker.yml/badge.svg)](https://github.com/odisha-ml/Awesome-Odia-AI/actions/workflows/deadlink_checker.yml)

Curated list of all AI related resources in Odia Language.

<details>
  <summary>
    Table of Contents
  </summary>

- [NLP](#nlp)
  - [Translation](#translation)
  - [Transliteration](#transliteration)
  - [Language Understanding](#language-understanding)
    - [Language Model](#language-model)
    - [Word Embedding](#word-embedding)
    - [Morphanalyzers](#morphanalyzers)
  - [Language Generation](#language-generation)
  - [Text Classification](#text-classification)
  - [Text Dataset](#text-dataset)
    - [Parallel Translation Corpus](#parallel-translation-corpus)
    - [Monolingual Corpus](#monolingual-corpus)
    - [Lexical Resources](#lexical-resources)
    - [POS Tagged Corpus](#pos-tagged-corpus)
    - [Dialect Detection Corpus](#dialect-detection-corpus)
  - [NLP Libraries / Tools](#nlp-libraries--tools)
  - [Other NLP Resources](#other-nlp-resources)
- [Audio](#audio)
  - [Speech Recognition](#speech-recognition)
  - [Text-to-Speech](#text-to-speech)
  - [Speech Dataset](#speech-dataset)
- [Computer Vision](#computer-vision)
  - [OCR](#ocr)
  - [Multimodal](#multimodal)
- [Applications](#applications)
- [Events](#events)
- [Community](#community)
  - [Policy & Ecosystem](#policy--ecosystem)

</details>

## NLP
### Translation

- Sua: Machine Translation from English to Odia language [[dataset](https://github.com/soumendrak/MTEnglish2Odia)[code](https://github.com/OdiaNLP/NMT)]
- IndicTrans: [[paper](https://arxiv.org/abs/2104.05596)] [[web](https://ai4bharat.iitm.ac.in/areas/model/NMT/IndicTrans)]
- IndicTrans2: [[paper](https://arxiv.org/abs/2305.16307)][[web](https://ai4bharat.iitm.ac.in/areas/model/NMT/IndicTrans2)][[code](https://github.com/AI4Bharat/IndicTrans2)]
- WAT 2025 NLLB-200 Fine-tuned Model: A highly optimized 3.3B parameter neural machine translation model scored on WAT 2025. [[model](https://huggingface.co/OdiaGenAI/facebook-nllb-200-3.3B-finetuned-odia)]
- NLTM-EILMT Bhashini Translation API System: The sovereign, Bhashini-backed open-source repository providing robust, production-ready bi-directional API/REST endpoints for English to Odia machine translation. [[code](https://github.com/eilmt/NLTM-EILMT)]
  
### Transliteration

- IndicXlit: [[paper](https://arxiv.org/abs/2205.03018)][[web](https://ai4bharat.iitm.ac.in/areas/xlit")][[code](https://github.com/AI4Bharat/IndicTrans2) ][[Demo](https://xlit.ai4bharat.org/)][[PyPi](https://pypi.org/project/ai4bharat-transliteration)]
- [open-source unicode converter](https://github.com/OdiaWikimedia/Converter) to transliterate between various languages to Odia language [Demo](https://or.wikipedia.org/s/1hv1).

### Language Understanding
#### Datasets
- IndicCorp: Large sentence-level monolingual corpora for 11 Indian languages and Indian English containing 8.5 billions words (250 million sentences) from multiple news domain sources. [[paper](https://aclanthology.org/2020.findings-emnlp.445/)][[code](https://github.com/AI4Bharat/Indic-BERT-v1)][[web](https://ai4bharat.iitm.ac.in/areas/llm)]  
- Naamapadam: Training and evaluation datasets for named entity recognition in multiple Indian language. [[paper](https://arxiv.org/abs/2212.10168)][[huggingface](https://huggingface.co/datasets/ai4bharat/naamapadam)][[web](https://ai4bharat.iitm.ac.in/areas/llm)]  
- IndicCorp v2: he largest collection of texts for Indic languages consisting of 20.9 billion tokens of which 14.4B tokens correspond to 23 Indic languages and 6.5B tokens of Indian English content curated from Indian websites. [[paper](https://arxiv.org/abs/2212.05409)][[code](https://github.com/AI4Bharat/IndicBERT/tree/main?tab=readme-ov-file#indiccorp-v2)]  
- L3Cube-IndicSQuAD (Odia): A meticulously engineered, 118,000+ sample extractive Question Answering dataset. [[dataset](https://huggingface.co/datasets/l3cube-pune/indic-squad)]

  
#### Language Model
- [Language Model](https://github.com/goru001/nlp-for-odia) : Pretrained Odia Language Model. 
- [BertOdia](https://colab.research.google.com/gist/satyapb2002/aeb7bf9a686a9c7294ec5725ff53fa49/odiabert_languagemodel.ipynb#scrollTo=xy_H5EjNTdRE) : Bert-based Odia Language Model.
- IndicBERT: Multilingual, compact ALBERT language model trained on IndicCorp covering 11 major Indian and English. Small model (18 million parameters) that is competitive with large LMs for Indian language tasks. [[paper](https://aclanthology.org/2020.findings-emnlp.445/)][[code](https://github.com/AI4Bharat/Indic-BERT-v1)][[web](https://ai4bharat.iitm.ac.in/areas/model/LLM/IndicBERT)]
- IndicNER: Named Entity Recognizer models for multiple Indian languages. The models are trained on the Naampadam NER dataset mined from Samanantar parallel corpora. [[paper](https://arxiv.org/abs/2212.10168)][[huggingface](https://huggingface.co/ai4bharat/IndicNER)][[web](https://ai4bharat.iitm.ac.in/language-understanding)]
- IndicBERTv2: Language model trained on IndicCorp v2 with competitive performance on IndicXTREME [[paper](https://arxiv.org/abs/2212.05409)][[code](https://github.com/AI4Bharat/IndicBERT)][[web](https://ai4bharat.iitm.ac.in/areas/model/LLM/IndicBERTv2)]
- Oriya-BERT-SQuAD: A highly capable 0.2B parameter OdiaBERT model rigorously fine-tuned directly on the IndicSQuAD dataset. [[model](https://huggingface.co/l3cube-pune/oriya-question-answering-squad-bert)]

#### Word Embedding
- [FastText (CommonCrawl + Wikipedia)](https://fasttext.cc/docs/en/crawl-vectors.html) : Pretrained Word vector (CommonCrawl + Wikipedia). Trained on Common Crawl and Wikipedia using fastText. Select the language "oriya" from the model list.
- [FastText (Wikipedia)](https://fasttext.cc/docs/en/pretrained-vectors.html) : Pretrained Word vector (Wikipedia). Trained on Wikipedia using fastText. Select the language "oriya" from the model list.
- IndicFT: Word embeddings for 11 Indian languages trained on IndicCorp. The embeddings are based on the fastText model and are well suited for the morphologically rich nature of Indic languages. [[paper](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf)][[code]()][[web](https://web.archive.org/web/20240304135630/https://ai4bharat.iitm.ac.in/indicft/)]

  
#### Morphanalyzers
* [IndicNLP Morphanalyzers](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : Unsupervised morphanalyzers for 10 Indian languages including Odia learnt using morfessor.

### Language Generation

#### Instruction Set

* [Odia master data llama2](https://huggingface.co/datasets/OdiaGenAI/odia_master_data_llama2): This dataset contains 180k Odia instruction sets translated from open-source instruction sets and Odia domain knowledge instruction sets.
* [Odia context 10k llama2 set](https://huggingface.co/datasets/OdiaGenAI/odia_context_10K_llama2_set): This dataset contains 10K instructions that span various facets of Odisha's unique identity. The instructions cover a wide array of subjects, ranging from the culinary delights in 'RECIPES,' the historical significance of 'HISTORICAL PLACES,' and 'TEMPLES OF ODISHA,' to the intellectual pursuits in 'ARITHMETIC,' 'HEALTH,' and 'GEOGRAPHY.' It also explores the artistic tapestry of Odisha through 'ART AND CULTURE,' which celebrates renowned figures in 'FAMOUS ODIA POETS/WRITERS', and 'FAMOUS ODIA POLITICAL LEADERS'. Furthermore, it encapsulates 'SPORTS' and the 'GENERAL KNOWLEDGE OF ODISHA,' providing an all-encompassing representation of the state.
* [Roleplay Odia](https://huggingface.co/datasets/OdiaGenAI/roleplay_odia): This dataset contains 1k Odia role play instruction set in conversation format.
* [OdiEnCorp translation instructions 25k](https://huggingface.co/datasets/OdiaGenAI/OdiEnCorp_translation_instructions_25k): This dataset contains 25k English-to-Odia translation instruction set.
* [Odia Reasoning Datasets (GU, OD, OpenThoughts)](https://huggingface.co/datasets/OdiaGenAIdata/Reasoning_OD): A suite of newly compiled, highly validated datasets designed to transition LLMs from basic conversational generation to step-by-step logical reasoning.

#### Pe-train Dataset

- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX): Multilingual dataset with monolingual data for Indic languages including Odia. [Paper](https://arxiv.org/abs/2309.09400) — Odia-only filtered version: [culturax-odia](https://huggingface.co/datasets/OdiaGenAIdata/culturax-odia)
- [Varta](https://huggingface.co/datasets/rahular/varta): The dataset contains 41.8 million news articles in 14 Indic languages and English, crawled from DailyHunt, a popular news aggregator in India that pulls high-quality articles from multiple trusted and reputed news publishers.

### Foundation LLM
- [Qwen 1.5 Odia 7B](https://huggingface.co/OdiaGenAI-LLM/qwen_1.5_odia_7b): This is a pre-trained Odia large language model with 7 billion parameters, and it is based on Qwen 1.5-7B. The model is pre-trained on the Culturex-Odia dataset, a filtered version of the original CulturaX dataset for Odia text. As per the authors, it is a model is a base model and not meant to be used as is. It is recommended to first finetune it on downstream tasks. [Blog](https://www.odiagenai.org/blog/introducing-odiagenai-s-qwen-based-pre-trained-llm-for-odia-language)
- [Odia-Gemma-2B-Base](https://huggingface.co/OdiaGenAI-LLM/odia-gemma-2b-base) : Pre-trained 2B-parameter decoder-only Odia LLM (Gemma 2B based).

### Fine-Tuned LLM
- [Odia llama2 7B base](https://huggingface.co/OdiaGenAI/odia_llama2_7B_base): odia_llama2_7B_base is based on Llama2-7b and finetuned with 180k Odia instruction set. [Paper](https://arxiv.org/pdf/2312.12624.pdf)
- [Llama3_8B_Odia_Unsloth & Llama 3.x R1 Scalable Series](https://huggingface.co/OdiaGenAI-LLM/Llama3_8B_Odia_Unsloth): Advanced Llama-3 based generative models optimized using the Unsloth library with 4-bit bnb quantization.
- [odia-t5-base](https://huggingface.co/mrSoul7766/odia-t5-base) : Multilingual Text-to-Text Transformer (mT5-based) fine-tuned for Odia translation, summarization, and QA.


### Benchmarking Set
* [Airavata Evaluation Suite](https://huggingface.co/collections/ai4bharat/airavata-evaluation-suite-65b13b7b68165de71ba0b333): A collection of benchmarks used for evaluation of Airavata, a Hindi instruction-tuned model on top of Sarvam's OpenHathi base model.
* [Indic LLM Benchmark](https://huggingface.co/Indic-Benchmark): A collection of LLM benchmark data in Gujurati, Nepali, Malayalam, Hindi, Telugu, Marathi, Kannada, Bengali.
* [MILU: Multi-task Indic Language Understanding](https://github.com/AI4Bharat/MILU): A comprehensive, culturally grounded evaluation framework featuring 4,525 highly verified Odia questions spanning 41 distinct domains.

### Text Dataset

#### Parallel Translation Corpus
* [OdiEnCorp 2.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3211) : This dataset contains 97K English-Odia parallel sentences and serving in [WAT2020](http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2020/index.html) for Odia-English machine translation task. [Paper](https://www.aclweb.org/anthology/2020.wildre-1.3.pdf) 
* [OPUS Corpus](http://opus.nlpl.eu/) : It contains parallel sentences of other languages with Odia. The collection of data are domain-specific and noisy.  
* [OdiEnCorp 1.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879) : This dataset contains 30K English-Odia parallel sentences. [Paper](https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47) 
* [IndoWordnet Parallel Corpus](https://github.com/anoopkunchukuttan/indowordnet_parallel) : Parallel corpora mined from IndoWordNet gloss and/or examples for Indian-Indian language corpora (6.3 million segments, 18 languages including Odia). [Paper](https://github.com/anoopkunchukuttan/indowordnet_parallel/blob/master/iwn_parallel_2020.pdf)
* [PMIndia](http://data.statmt.org/pmindia/) : Parallel corpus for En-Indian languages mined from Mann ki Baat speeches of the PM of India. It contains 38K English-Odia parallel sentences.[Paper](https://arxiv.org/abs/2001.09907) 
* [CVIT PIB](https://web.archive.org/web/20231205223617/http://preon.iiit.ac.in/~jerin/bhasha/) : Parallel corpus for En-Indian languages mined from press information bureau website of India. It contains 60K English-Odia parallel sentences.
* [Samanantar](https://datasets.ai4bharat.org/samanantar/) is the largest publicly available parallel corpora collection for Indic languages. The corpus has 49.6M sentence pairs between English to Indian Languages.
* [BPCC](https://ai4bharat.iitm.ac.in/areas/nmt) is a comprehensive and publicly available parallel corpus containing a mix of Human labelled data and automatically mined data; totaling to approximately 230 million bitext pairs[[Paper](https://arxiv.org/abs/2305.16307)]].
* [English–Odia Entertainment Parallel Corpus](https://www.futurebeeai.com/dataset/parallel-corpora/oriya-odia-english-translated-parallel-corpus-for-entertainment-domain) : Domain-specific English–Odia sentence-aligned parallel corpus (50k+ pairs) focused on entertainment content.

#### Monolingual Corpus
* [Odia News Corpus](https://www.soumendrak.com/blog/scrape-news-website-using-scrapy/) Odia Monolingual News Corpus of more than 1.5GB. [Dataset](https://www.kaggle.com/datasets/soumendrak/odiamonolingualnewscorpus)
* [EMILLE Corpus](https://www.lancaster.ac.uk/fass/projects/corpus/emille/) : It contains fourteen monolingual corpora for Indian languages including Odia.[Manual](https://www.lancaster.ac.uk/fass/projects/corpus/emille/MANUAL.htm) 
* [OdiEnCorp 1.0](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2879) : This dataset contains 221K Odia sentences.[Paper](https://link.springer.com/chapter/10.1007/978-981-13-9282-5_47) 
* [AI4Bharat-IndicNLP Corpus](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : The text corpus not available now (will be available later). It used 3.5M Odia sentences to build the embedding. Vocabulary frequency files are available.[Paper](https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf)
* [OSCAR Corpus](https://oscar-corpus.com/) : It contains around 300K Odia sentences.
* [Large Odia LLM Pre-training Corpus (pre_train_odia_data_processed)](https://huggingface.co/datasets/OdiaGenAIdata/pre_train_odia_data_processed): A uniformly processed, aggressively deduplicated amalgamation of massive Odia text data of 6.4GB.
* [IndicDialogue (Odia Subtitles)](https://data.mendeley.com/datasets/wcb4bxbyxx) : Large subtitle and dialogue corpus from OpenSubtitles in 10 Indic languages including Odia, with 6.8M+ Odia dialogues.
* [Odia-data-collection](https://huggingface.co/datasets/Minutor/Odia-data-collection) : Aggregated Odia text dataset on Hugging Face created for language training.

#### Lexical Resources
* [IndoWordNet](http://www.cfilt.iitb.ac.in/indowordnet/) : Wordnet for Indian languages including Odia.


#### POS Tagged corpus
* [Indian Language Corpora Initiative](https://web.archive.org/web/20240414185340/http://sanskrit.jnu.ac.in/ilci/index.jsp/) : It contains parallel annotated corpora in 12 Indian languages including Odia (tourism and health domain). 
- [Odia Treebank (ODTB)](https://github.com/UniversalDependencies/UD_Odia-ODTB/tree/dev) : Odia UD Treebank, part of **Universal Dependencies v2.18** (24th release, 353 treebanks across 193 languages). Expanded from 1,029 to **5,818 annotated instances**. Links: [UD v2.18 Release](https://universaldependencies.org/#download) — [Odia ODTB Treebank Page](https://universaldependencies.org/treebanks/or_odtb/index.html)

#### Dialect Detection corpus
* [Odia-Santali Dialect Detection Corpus](https://github.com/shantipriyap/Odia-Santali-Dialect-Detection-Dataset/) : This corpus contains text data of Odia and Santali written in Odia script. 

### Text Classification
* [Odia News Article Classification](https://www.kaggle.com/disisbig/odia-news-dataset) : This dataset contains approxmately 19,000 news article headlines collected from Odia news websites. The labeled dataset is splitted into training and testset suitable for supervised text classification. 
* [AI4Bharat IndicNLP News Articles](https://github.com/ai4bharat-indicnlp/indicnlp_corpus) : This datasets comprising news articles and their categories for 9 languages including Odia. For Odia language, it has 4 classes (business, crime, entertainment, sports) and each class contains 7.5K news articles. The dataset is balanced across classes. [Paper](https://github.com/ai4bharat-indicnlp/indicnlp_corpus/blob/master/ai4bharat-indicnlp-corpus-2020.pdf)
* [MTEB: OdiaNewsClassification](https://huggingface.co/datasets/mteb/OdiaNewsClassification) : A heavily curated 3-class classification testbed comprising over 17,200 journalistic articles. Officially integrated into the MTEB standard.
* [Odia Sentiment MuRIL v4](https://huggingface.co/Baps24/odia-sentiment-muril-v4) : An advanced sentiment classifier built natively upon the MuRIL framework.

### NLP Libraries / Tools 
* [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) : It is a python based NLP library for Indian language text processing including Odia.
* [Odia Romanization Script](https://github.com/shantipriyap/odia_nlp) : The perl script "odiaroman" maps the Devnagri (Odia) to Latin.
- [OpenOdia : Tools for Odia language](https://github.com/soumendrak/openodia)
- [BrahmicTokenizer-131K](https://github.com/theschoolofai/BrahmicTokenizer-131K): A 131K-vocabulary multilingual tokenizer optimized for Brahmic scripts including Odia, outperforming standard tokenizers in token fertility and efficiency. [[paper](https://arxiv.org/abs/2605.29379)][[model](https://huggingface.co/theschoolofai/BrahmicTokenizer-131K)]

### Other NLP Resources
* [TDIL](http://tdil-dc.in/index.php?lang=en) : It contains language application, resources, and tools for Indian languages including Odia. It contains many language applications, resources, and tools for Odia such as Odia terminology application, Odia language search engine, wordnet, English-Odia parallel text corpus, English-Odia machine-assisted translation, text-to-speech software, and many more.  

## Audio

### Speech Recognition
- [IndicWav2Vec-Odia](https://huggingface.co/ai4bharat/indicwav2vec-odia) : A foundational acoustic architecture applying the self-supervised wav2vec 2.0 framework directly to unannotated Odia phonetics.
- [wav2vec2-large-xlsr-53-odia](https://huggingface.co/theainerd/wav2vec2-large-xlsr-53-odia) : Fine-tuned Wav2Vec2-Large-XLSR-53 model for Odia ASR.
- [whisper-small-odia-finetuned](https://huggingface.co/Mohan-diffuser/whisper-small-odia-finetuned) : Whisper-small model fine-tuned with LoRA on an Odia-English bilingual ASR dataset.
- [Olive_Odia_ASR](https://github.com/OdiaGenAI/Olive_Odia_ASR) : OdiaGenAI toolkit and scripts for fine-tuning and serving Whisper-based Odia ASR models.

### Text-to-Speech
- Indic-TTS [[Paper](https://arxiv.org/abs/2211.09536)][[Code](https://github.com/AI4Bharat/Indic-TTS)] [[Try It Live](https://models.ai4bharat.org/#/tts)]][[Video](https://youtu.be/I3eo8IUAP7s)]


### Speech Dataset
* [IIT Madras IndicTTS](https://www.iitm.ac.in/donlab/tts/index.php) : The Indic TTS project develops the text-to-speech (TTS) synthesis system for Indian languages including Odia. The database contains spoken sentences/utterances recorded by both Male and Female native speakers.
* [LDC-IL](https://data.ldcil.org/speech/speech-raw-corpus/odia-raw-speech-corpuss) :  It includes Odia annotated speech corpora which has voices of 450 different native speakers.
- [Mozilla Common Voice](https://mozilladatacollective.com/datasets/cmn2cww3s01dfo107etbjbif1) : The Mozilla Common Voice project is a community-led project to build a large multilingual dataset for speech recognition.
- [Odia text to speech dataset](https://commons.wikimedia.org/wiki/Category:Odia_pronunciation) : 55000 odia words pronunced in various dialets of Odisha like Baleswari, Puri, Cuttack, etc.
- [Odia ASR Benchmark Dataset for Noisy Speech Recognition](https://aikosh.indiaai.gov.in/home/datasets/details/odia_asr_benchmark_dataset_for_noisy_speech_recognition_kathath_odia_noisy_test_known.html) : About 500 MB of CC-BY-4.0 data. 
- [ODEN-speech Corpus](https://huggingface.co/datasets/BBSRguy/ODEN-speech) : A paradigm-shifting 462-hour speech ensemble derived from seamlessly merging 8 disparate corpora, standardized to 16kHz mono audio of size 51.9GB.
- [Odia General Conversation Speech Dataset](https://www.futurebeeai.com/dataset/speech-dataset/general-conversation-oriya-odia-india) : Real-world, unscripted conversational Odia speech with detailed metadata.
- [Odia Speech Datasets Collection](https://www.futurebeeai.com/dataset/speech-data/odia-dataset) : Commercial suite of Odia speech datasets for ASR, TTS, and voice-assistant training.
  
## Computer Vision

### OCR
* [Indic-OCR](https://indic-ocr.github.io/) : OCR tools for Indic scripts including Odia. Also, supports _Ol Chiki_ (Santali).
* [IndicPhotoOCR (Scene Text Recognition)](https://github.com/Bhashini-IITJ/IndicPhotoOCR) : A high-velocity, end-to-end scene text recognition toolkit capable of parsing complex, visually noisy Odia script in organic environments.
* [DocuExtract: Odia Handwritten OCR](https://github.com/tell2jyoti/docuextract-odia-ocr) : An advanced computational vision system employing progressive vision-language fine-tuning to transcribe stochastic, highly erratic handwritten Odia text.
* [Odia Lipi OCR Data](https://huggingface.co/datasets/OdiaGenAIOCR/Odia-lipi-ocr-data) : Open Odia OCR benchmark dataset of scanned Odia text with validated Unicode annotations.
* [Odia OCR Merged Multi-Source Dataset](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) : Merged OCR dataset combining OdiaGenAIOCR and other sources (≈192k samples) for printed and handwritten Odia OCR.
* [odia-ocr-qwen-finetuned](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned) : Production-ready Qwen2.5-VL-3B-Instruct VLM fine-tuned on 58,720 Odia text-image pairs for robust OCR.
* [odia-ocr-qwen-finetuned_v2](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v2) : Updated Odia OCR Qwen model trained on ~73k word-level Odia crops spanning diverse fonts and print qualities.

### Multimodal
* [Odia Visual Genome (OVG)](http://hdl.handle.net/11234/1-5979) : Multimodal English-Odia dataset of Visual Genome image-caption pairs.
* [Odia Image Captioning Dataset](https://www.futurebeeai.com/dataset/multi-modal-dataset/odia-image-caption-dataset) : FutureBeeAI multimodal dataset of diverse images with Odia captions and rich metadata.
* [odia-llava-dataset](https://huggingface.co/datasets/sam2ai/odia-llava-dataset) : Hugging Face dataset of Odia instruction-style image-text pairs suitable for training LLaVA-style multimodal Odia models.

## Applications
- [Odia Lingua Chatbot](https://github.com/HimanshuMohanty-Git24/Odia_Lingua) : AI-powered Odia language chatbot built with Groq API, Odia TTS (MMS-TTS-ory), and Google Search integration.

## Events
- Global Conference: [ICON 2024](https://au-kbc.org/icon2024/) | [WAT 2025](https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation) | [Odisha AI Summit 2025 & Conference 2024](https://odishaaisummit.org/) | [2023 Pt 2](https://www.youtube.com/live/KZB9bfKkLgM?si=3i9eY22xT-1yZTD8) [2023 Pt 1](https://www.youtube.com/live/GPkWL-9akQc?si=uh0Ay0SKEVlRnX3U) | [2022](https://www.youtube.com/live/MPrU-3s8ccw?si=gxbOFyfI3j3g8UsH) | [2021](https://www.youtube.com/live/iX59_YJzINs?si=TiZmMMeB6Hy28JcZ) | [2020](https://www.youtube.com/live/PF5DScCr5SI?si=znfuwHbrIgHSzgnO)
- Summer School / Workshop: [OdiaGenAI Generative AI Workshops 2025](https://www.odiagenai.org/workshop-2025) | [2022](https://youtube.com/playlist?list=PLQCNXbSwgbGwMW4rGHr_LIfSCMh-7lgbR&si=f_b94K73yVAKST1E)

## Community
- [Odisha AI](https://www.odishaai.org/)
- [Odia Generative AI](https://www.odiagenai.org/)

### Policy & Ecosystem
- [Odisha AI Policy 2025 & Taskforce](https://dowr.odisha.gov.in/sites/default/files/2025-07/Odisha%20AI%20Policy-2025.pdf)
- [National IndiaAI Mission](https://indiaai.gov.in/)
