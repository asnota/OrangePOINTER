# OrangePOINTER

The repository contains the scripts for the model, described in a paper submitted to NeurIPS 2021, titled 
"OrangePOINTER: Constraint progressive generation of texts in French language", which exploits the approach published in <a href="https://arxiv.org/abs/2005.00558">this article</a>.
The goal of this approach is a text generation under specific constrains (keywords) in a progressive, non-autoregressive manner.

## Project predecessor relation
The current project is an adaptation to a French language context of the <a href="https://github.com/dreasysnail/POINTER">initial code repository</a>, distributed under the MIT license. 
The main changes are (but to limited to):
1. Making code executable as Jupiter notebooks on cloud platforms (Google Cloud Platform, Colab, Kaggle)   
2. Transitionning the training script from GPU to TPU.
3. Rewriting of the keywords extraction script, since the YAKE extractor didn't result in acceptable output for French language.
4. Addition of the CodeCarbon tracking, reported to the Comet ML platform for the scripts requiring considerable energy consumption.
5. Amelioration of the cleansing procedure in ```generate_training_data.ipynb``` script, based on empirical observations of the output.
6. Addition of the ```join_train_data.ipynb``` script. The necessity of this script was dictated by the fact that 100Mo of the raw French text takes about 3 hours to output the data consumable by the model, therefore we made turning several GCP instances at the same time to speed up the generation of the pretraining data and join them into one big file in the end.
7. Addition of the ```postprocessing.ipynb``` script, since the result of the inference contained undesired tags.
8. Numerous comments and structurization of the code blocks.

## Project description
Project consists of seven Jupiter notebooks executable on cloud platforms (Colab, Kaggle, GCP).
The order of the scripts execution might be described in the following scenarios:

### Training data generation
1) ```generate_training_data.ipynb``` consumes a .txt file of raw data and outputs six .json files, joined into two zip folders, containing 
3 files of metrics and 3 files for the text data itself. The number "3" corresponds to the number of data epochs.
You must put your key and name in the code from <a href="https://www.comet.ml/site/">Comet ML</a> in order to make work the CodeCarbon reporting. If you don't have CometML account or you don't want to create one, do not execute CometML-related cells to avoid login errors.

2) ```join_train_data.ipynb``` might be applied if the script was running on different VM instances or several times on the same VM, to consolidate the pretraining data.
It takes zip folders coming from ```generate_training_data.ipynb```. You must manually rename zip folders by adding _[number], starting from 0, for example:
```
metrics_data_0.zip
training_data_0.zip
```
the script generates two zip folders, similar to the ones you might have received directly from `generate_training_data.ipynb` script.

### Pre-training
```pretraining_on_TPU.ipynb``` must be executed on TPU enabled device. It takes zip folders from training data generation step and outputs a ```pytorch_model.bin``` file. 
Put the file inside a folder titled ```model``` containing configuration files - you may download it from this repository, and zip it.
 
### Finetunning
Similar to the pretraining, but uses different set of parameters (learning rate, number of epochs) and, most importantly, it takes a pretrained model as an input. 
For the sake of clarity we have devided a finetunning script to a separate file, containing all configuration needed: ```finetunning_on_TPU.ipynb```
The example of a pretrained model might be found in a current repository.
 
### Keywords extraction
```spacy_keywords_extraction.ipynb``` takes a raw text file and returns a text file of extracted keywords. The amont of generated keywords may be modulated.

### Inference
```inference.ipynb``` takes the finetuned model along with configuration files and a keywords.txt file. 
The decoding strategy might be switched between 'greedy' and 'sampling'. 
Such parameters as the top-k, top-p and temperature for the 'sampling' decoding strategy, might be modulated as well.

## Data sources description

### Pretraining dataset: CC-100

| Dataset               | Size |
| ---                   | ---  |
| CC-100 (French part)  | 54Go |

### Finetunning dataset

Split sizes are given in thousands of documents.
Vocab sizes are given in thousands of tokens.

| Dataset | train/val/test sizes |avg. document length in words | avg. document length in sentences | avg. summary length in words | avg. summary length in sentences | vocabulary size: document | vocabulary size: summary |
| ---     | ---                  | ---                          | ---                               | ---                          | ---                              | ---                       | ---                      |
| OrangeSUM | 21.4/1.5/1.5       | 350                          | 12.06 							   | 32.12						  | 1.43							 | 420						 | 71                       |

The pregenerated data used during the finetunning is available for downloading: 
<a href="https://orangepointer.blob.core.windows.net/files/metrics_data.zip">metrics folder</a> and 
<a href="https://orangepointer.blob.core.windows.net/files/training_data.zip">training text data folder</a>.

## Models

| Model            | Link to download    |
| ---              | ---                 |
| pretrained model | <a href="https://orangepointer.blob.core.windows.net/files/pretrained_model.zip">link</a> |
| finetuned model | <a href="https://orangepointer.blob.core.windows.net/files/finetunned_model.zip">link</a> |

## Examples of generation

1. Sampling decoding strategy:

| Keys             | Generated sentences |
| ---              | ---                 |
| keys | sentences |
| keys | sentences |
| keys | sentences |

2. Greedy decoding strategy:

| Keys             | Generated sentences |
| ---              | ---                 |
| keys | sentences |
| keys | sentences |
| keys | sentences |

## Demo
If you wish to run a demo of the inference using a finetuned model, you may do so in <a href="https://colab.research.google.com/">Colab</a> or <a href="https://www.kaggle.com/">Kaggle</a> (for free). 
In either of platforms you need import the ```finetunning_on_TPU.ipynb```, downloaded from a current repository.
Once done, execute all the cells - the script aready contains wget commands downloading the latest version of <a href="https://orangepointer.blob.core.windows.net/files/finetunned_model.zip">OrangePOINTER finetuned model</a> and the <a href="https://orangepointer.blob.core.windows.net/files/keywords.txt">keywords</a> for the text generation, which were extracted from the summaries of OrangeSUM dataset, contained in a test split (1500 entries).
The generated txt file will be saved to the same folder, where the model was unpacked (./model).