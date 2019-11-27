
# Custom Fasttext Embeddings with Flair 

<p align="center">      
  <img src="header.png" alt="Header">      
</p>

To get a better overview and understanding of the need for this project, I would recommend to read [my article on the same.](https://medium.com/@pranaychandekar/custom-fasttext-embeddings-with-flair-da642be3505e?source=friends_link&sk=fc4cab36b2204492dcb5c4fd343032f3)   
    
The objective of this script is to be able to use our custom fasttext embeddings with the *[Flair](https://github.com/zalandoresearch/flair)*.  

## Usage    

 The integration is very simple:  
  
 1. Copy the `fasttext_custom_embeddings_with_flair.py` file to your project.   
 2. Install the packages from `requirements.txt`.  
 3. Import the `FastTextEmbeddings` from `fasttext_custom_embeddings_with_flair.py` file.  
 4. Instantiate an object of `FastTextEmbeddings` by passing either the ***local path*** or the ***remote http(s) url*** to the constructor.   
 5. Use it like you use any other embedding object in **Flair**.  
  
 ---
 ### Usage with local file    
    
```python   
from fasttext_custom_embeddings_with_flair import FastTextEmbeddings 
from flair.data import Sentence 

ft_embeddings = FastTextEmbeddings('/path/to/custom_fasttext_embeddings.bin', use_local=True)

sentence = Sentence('The quick brown fox jumps over a lazy dog.', use_tokenizer=True) '
ft_embeddings.embed(sentence)
```  
  
 1. Initialize the constructor by passing the ***local path*** to embeddings `.bin` file.   
 2. Set the boolean `use_local` to `True`.   
  
  Support for embedding file with ".vec" extension is also added. We need to add one more parameter `extension='.vec'` to the constructor.
```python   
ft_embeddings = FastTextEmbeddings('/path/to/custom_fasttext_embeddings.vec', use_local=True, extension='.vec')
```  

>**Note:** The ".vec" extension files do not support out of vocabulary(OOV) embeddings. In this case, for OOV words, zero vectors are assigned by default.    
  ---  
### Usage with remote file    
    
```python 
from fasttext_custom_embeddings_with_flair import FastTextEmbeddings 
from flair.data import Sentence    

ft_embeddings = FastTextEmbeddings('/url/to/custom_fattext_embeddings.bin', use_local=False)    

sentence = Sentence('The quick brown fox jumps over a lazy dog.', use_tokenizer=True) 
ft_embeddings.embed(sentence) 
```   
 
 1. Initialize the constructor by passing the ***remote url*** to embeddings `.bin` file. (*I have tested this with my embeddings in S3 with an https url.*)  
 
 2. Set the boolean `use_local` to `False`. (*Mandatory step if you are using a remote url.*)  
  
    
---  
### Usage with StackedEmbeddings    

```python 
from fasttext_custom_embeddings_with_flair import FastTextEmbeddings 
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.data import Sentence 

ft_embeddings = FastTextEmbeddings('/url/to/custom_fattext_embeddings.bin', use_local=False) 
glove_embeddings = WordEmbeddings('glove')    

stacked_embeddings = StackedEmbeddings([ft_embeddings, glove_embeddings])

sentence = Sentence('The quick brown fox jumps over a lazy dog.', use_tokenizer=True) 
stacked_embeddings.embed(sentence) 
```    
 ---  
### Usage with DocumentEmbeddings    

```python 
from fasttext_custom_embeddings_with_flair import FastTextEmbeddings 
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings 
from flair.data import Sentence    

ft_embeddings = FastTextEmbeddings('/url/to/custom_fattext_embeddings.bin', use_local=True) 
glove_embeddings = WordEmbeddings('glove')    

document_rnn_embeddings = DocumentRNNEmbeddings([ft_embeddings, glove_embeddings])

sentence = Sentence('The quick brown fox jumps over a lazy dog.', use_tokenizer=True) 
document_rnn_embeddings.embed(sentence) 
``` 
--- 
In case of any doubts, get in touch.  
  
  **Author**: [Pranay Chandekar](https://www.linkedin.com/in/pranaychandekar/)
    
    
 ---  
  **Refernces:**    
    
1. [Flair](https://github.com/zalandoresearch/flair)  
 2. [fastText](https://fasttext.cc/docs/en/python-module.html)  
    
---    
 ### License:    
 This repository is licensed under the Apache 2.0 License.
 