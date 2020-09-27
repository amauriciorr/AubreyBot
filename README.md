# AubreyBot
![aubrey](/assets/IMG_4077.jpg)
Create your own chatbot trained on lyrics pulled from [Genius](https://genius.com/)! This project was named after Aubrey "Drake" Graham as his lyrics were the first I chose to bring this chatbot to life. 


## Getting started
You'll need Genius API keys in order to collect a musician's lyrics. You can create your own keys [here](https://docs.genius.com/#/getting-started-h1). Make sure to save your generated keys in a file named `genius_keys.py`. Your secret access token must be saved as `GENIUS_ACCESS_TOKEN='YOUR-ACCESS-TOKEN-HERE'` within this file.

 Additionally, access to a GPU in order to train your chatbot. [Google Cloud](https://cloud.google.com/) and [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) both provide access to GPUs. Colab is free to use while Cloud gives you $300 free credits to use before charging you.

### Install required packages
Run `pip install -r requirements`

### Files explained
* `setup.py`: handles downloading, preprocessing, and creating both your training and validation datasets. 
* `build_rapbot.py`: trains your chatbot
* `Aubrey.py`: used for loading your pretrained model and starting a conversation with your chatbot
*`processing_utils.py`: utilities for downloading and processing your lyrics
*`chat_utils.py`: utilities for how chatbot generates responses


### Downloading and preprocessing
Before training your chatbot, you will first need to pull lyrics from Genius, you can do this like so 

`python setup.py --artist_name "Drake" `

Alternatively, if you've already pulled lyrics through Genius' API, you can run

`python setup.py --download_lyrics false --load_path ./path/to/your/lyrics.json`


### Training Aubrey
To start training your chatbot run `python build_rapbot.py`. Files preprocessed in the previous step will automatically be read. 


There are a variety of flags to adjust the neural network training and architecture. To see a full list of flags, simply run `python build_rapbot.py -h`. 

#### FAQ flags
* `--num_epochs`: specify how many epochs to train your model for. This is set to 30 by default.
* `--batch_size`: number of training examples used in single iteraiton. This is set to 32 by default
* `learning_rate`: rate at which to adjust weights when doing gradient descent
*`--save_dir`: directory to save your trained model. By default, this is set to `./models/`. This folder will be created for you if it does not exist.
* [WIP] ~~`--use_BERT`: use transfer learning to train your chatbot using an encoder-decoder BERT model~~

### Talking to Aubrey

To talk to your chatbot, simply run `python Aubrey.py --model_path ./path/to/your/pretrained_model.pt`. 

This will automatically load the model trained from the previous step along with the a pickled dictionary object containing the tokenized vocabulary that was recorded during preprocessing. This latter file is important for converting your chatbot's  output into something that's human-readable. 

![aubrey_chatlog](/assets/chatting_with_aubrey.gif)

To stop talking to Aubrey at any moment, enter `q` or `quit` in the chat. 




## Resources
### Seq2seq
* https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
### Attention
* https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#transformer
* https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
* https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
### Transformers
* https://towardsdatascience.com/transformers-141e32e69591
* http://jalammar.github.io/illustrated-transformer/
* https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/

### BERT
* https://arxiv.org/abs/1810.04805
* https://huggingface.co/transformers/model_doc/bert.html
* https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
