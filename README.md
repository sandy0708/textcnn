TextCNN implementation
=============
How to run
-------------   
```
./downloads.sh
python main.py --model_ver "CNN-static" --batch_size 50 --epoch 20 --lrate 0.1
```


File structure
-------------   
```bash
textcnn    
├── downloads.sh    
├── main.py    
├── data    
│   └── rt-polaritydata    
│       ├── rt-polarity.neg    
│       └── rt-polarity.pos    
├── embeddings    
│   ├── word_embedding.py    
│   └── GoogleNews-vectors-negative300.bin    
├── vectorizer    
│   ├── english_preprocessing.py    
│   ├── english_tokenizer.py    
│   └── vectorizer.py    
└── pytorch    
    ├── dataloader.py    
    ├── model.py
    └── train.py
```


  
Result
-------------   
* CNN-rand: 73.39%    
* CNN-static: 75.28%    
* CNN-non-static     
* CNN-multichannel    