```
python main.py --model_ver "CNN-static" --batch_size 50 --epoch 20 --lrate 0.1
```

textcnn
├── english_preprocessing.py
├── english_tokenizer.py
├── word_embedding.py
├── vectorizer.py
├── model.py
├── dataloader.py
├── train.py
└── main.py


CNN-rand = 73.39%
CNN-static = 75.28%
CNN-non-static 
CNN-multichannel