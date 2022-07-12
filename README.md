# birdnet_embedding_analyzer
Steps to generate the embeddings used in the experiments:

1) Install the BirdNET-Analyzer in the correct spot (see steps 3 and 4)
2) Execute fetcher.py
3) run `python .\BirdNET-Analyzer-main\analyze.py --i wav_files --o analyzed --min_conf 0.5 --threads 6`
4) run `python .\BirdNET-Analyzer-main\embeddings.py --i wav_files --o embeddings --threads 6`
5) Execute embedder.py

Now, experiments in embedder.py can be executed.
