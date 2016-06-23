# aRNNie
aRNNie is a character-level recurrent neural network. Named after my neural net teacher, [Arnie Azcarraga](http://www.dlsu.edu.ph/faculty/fis/faculty_info.asp?fac_id=103957073).

Note: This is a rewrite of [Karpathy's minimal RNN](https://gist.github.com/karpathy/d4dee566867f8291f086).

The sample data in this repository is a collection of speeches by President Noynoy Aquino scraped from [Gov.ph](http://www.gov.ph/section/speeches/)

## To use:
```
pip install -r requirements.txt

# generate the model first:
python train.py

# If loss seems stagnant (maybe about 5 minutes), ctrl+c to cancel training.

# to generate a sample text:
python run_model.py
```
