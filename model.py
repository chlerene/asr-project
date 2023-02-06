# -*- coding: utf-8 -*-
"""Kopie von Blatt 3: Automatic Speech Recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NmpBDw7vpT-IlBLIhbnXKTrJj82KPtaK

# Praktikum Automatische Spracherkennung
## Blatt 3: Automatich Speech recognition using CTC

### In this lab exercise, you will use the huggingface 🤗 framework and pre-trained models for automatich speech recognition.

## 1. Create and Audiodataset

First look at the data files. How is the file formated is the data clean or do you need to clean special characters. In our case the transcripts are already quite clean so lucky you.
"""

#imports
from torch.utils.data import Dataset
import soundfile as sf
from transformers import Trainer
from transformers import TrainingArguments,IntervalStrategy
from transformers import WhisperForConditionalGeneration
from torch import nn
from collections import OrderedDict
from tqdm import tqdm
from dataclasses import dataclass
from transformers import Wav2Vec2CTCTokenizer,WhisperFeatureExtractor
from typing import List, Dict, Union
from transformers import WhisperPreTrainedModel, WhisperModel
from typing import Optional
import torch
from torch import nn
from collections import OrderedDict
from transformers.modeling_outputs import CausalLMOutput
import json

class AudioDataset(Dataset):
  def __init__(self, stm_path, sort=True,
                 min_utt_length=3000, max_utt_length=30000,
                 max_target_length=448,
              ):
    
      #TODO call __load_utts()
      self.stm_path = stm_path
      self.sort = sort
      self.min_utt_length = min_utt_length
      self.max_utt_length = max_utt_length
      self.max_target_length = max_target_length

      self.__load_utts()
    
      # TODO: Load all data and labels into a list using __load_utts function (Your file \t between columns) column 4 gives audio length in ms

  def __load_utts(self):
      utts = list()
      # TODO: Load all data into list (utts) (Your file \t between columns) column 4 gives audio length in ms
      # Elements of utts should look like (utt_id, wav_path, utt_length, transcript)
      # Exclude audio files shorter than min_utt_length and longer than max_utt_length
      # Exclude samples with transcripts longer than max_target_length
      lines = list(open(self.stm_path))

      # Format list
      for l in lines:
          # Split columns
          l = l.split("\t")
          # Delete \n at the end of transcript 
          l[-1] = l[-1][:-1]
          # Convert audio length to float
          l[4] = float(l[4])
          # Exclude too short and too long audio files as well as too long transcripts
          if l[4] >= self.min_utt_length and l[4] <= self.max_utt_length and len(list(l[5])) <= self.max_target_length:
              utts.append(l)

      #TODO sort list according to utt_length
      if self.sort == True:
          utts = sorted(utts, key=lambda x: x[4])
      self.stm_set = utts

      #print("{}/{} rest had audio length <{} or text='' or length>{}".format(self.total_utts, self.stm_lines, self.min_utt_length, self.max_utt_length))

  def __len__(self):
      #TODO return number of samples in dataset 

      return len(self.stm_set)

  def __getitem__(self, index):
      #TODO read audio from wav_path using sf.read

      # Get dataset element of given index
      item = self.stm_set[index]
      
      # Extract utt_id and transcript of dataset element
      utt_id = item[0]
      transcript = item[5]

      # Read audio from wav_path
      audio, _ = sf.read(item[1])

      return utt_id, audio, transcript

"""### 2. In order to use ASR with CTC loss we need a vocabulary of all appearing letters in our text.

Read in the training file of the to provided stm files. And generate a vocabulary only considering the transcipt column of that file. 

As we will be using Huggingface's Wav2Vec2CTCTokenizer we will need to add two more tokens to your voacb. 


1.   [UNK] = unkown token
2.   [PAD] = padding token

Additionall you should replace " " with | so it predicted spaces are more easily visible.

The [UNK] is used in cases you need to encode letters which do not appear in your vocabulary.
The [PAD] token is very important for CTC models. CTC models in contrast to Sequence-to-Sequence models (Blatt 2) are different in that you dont predict autoregressively. For each input frame your model predicts an output. As the  input speech and the target text are very different modalitys this results in very long input sequences but rather short output lengths. 
In order to cope with that and also in order to account for pauses in speech CTC-models use the [PAD] tokens.

Basically the CTC-model predicts for each acoustic frame a possible output token. This results in predictions possibly like folliowing example:

"HHHelll|ll|ooo[PAD]|[PAD]mmmmyy"

The CTC-decoding will squash repetitions and than remove the [PAD] tokens resulting in: 

"Hello|my" or "Hello my"


[Link to more detailed CTC explanation. Very interesting.](https://distill.pub/2017/ctc/)

2.1

Your task here is to read in the training file of the stms. And create a vocabulary containing each character in the transcripts.
Followed by steps:
   

1.   replacing " " (space) with |
2.   Add [UNK]
3.   Add [PAD]

Afterwards save your vocab in a json file.
"""
def do_vocab(stm_file, idx):
    #TODO create your initial vocabulary based on transcripts in training stm
    character_count = 0
    for item in stm_file:
        # Extract transcript and convert to character list
        transcript = list(item[-1])
        for character in transcript:
            # Add character to vocab if not already added
            if (character not in vocab_dict.values()):
                vocab_dict[character_count] = character
                character_count += 1

    return vocab_dict

"""### 3. Lets train something

As already mentioned you will be training a CTC-model today.
One of the most classic models used for this task is probably the Wav2Vec2 model however doing what everyone does is boring.

We will use parts of the Whisper model here more specfically the encoder.
Whisper was trained on massive amount of data shows what can be accomplished with these amounts of data.

[Link to Whisper Paper](https://cdn.openai.com/papers/whisper.pdf)

However Whisper is an Encoder-Decoder based Sequnce-to-sequence model. This is why we will be using only the Encoder part. Otherwise the model would take to much GPU memory during training.

The whisper model was trained with audios of length 30 sec with sample rate 16000Hz and 80 bin log-Mel features. Luckily our data already has the correct sampel rate.

3.1 In order to accomplish this task you will need to define the Wav2VecCTCTokenizer and WhisperFeatureExtractor.



*   WhisperFeatureExtractor: This will zero pad our audios to 30 seconds length and afterwards extract our 80 bin log-Mel features. As such all inputs will have the same shape and no further masking will be needed for our inputs. 
*   Wav2VecCTCTokenizer: This will encode our transcripts into appropriate tokens defined in the parameters during intialization. It also provides the decoding algorithm for CTC predictions.
"""

def get_vocab_size(vocab_path):
    #TODO return len of vocabulary. Wav2Vec2CTCTokenizer needs this.
    with open(vocab_path) as json_file:
        vocab = json.load(json_file)
    
    return len(vocab)

def get_processors(vocab_file):
  whisper_x = "openai/whisper-tiny"
  vocab_size = get_vocab_size(vocab_file)

  feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_x)
  tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", vocab_size=vocab_size)
  
  return feature_extractor, tokenizer

"""3.2 We will be using the Huggingface training pipeline. In order to use that we need to initialize our training and validation datasets.

"""

def get_datasets(tr_stm, val_stm):
  tr_dataset = AudioDataset(tr_stm)
  val_dataset = AudioDataset(val_stm)

  return tr_dataset, val_dataset

"""3.3 As already mentioned we need to define a custom model in order to add a couple of layers on the encoder output of the whisper model.

"""
class WhisperForCTC(WhisperPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.version",
        r"decoder.version",
        r"proj_out.weight",
    ]
    _keys_to_ignore_on_save = [
        r"proj_out.weight",
    ]

    def __init__(self, config):
        super().__init__(config)


        self.model = WhisperModel(config)

        self.dropout = nn.Dropout(config.final_dropout)

        self.feature_transform = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn1', nn.BatchNorm1d(config.hidden_size)),
                ('activation1', nn.LeakyReLU()),
                ('drop1', nn.Dropout(config.final_dropout)),
            ]))
        
        self.lm_head = nn.Linear(config.d_model, config.custom_vocab)

        self.config = config

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        head_mask = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO run the encoder of the whisper model
        enc_outputs = self.model.encoder(
                      input_features,
                      head_mask=head_mask,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                      return_dict=return_dict,
                      )
        
        hidden_states = enc_outputs[0] # TODO pass only hidden_states of the enc_outputs
        
        hidden_states = self.dropout(hidden_states)

        # TODO pass hidden_states  through self.feature_transform. 
        # make sure you understand what each dimension of hidden_states represents. E.x batch_size,...
        B, T, F = hidden_states.size()
        hidden_states = hidden_states.view(B * T, F)
        hidden_states = self.feature_transform(hidden_states)
        hidden_states = hidden_states.view(B, T, F)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
            
            input_lengths = torch.tensor([1500] * B) #B=Batch_size

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0  #TODO generate label_mask with values true if labels is not padded.
      
            target_lengths = labels_mask.sum(-1) #Get length of each target in batch 

            flattened_targets = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1) #TODO calculate log_softmax over your probabilities and put result in correct shape for ctc_loss 
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets.cpu(),
                    input_lengths.cpu(),
                    target_lengths.cpu(),
                    blank=self.config.pad_token_id,
                    reduction="mean",#self.config.ctc_loss_reduction,
                )

        if not return_dict:

            output = (logits,) +enc_outputs[0:]  #enc_outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=enc_outputs.hidden_states, attentions=enc_outputs.attentions
        )

def get_model(vocab_file):
  whisper_x = "openai/whisper-tiny"

  vocab_size = get_vocab_size(vocab_file)

  model_orig = WhisperForConditionalGeneration.from_pretrained(
          whisper_x,
  )
  model_orig.config.forced_decoder_ids = None
  model_orig.config.suppress_tokens = []
  model_orig.config.use_cache = False
  model_orig.config.custom_vocab = vocab_size
  model_orig.config.final_dropout = 0.2
  model_orig.config.pad_token_id= vocab_size-1 #This should be the id of the [PAD] token in your vocab
  #print(model_orig.config)
  model = WhisperForCTC(model_orig.config)

  missing_keys, unexpected_keys = model.load_state_dict(model_orig.state_dict(),strict=False)
  print("Missing keys: {} Unexpected keys: {}".format(missing_keys, unexpected_keys))

  return model

"""3.4 

Lokking at the Huggingface docu we see that we also need to define a Collator. The collator gets passed a batch as input.
The batch consits of elements given by **\_\_getitem__** in our Dataset.
"""
@dataclass
class CollatorForCTC:
    feature_extractor: WhisperFeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    fp16: Optional[bool] = False
    return_utt_ids: Optional[bool] = False
    return_cleartext: Optional[bool] = False

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        uid = [b[0] for b in batch]
        src = [b[1] for b in batch]
        tgt = [b[2] for b in batch]

        # TODO extract features using feature_extractor we want pytorch tensors
        batch = self.feature_extractor(src, sampling_rate=16000, return_tensors='pt')
        
        # TODO tokenize target transcript using self.tokenizer we want them to be padded and pytorch tensors
        tgt_batch = self.tokenizer(tgt, padding=True, return_tensors='pt')

        # TODO tgt_batch contains several keys we want the padded tokens to be replaced with the value -100 
        #labels = tgt_batch["input_ids"]
        #labels[labels==self.tokenizer.pad_token_id] = -100
        labels = tgt_batch["input_ids"].masked_fill(tgt_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        if self.return_utt_ids:
            batch['utt_ids'] = uid
        if self.return_cleartext:
            batch['text'] = tgt
            
        return batch

if __name__ == '__main__':
    vocab_dict = {}

    # Read in training file
    lines =  list(open('train.stm'))

    tr_stm = []

    # Format list
    for l in lines:
        # Split columns
        l = l.split("\t")
        # Delete \n at the end of transcript
        l[-1] = l[-1][:-1]
        # Convert audio length to float
        l[4] = float(l[4])

        tr_stm.append(l)

    text_idx = [item[0] for item in tr_stm]

    vocab_dict = do_vocab(tr_stm, text_idx)

    #TODO DO steps described in 2.1
    # Replace " " (space) with |
    key = [k for k, v in vocab_dict.items() if v == ' '][0]
    vocab_dict[key] = '|'
    # Add [UNK] token
    if ('[UNK]' not in vocab_dict.values()):
        vocab_dict[len(vocab_dict)] = '[UNK]'
    # Add [PAD] token
    if ('[PAD]' not in vocab_dict.values()):
        vocab_dict[len(vocab_dict)] = '[PAD]'

    # Switch keys and values
    vocab_dict = {y: x for x, y in vocab_dict.items()}

    # Save vocab in json file
    with open('vocab_dict.json', 'w') as fp:
        json.dump(vocab_dict, fp)

    vocab_file='vocab_dict.json'
    feature_extractor, tokenizer = get_processors(vocab_file)

    model = get_model(vocab_file)
    feature_extractor.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    model.save_pretrained("./model")

    tr_stm = 'train.stm'
    val_stm = 'dev-libri_time.stm'
    tr_dataset, val_dataset = get_datasets(tr_stm, val_stm)

    data_collator = CollatorForCTC(feature_extractor=feature_extractor, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="model",
        #group_by_length=args.group_by_length,
        #length_column_name="audio_length",
        per_device_train_batch_size=15,
        per_device_eval_batch_size=15,
        evaluation_strategy=IntervalStrategy.STEPS, #steps
        num_train_epochs=10,
        fp16=False,
        gradient_checkpointing=False,
        save_steps=1000,
        save_strategy = "steps",  ##use custom callback for save-x-best set to no
        load_best_model_at_end=True,
        metric_for_best_model = 'loss', #'wer',
        greater_is_better=False,
        eval_steps=1000,
        logging_dir='logs',
        logging_steps=25,
        learning_rate=0.0001,
        weight_decay=0.005,
        warmup_steps=500,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=5,
        save_total_limit=2,
        dataloader_num_workers=6,
        remove_unused_columns=False,
        push_to_hub=False
        )

    tr_dataset.__len__

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        #compute_metrics=compute_metrics_wer,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        )

    trainer.train()