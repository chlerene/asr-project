import argparse
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer
import torchaudio
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder, download_pretrained_files
from typing import List
from model import AudioDataset, WhisperForCTC, CollatorForCTC 

parser = argparse.ArgumentParser(description='asr-project')

parser.add_argument('--model', help="path to model to evalute", type=str, required=True)
parser.add_argument('--processor', help="path to processor", type=str, default=None)
parser.add_argument('--stm', help="test stm: uid,wavpath,from,to,length,text", type=str, default=None)
parser.add_argument('--vocab', help="vocab for decoding", type=str, default=None)
parser.add_argument('--input-name', help="what is the key for in src", default="input_values")

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=29):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()

class CustomLM(CTCDecoderLM):
    def __init__(self, lm: np.array):
        CTCDecoderLM.__init__(self)
        self.lm = lm
        self.sil = -1  # index for silent token in the language model
        self.states = {}

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        score = np.amax(self.lm[self.sil])

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = np.amax(self.lm[token_index-1])
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load model
    model = WhisperForCTC.from_pretrained(args.model)
    
    # Load processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.processor)
    tokenizer = AutoTokenizer.from_pretrained(args.processor)
    processor = (feature_extractor, tokenizer)

    # Load test set  and create data_collator + data_loader
    data = AudioDataset(args.stm)
    data_collator = CollatorForCTC(feature_extractor=processor[0], tokenizer=processor[1], fp16=True, return_cleartext=True, return_utt_ids=True)
    data_loader = DataLoader(data, collate_fn=data_collator, num_workers=2, pin_memory=False)

    model.to("cuda")
    model.eval()

    # Get vocab
    with open(args.vocab) as json_file:
        vocab = json.load(json_file)
    
    # Create own LM 
    content = np.loadtxt("n-grams/1-gram.txt")
    custom_lm = CustomLM(content)

    # Load pretained LM
    files = download_pretrained_files("librispeech-4-gram")

    # Construct beam search decoder
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26

    beam_search_decoder = ctc_decoder(
        #lexicon=files.lexicon,
        lexicon="lexicon.txt",
        tokens=list(vocab),
        #lm=files.lm,
        lm=custom_lm,
        nbest=3,
        beam_size=1500,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
        blank_token = "[PAD]",
        unk_word= "[UNK]"
    )

    # Construct greedy decoder
    greedy_decoder = GreedyCTCDecoder(list(vocab))

    bs_wer_dataset = 0.0
    greedy_wer_dataset = 0.0

    sample_count = 1

    with torch.no_grad():
        for batch in data_loader:
            lst = batch['text']
            uids = batch["utt_ids"]
            src =  batch[args.input_name]
            labels = batch["labels"]
 
            # Calculate emission with acoustic model
            logits = model(src.to("cuda")).logits
            
            # Decode with beam search decoder
            beam_search_result = beam_search_decoder(logits.cpu())
            beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
            
            # Decode with greedy decoder
            greedy_result = greedy_decoder(logits[0].cpu())
            greedy_transcript = " ".join(greedy_result)

            # Calculate WER for beam search decoder and greedy decoder (per sample)
            bs_wer = torchaudio.functional.edit_distance(lst[0].split(), beam_search_transcript.split()) / len(lst[0].split())
            greedy_wer = torchaudio.functional.edit_distance(lst[0].split(), greedy_transcript.split()) / len(lst[0].split())

            print(f"{sample_count}/{len(data)}:")
            sample_count += 1
            print(f"Actual Transcript: {lst[0]}")
            print(f"BS Transcript: {beam_search_transcript}, BS WER: {bs_wer}")
            print(f"Greedy Transcript: {greedy_transcript}, Greedy WER: {greedy_wer}\n")
            
            bs_wer_dataset += bs_wer
            greedy_wer_dataset += greedy_wer

    # Calculate WER for beam search decoder and greedy decoder (whole test set)
    bs_wer_dataset /= sample_count
    greedy_wer_dataset /= sample_count

    print(f"BS WER Dataset: {bs_wer_dataset}")
    print(f"Greedy WER Dataset: {greedy_wer_dataset}")
