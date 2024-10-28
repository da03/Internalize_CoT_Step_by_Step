
from torch.utils.data import Dataset, DataLoader

from deq_reasoning.dataset.digit_multiplication import CustomTokenizer, CoTDataset, CoTDataCollator, vocab, char2id, id2char

def load_dataloader(config):
    if config.dataset.name == "digit_multiplication":

        tokenizer = CustomTokenizer(vocab, char2id, id2char)

        vocab_size = tokenizer.vocab_size
        max_seq_len = config.max_length

        train_dataset = CoTDataset(tokenizer, config.train_path, max_length=config.max_length, max_size=config.max_size, with_cot=config.dataset.cot)
        train_data_collator = CoTDataCollator(tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_data_collator)

        val_dataset = CoTDataset(tokenizer, config.val_path, max_length=config.max_length)
        val_data_collator = CoTDataCollator(tokenizer=tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=val_data_collator)

        return tokenizer, train_loader, val_loader, val_dataset
    elif config.dataset.name == "":
        pass
    else:
        raise NotImplementedError(f"Dataset {config.dataset} is not implemented.")
