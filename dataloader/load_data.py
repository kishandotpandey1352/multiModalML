from torch.utils.data import DataLoader
from text_dataloader import TextDataset

dataset = TextDataset(csv_file='text_data.csv', text_column='Text')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    raw_texts = batch['text']
    print('raw text from dataloader : ',raw_texts)
    print('batch from dataloader : ',batch)
    # model(input_ids, attention_mask)