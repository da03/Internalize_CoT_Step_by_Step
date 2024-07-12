from datasets import load_dataset


for language in ['eng',]:
    data = load_dataset('masakhane/afrimgsm', language)['test']
    for example in data:
        print (example['question'])
        print (example['answer_number'])

