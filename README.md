# AICUP

Description
本項目是一個使用 EleutherAI 的 Pythia模型進行自然語言處理的示例。


### 上手指南

Google Colab上安裝和配置Python環境，以運行基於Python 3.x的機器學習和自然語言處理腳本。



###### 開發前的配置要求

平台: 推薦使用Google Colab，因為它提供了大量的計算資源。
Python版本: Python 3.x。

###### **安裝步驟**

!pip install transformers
!pip install datasets


######**模型選擇**
from transformers import AutoTokenizer, AutoModelForCausalLM

plm = "EleutherAI/pythia-410m-deduped" #此處可以修改你想要的模型
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep ='\n\n####\n\n'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}

tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.padding_side = 'left'

from datasets import load_dataset, Features, Value
dataset = load_dataset("csv", data_files="opendid_set1.tsv", delimiter='\t',
                       features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                       column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)

######**模型預測**
主要用於測試訓練好的模型，測試它在給定的seed text基礎上生成文本。這個過程可以幫助評估模型的表現。
import torch
from tqdm import tqdm#, tqdm_notebook
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_text(model, tokenizer, seed, n_words=20):
    model = model.to(device)
    model.eval()
    text = tokenizer.encode(seed)
    inputs, past_key_values = torch.tensor([text]), None
    with torch.no_grad():
        for _ in tqdm(range(n_words)):
            out = model(inputs.to(device), past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            text.append(inputs.item())
            if tokenizer.decode(inputs.item()) == eos:
                break


    return tokenizer.decode(text)

sample_text(model, tokenizer, seed=f"{bos} DR AADLAND ABRAHAM {sep}")

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

EPOCHS = 5 # 設定你的訓練次數
optimizer = AdamW(model.parameters(),lr=5e-5)

steps = len(bucket_train_dataloader)
total_steps = steps * EPOCHS
print(steps, total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps*0.1,
    num_training_steps=total_steps
)

model.resize_token_embeddings(len(tokenizer))
model.to(device)
print(f'Total numbers of steps: {total_steps}')
model


from tqdm import tqdm,trange

global_step = 0
total_loss = 0

model.train()
for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    total_loss = 0

    predictions , true_labels = [], []

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels)#, attention_mask=masks)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files="opendid_valid.tsv", delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])
valid_list

from tqdm import tqdm#, tqdm_notebook

tokenizer.padding_side = 'left'
def sample_batch(model, tokenizer, input):
    """Generate text from a trained model."""
    model.eval()
    seeds = [f"{bos} {text['content']} {sep}" for text in input]
    texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
    outputs = []
    #return
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = PAD_IDX,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        for idx , pred in enumerate(preds):
            pred = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            if pred == "PHI: NULL":
                continue
            phis = pred.split('\n')
            lidxs = {}
            for p in phis:
                tid = p.find(':')
                if tid > 0:
                    text = p[tid+1:].strip()
                    nv = text.find('=>')
                    normalizedV = None
                    # 處理時間正規化
                    # YOU IMPLEMENTATION
                    #
                    #
                    #
                    lidx = 0
                    if text in lidxs:
                        lidx = lidxs[text]
                    lidx = input[idx]['content'].find(text, lidx)
                    eidx = lidx+len(text)
                    lidxs[text] = eidx
                    sidx=int(input[idx]['idx'])
                    if normalizedV is None:
                        outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}')
                    else:
                        outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}\t{normalizedV}')
    return outputs

f = open("answer.txt", "w")
BATCH_SIZE = 8
for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
    with torch.no_grad():
        seeds = valid_list[i:i+BATCH_SIZE]
        outputs = sample_batch(model, tokenizer, input=seeds)
        for o in outputs:
            f.write(o)
            f.write('\n')
f.close()

### 作者
c110154217@nkust.edu.tw
c110154244@nkust.edu.tw
https://colab.research.google.com/drive/1lXSfSHWIK8y7COfMTSMNjVgU3O9-B5bG?authuser=1#scrollTo=iDaHQrOkhzPQ
 *您也可以在貢獻者名單中參看所有參與該專案的開發者。*

### 版權說明

該項目簽署了MIT 授權許可，詳情請參閱 [LICENSE.txt](https://github.com/your_github_name/your_repository/blob/master/LICENSE.txt)



