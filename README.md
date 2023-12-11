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


######**模型選擇**
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

### 作者
c110154217@nkust.edu.tw
c110154244@nkust.edu.tw
https://colab.research.google.com/drive/1lXSfSHWIK8y7COfMTSMNjVgU3O9-B5bG?authuser=1#scrollTo=iDaHQrOkhzPQ
 *您也可以在貢獻者名單中參看所有參與該專案的開發者。*

### 版權說明

該項目簽署了MIT 授權許可，詳情請參閱 [LICENSE.txt](https://github.com/your_github_name/your_repository/blob/master/LICENSE.txt)



