import torch
import transformers


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "t5-3b"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

input_ids = torch.tensor([tokenizer.encode('"Checkmate," Rosaline announced with glee.  She was getting to be really good at <extra_id_0>.', add_special_tokens=True)]).to(device)

print(input_ids)
print(tokenizer.decode(input_ids[0]))

print("---MLM Method---")
with torch.no_grad():
    decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    predictions = model(input_ids, decoder_input_ids=decoder_ids).logits
    predictions = torch.softmax(predictions[0, 1], dim=0)  # 1 is position of <extra_id_0>
    top_inds = torch.argsort(predictions,descending=True)[:15].cpu().numpy()
    for top_ind in top_inds:
        print(tokenizer.decode([top_ind]))

