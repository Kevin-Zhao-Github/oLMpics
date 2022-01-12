import torch
import transformers


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "t5-3b"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

aff = ["A trout is", "A salmon is", "An ant is", "A bee is", "A robin is", "A sparrow is", "An oak is", "A pine is", "A rose is", "A daisy is", "A carrot is", "A pea is", "A hammer is", "A saw is", "A car is", "A truck is", "A hotel is", "A house is"]
aff_a = []
aff_an = []
neg_a = []
neg_an = []

for question in aff:
    aff_a.append(question + " a <extra_id_0>.")
    aff_an.append(question + " an <extra_id_0>.")
    neg_a.append(question + " not a <extra_id_0>.")
    neg_an.append(question + " not an <extra_id_0>.")


for i in range(len(aff)):
    input_ids = torch.tensor([tokenizer.encode(aff_a[i], add_special_tokens=True)]).to(device)
    with torch.no_grad():
        decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        predictions_aff_a = model(input_ids, decoder_input_ids=decoder_ids).logits
    input_ids = torch.tensor([tokenizer.encode(neg_a[i], add_special_tokens=True)]).to(device)
    with torch.no_grad():
        decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        predictions_neg_a = model(input_ids, decoder_input_ids=decoder_ids).logits

    aff_a_preds = []
    predictions_aff_a = torch.softmax(predictions_aff_a[0, 1], dim=0)  # 1 is position of <extra_id_0>
    top_inds = torch.argsort(predictions_aff_a,descending=True)[:5].cpu().numpy()
    for top_ind in top_inds:
        aff_a_preds.append(tokenizer.decode([top_ind]))

    neg_a_preds = []
    predictions_neg_a = torch.softmax(predictions_neg_a[0, 1], dim=0)  # 1 is position of <extra_id_0>
    top_inds = torch.argsort(predictions_neg_a,descending=True)[:5].cpu().numpy()
    for top_ind in top_inds:
        neg_a_preds.append(tokenizer.decode([top_ind]))

    print(aff_a_preds, neg_a_preds)

