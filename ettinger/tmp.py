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

print(aff_a)
print(aff_an)
print(neg_a)
print(neg_an)
