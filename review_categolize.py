from transformers import pipeline

# clf = pipeline("sentiment-analysis")
# print(clf(["The check-in took forever and the room was not clean.", "The room was clean and the check-in was fast."]))

labels = ["cleanliness","staff","facility","noise","location","price/value","food","checkin/checkout","wifi","other"]
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The room was clean but the staff was rude at check-in."
print(zs(text, labels))
