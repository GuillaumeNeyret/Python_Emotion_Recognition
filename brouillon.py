import random
from collections import Counter

def random_list(min,max, nb_items):
    res = [random.randint(min, max) for _ in range(nb_items)]
    return res

def int_to_emotion(lst, label):
    res = [label[i] for i in lst]
    return res

def freq_items(list, label):
    # my_dict = {label[i]: list.count(i) for i in range(0,len(label))}
    my_dict = {i: list.count(i) for i in range(0, len(label))}
    return my_dict

# def freq_items2(list, label):
#     return dict(zip(label, Counter(list)))

def freq_items2(lst, label):
    counter = Counter(lst)
    return {l: counter[l] for l in label}

labels1 = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
toto = random_list(0,4,10)

print(toto)
print(labels1)
emotions = int_to_emotion(toto,labels1)
print('traduction :',emotions)
print(freq_items(toto,labels1))
lulu = Counter(emotions)
print(lulu)
print("Surprise :", lulu['Surprise'])

print('-----------------------')
tata = Counter(toto)
for i in range(0,len(labels1)):
    print(labels1[i]+' :'+str(tata[i]))


