import re
'''
teste="Emotional connection to a child rather than an adult may lead to child sexual abuse ChildOrientedSexuality. Providing support to people with sexual interest in children is key in preventing CSA. CSAPE will produce material &amp; training on the phenomenon.\nhttps://t.co/xasK8B9PlV"
teste= "إيران: قوات الأمن تقتل وتعذّب وتعتدي على الأطفال https://t.co/UvcvVK43k4"

hash=re.sub(r'\n',r' ', teste)
hash=re.search(r'(?<!https\?:\/\/)(?<!www\.)[^ \n]+(?: [^ \n]+)*', hash)

text_original = "https://t.co/xasK8B9PlV Emotional connection to a child rather than an adult may lead to child sexual abuse ChildOrientedSexuality. Providing support to people with sexual interest in children is key in preventing CSA. CSAPE will produce material &amp; training on the phenomenon.\nhttps://t.co/xasK8B9PlV"
text_original= "إيران: قوات الأمن تقتل وتعذّب وتعتدي على الأطفال https://t.co/UvcvVK43k4"
text_wanted = ' '.join(re.findall(r'(?:https?:\/\/|www\.)\S+|(\S+)', text_original))

links= re.findall(r'(?:https?:\/\/|www\.)\S+', text_original)
print(text_wanted)
print(links)

'''

with open('tweets_april.json', 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

with open('output_file.json', 'w',encoding='utf-8') as output_file:
    output_file.writelines(lines[68914:])
