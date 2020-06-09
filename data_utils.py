def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='UTF-8') as f:
        for line in f:
            x = line.strip() + "]"
            x = x.split(":")[1]
            if len(x) <= 7:
                continue
            if x[7] == '，':
                files_content += x

    # 删除标点
    # files_content = files_content.replace('，', '').replace('。', '').replace('？', '').replace('！', '')

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    Word_Pairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*Word_Pairs)
    # words += (" ",)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    Word_2numF = lambda x: word2num.get(x, len(words) - 1)
    return Word_2numF, num2word, words, files_content

if __name__ == '__main__':
    from config import Config
    Config.poetry_file = 'poetry.txt'
    Word_2numF, num2word, words, files_content = preprocess_file(Config)

    print(files_content)
    print(num2word)
    print(words)
    print(Word_2numF('乃'))

    poems = files_content.split(']')
    with open('poetry6min1.txt', 'a+') as file:
        for poem in poems:
            file.write(poem + '\n')
