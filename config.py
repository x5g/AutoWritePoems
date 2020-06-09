class Config(object):
    poetry_file = 'poetry4min.txt'
    weight_file = 'poetry_model_4_1.h5'
    # 根据前六个字预测第七个字
    max_len = 4
    batch_size = 128
    learning_rate = 0.0005
